import os, glob, functools, fnmatch, io, shutil
from zipfile import ZipFile
from itertools import product
import numpy as np
import pandas as pd
from PIL import Image


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl

import rasterio
from rasterio import features, mask

import geopandas as gpd

import tensorflow as tf
from tensorflow.python.keras import layers, losses, models
from tensorflow.python.keras import backend as K  
import tensorflow_addons as tfa

from tensorflow_examples.models.pix2pix import pix2pix
from segmentation_models.metrics import iou_score
from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import datetime

from focal_loss import SparseCategoricalFocalLoss
from sklearn.metrics import confusion_matrix, f1_score
import skimage.io as skio

from time import sleep
from tqdm.notebook import tqdm

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("physical devices: ", physical_devices)

root_dir = '/home/ubuntu/data/'
workshop_dir = '/home/ubuntu/models/'

img_dir = os.path.join(root_dir,'tiled_images/')
label_dir = os.path.join(root_dir,'tiled_labels/')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

x_test_filenames_partition_fn = os.path.join(root_dir,'x_test_filenames_partition.txt')
y_test_filenames_partition_fn = os.path.join(root_dir,'y_test_filenames_partition.txt')

def get_test_lists(imdir, lbldir):
  imgs = glob.glob(os.path.join(imdir,"*.png"))
  dset_list = []
  for img in imgs:
    filename_split = os.path.splitext(img) 
    filename_zero, fileext = filename_split 
    basename = os.path.basename(filename_zero) 
    dset_list.append(basename)

  x_filenames = []
  y_filenames = []
  for img_id in dset_list:
    x_filenames.append(os.path.join(imdir, "{}.png".format(img_id)))
    y_filenames.append(os.path.join(lbldir, "{}.png".format(img_id)))

  print("number of images: ", len(dset_list))
  return dset_list, x_filenames, y_filenames

if os.path.isfile(fn) for fn in [x_test_filenames_partition_fn, y_test_filenames_partition_fn]:
  x_test_filenames = [line.strip() for line in open(os.path.join(root_dir,'x_test_filenames_partition.txt'), 'r')]
  y_test_filenames = [line.strip() for line in open(os.path.join(root_dir,'y_test_filenames_partition.txt'), 'r')]
else:
  test_list, x_test_filenames, y_test_filenames = get_test_lists(img_dir, label_dir)

print("!!!!! number of images: ", len(x_test_filenames))

# set input image shape
img_shape = (96, 96, 3)
# set batch size for model
batch_size = 2 

EPOCHS = 5

# Function for reading the tiles into TensorFlow tensors 
# See TensorFlow documentation for explanation of tensor: https://www.tensorflow.org/guide/tensor
def _process_pathnames(fname, label_path):
  # We map this function onto each pathname pair 
  print("FILENAME: ", fname)
  img_str = tf.io.read_file(fname)
  img = tf.image.decode_png(img_str, channels=3)

  label_img_str = tf.io.read_file(label_path)

  # These are png images so they return as (num_frames, h, w, c)
  label_img = tf.image.decode_png(label_img_str, channels=1)
  # The label image should have any values between 0 and 8, indicating pixel wise
  # foreground class or background (0). We take the first channel only. 
  label_img = label_img[:, :, 0]
  label_img = tf.expand_dims(label_img, axis=-1)
  return img, label_img

# Function to augment the data with horizontal flip
def flip_img_h(horizontal_flip, tr_img, label_img):
  if horizontal_flip:
    flip_prob = tf.random.uniform([], 0.0, 1.0)
    tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                lambda: (tr_img, label_img))
  return tr_img, label_img

# Function to augment the data with vertical flip
def flip_img_v(vertical_flip, tr_img, label_img):
  if vertical_flip:
    flip_prob = tf.random.uniform([], 0.0, 1.0)
    tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_up_down(tr_img), tf.image.flip_up_down(label_img)),
                                lambda: (tr_img, label_img))
  return tr_img, label_img

# Function to augment the images and labels
def _augment(img,
             label_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=None,  # Scale image e.g. 1 / 255.
             horizontal_flip=False,
             vertical_flip=False): 
  if resize is not None:
    # Resize both images
    label_img = tf.image.resize(label_img, resize)
    img = tf.image.resize(img, resize)
  
  img, label_img = flip_img_h(horizontal_flip, img, label_img)
  img, label_img = flip_img_v(vertical_flip, img, label_img)
  img = tf.cast(img, tf.float32) 
  if scale is not None:
    img = tf.cast(img, tf.float32) * scale
    #img = tf.keras.layers.Rescaling(scale=scale, offset=-1)
  #label_img = tf.cast(label_img, tf.float32) * scale
  #print("tensor: ", tf.unique(tf.keras.backend.print_tensor(label_img)))
  return img, label_img

# Main function to tie all of the above four dataset processing functions together 
def get_baseline_dataset(filenames, 
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5, 
                         batch_size=batch_size,
                         shuffle=True):           
  num_x = len(filenames)
  # Create a dataset from the filenames and labels
  print("BSL: ", filenames[0:10], labels[0:10])
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  print("dataset: ", dataset)
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
  if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
    assert batch_size == 1, "Batching images must be of the same size"

  dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
  
  if shuffle:
    dataset = dataset.shuffle(num_x)
  
  
  # It's necessary to repeat our data for all epochs 
  dataset = dataset.repeat().batch(batch_size)
  return dataset


# dataset configuration for testing
test_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
}
test_preprocessing_fn = functools.partial(_augment, **test_cfg)

print(x_test_filenames[0:10])
print(y_test_filenames[0:10])

test_ds = get_baseline_dataset(x_test_filenames,
                              y_test_filenames, 
                              preproc_fn=test_preprocessing_fn,
                              batch_size=batch_size)

# reset the forground list to capture the test images
foreground_list_x = []
foreground_list_y = []
for x,y in zip(x_test_filenames, y_test_filenames): 
    try:
      filename_split = os.path.splitext(y) 
      filename_zero, fileext = filename_split 
      basename = os.path.basename(filename_zero) 
      img = np.array(Image.open(y))
      if img.max()>0:
      #if basename not in background_list_train:
        foreground_list_x.append(x)
        foreground_list_y.append(y)
      else:
        continue
    except:
      continue

num_foreground_examples = len(foreground_list_y)
print("num_foreground_examples: ", num_foreground_examples)

display_num = 1
r_choices = np.random.choice(num_foreground_examples, 1) #num_foreground_examples, 1)
for i in range(0, display_num * 2, 2):
  img_num = r_choices[i // 2]

temp_ds = get_baseline_dataset(foreground_list_x[img_num:img_num+1], 
                               foreground_list_y[img_num:img_num+1],
                               preproc_fn=test_preprocessing_fn,
                               batch_size=1,
                               shuffle=False)

# Let's examine some of these augmented images

iterator = iter(temp_ds)
next_element = iterator.get_next()

batch_of_imgs, label = next_element

# Running next element in our graph will produce a batch of images

sample_image, sample_mask = batch_of_imgs[0], label[0,:,:,:]


# Optional, you can load the model from the saved version
load_from_checkpoint = True
if load_from_checkpoint == True:
  save_model_path = os.path.join(workshop_dir,'model_out_batch_{}_ep{}_pretrain_focalloss/'.format(batch_size, EPOCHS))
  model = tf.keras.models.load_model(save_model_path, custom_objects={"loss": SparseCategoricalFocalLoss, "iou_score": iou_score})
else:
  print("inferencing from in memory model")


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def get_predictions(image= None, dataset=None, num=1):
  if image is None and dataset is None:
    return ValueError("At least one of image or dataset must not be None.")
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      return pred_mask
  else:
    pred_mask = create_mask(model.predict(image[tf.newaxis, ...]))
    pred_mask = tf.keras.backend.eval(pred_mask)
    return pred_mask

tiled_prediction_dir = os.path.join(root_dir,'predictions_test_focal_loss/')
if not os.path.exists(tiled_prediction_dir):
    os.makedirs(tiled_prediction_dir)
    
pred_masks = []
pred_paths = []
true_masks = []

for i in range(0, len(x_test_filenames)):
    img_num = i

    try:
      temp_ds = get_baseline_dataset(x_test_filenames[img_num:img_num+1], 
                                   y_test_filenames[img_num:img_num+1],
                                   preproc_fn=test_preprocessing_fn,
                                   batch_size=1,
                                   shuffle=False)
    except Exception as e: 
      print(str(e))

    # Let's examine some of these augmented images

    iterator = iter(temp_ds)
    next_element = iterator.get_next()

    batch_of_imgs, label = next_element

    # Running next element in our graph will produce a batch of images
    image, mask = batch_of_imgs[0], label[0,:,:,:]
    mask_int = tf.dtypes.cast(mask, tf.int32)
    true_masks.append(mask_int)
    print(y_test_filenames[img_num:img_num+1])
    print(np.unique(mask_int))

    # run and plot predictions
    pred_mask = get_predictions(image)
    pred_masks.append(pred_mask)
    
    # save prediction images to file

    filename_split = os.path.splitext(x_test_filenames[img_num]) 
    filename_zero, fileext = filename_split 
    basename = os.path.basename(filename_zero) 
    pred_path = os.path.join(tiled_prediction_dir, "{}.png".format(basename))
    pred_paths.append(pred_path)
    tf.keras.preprocessing.image.save_img(pred_path,pred_mask, scale=False) # scaling is good to do to cut down on file size, but adds an extra dtype conversion step.    

path_df = pd.DataFrame(list(zip(x_test_filenames, y_test_filenames, pred_paths)), columns=["img_names", "label_names", "pred_names"])
path_df.to_csv(os.path.join(root_dir, "test_file_paths.csv"))

