import datetime, os, fnmatch, functools, io, glob, random, shutil
from itertools import product
from time import sleep
from zipfile import ZipFile

from focal_loss import SparseCategoricalFocalLoss
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import skimage.io as skio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl

from rasterio import features, mask

from segmentation_models.metrics import iou_score
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.python.keras import layers, losses, models
from tensorflow.python.keras import backend as K  
from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback
from tqdm.notebook import tqdm

tfds.disable_progress_bar()

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("physical devices: ", physical_devices)

ROOT_DIR = '/home/ubuntu/data/'
OUTPUT_DIR = '/home/ubuntu/models/'

IMG_DIR = os.path.join(ROOT_DIR,'tiled_images/') 
LABEL_DIR = os.path.join(ROOT_DIR,'tiled_labels/') 

# input image shape
IMG_SHAPE = (96, 96, 3)
# batch size for model
BATCH_SIZE = 8

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

def get_train_test_lists(imdir, lbldir):
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
    y_filenames.append(os.path.join(lbldir, "RDSISC4_{}_classified{}.png".format(img_id[:-4], img_id[-4:])))

  print("number of images: ", len(dset_list))
  return dset_list, x_filenames, y_filenames

train_list_fn = os.path.join(ROOT_DIR,"train_list_filtered_07.txt")
x_train_filenames_fn = os.path.join(ROOT_DIR,'x_train_filenamesfiltered_07.txt')
y_train_filenames_fn = os.path.join(ROOT_DIR,'y_train_filenamesfiltered_07.txt')

bad_groundtruth_examples = [line.strip() for line in open(os.path.join(ROOT_DIR,"flagged_melt_pcts_fns.csv"), 'r')]

# CODE USED FOR FILTERING
"""
x_train_filenames = []
y_train_filenames = []

for img_id in train_list_fn:
    if "RDSISC4_{}_classified{}".format(img_id[:-4], img_id[-4:]) in bad_groundtruth_examples:
        print("flagged image: ", img_id)
        train_list.remove(img_id)
    else:
        continue
    x_train_filenames.append(os.path.join(IMG_DIR, "{}.png".format(img_id)))
    y_train_filenames.append(os.path.join(LABEL_DIR, "RDSISC4_{}_classified{}.png".format(img_id[:-4], img_id[-4:])))

with open(os.path.join(ROOT_DIR,'train_list_filtered_07.txt'), 'w') as f:
   for item in train_list:
       f.write("%s\n" % item)

with open(os.path.join(ROOT_DIR,'x_train_filenames_filtered_07.txt'), 'w') as f:
   for item in x_train_filenames:
       f.write("%s\n" % item)

with open(os.path.join(ROOT_DIR,'y_train_filenames_filtered_07.txt'), 'w') as f:
   for item in y_train_filenames:
       f.write("%s\n" % item)

"""
try:
  train_list = [line.strip() for line in open(train_list_fn, 'r')]
  x_train_filenames = [line.strip() for line in open(x_train_filenames_fn, 'r')]
  y_train_filenames = [line.strip() for line in open(y_train_filenames_fn, 'r')]
except:
  train_list, x_train_filenames, y_train_filenames = get_train_test_lists(IMG_DIR, LABEL_DIR)
  with open(os.path.join(ROOT_DIR,'train_list_filtered_07.txt'), 'w') as f:
    for item in train_list:
        f.write("%s\n" % item)

  with open(os.path.join(ROOT_DIR,'x_train_filenames_filtered_07.txt'), 'w') as f:
    for item in x_train_filenames:
        f.write("%s\n" % item)

  with open(os.path.join(ROOT_DIR,'y_train_filenames_filtered_07.txt'), 'w') as f:
    for item in y_train_filenames:
        f.write("%s\n" % item)


print("number of images: ", len(train_list))


skip = False

if not skip:
  background_list_train = []
  for i in train_list: 
      # read in each labeled images
      img = np.array(Image.open(os.path.join(LABEL_DIR,"RDSISC4_{}_classified{}.png".format(i[:-4], i[-4:]))))  
      # check if no values in image are greater than zero (background value)
      if img.max()==0:
          background_list_train.append(i)

  print("Number of background images: ", len(background_list_train))

  with open(os.path.join(ROOT_DIR,'background_list_train.txt'), 'w') as f:
    for item in background_list_train:
        f.write("%s\n" % item)

else:
  background_list_train = [line.strip() for line in open(os.path.join(ROOT_DIR,"background_list_train.txt"), 'r')]
  print("Number of background images: ", len(background_list_train))

background_removal = len(background_list_train) * 0.9
train_list_clean = [y for y in train_list if y not in background_list_train[0:int(background_removal)]]

x_train_filenames = []
y_train_filenames = []

for i, img_id in zip(tqdm(range(len(train_list_clean))), train_list_clean):
  pass 
  x_train_filenames.append(os.path.join(IMG_DIR, "{}.png".format(img_id)))
  y_train_filenames.append(os.path.join(LABEL_DIR, "RDSISC4_{}_classified{}.png".format(img_id[:-4], img_id[-4:])))

print("Number of background tiles: ", background_removal)
print("Remaining number of tiles after 90% background removal: ", len(train_list_clean))

x_train_filenames_partition_fn = os.path.join(ROOT_DIR,'x_train_filenames_partition_filtered_07.txt')
y_train_filenames_partition_fn = os.path.join(ROOT_DIR,'y_train_filenames_partition_filtered_07.txt')
x_val_filenames_partition_fn = os.path.join(ROOT_DIR,'x_val_filenames_partition_filtered_07.txt')
y_val_filenames_partition_fn = os.path.join(ROOT_DIR,'y_val_filenames_partition_filtered_07.txt')
x_test_filenames_partition_fn = os.path.join(ROOT_DIR,'x_test_filenames_partition_filtered_07.txt')
y_test_filenames_partition_fn = os.path.join(ROOT_DIR,'y_test_filenames_partition_filtered_07.txt')

try:
  x_train_filenames = [line.strip() for line in open(x_train_filenames_partition_fn, 'r')]
  y_train_filenames = [line.strip() for line in open(y_train_filenames_partition_fn, 'r')]
  x_val_filenames = [line.strip() for line in open(x_val_filenames_partition_fn, 'r')]
  y_val_filenames = [line.strip() for line in open(y_val_filenames_partition_fn, 'r')]
  x_test_filenames = [line.strip() for line in open(x_test_filenames_partition_fn, 'r')]
  y_test_filenames = [line.strip() for line in open(y_test_filenames_partition_fn, 'r')]
except:
  x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(x_train_filenames, y_train_filenames, test_size=0.3, random_state=42)
  x_val_filenames, x_test_filenames, y_val_filenames, y_test_filenames = train_test_split(x_val_filenames, y_val_filenames, test_size=0.33, random_state=42)

num_train_examples = len(x_train_filenames)
num_val_examples = len(x_val_filenames)
num_test_examples = len(x_test_filenames)

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))
print("Number of test examples: {}".format(num_test_examples))

vals_train = []
vals_val = []
vals_test = []

def get_vals_in_partition(partition_list, x_filenames, y_filenames):
  for x,y,i in zip(x_filenames, y_filenames, tqdm(range(len(y_filenames)))):
      pass 
      try:
        img = np.array(Image.open(y)) 
        vals = np.unique(img)
        partition_list.append(vals)
      except:
        continue

def flatten(partition_list):
    return [item for sublist in partition_list for item in sublist]

get_vals_in_partition(vals_train, x_train_filenames, y_train_filenames)
get_vals_in_partition(vals_val, x_val_filenames, y_val_filenames)
get_vals_in_partition(vals_test, x_test_filenames, y_test_filenames)

print("Values in training partition: ", set(flatten(vals_train)))
print("Values in validation partition: ", set(flatten(vals_val)))
print("Values in test partition: ", set(flatten(vals_test)))

if set([0, 1, 2, 3, 4, 5]).issubset(set(flatten(vals_train))) == True and set([0, 1, 2, 3, 4, 5]).issubset(set(flatten(vals_val))) == True and set([0, 1, 2, 3, 4, 5]).issubset(set(flatten(vals_test))) == True:
    print("each partition has all values")
else:
    print("re-partitioning")
    x_train_filenames = []
    y_train_filenames = []

    for i, img_id in zip(tqdm(range(len(train_list_clean))), train_list_clean):
        pass
        x_train_filenames.append(os.path.join(IMG_DIR, "{}.png".format(img_id)))
        y_train_filenames.append(os.path.join(LABEL_DIR, "RDSISC4_{}_classified{}.png".format(img_id[:-4], img_id[-4:])))
    seed = random.randint(1, 100)
    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(x_train_filenames, y_train_filenames, test_size=0.3, random_state=seed)
    x_val_filenames, x_test_filenames, y_val_filenames, y_test_filenames = train_test_split(x_val_filenames, y_val_filenames, test_size=0.33, random_state=seed)
    get_vals_in_partition(vals_train, x_train_filenames, y_train_filenames)
    get_vals_in_partition(vals_val, x_val_filenames, y_val_filenames)
    get_vals_in_partition(vals_test, x_test_filenames, y_test_filenames)
    print("Values in training partition: ", set(flatten(vals_train)))
    print("Values in validation partition: ", set(flatten(vals_val)))
    print("Values in test partition: ", set(flatten(vals_test)))



if not os.path.isfile(fn) for fn in [x_train_filenames_partition_fn, y_train_filenames_partition, x_val_filenames_partition, y_val_filenames_partition, x_test_filenames_partition, y_test_filenames_partition]:
  with open(os.path.join(ROOT_DIR,'x_train_filenames_partition_filtered_07.txt'), 'w') as f:
    for item in x_train_filenames:
        f.write("%s\n" % item)

  with open(os.path.join(ROOT_DIR,'y_train_filenames_partition_filtered_07.txt'), 'w') as f:
    for item in y_train_filenames:
        f.write("%s\n" % item)

  with open(os.path.join(ROOT_DIR,'x_val_filenames_partition_filtered_07.txt'), 'w') as f:
    for item in x_val_filenames:
        f.write("%s\n" % item)

  with open(os.path.join(ROOT_DIR,'y_val_filenames_partition_filtered_07.txt'), 'w') as f:
    for item in y_val_filenames:
        f.write("%s\n" % item)

  with open(os.path.join(ROOT_DIR,'x_test_filenames_partition_filtered_07.txt'), 'w') as f:
    for item in x_test_filenames:
        f.write("%s\n" % item)

  with open(os.path.join(ROOT_DIR,'y_test_filenames_partition_filtered_07.txt'), 'w') as f:
    for item in y_test_filenames:
        f.write("%s\n" % item)
else:
  continue

background_list_train = [line.strip() for line in open(os.path.join(ROOT_DIR,"background_list_train.txt"), 'r')]

display_num = 3
# select only for tiles with foreground labels present
foreground_list_x = []
foreground_list_y = []
for x,y in zip(x_train_filenames, y_train_filenames): 
    try:
      filename_split = os.path.splitext(y) 
      filename_zero, fileext = filename_split 
      basename = os.path.basename(filename_zero) 
      if basename not in background_list_train:
        foreground_list_x.append(x)
        foreground_list_y.append(y)
      else:
        continue
    except:
      continue

num_foreground_examples = len(foreground_list_y)

# randomlize the choice of image and label pairs
r_choices = np.random.choice(num_foreground_examples, display_num)

# Function for reading the tiles into TensorFlow tensors 
# See TensorFlow documentation for explanation of tensor: https://www.tensorflow.org/guide/tensor
def _process_pathnames(fname, label_path):
  # We map this function onto each pathname pair  
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

# Function to augment the data with contrast adjustment
def adjust_contrast_img(contrast_adj, tr_img, contrast_range):
  if contrast_adj:
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])

    adj_prob = tf.random.uniform([], 0.0, 1.0)

    tr_img = tf.cond(tf.less(adj_prob, 1.0), #0.5
                                lambda: (tf.image.adjust_contrast(tr_img, contrast)),
                                lambda: (tr_img))
    #image = tf.image.adjust_contrast(images, contrast)
  return tr_img

# Function to augment the data with brightness adjustment
def adjust_brightness_img(brightness_adj, tr_img, brightness_delta):
  if brightness_adj:
    brightness = np.random.uniform(brightness_delta[0], brightness_delta[1])

    adj_prob = tf.random.uniform([], 0.0, 1.0)

    tr_img = tf.cond(tf.less(adj_prob, 1.0), #0.5
                                lambda: (tf.image.adjust_brightness(tr_img, brightness)),
                                lambda: (tr_img))
    #image = tf.image.adjust_brightness(images, brightness)
  return tr_img

# Function to augment the images and labels
# Function to augment the images and labels
def _augment(img,
             label_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=None,  # Scale image e.g. 1 / 255.
             horizontal_flip=False,
             vertical_flip=False,
             contrast_adj =False,
             brightness_adj=False,
             contrast_range=[0.5, 1.5],
             brightness_delta=[-0.2, 0.2]):
  if resize is not None:
    # Resize both images
    label_img = tf.image.resize(label_img, resize)
    img = tf.image.resize(img, resize)

  img, label_img = flip_img_h(horizontal_flip, img, label_img)
  img, label_img = flip_img_v(vertical_flip, img, label_img)
  img = adjust_contrast_img(contrast_adj, img, contrast_range)
  img = adjust_brightness_img(brightness_adj, img, brightness_delta)
  img = tf.cast(img, tf.float32)
  if scale is not None:
    img = tf.cast(img, tf.float32) * scale
  return img, label_img

# Main function to tie all of the above four dataset processing functions together 
def get_baseline_dataset(filenames, 
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5, 
                         batch_size=BATCH_SIZE,
                         shuffle=True):           
  num_x = len(filenames)
  # Create a dataset from the filenames and labels
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
  if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
    assert BATCH_SIZE == 1, "Batching images must be of the same size"

  dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
  
  if shuffle:
    dataset = dataset.shuffle(num_x)
  
  
  # It's necessary to repeat our data for all epochs 
  dataset = dataset.repeat().batch(BATCH_SIZE)
  return dataset

# dataset configuration for training
tr_cfg = {
    'resize': [IMG_SHAPE[0], IMG_SHAPE[1]],
    'scale': 1 / 255.,
    'horizontal_flip': True,
    'vertical_flip': True,
    'contrast_adj': True,
    'brightness_adj': True
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

# dataset configuration for validation
val_cfg = {
    'resize': [IMG_SHAPE[0], IMG_SHAPE[1]],
    'scale': 1 / 255.,
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)

# dataset configuration for testing
test_cfg = {
    'resize': [IMG_SHAPE[0], IMG_SHAPE[1]],
    'scale': 1 / 255.,
}
test_preprocessing_fn = functools.partial(_augment, **test_cfg)

# create the TensorFlow datasets
train_ds = get_baseline_dataset(x_train_filenames,
                                y_train_filenames,
                                preproc_fn=tr_preprocessing_fn,
                                batch_size=BATCH_SIZE)
val_ds = get_baseline_dataset(x_val_filenames,
                              y_val_filenames, 
                              preproc_fn=val_preprocessing_fn,
                              batch_size=BATCH_SIZE)
test_ds = get_baseline_dataset(x_test_filenames,
                              y_test_filenames, 
                              preproc_fn=test_preprocessing_fn,
                              batch_size=BATCH_SIZE)

# Now we will display some samples from the datasets
display_num = 1
r_choices = np.random.choice(num_foreground_examples, 1)
for i in range(0, display_num * 2, 2):
  img_num = r_choices[i // 2]

temp_ds = get_baseline_dataset(foreground_list_x[img_num:img_num+1], 
                               foreground_list_y[img_num:img_num+1],
                               preproc_fn=tr_preprocessing_fn,
                               batch_size=1,
                               shuffle=False)

# Let's examine some of these augmented images

iterator = iter(temp_ds)
next_element = iterator.get_next()

batch_of_imgs, label = next_element

# Running next element in our graph will produce a batch of images

sample_image, sample_mask = batch_of_imgs[0], label[0,:,:,:]

# reset the forground list to capture the validation images
foreground_list_x = []
foreground_list_y = []
for x,y in zip(x_val_filenames, y_val_filenames): 
    try:
      filename_split = os.path.splitext(y) 
      filename_zero, fileext = filename_split 
      basename = os.path.basename(filename_zero) 
      if basename not in background_list_train:
        foreground_list_x.append(x)
        foreground_list_y.append(y)
      else:
        continue
    except:
      continue

num_foreground_examples = len(foreground_list_y)

display_num = 1
r_choices = np.random.choice(num_foreground_examples, 1)
for i in range(0, display_num * 2, 2):
  img_num = r_choices[i // 2]

temp_ds = get_baseline_dataset(foreground_list_x[img_num:img_num+1], 
                               foreground_list_y[img_num:img_num+1],
                               preproc_fn=val_preprocessing_fn,
                               batch_size=1,
                               shuffle=False)

# Let's examine some of these augmented images

iterator = iter(temp_ds)
next_element = iterator.get_next()

batch_of_imgs, label = next_element

# Running next element in our graph will produce a batch of images

sample_image, sample_mask = batch_of_imgs[0], label[0,:,:,:]

# reset the forground list to capture the test images
foreground_list_x = []
foreground_list_y = []
for x,y in zip(x_test_filenames, y_test_filenames): 
    try:
      filename_split = os.path.splitext(y) 
      filename_zero, fileext = filename_split 
      basename = os.path.basename(filename_zero) 
      if basename not in background_list_train:
        foreground_list_x.append(x)
        foreground_list_y.append(y)
      else:
        continue
    except:
      continue

num_foreground_examples = len(foreground_list_y)

display_num = 1
r_choices = np.random.choice(num_foreground_examples, 1)
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

# set number of model output channels to the number of classes (including background)
OUTPUT_CHANNELS = 6

base_model = tf.keras.applications.MobileNetV2(input_shape=[96, 96, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu', 
    'block_13_expand_relu',
    'block_16_project', 
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False #True # Set this to False if using pre-trained weights

up_stack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[96, 96, 3], name='first_layer')
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same', name='last_layer')

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)

for layer in model.layers:
    print(layer.name, layer.output_shape)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=SparseCategoricalFocalLoss(gamma=2, from_logits=True), #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', iou_score])

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

# Tensorboard

log_dir = os.path.join(OUTPUT_DIR,'logs/')
log_fit_dir = os.path.join(OUTPUT_DIR,'logs', 'fit')
log_fit_session_dir = os.path.join(OUTPUT_DIR,'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
visualizations_dir = os.path.join(OUTPUT_DIR,'logs', 'vizualizations')
visualizations_session_dir = os.path.join(OUTPUT_DIR,'logs', 'vizualizations', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

dirs = [log_fit_dir, visualizations_dir]
for dir in dirs:
  if (os.path.isdir(dir)):
    print("Making fresh log dir.")
    shutil.rmtree(dir)
  else:
    print("Fresh log dir exists.")

dirs = [log_dir, log_fit_dir, log_fit_session_dir, visualizations_dir, visualizations_session_dir]
for dir in dirs:
  if (not os.path.isdir(dir)):
    os.mkdir(dir)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_fit_session_dir, histogram_freq=1, write_graph=True)

# get a batch of validation samples to plot activations for
for example in val_ds.take(1):
  image_val, label_val = example[0], example[1]

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

callbacks = [
    ActivationsVisualizationCallback(
        validation_data=(image_val, label_val),
        layers_name=["last_layer"],
        output_dir=visualizations_session_dir,
    ),
    tensorboard_callback,
    early_stopping_callback
]

EPOCHS = 100

model_history = model.fit(train_ds,
                   steps_per_epoch=int(np.ceil(num_train_examples / float(BATCH_SIZE))),
                   epochs=EPOCHS,
                   validation_data=val_ds,
                   validation_steps=int(np.ceil(num_val_examples / float(BATCH_SIZE))),
                   callbacks=callbacks)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

final_epochs = len(model_history.history['loss'])
print("final number of epochs: ", final_epochs)

if (not os.path.isdir(workshop_dir)):
  os.mkdir(workshop_dir)
save_model_path = os.path.join(OUTPUT_DIR,'model_out_batch_{}_ep{}_earlystopping_pretrain_focalloss/'.format(BATCH_SIZE, final_epochs))
if (not os.path.isdir(save_model_path)):
  os.mkdir(save_model_path)
model.save(save_model_path)

