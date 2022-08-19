import datetime, os, fnmatch, functools, io, glob, random, shutil, json
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

from utils import model_utils, data_utils

tfds.disable_progress_bar()

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("physical devices: ", physical_devices)

with open("../configs/datasets.json", "r") as f:
    datasets = json.load(f)

ROOT_DIR = datasets['ROOT_DIR']
OUTPUT_DIR = '/home/ubuntu/models/'

IMG_DIR = os.path.join(ROOT_DIR, datasets['icebridge_image_dir/']) 
LABEL_DIR = os.path.join(ROOT_DIR,datasets['icebridge_label_dir/']) 

# input image shape
IMG_SHAPE = datasets['image_shape']
# batch size for model
BATCH_SIZE =  datasets['batch_size']
# set number of model output channels to the number of classes (including background)
OUTPUT_CHANNELS = datasets['output_channels']
# number of epochs completed for model training using early stopping
EPOCHS = 60 

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

x_test_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['x_test_filenames_partition'])
y_test_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['y_test_filenames_partition'])

if os.path.isfile(fn) for fn in [x_test_filenames_partition_fn, y_test_filenames_partition_fn]:
    x_test_filenames = [line.strip() for line in open(x_test_filenames_partition_fn, 'r')]
    y_test_filenames = [line.strip() for line in open(y_test_filenames_partition_fn, 'r')]
else:
    test_list, x_test_filenames, y_test_filenames = data_utils.get_test_lists_planetscope(IMG_DIR, LABEL_DIR)

print("!!!!! number of images: ", len(x_test_filenames))

# dataset configuration for testing
test_cfg = {
    'resize': [IMG_SHAPE[0], IMG_SHAPE[1]],
    'scale': 1 / 255.,
}
test_preprocessing_fn = functools.partial(_augment, **test_cfg)

print(x_test_filenames[0:10])
print(y_test_filenames[0:10])

test_ds = model_utils.get_baseline_dataset(x_test_filenames,
                              y_test_filenames, 
                              preproc_fn=test_preprocessing_fn,
                              batch_size=BATCH_SIZE)

# reset the forground list to capture the test images
foreground_list_x = []
foreground_list_y = []
for x,y in zip(x_test_filenames, y_test_filenames): 
    try:
        file_name, file_ext = os.path.splittext(y)
        basename = os.path.basename(file_name) 
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

temp_ds = model_utils.get_baseline_dataset(foreground_list_x[img_num:img_num+1], 
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
    save_model_path = os.path.join(OUTPUT_DIR,'model_out_batch_{}_ep{}_pretrain_focalloss/'.format(BATCH_SIZE, EPOCHS))
    model = tf.keras.models.load_model(save_model_path, custom_objects={"loss": SparseCategoricalFocalLoss, "iou_score": iou_score})
else:
    print("inferencing from in memory model")


def get_predictions(image= None, dataset=None, num=1):
    if not(image) and not(dataset):
        return ValueError("At least one of image or dataset must not be None.")
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            return pred_mask
    else:
        pred_mask = model_utils.create_mask(model.predict(image[tf.newaxis, ...]))
        pred_mask = tf.keras.backend.eval(pred_mask)
        return pred_mask

tiled_prediction_dir = os.path.join(ROOT_DIR,'predictions_test_focal_loss_batch_{}_ep{}/'.format(BATCH_SIZE, EPOCHS))
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
    file_name, file_ext = os.path.splittext(x_test_filenames[img_num])
    basename = os.path.basename(file_name) 
    pred_path = os.path.join(tiled_prediction_dir, "{}.png".format(basename))
    pred_paths.append(pred_path)
    tf.keras.preprocessing.image.save_img(pred_path,pred_mask, scale=False) # scaling is good to do to cut down on file size, but adds an extra dtype conversion step.    

path_df = pd.DataFrame(list(zip(x_test_filenames, y_test_filenames, pred_paths)), columns=["img_names", "label_names", "pred_names"])
path_df.to_csv(os.path.join(ROOT_DIR, "test_file_paths_{}_ep{}.csv".format(BATCH_SIZE, EPOCHS)))

