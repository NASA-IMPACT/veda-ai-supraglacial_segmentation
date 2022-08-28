import datetime, os, fnmatch, functools, io, glob, random, shutil, json
from itertools import product
from time import sleep
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as skio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl

from rasterio import features, mask
import tensorflow as tf
from focal_loss import SparseCategoricalFocalLoss

#from tqdm.notebook import tqdm

from utils import model_utils, data_utils

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
# maximum number of epochs to let the model train for
MAX_EPOCHS = datasets['max_epochs']
# set number of model output channels to the number of classes (including background)
OUTPUT_CHANNELS = datasets['output_channels']

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


train_list_fn = os.path.join(ROOT_DIR,datasets['train_filenames'])
x_train_filenames_fn = os.path.join(ROOT_DIR,datasets['x_filenames'])
y_train_filenames_fn = os.path.join(ROOT_DIR,datasets['y_filenames'])

bad_groundtruth_examples = [line.strip() for line in open(os.path.join(ROOT_DIR,"flagged_melt_pcts_fns.csv"), 'r')]

try:
    train_list = [line.strip() for line in open(train_list_fn, 'r')]
    x_train_filenames = [line.strip() for line in open(x_train_filenames_fn, 'r')]
    y_train_filenames = [line.strip() for line in open(y_train_filenames_fn, 'r')]
except:
    train_list, x_train_filenames, y_train_filenames = read_data.get_train_test_lists_icebridge(IMG_DIR, LABEL_DIR)
    with open(os.path.join(ROOT_DIR,datasets['train_filenames']), 'w') as f:
        for item in train_list:
            f.write("%s\n" % item)

    with open(os.path.join(ROOT_DIR,datasets['x_filenames']), 'w') as f:
        for item in x_train_filenames:
            f.write("%s\n" % item)

    with open(os.path.join(ROOT_DIR,datasets['y_filenames']), 'w') as f:
        for item in y_train_filenames:
            f.write("%s\n" % item)


print("Number of images available for training: ", len(train_list))

skip = False
proportion_remove = 0.9
background_removal, train_list_clean = data_utils.read_background(train_list, ROOT_DIR, LABEL_DIR, skip, proportion_remove)

x_train_filenames = []
y_train_filenames = []

for i, img_id in zip(tqdm(range(len(train_list_clean))), train_list_clean):
    pass 
    x_train_filenames.append(os.path.join(IMG_DIR, "{}.png".format(img_id)))
    y_train_filenames.append(os.path.join(LABEL_DIR, "RDSISC4_{}_classified{}.png".format(img_id[:-4], img_id[-4:])))

print("Number of background tiles: ", background_removal)
print("Remaining number of tiles after 90% background removal: ", len(train_list_clean))

x_train_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['x_train_filenames_partition'])
y_train_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['y_train_filenames_partition'])
x_val_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['x_val_filenames_partition'])
y_val_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['y_val_filenames_partition'])
x_test_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['x_test_filenames_partition'])
y_test_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['y_test_filenames_partition'])

x_train_filenames, x_val_filenames, x_test_filenames, y_train_filenames, y_val_filenames, y_test_filenames = data_utils.get_lists_partitions(train_list_clean, x_train_filenames, y_train_filenames, x_train_filenames_partition_fn, y_train_filenames_partition_fn, x_val_filenames_partition_fn, y_val_filenames_partition_fn, x_test_filenames_partition_fn, y_test_filenames_partition_fn)

num_train_examples = len(x_train_filenames)
num_val_examples = len(x_val_filenames)
num_test_examples = len(x_test_filenames)

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))
print("Number of test examples: {}".format(num_test_examples))

vals_train = []
vals_val = []
vals_test = []

vals_train = data_utils.get_vals_in_partition(vals_train, x_train_filenames, y_train_filenames)
vals_val = data_utils.get_vals_in_partition(vals_val, x_val_filenames, y_val_filenames)
vals_test = data_utils.get_vals_in_partition(vals_test, x_test_filenames, y_test_filenames)

print("Values in training partition: ", set(flatten(vals_train)))
print("Values in validation partition: ", set(flatten(vals_val)))
print("Values in test partition: ", set(flatten(vals_test)))

x_train_filenames, x_val_filenames, x_test_filenames, y_train_filenames, y_val_filenames, y_test_filenames = check_write_lists_partitions(vals_train, vals_val, vals_test, train_list_clean, x_train_filenames, y_train_filenames)

background_list_train = [line.strip() for line in open(os.path.join(ROOT_DIR,datasets['background_files']), 'r')]

display_num = 3
# select only for tiles with foreground labels present
foreground_list_x = []
foreground_list_y = []
for x,y in zip(x_train_filenames, y_train_filenames): 
    try:
        file_name, file_ext = os.path.splittext(y)
        basename = os.path.basename(file_name) 
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

# dataset configuration for training
tr_cfg = {
    'resize': [IMG_SHAPE[0], IMG_SHAPE[1]],
    'scale': 1 / 255.,
    'horizontal_flip': True,
    'vertical_flip': True,
    'contrast_adj': True,
    'brightness_adj': True
}
tr_preprocessing_fn = functools.partial(model_utils._augment, **tr_cfg)

# dataset configuration for validation
val_cfg = {
    'resize': [IMG_SHAPE[0], IMG_SHAPE[1]],
    'scale': 1 / 255.,
}
val_preprocessing_fn = functools.partial(model_utils._augment, **val_cfg)

# dataset configuration for testing
test_cfg = {
    'resize': [IMG_SHAPE[0], IMG_SHAPE[1]],
    'scale': 1 / 255.,
}
test_preprocessing_fn = functools.partial(model_utils._augment, **test_cfg)

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
        file_name, file_ext = os.path.splittext(y)
        basename = os.path.basename(file_name) 
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
        file_name, file_ext = os.path.splittext(y)
        basename = os.path.basename(file_name) 
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

model = model_utils.unet_model(OUTPUT_CHANNELS)

#for layer in model.layers:
#    print(layer.name, layer.output_shape)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=datasets['learning_rate']),
              loss=SparseCategoricalFocalLoss(gamma=2, from_logits=True), #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', iou_score])

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

model_history = model.fit(train_ds,
                   steps_per_epoch=int(np.ceil(num_train_examples / float(BATCH_SIZE))),
                   epochs=MAX_EPOCHS,
                   validation_data=val_ds,
                   validation_steps=int(np.ceil(num_val_examples / float(BATCH_SIZE))),
                   callbacks=callbacks)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

final_epochs = len(model_history.history['loss'])
print("Final number of epochs: ", final_epochs)

if (not os.path.isdir(OUTPUT_DIR)):
    os.mkdir(OUTPUT_DIR)
save_model_path = os.path.join(OUTPUT_DIR,'model_out_batch_{}_ep{}_earlystopping_pretrain_focalloss/'.format(BATCH_SIZE, final_epochs))
if (not os.path.isdir(save_model_path)):
    os.mkdir(save_model_path)
model.save(save_model_path)

