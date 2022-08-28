
import glob
import json
import os
import skimage.io as skio
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from focal_loss import SparseCategoricalFocalLoss
from PIL import Image
from segmentation_models.metrics import iou_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.python.keras import backend as K  
from tensorflow.python.keras import layers, losses, models
from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback

tfds.disable_progress_bar()

with open("../configs/datasets.json", "r") as f:
    datasets = json.load(f)

IMG_SHAPE = datasets['image_shape']

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False)

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

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

