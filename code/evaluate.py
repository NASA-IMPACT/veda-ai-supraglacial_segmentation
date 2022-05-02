import os, fnmatch, functools, glob
from itertools import product
from zipfile import ZipFile

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np

import pandas as pd
from PIL import Image
import geopandas as gpd
from IPython.display import clear_output
from time import sleep

import skimage.io as skio # lighter dependency than tensorflow for working with our tensors/arrays
from sklearn.metrics import confusion_matrix, f1_score

ROOT_DIR = '/home/ubuntu/data/'
path_df = pd.read_csv(os.path.join(ROOT_DIR, "test_file_paths.csv"))

# reading in preds
label_arr_lst = path_df["label_names"].apply(skio.imread)
pred_arr_lst = path_df["pred_names"].apply(skio.imread)

pred_arr_lst_valid = []
label_arr_lst_valid = []
for i in range(0, len(pred_arr_lst)):
    if pred_arr_lst[i].shape != label_arr_lst[i].shape:
        
        print(f"The {i}th label has an incorrect dimension, skipping.")
        print(pred_arr_lst[i])
        print(label_arr_lst[i])
        print(pred_arr_lst[i].shape)
        print(label_arr_lst[i].shape)
        
    else:
        pred_arr_lst_valid.append(pred_arr_lst[i])
        label_arr_lst_valid.append(label_arr_lst[i])

# Compute per class IoU        
def maskIOU(mask1, mask2, class_val):   # From the question.
    mask1_area = np.count_nonzero(mask1 == int(class_val))
    mask2_area = np.count_nonzero(mask2 == int(class_val))
    intersection = np.count_nonzero(np.logical_and( mask1==int(class_val),  mask2==int(class_val) ))
    try:
        iou = intersection/(mask1_area+mask2_area-intersection)
    except Exception as e:
        print("couldn't compute iou because: ", e)
        iou = 0
    return iou

iou_0 = []
iou_1 = []
iou_2 = []
iou_3 = []
iou_4 = []
iou_5 = []


for l, p in zip(label_arr_lst_valid, pred_arr_lst_valid):
    iou0 = maskIOU(l, p, 0)
    iou_0.append(iou0)
    iou1 = maskIOU(l, p, 1)
    iou_1.append(iou1)
    iou2 = maskIOU(l, p, 2)
    iou_2.append(iou2)
    iou3 = maskIOU(l, p, 3)
    iou_3.append(iou3)
    iou4 = maskIOU(l, p, 4)
    iou_4.append(iou4)
    iou5 = maskIOU(l, p, 5)
    iou_5.append(iou5)

iou_avg_0 = np.mean(iou_0)
iou_avg_1 = np.mean(iou_1)
iou_avg_2 = np.mean(iou_2)
iou_avg_3 = np.mean(iou_3)
iou_avg_4 = np.mean(iou_4)
iou_avg_5 = np.mean(iou_5)


# flatten our tensors and use scikit-learn to create a confusion matrix
flat_preds = np.concatenate(pred_arr_lst_valid).flatten()
flat_truth = np.concatenate(label_arr_lst_valid).flatten()
OUTPUT_CHANNELS = 6
cm = confusion_matrix(flat_truth, flat_preds, labels=list(range(OUTPUT_CHANNELS)))

classes = [0,1,2,3,4,5]

#%matplotlib inline
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=list(range(OUTPUT_CHANNELS)), yticklabels=list(range(OUTPUT_CHANNELS)),
       title='Normalized Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f' #'d' # if normalize else 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
ax.set_ylim(len(classes)-0.5, -0.5)

plt.savefig(f'{ROOT_DIR}cm.png')


# compute f1 score
f1 = f1_score(flat_truth, flat_preds, average='macro')

print("cm: ", cm)
print("f1: ", f1)
print("iou_avg_0, iou_avg_1, iou_avg_2, iou_avg_3, iou_avg_4, iou_avg_5: ", iou_avg_0, iou_avg_1, iou_avg_2, iou_avg_3, iou_avg_4, iou_avg_5)

