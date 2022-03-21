import os, glob, functools, fnmatch
from zipfile import ZipFile
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.rcParams['axes.grid'] = False
#mpl.rcParams['figure.figsize'] = (12,12)
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
import geopandas as gpd
from IPython.display import clear_output
from time import sleep

import skimage.io as skio # lighter dependency than tensorflow for working with our tensors/arrays
from sklearn.metrics import confusion_matrix, f1_score

root_dir = '/home/ubuntu/data/'
path_df = pd.read_csv(os.path.join(root_dir, "test_file_paths.csv"))

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

# flatten our tensors and use scikit-learn to create a confusion matrix
flat_preds = np.concatenate(pred_arr_lst_valid).flatten()
flat_truth = np.concatenate(label_arr_lst_valid).flatten()
OUTPUT_CHANNELS = 6
cm = confusion_matrix(flat_truth, flat_preds, labels=list(range(OUTPUT_CHANNELS)))

classes = [0,1,2,3,4,5]

# compute f1 score
f1 = f1_score(flat_truth, flat_preds, average='macro')

print("confusion matrix: ", cm)
print("f1 score: ", f1)

