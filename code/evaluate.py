import fnmatch, functools, glob
import numpy as np
import os
import pandas as pd
from itertools import product
import skimage.io as skio # lighter dependency than tensorflow for working with our tensors/arrays

from utils import eval_utils
from utils.read_data import get_image_label_arrays

with open("../configs/datasets.json", "r") as f:
    datasets = json.load(f)

path_df = pd.read_csv(os.path.join(datasets['ROOT_DIR'], "test_file_paths.csv"))

# reading in preds
label_arr_lst = path_df["label_names"].apply(skio.imread)
pred_arr_lst = path_df["pred_names"].apply(skio.imread)

pred_arr_lst_valid, label_arr_lst_valid = get_image_label_arrays(path_df)

# Compute per class IoU       
ious = {'iou_0': [], 'iou_1': [], 'iou_2': [], 'iou_3': [], 'iou_4': []}
#iou_5 = [] # commented out because ridge shadows was dropped as a class

for l, p in zip(label_arr_lst_valid, pred_arr_lst_valid):
    iou0 = eval_utils.maskIOU(l, p, 0)
    ious['iou_0'].append(iou0)
    iou1 = eval_utils.maskIOU(l, p, 1)
    ious['iou_0'].append(iou1)
    iou2 = eval_utils.maskIOU(l, p, 2)
    ious['iou_0'].append(iou2)
    iou3 = eval_utils.maskIOU(l, p, 3)
    ious['iou_0'].append(iou3)
    iou4 = eval_utils.maskIOU(l, p, 4)
    ious['iou_0'].append(iou4)
    #iou5 = eval_utils.maskIOU(l, p, 5)
    #iou_5.append(iou5)

# Get per class averages
iou_avg_0 = np.mean(iou_0)
iou_avg_1 = np.mean(iou_1)
iou_avg_2 = np.mean(iou_2)
iou_avg_3 = np.mean(iou_3)
iou_avg_4 = np.mean(iou_4)
#iou_avg_5 = np.mean(iou_5)


# flatten our tensors and use scikit-learn to create a confusion matrix
flat_preds, flat_truth = eval_utils.flatten_arrays(pred_arr_lst_valid, label_arr_lst_valid)
flat_preds, flat_truth = eval_utils.denullify_arrays(pred_arr_lst_valid, label_arr_lst_valid)

OUTPUT_CHANNELS = 5 #6
cm = eval_utils.get_cm(flat_truth, flat_preds, OUTPUT_CHANNELS)

# compute F1 score
f1, f1_scores_with_labels = get_f1(flat_truth, flat_preds, OUTPUT_CHANNELS)
print("f1 scores with labels: ", f1_scores_with_labels)
print("overall f1 score: ", f1)
print("cm: ", cm)
print("iou_avg_0, iou_avg_1, iou_avg_2, iou_avg_3, iou_avg_4: ", iou_avg_0, iou_avg_1, iou_avg_2, iou_avg_3, iou_avg_4)
