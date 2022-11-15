import numpy as np
import os
import pandas as pd
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

ious, iou_avg_0, iou_avg_1, iou_avg_2, iou_avg_3, iou_avg_4 = eval_utils.get_ious(label_arr_lst_valid, pred_arr_lst_valid)

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
