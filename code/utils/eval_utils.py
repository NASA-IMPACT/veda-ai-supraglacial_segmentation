import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

# Compute per class IoU        
def maskIOU(mask_gt, mask_pr, class_val):  
    """Computes a per-class Intersection over Union (IoU) score between two arrays.
    Args:
        mask_gt (nd.array): The ground truth array.
        mask_pr (nd.array): The prediction array.
        class_val (int): The class value to compute an IoU score for.
    Returns:
        iou (float): The IoU score for the specified class.
    """
    mask_gt_area = np.count_nonzero(mask_gt == int(class_val))
    mask_pr_area = np.count_nonzero(mask_pr == int(class_val))
    intersection = np.count_nonzero(np.logical_and( mask_gt==int(class_val),  mask_pr==int(class_val) ))
    try:
        iou = intersection/(mask_gt_area+mask_pr_area-intersection)
    except Exception as e:
        print("couldn't compute iou because: ", e)
        iou = 0
    return iou


def flatten_arrays(pred_arr_lst_valid, label_arr_lst_valid):
    # flatten our tensors and use scikit-learn to create a confusion matrix
    flat_preds = np.concatenate(pred_arr_lst_valid).flatten()
    flat_truth = np.concatenate(label_arr_lst_valid).flatten()
    return flat_preds, flat_truth

def denullify_arrays(pred_arr_lst_valid, label_arr_lst_valid):
    flat_preds_cp = flat_preds.copy()
    flat_preds = flat_preds[flat_preds!=0]
    flat_truth = flat_truth[flat_preds_cp!=0]
    return flat_preds, flat_truth

def get_cm(flat_truth, flat_preds, num_classes):
    cm = confusion_matrix(flat_truth, flat_preds, labels=list(range(num_classes)))
    classes = list(range(0, num_classes))

    #%matplotlib inline
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)),
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
    plt.savefig(f'{datasets['ROOT_DIR']}/cm.png')
    return cm 

def get_f1(flat_truth, flat_preds, num_classes)
    # compute F1 score
    labels = list(range(0, num_classes))
    f1 = f1_score(flat_truth, flat_preds, average='macro')
    f1_scores = f1_score(flat_truth, flat_preds, average=None, labels=labels)
    f1_scores_with_labels = {label:score for label,score in zip(labels, f1_scores)}
    return f1, f1_scores_with_labels

print("f1 scores with labels: ", f1_scores_with_labels)
print("overall f1 score: ", f1)
print("cm: ", cm)
print("iou_avg_0, iou_avg_1, iou_avg_2, iou_avg_3, iou_avg_4: ", iou_avg_0, iou_avg_1, iou_avg_2, iou_avg_3, iou_avg_4)
