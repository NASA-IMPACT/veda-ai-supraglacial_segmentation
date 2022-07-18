import glob, io, os
import numpy as np

BLUE_THRESH = 220


def mask_blue(image_nominal, image_pred):
    """Takes a [r,g,b] image (pre-histogram matching) and masks areas of its prediction counterpart where 
    pixels are detected as melt pond but do not exceed a certain intensity in the blue channel.
    Args:
        image_nominal (nd.array): A [r,g,b] image (pre-histogram matching).
        image_pred (nd.array): The corresponding single channel prediction image. Can be derived from a
        histogram matched counterpart.
    Returns:
        img_pred (nd.array): The prediction image array, masked as needed.
    """
    if 3 in np.unique(image_pred):
        r,g,b = np.dsplit(image_nominal,image_nominal.shape[-1])
        img_pred = np.expand_dims(image_pred, axis=2)
        img_pred[np.where((img_pred == 3) & (b <= BLUE_THRESH))] = 0

    else:
        img_pred = image_pred
    return img_pred
