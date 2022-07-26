import glob, io, os
import numpy as np

#BLUE_THRESH = 50
#NDWI_THRESH = 0.3
#NIR_THRESH = 3000

def mask(img_thresh, img_pred, thresh):
    img_pred[np.where((img_pred == 3) & (img_thresh >= thresh))] = 0
    img_pred[np.where((img_pred == 4) & (img_thresh >= thresh))] = 0
    return img_pred

def mask_ndwi_nir_blue(image_rgb, image_nir, image_pred, mask_type='nir'):
    """Takes a [r,g,b] image (pre-histogram matching) and masks areas of its prediction counterpart where 
    pixels are detected as melt pond but do not exceed a certain intensity in the threshold channel.
    Args:
        image_rgb (nd.array): A 3 channel [r,g,b] uint16 image (pre-histogram matching).
        image_nir (nd.array): A single channel [nir] uint16 image (pre-histogram matching).
        image_pred (nd.array): The corresponding single channel prediction image. Can be derived from a
        histogram matched counterpart.
        mask_type (string): Which information will be used to threshold and mask rocky pixels. Options
        are ['nir', 'ndwi', 'blue'].
    Returns:
        img_pred (nd.array): The prediction image array, masked as needed.
    """
    if 3 in np.unique(image_pred):
        if mask_type == ndwi:
            r,g,b = np.dsplit(image_nominal,image_nominal.shape[-1])
            
            img_ndwi = ((g-image_nir))/((g+image_nir))
            ndwi_thresh = image_ndwi.max() * 0.1
            if ndwi_thresh < image_ndwi.min():
                range_ndwi = image_ndwi.max() - image_ndwi.min()
                ndwi_thresh = image_ndwi.max() - (0.9 * range_ndwi) 
            img_pred = mask(img_ndwi, image_pred, ndwi_thresh)
        elif mask_type == nir: 
            nirthresh = image_nir.max() * 0.1
            if nirthresh < image_nir.min():
                range_nir = image_nir.max() - image_nir.min()
                nir_thresh = image_nir.max() - (0.9 * range_nir)
            img_pred = mask(image_nir, img_pred, nir_thresh)
        elif mask_type == blue:
            r,g,b = np.dsplit(image_nominal,image_nominal.shape[-1])
            blue_thresh = b.max() * 0.1
            if blue_thresh < b.min():
                range_blue = b.max() - b.min()
                blue_thresh = b.max() - (0.9 * range_blue) 
            img_pred = np.expand_dims(image_pred, axis=2)
            img_pred = mask(b, img_pred, blue_thresh)
        else:
            img_pred = image_pred
    else:
        img_pred = image_pred
    return img_pred
