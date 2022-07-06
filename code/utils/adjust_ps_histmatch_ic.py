import os, glob
import numpy as np
import skimage.io as skio
from skimage.exposure import match_histograms, adjust_log
from skimage.color import rgb2ycbcr, ycbcr2rgb


def hist_match(image, reference, out_dir):
    """Takes one planetscope image and the reference image. Performs histogram matching such that the 
    planetscope image resembles the icebridge reference.

    Args:
        image (string): The path to a planetscope image.
        reference (string): The path to a reference icebridge image.
        out_dir (string): The output directory to write the histogram matched planetscope image to.
    Returns:
        matched (nd.array): A histogram matched planetscope image.
    """
    img = skio.imread(image)
    filename_split = os.path.splitext(image)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    reference = skio.imread(reference)
    img = rgb2ycbcr(img, channel_axis=-1)
    reference = rgb2ycbcr(reference, channel_axis=-1)
    matched = match_histograms(img, reference, channel_axis=-1, multichannel=True)
    matched = ycbcr2rgb(matched, channel_axis=-1)
    matched = adjust_log(matched, 1)
    skio.imsave(f'{out_dir}/{basename}.png', matched)
    return matched


