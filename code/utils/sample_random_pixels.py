import cv2
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
import random

with open("../configs/datasets.json", "r") as f:
    datasets = json.load(f)
ROOT_DIR = datasets['ROOT_DIR']
gtpath = os.path.join(ROOT_DIR,datasets['icebridge_label_dir/']) 
predpath = os.path.join(ROOT_DIR,datasets['icebridge_pred_dir/']) 
outpath = os.path.join(ROOT_DIR,datasets['icebridge_random_sample_dir'])


def process_image(image_array, image_type, outpath):
    """Takes an image array and writes it to a prescribed path.
    Args:
        image_array (nd.array): Image array.
        image_type (string): The type of image, 'gt' for ground truth and 'pr' for prediction.
        outpath (path): Directory to write masked tiles and sampled pixel coordinate lists to.
    Returns:
        n/a
    """
    img = Image.fromarray(image_array)
    if (not os.path.isdir(outpath+str(image_type)+'/')):
        os.mkdir(outpath+str(image_type)+'/')
    img.save(outpath+str(image_type)+'/'+basename+'.png')
    return
    

def random_sample(gtpath, predpath, outpath, number_tiles, number_pixels):
    """Randomly samples matching ground truth and prediction tiles along with pixels within tiles . 
    Masks all pixels not sampled as zero.
    Args:
        gtpath (path): Directory containing the ground truth label tiles (from test partition).
        predpath (path): Directory containing the matching predicted tiles.
        outpath (path): Directory to write masked tiles and sampled pixel coordinate lists to.
        number_tiles (int): The number of tiles to sample.
        number_pixels (int): The number of pixels to sample within a single tile.
    Returns:
        n/a
    """
    filenames = random.sample(os.listdir(predpath), number_tiles)
    for f in filenames:
        filename_split = os.path.splitext(f)
        filename_zero, fileext = filename_split
        basename = os.path.basename(filename_zero)
        im = cv2.imread(predpath+f)
        X,Y = np.where(im[...,0]>=0)
        coords = np.column_stack((X,Y))
        np.random.shuffle(coords)
        coordsn = coords[0:number_pixels]
        coords = coords[number_pixels:]

        img = np.array(Image.open(gtpath+basename+'.png'))
        img1 = np.array(Image.open(predpath+basename+'.png'))

        img_gt = img.copy()
        img_ps = img1.copy()

        for coord in coords:
            img_gt[coord[1],coord[0]] = 0
            img_ps[coord[1],coord[0]] = 0

        with open(outpath+basename+'_coords_icebridge.txt', 'w') as file_handler:
            for item in coordsn:
                file_handler.write("{}\n".format(item))

        process_image(img_gt, 'gt', outpath)
        process_image(img_ps, 'pr', outpath)

    df = pd.DataFrame(columns=['label_names', 'pred_names'])
    label_names = glob.glob(outpath+'gt/*')
    pred_names = glob.glob(outpath+'pr/*')
    df.label_names = label_names
    df.pred_names = pred_names
    df.to_csv(outpath+'randomsample.csv')
    return
