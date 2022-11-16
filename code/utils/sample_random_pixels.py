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
GT_PATH = os.path.join(ROOT_DIR,datasets['icebridge_label_dir/']) 
PRED_PATH = os.path.join(ROOT_DIR,datasets['icebridge_pred_dir/']) 
OUT_PATH = os.path.join(ROOT_DIR,datasets['icebridge_random_sample_dir'])


def process_image(image_array, image_type, OUT_PATH):
    """Takes an image array and writes it to a prescribed path.
    Args:
        image_array (nd.array): Image array.
        image_type (string): The type of image, 'gt' for ground truth and 'pr' for prediction.
        OUT_PATH (path): Directory to write masked tiles and sampled pixel coordinate lists to.
    Returns:
        n/a
    """
    img = Image.fromarray(image_array)
    if (not os.path.isdir(f"{OUT_PATH}/{str(image_type)}/")):
        os.mkdir(f"{OUT_PATH}/{str(image_type)}/")
    img.save(f"{OUT_PATH}/{str(image_type)}/{basename}.png")

    

def random_sample(GT_PATH, PRED_PATH, OUT_PATH, number_tiles, number_pixels):
    """Randomly samples matching ground truth and prediction tiles along with pixels within tiles . 
    Masks all pixels not sampled as zero.
    Args:
        GT_PATH (path): Directory containing the ground truth label tiles (from test partition).
        PRED_PATH (path): Directory containing the matching predicted tiles.
        OUT_PATH (path): Directory to write masked tiles and sampled pixel coordinate lists to.
        number_tiles (int): The number of tiles to sample.
        number_pixels (int): The number of pixels to sample within a single tile.
    Returns:
        n/a
    """
    filenames = random.sample(os.listdir(PRED_PATH), number_tiles)
    for f in filenames:
        filename_split = os.path.splitext(f)
        filename_zero, fileext = filename_split
        basename = os.path.basename(filename_zero)
        im = cv2.imread(PRED_PATH+f)
        X,Y = np.where(im[...,0]>=0)
        coords = np.column_stack((X,Y))
        np.random.shuffle(coords)
        coordsn = coords[0:number_pixels]
        coords = coords[number_pixels:]

        img_gt = np.array(Image.open(f"{GT_PATH}/{basename}.png"))
        img_ps = np.array(Image.open(f"{PRED_PATH}/{basename}.png"))

        for i in coords:
            img_gt[i[1],i[0]] = 0
            img_ps[i[1],i[0]] = 0

        with open(f"{OUT_PATH}/{basename}_coords_icebridge.txt", 'w') as file_handler:
            for item in coordsn:
                file_handler.write("{}\n".format(item))

        process_image(img_gt, 'gt', OUT_PATH)
        process_image(img_ps, 'pr', OUT_PATH)

    df = pd.DataFrame(columns=['label_names', 'pred_names'])
    label_names = glob.glob(f"{OUT_PATH}/gt/*")
    pred_names = glob.glob(f"{OUT_PATH}/pr/*")
    df.label_names = label_names
    df.pred_names = pred_names
    df.to_csv(f"{OUT_PATH}/randomsample.csv")
