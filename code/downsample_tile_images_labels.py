import os, sys, glob
import boto3
from PIL import Image
import io
from configparser import ConfigParser
import numpy as np
import cv2
import math
import rasterio as rio
import skimage.transform
from utils import s3_utils

# This script requires access to a file containing your aws credentials formatted as such:
"""
[credentials]
AWS_ACCESS_KEY_ID = xxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY = xxxxxxxxxxxxxxxxx
"""

# authenticate with AWS credentials
config = ConfigParser()
configFilePath = './access_keys.csv'
with open(configFilePath) as f:
    config.read_file(f)
AWS_ACCESS_KEY_ID = config.get('credentials', 'AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config.get('credentials', 'AWS_SECRET_ACCESS_KEY')

downsample_factor = 20.0
tile_size = 224


s3 = boto3.resource('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

rio_session = rio.env.Env(aws_access_key_id=AWS_ACCESS_KEY_ID,
             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
             region_name='us-east')


# Get image paths in S3.
items_s3 = []
urls_s3 = []
items_urls_s3 = []

for i in s3_utils.iterate_bucket_items(bucket='veda-ai-supraglacial-meltponds'):
    bucket='veda-ai-supraglacial-meltponds'
    ik = i["Key"]
    items_s3.append(ik)
    url = f"s3://{str(bucket)}/{str(ik)}"
    urls_s3.append(url)
    item_url = (ik, url)
    items_urls_s3.append(item_url)


def downsample(image, image_name, option):
    """Downsamples images using the downsample_factor parameter.
    Args:
        image (PIL image): An image opened in PIL.
        image_name (string): The image filename.
        option (string): Choice of r,g,b image or label image.
    Returns:
        n/a
    """
    bucket = s3.Bucket('veda-ai-supraglacial-meltponds')
    filename_split = os.path.splitext(image_name) 
    filename_zero, fileext = filename_split 
    basename = os.path.basename(filename_zero) 
    if option == "image":
        outfile = f"downsampled_images/{basename}.png"
    else:
        outfile = f"downsampled_labels/{basename}.png"
    if i != outfile:
        try:

            im = image.copy()
            size = (np.array(im).shape[0])/20.0, (np.array(im).shape[1])/20.0
            im = np.array(im)
            im = skimage.transform.resize(im,(size),mode='edge',anti_aliasing=False,anti_aliasing_sigma=None,preserve_range=True,order=0) 
            im = im.astype(np.uint8)
            im = Image.fromarray(im) 
            print("Downsampled image dimensions: ", np.array(im).shape)
            print("Downsampled image values: ",  np.unique(im))
            in_mem_file = io.BytesIO()
            im.save(in_mem_file, format="PNG") 
            in_mem_file.seek(0)
            filename_split = os.path.splitext(str(i)) 
            filename_zero, fileext = filename_split 
            basename = os.path.basename(filename_zero) 
            key = outfile

            # Upload image to s3
            s3.Bucket('veda-ai-supraglacial-meltponds').put_object(Key=key, Body=in_mem_file)
        except IOError:
            print("cannot create thumbnail for '%s'" % i)
    return


# Downsample images

for i_url in items_urls_s3:
    # Option can be "image" or "label"
    substring = "original"
    if option == "image":
        substring1 = ".JPG" 
    else:
        substring1 = "_classified.tif"
    if substring in str(i_url) and substring1 in str(i_url):
        i = i_url[0]
        url = i_url[1]
        if option == "image":
            image_in_mem = s3_utils.image_from_s3("veda-ai-supraglacial-meltponds", str(i))
        else:
            image_in_mem = s3_utils.image_from_s3_rio(url)
            #option = "label"
        downsample(image_in_mem, i, option)


def tile(image, image_name, option):
    """Tiles images using the tile_size parameter.
    Args:
        image (PIL image): An image opened in PIL.
        image_name (string): The image filename.
        option (string): Choice of r,g,b image or label image.
    Returns:
        n/a
    """
    open_cv_image = np.array(image)
    image = np.array(image)
    print("downsampled image values pre tiling: ", np.unique(image))
    if option == "image":
        # Convert RGB to BGR 
        image = open_cv_image[:, :, ::-1].copy() 
    else:
        continue

    n_tiles_x = math.ceil(image.shape[1]/tile_size)
    n_tiles_y = math.ceil(image.shape[0]/tile_size)

    make_last_part_full = True; 

    for n_tile_x in range(n_tiles_x):
        for n_tile_y in range(n_tiles_y):
            print("n_tile_x, n_tiles_y: ", n_tile_x, n_tiles_y)
            start_x = n_tile_x*tile_size
            end_x = start_x + tile_size
            start_y = n_tile_y*tile_size
            end_y = start_y + tile_size;

            if(end_y > image.shape[0]):
                end_y = image.shape[0]

            if(end_x > image.shape[1]):
                end_x = image.shape[1]

            if( make_last_part_full == True and (n_tile_x == n_tiles_x-1 or n_tile_y == n_tiles_y-1) ):
                start_x = end_x - tile_size
                start_y = end_y - tile_size

            current_tile = image[start_y:end_y, start_x:end_x]
            if option == "image":
                current_tile = cv2.cvtColor(current_tile, cv2.COLOR_BGR2RGB)
            else:
                continue
            print("current_tile shape: ", current_tile.shape)
            in_mem_file = io.BytesIO()
            print("output values in tile: ", np.unique(current_tile))
            current_tile_pil = Image.fromarray(current_tile)
            current_tile_pil.save(in_mem_file, format="PNG")
            in_mem_file.seek(0)
            filename_split = os.path.splitext(image_name) 
            filename_zero, fileext = filename_split 
            basename = os.path.basename(filename_zero) 
            if option == "image":
                key = f"tiled_images/{basename}_{str(n_tile_x)}_({str(n_tile_y)}).png"
            else:
                key = f"tiled_labels/{basename}/{str(n_tile_x)}_{str(n_tile_y)}.png"
            s3.Bucket('veda-ai-supraglacial-meltponds').put_object(Key=key, Body=in_mem_file)
    return

# Tile images

items_s3 = []

for i in iterate_bucket_items(bucket='veda-ai-supraglacial-meltponds'):
    ik = i["Key"]
    items_s3.append(ik)

for i in items_s3:
    # Option can be "image" or "label"
    if option == "image":
        substring = f"downsampled_{option}s"
    else:
        substring = f"downsampled_{option}s"
    substring1 = "png"
    if substring in str(i) and substring1 in str(i): 
        print(i)
        image_in_mem = image_from_s3("veda-ai-supraglacial-meltponds", str(i))
        tile(image_in_mem, i, option)
    else:
        continue



