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

downsample_factor = 5.0
tile_size = 224


s3 = boto3.resource('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

rio_session = rio.env.Env(aws_access_key_id=AWS_ACCESS_KEY_ID,
             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
             region_name='us-east')


def image_from_s3(bucket, key):
    bucket = s3.Bucket(bucket)
    image = bucket.Object(key)
    img_data = image.get().get('Body').read()
    return Image.open(io.BytesIO(img_data)) 

def image_from_s3_rio(url):
    with rio_session:
        with rio.open(url) as dataset:
            dataset_array = dataset.read()
            dataset_array = dataset_array.transpose(1,2,0)
    return Image.fromarray(np.squeeze(np.array(dataset_array).astype(np.uint8)))

def iterate_bucket_items(bucket):
    """
    Generator that iterates over all objects in a given s3 bucket

    See http://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.list_objects_v2 
    for return data format
    :param bucket: name of s3 bucket
    :return: dict of metadata for an object
    """


    client = client = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY_ID,aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    paginator = client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket = bucket)

    for page in page_iterator:
        if page['KeyCount'] > 0:
            for item in page['Contents']:
                yield item

items_s3 = []
urls_s3 = []
items_urls_s3 = []

for i in iterate_bucket_items(bucket='veda-ai-supraglacial-meltponds'):
    bucket='veda-ai-supraglacial-meltponds'
    ik = i["Key"]
    items_s3.append(ik)
    url = 's3://'+str(bucket)+'/'+str(ik)
    urls_s3.append(url)
    item_url = (ik, url)
    items_urls_s3.append(item_url)


def downsample(image, image_name, option):
    bucket = s3.Bucket('veda-ai-supraglacial-meltponds')
    filename_split = os.path.splitext(image_name) 
    filename_zero, fileext = filename_split 
    basename = os.path.basename(filename_zero) 
    if option == "image":
        outfile = "downsampled_images/"+basename+".png"
    else:
        outfile = "downsampled_labels/"+basename+".png"
    if i != outfile:
        try:

            im = image.copy()
            size = (np.array(im).shape[0])/5.0, (np.array(im).shape[1])/5.0
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



for i_url in items_urls_s3:
    substring = "original"
    if option == "image":
        substring1 = ".JPG" 
    else:
        substring1 = "_classified.tif"
    if substring in str(i_url) and substring1 in str(i_url):
        i = i_url[0]
        url = i_url[1]
        if option == "image":
            image_in_mem = image_from_s3("veda-ai-supraglacial-meltponds", str(i))
        else:
            image_in_mem = image_from_s3_rio(url)
        downsample(image_in_mem, i)


def tile(image, image_name):
    open_cv_image = np.array(image)
    image = np.array(image)
    print("downsampled image values pre tiling: ", np.unique(image))
    if option == "image":
        # Convert RGB to BGR 
        image = open_cv_image[:, :, ::-1].copy() 
    else:
        continue

    tileSizeX = 224;
    tileSizeY = 224;
    numTilesX = math.ceil(image.shape[1]/tileSizeX)
    numTilesY = math.ceil(image.shape[0]/tileSizeY)

    makeLastPartFull = True; 

    for nTileX in range(numTilesX):
        for nTileY in range(numTilesY):
            print("nTileX, nTileY: ", nTileX, nTileY)
            startX = nTileX*tileSizeX
            endX = startX + tileSizeX
            startY = nTileY*tileSizeY
            endY = startY + tileSizeY;

            if(endY > image.shape[0]):
                endY = image.shape[0]

            if(endX > image.shape[1]):
                endX = image.shape[1]

            if( makeLastPartFull == True and (nTileX == numTilesX-1 or nTileY == numTilesY-1) ):
                startX = endX - tileSizeX
                startY = endY - tileSizeY

            currentTile = image[startY:endY, startX:endX]
            if option == "image":
                currentTile = cv2.cvtColor(currentTile, cv2.COLOR_BGR2RGB)
            else:
                continue
            print("currentTile shape: ", currentTile.shape)
            in_mem_file = io.BytesIO()
            print("output values in tile: ", np.unique(currentTile))
            currentTile_pil = Image.fromarray(currentTile)
            currentTile_pil.save(in_mem_file, format="PNG")
            in_mem_file.seek(0)
            filename_split = os.path.splitext(image_name) 
            filename_zero, fileext = filename_split 
            basename = os.path.basename(filename_zero) 
            if option == "image":
                key = "tiled_images/" + basename + "_" + str(nTileX) + "_" + str(nTileY) +".png"
            else:
                key = "tiled_labels/" + basename + "_" + str(nTileX) + "_" + str(nTileY) +".png"
            s3.Bucket('veda-ai-supraglacial-meltponds').put_object(Key=key, Body=in_mem_file)

items_s3 = []

for i in iterate_bucket_items(bucket='veda-ai-supraglacial-meltponds'):
    ik = i["Key"]
    items_s3.append(ik)

for i in items_s3:
    if option == "image":
        substring = "downsampled_images"
    else:
        substring = "downsampled_labels"
    substring1 = "png"
    if substring in str(i) and substring1 in str(i): 
        print(i)
        image_in_mem = image_from_s3("veda-ai-supraglacial-meltponds", str(i))
        tile(image_in_mem, i)
    else:
        continue



