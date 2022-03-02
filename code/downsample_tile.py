import os, sys, glob
import boto3
from PIL import Image
import io
from configparser import ConfigParser
import numpy as np
import cv2
import math


# This script requires access to a file containing your aws credentials formatted as such:
'''
[credentials]
AWS_ACCESS_KEY_ID = xxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY = xxxxxxxxxxxxxxxxx
'''

# authenticate with AWS credentials
config = ConfigParser()
configFilePath = '/path/to/accessKeys.csv'
with open(configFilePath) as f:
    config.read_file(f)
AWS_ACCESS_KEY_ID = config.get('credentials', 'AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config.get('credentials', 'AWS_SECRET_ACCESS_KEY')

downsample_factor = 5.0
tile_size = 224


s3 = boto3.resource('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

def image_from_s3(bucket, key):
    bucket = s3.Bucket(bucket)
    image = bucket.Object(key)
    img_data = image.get().get('Body').read()
    return Image.open(io.BytesIO(img_data)) 


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

for i in iterate_bucket_items(bucket='veda-ai-supraglacial-meltponds'):
    ik = i["Key"]
    items_s3.append(ik)


def downsample(image, image_name):
    bucket = s3.Bucket('veda-ai-supraglacial-meltponds')
    filename_split = os.path.splitext(image_name) 
    filename_zero, fileext = filename_split 
    basename = os.path.basename(filename_zero) 
    outfile = "downsampled/"+basename+".jpeg"
    if i != outfile:
        try:
            im = image_from_s3("veda-ai-supraglacial-meltponds", str(i))
            print("Original image dimensions: ", np.array(im).shape)
            size = (np.array(im).shape[0])/5.0, (np.array(im).shape[1])/5.0
            im.thumbnail(size, Image.ANTIALIAS)
            print("Downsampled image dimensions: ", np.array(im).shape)
            in_mem_file = io.BytesIO()
            im.save(in_mem_file, format=im.format) 
            in_mem_file.seek(0)
            filename_split = os.path.splitext(str(i)) 
            filename_zero, fileext = filename_split 
            basename = os.path.basename(filename_zero) 
            key = outfile

            # Upload image to s3
            s3.Bucket('veda-ai-supraglacial-meltponds').put_object(Key=key, Body=in_mem_file)
        except IOError:
            print("cannot create thumbnail for '%s'" % i)


for i in items_s3:
    substring = "original"
    substring1 = "JPG"
    if substring in str(i) and substring1 in str(i):
        print(i)
        image_in_mem = image_from_s3("veda-ai-supraglacial-meltponds", str(i))
        downsample(image_in_mem, i)


def tile(image, image_name):
    open_cv_image = np.array(image) 
    # Convert RGB to BGR 
    image = open_cv_image[:, :, ::-1].copy() 

    tileSizeX = 224;
    tileSizeY = 224;
    numTilesX = math.ceil(image.shape[1]/tileSizeX)
    numTilesY = math.ceil(image.shape[0]/tileSizeY)

    makeLastPartFull = True; # in case you need even siez

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
            currentTile = cv2.cvtColor(currentTile, cv2.COLOR_BGR2RGB)
            print("currentTile shape: ", currentTile.shape)
            in_mem_file = io.BytesIO()
            currentTile_pil = Image.fromarray(currentTile)
            currentTile_pil.save(in_mem_file, format="JPEG")
            in_mem_file.seek(0)
            filename_split = os.path.splitext(image_name) 
            filename_zero, fileext = filename_split 
            basename = os.path.basename(filename_zero) 
            key = "tiled/" + basename + "_" + str(nTileX) + "_" + str(nTileY) +".jpeg"
            s3.Bucket('veda-ai-supraglacial-meltponds').put_object(Key=key, Body=in_mem_file)


for i in items_s3:
    substring = "downsampled"
    substring1 = "jpeg"
    if substring in str(i) and substring1 in str(i): 
        print(i)
        image_in_mem = image_from_s3("veda-ai-supraglacial-meltponds", str(i))
        tile(image_in_mem, i)
    else:
        continue