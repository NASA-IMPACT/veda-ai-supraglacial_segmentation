import os, sys, glob
import boto3
from PIL import Image
import io
import rasterio as rio

def image_from_s3(bucket, key):
    """Obtains an AWS S3 path for image files meeting a certain criteria and opens them with PIL.
    Args:
        bucket (string): The AWS bucket in which the image is stored.
        key (string): The image path within the bucket
    Returns:
        Image.open(io.BytesIO(img_data)) (PIL Image): an Image opened with PIL.
    """
    bucket = s3.Bucket(bucket)
    image = bucket.Object(key)
    img_data = image.get().get('Body').read()
    return Image.open(io.BytesIO(img_data)) 

def image_from_s3_rio(url):
    """Obtains an AWS S3 path for image files meeting a certain criteria and opens them with Rasterio.
    Args:
        url (string): The AWS S3 url for an image file.
    Returns:
        Image.fromarray(np.squeeze(np.array(dataset_array).astype(np.uint8))) (PIL Image): an Image opened with PIL.
    """
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
                
