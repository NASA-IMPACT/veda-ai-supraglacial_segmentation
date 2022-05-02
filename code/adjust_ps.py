import os, glob, functools, fnmatch, io, shutil
from zipfile import ZipFile
from itertools import product
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as skio
from skimage import exposure

ROOT_DIR = '/home/ubuntu/'
IMG_DIR = os.path.join(root_dir,'planetscope/')

def get_test_lists(imdir):
  imgs = glob.glob(os.path.join(imdir,"*.png"))
  dset_list = []
  for img in imgs:
    filename_split = os.path.splitext(img)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    dset_list.append(basename)

  x_filenames = []
  for img_id in dset_list:
    x_filenames.append(os.path.join(imdir, "{}.png".format(img_id)))

  print("number of images: ", len(dset_list))
  return x_filenames

x_test_filenames = get_test_lists(IMG_DIR, LABEL_DIR)

print("!!!!! number of images: ", len(x_test_filenames))

kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}

def adjust_planet_image(image):
    # Load an example image
    img = skio.imread(image)
    # Gamma
    gamma_corrected = exposure.adjust_gamma(img, 3) #2)

    # Logarithmic
    logarithmic_corrected = exposure.adjust_log(img, 2) #1)

    def convert_temp(image, temp):
        image = Image.fromarray(image)
        r, g, b = kelvin_table[temp]
        matrix = ( r / 255.0, 0.0, 0.0, 0.0,
                   0.0, g / 255.0, 0.0, 0.0,
                   0.0, 0.0, b / 255.0, 0.0 )
        return np.array(image.convert('RGB', matrix))

    gamma_corrected = convert_temp(gamma_corrected, 9500)
    skio.imsave('{}_gamma_corrected_cooled9500K.png'.format(image[:-4]), gamma_corrected)

for i in x_test_filenames:
  adjust_planet_image(i)