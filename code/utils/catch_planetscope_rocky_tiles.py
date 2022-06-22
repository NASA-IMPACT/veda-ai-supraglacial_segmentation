import glob, io, os
import numpy as np
from PIL import Image

# Range of RGB values for exposed rock in PlanetScope imagery; CAUTION: prone to PlanetScope's image heterogeneity
rock_red = range(110,180)
rock_green = range(110,180)
rock_blue = range(110,180)
rocks = [rock_red,rock_green,rock_blue]
rocks_range = list(itertools.product(*rocks))
print("rocks_range: ", list(itertools.product(*rocks)))


def catch_rocky_tiles(image_dir):
    """Takes a directory of planetscope images and checks if any pixels fall within a certain range 
    identified with exposed rock in planetscope images.
    Args:
        image_dir (string): The path to a planetscope images.
    Returns:
        rocky_tiles (list): A list of images with exposed rock.
    """
    rocky_tiles = []
    images = glob.glob(f'{image_dir}/*.png')
    for image in images:
        filename_split = os.path.splitext(image) 
        filename_zero, fileext = filename_split 
        basename = os.path.basename(filename_zero) 
        #print(basename)
        if not os.path.isfile(f'{plot_dir}{basename}_vals.png'):
            try:
                image_in_mem = Image.open(image)
                img_rgb = np.array(image_in_mem)
                for rock in rocks_range:
                  a,b,c = rock[0], rock[1], rock[2]
                  ind = np.where((img_rgb[:, :, 0]==a) & (img_rgb[:, :, 1]==b) & (img_rgb[:, :, 2]==c))
                  #print(ind)
                  answer = list(zip(ind[0], ind[1]))
                  if answer:
                    #print(answer, a,b,c)
                    #print(basename)
                    rocky_tiles.append(basename)
    return rocky_tiles
