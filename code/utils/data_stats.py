import numpy as np
from PIL import Image

def return_class_count(class_int,c):
    """Returns the class count in unis of pixels per tile."""
    if class_int in colors:
        c+=color_count_dict[class_int]
        return c

def return_class_pct(class_int,pctl):
    """Returns the class-wise percentage of pixels per tile."""
    if class_int in colors:
        pct = float(color_count_dict[class_int])/img.size
        pctl.append((i, pct))
        return pctl

def get_class_counts_partitions(partition_list, c0,c1,c2,c3,c4,c5, num_classes):
    """Takes a list of filenames in a partition and counters (per class). Calculates the
    number of pixels per class in each image in the partition and adds them to the 
    partition sum.
    Args:
        partition_list (list): A list of images paths.
        c0,c1,c2,c3,c4,c5 (int): The counters.
    Returns:
        c0,c1,c2,c3,c4,c5 (int): The counts.
    """
    for i in partition_list:
        img = np.array(Image.open(i))
        #print(img.shape)
        colors, counts = np.unique(img, return_counts = True)
        #print(colors, counts)
        color_count_dict = dict(zip(colors, counts))
        print(color_count_dict)
        c0=return_class_count(0,c0)
        c1=return_class_count(1,c1)
        c2=return_class_count(2,c2)
        c3=return_class_count(3,c3)
        c4=return_class_count(4,c4)
        c5=return_class_count(5,c5)
    return c0,c1,c2,c3,c4,c5

def get_class_pcts_partitions(partition_list, pcts_0,pcts_1,pcts_2,pcts_3,pcts_4,pcts_5):
    """Takes a list of filenames in a partition. Calculates the
    percentage of each class (in terms of pixels) in each image in the 
    partition and adds them to the partition sum.
    Args:
        partition_list (list): A list of images paths.
        pcts_0,pcts_1,pcts_2,pcts_3,pcts_4,pcts_5 (list): Lists containing class percentages.
    Returns:
        n/a
    """
    for i in partition_list:
        img = np.array(Image.open(i))
        colors, counts = np.unique(img, return_counts = True)
        color_count_dict = dict(zip(colors, counts))
        print(color_count_dict)
        pcts_0=return_class_pct(0,pcts_0)
        pcts_1=return_class_pct(1,pcts_1)
        pcts_2=return_class_pct(2,pcts_2)
        pcts_3=return_class_pct(3,pcts_3)
        pcts_4=return_class_pct(4,pcts_4)
        pcts_5=return_class_pct(5,pcts_5)
    return

def get_class_rgb_ranges_partitions(partition_list_x, partition_list_y, r0,r1,r2,r3,r4,r5, g0,g1,g2,g3,g4,g5, b0,b1,b2,b3,b4,b5):
    """Takes lists of image filenames and label filenames for a partition, as well as rgb value counters. 
    Calculates the percentage of each class (in terms of pixels) in each image in the 
    partition and adds them to the partition sum.
    Args:
        partition_list (list): A list of images paths.
        r0,r1,r2,r3,r4,r5, g0,g1,g2,g3,g4,g5, b0,b1,b2,b3,b4,b5 (int): The rgb value counters.
    Returns:
        n/a
    """
    for i, y in zip(partition_list_x, partition_list_y):
        rgb = np.array(Image.open(i))
        img = np.array(Image.open(y))
        #print(img.shape)
        colors, counts = np.unique(img, return_counts = True)
        #print(colors, counts)
        color_count_dict = dict(zip(colors, counts))
        print(color_count_dict)
        def return_rgb_values(r,g,b,img, class_int):
            if class_int in colors:
                coords = np.column_stack(np.where(img == class_int))
                for coord in coords:
                    x,y = coord
                    rv, gv, bv = rgb[x,y]
                    r.append(rv)
                    g.append(gv)
                    b.append(bv)
            return

        return_rgb_values(r0,g0,b0,img, 0)
        return_rgb_values(r1,g1,b1,img, 1)
        return_rgb_values(r2,g2,b2,img, 2)
        return_rgb_values(r3,g3,b3,img, 3)
        return_rgb_values(r4,g4,b4,img, 4)
        return_rgb_values(r5,g5,b5,img, 5)
        return 