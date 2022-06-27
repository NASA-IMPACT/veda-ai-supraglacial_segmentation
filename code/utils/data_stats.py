import os, collections
import numpy as np
from PIL import Image

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
        if 0 in colors:
            print("count for class 0: ", color_count_dict[0])
            c0+=color_count_dict[0]
        if 1 in colors:
            print("count for class 1: ", color_count_dict[1])
            c1+=color_count_dict[1]
        if 2 in colors:
            print("count for class 2: ", color_count_dict[2])
            c2+=color_count_dict[2]
        if 3 in colors:
            print("count for class 3: ", color_count_dict[3])
            c3+=color_count_dict[3]
        if 4 in colors:
            print("count for class 4: ", color_count_dict[4])
            c4+=color_count_dict[4]
        if 5 in colors:
            print("count for class 5: ", color_count_dict[5])
            c5+=color_count_dict[5]
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
        if 0 in colors:
            pct_0 = float(color_count_dict[0])/img.size
            pcts_0.append((i, color_count_dict[0]))
        if 1 in colors:
            pct_1 = float(color_count_dict[1])/img.size
            pcts_1.append((i, color_count_dict[1]))
        if 2 in colors:
            pct_2 = float(color_count_dict[2])/img.size
            pcts_2.append((i, color_count_dict[2]))
        if 3 in colors:
            pct_3 = float(color_count_dict[3])/img.size
            pcts_3.append((i, color_count_dict[3]))
        if 4 in colors:
            pct_4 = float(color_count_dict[4])/img.size
            pcts_4.append((i, color_count_dict[4]))
        if 5 in colors:
            pct_5 = float(color_count_dict[5])/img.size
            pcts_5.append((i, color_count_dict[5]))
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
        if 0 in colors:
            coords = np.column_stack(np.where(img == 0))
            print("coords for 0: ", coords)
            for coord in coords:
                x,y = coord
                print("rgb values for class 0: ", rgb[x,y])
                r0v, g0v, b0v = rgb[x,y]
                r0.append(r0v)
                g0.append(g0v)
                b0.append(b0v)
        if 1 in colors:
            coords = np.column_stack(np.where(img == 1))
            for coord in coords:
                x,y = coord
                r1v, g1v, b1v = rgb[x,y]
                print("rgb values for class 1: ", rgb[x,y])
                r1.append(r1v)
                g1.append(g1v)
                b1.append(b1v)
        if 2 in colors:
            coords = np.column_stack(np.where(img == 2))
            for coord in coords:
                x,y = coord
                r2v, g2v, b2v = rgb[x,y]
                print("rgb values for class 2: ", rgb[x,y])
                r2.append(r2v)
                g2.append(g2v)
                b2.append(b2v)
        if 3 in colors:
            coords = np.column_stack(np.where(img == 3))
            for coord in coords:
                x,y = coord
                r3v, g3v, b3v = rgb[x,y]
                print("rgb values for class 3: ", rgb[x,y])
                r3.append(r3v)
                g3.append(g3v)
                b3.append(b3v)
        if 4 in colors:
            coords = np.column_stack(np.where(img == 4))
            for coord in coords:
                x,y = coord
                r4v, g4v, b4v = rgb[x,y]
                print("rgb values for class 4: ", rgb[x,y])
                r4.append(r4v)
                g4.append(g4v)
                b4.append(b4v)
        if 5 in colors:
            coords = np.column_stack(np.where(img == 5))
            for coord in coords:
                x,y = coord
                r5v, g5v, b5v = rgb[x,y]
                print("rgb values for class 5: ", rgb[x,y])
                r5.append(r5v)
                g5.append(g5v)
                b5.append(b5v)
        return 