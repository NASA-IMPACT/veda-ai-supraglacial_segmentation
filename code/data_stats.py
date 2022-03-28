import os, collections
import numpy as numpy
from PIL import Image

root_dir = '/home/ubuntu/data/'

x_train_filenames_partition_fn = os.path.join(root_dir,'x_train_filenames_partition.txt')
y_train_filenames_partition_fn = os.path.join(root_dir,'y_train_filenames_partition.txt')
x_val_filenames_partition_fn = os.path.join(root_dir,'x_val_filenames_partition.txt')
y_val_filenames_partition_fn = os.path.join(root_dir,'y_val_filenames_partition.txt')
x_test_filenames_partition_fn = os.path.join(root_dir,'x_test_filenames_partition.txt')
y_test_filenames_partition_fn = os.path.join(root_dir,'y_test_filenames_partition.txt')

try:
  x_train_filenames = [line.strip() for line in open(x_train_filenames_partition_fn, 'r')]
  y_train_filenames = [line.strip() for line in open(y_train_filenames_partition_fn, 'r')]
  x_val_filenames = [line.strip() for line in open(x_val_filenames_partition_fn, 'r')]
  y_val_filenames = [line.strip() for line in open(y_val_filenames_partition_fn, 'r')]
  x_test_filenames = [line.strip() for line in open(x_test_filenames_partition_fn, 'r')]
  y_test_filenames = [line.strip() for line in open(y_test_filenames_partition_fn, 'r')]
except:
  print("partition files do not exist")


x_train_filenames_d = [d[53:63] for d in x_train_filenames]
x_val_filenames_d = [d[53:63] for d in x_val_filenames]
x_test_filenames_d = [d[53:63] for d in x_test_filenames]


countertr=collections.Counter(x_train_filenames_d)
counterv=collections.Counter(x_val_filenames_d)
countert=collections.Counter(x_test_filenames_d)

print("frequency of dates in training partition: ", countertr)
print("frequency of dates in validation partition: ", counterv)
print("frequency of dates in testing partition: ", countert)

def get_class_counts_partitions(partition_list, c0,c1,c2,c3,c4,c5):
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


train_0_count, train_1_count, train_2_count, train_3_count, train_4_count, train_5_count = 0,0,0,0,0,0
val_0_count, val_1_count, val_2_count, val_3_count, val_4_count, val_5_count = 0,0,0,0,0,0
test_0_count, test_1_count, test_2_count, test_3_count, test_4_count, test_5_count = 0,0,0,0,0,0
train_0_count, train_1_count, train_2_count, train_3_count, train_4_count, train_5_count = get_class_counts_partitions(y_train_filenames, train_0_count, train_1_count, train_2_count, train_3_count, train_4_count, train_5_count)
val_0_count, val_1_count, val_2_count, val_3_count, val_4_count, val_5_count = get_class_counts_partitions(y_val_filenames, val_0_count, val_1_count, val_2_count, val_3_count, val_4_count, val_5_count)
test_0_count, test_1_count, test_2_count, test_3_count, test_4_count, test_5_count = get_class_counts_partitions(y_test_filenames, test_0_count, test_1_count, test_2_count, test_3_count, test_4_count, test_5_count)


print("number of pixels per class in training partition: ", "no data = ", train_0_count, "snow and bright ice = ", train_1_count, "dark and thin ice = ", train_2_count, "melt pond and submerged ice = ", train_3_count, "open water = ", train_4_count, "ridge shadows = ", train_5_count)
print("number of pixels per class in validation partition: ", "no data = ", val_0_count, "snow and bright ice = ", val_1_count, "dark and thin ice = ", val_2_count, "melt pond and submerged ice = ", val_3_count, "open water = ", val_4_count, "ridge shadows = ", val_5_count)
print("number of pixels per class in testing partition: ", "no data = ", test_0_count, "snow and bright ice = ", test_1_count, "dark and thin ice = ", test_2_count, "melt pond and submerged ice = ", test_3_count, "open water = ", test_4_count, "ridge shadows = ", test_5_count)