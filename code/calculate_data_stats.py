import collections
import json
import os
import numpy as np
from utils.datastats import get_class_counts_partitions, get_class_rgb_ranges_partitions

with open("../configs/datasets.json", "r") as f:
    datasets = json.load(f)

ROOT_DIR = datasets['ROOT_DIR']

x_train_filenames_partition_fn = os.path.join(ROOT_DIR, datasets['ROOT_DIR'])
y_train_filenames_partition_fn = os.path.join(ROOT_DIR, datasets['y_train_filenames_partition'])
x_val_filenames_partition_fn = os.path.join(ROOT_DIR, datasets['x_val_filenames_partition'])
y_val_filenames_partition_fn = os.path.join(ROOT_DIR, datasets['y_val_filenames_partition'])
x_test_filenames_partition_fn = os.path.join(ROOT_DIR, datasets['x_test_filenames_partition'])
y_test_filenames_partition_fn = os.path.join(ROOT_DIR, datasets['y_test_filenames_partition'])

try:
    x_train_filenames = [line.strip() for line in open(x_train_filenames_partition_fn, 'r')]
    y_train_filenames = [line.strip() for line in open(y_train_filenames_partition_fn, 'r')]
    x_val_filenames = [line.strip() for line in open(x_val_filenames_partition_fn, 'r')]
    y_val_filenames = [line.strip() for line in open(y_val_filenames_partition_fn, 'r')]
    x_test_filenames = [line.strip() for line in open(x_test_filenames_partition_fn, 'r')]
    y_test_filenames = [line.strip() for line in open(y_test_filenames_partition_fn, 'r')]
except:
    print("partition files do not exist")

train_list = [line.strip() for line in open(os.path.join(root_dir, datasets['train_filenames']), 'r')]

x_train_filenames_d = [d[53:63] for d in x_train_filenames]
x_val_filenames_d = [d[53:63] for d in x_val_filenames]
x_test_filenames_d = [d[53:63] for d in x_test_filenames]


countertr=collections.Counter(x_train_filenames_d)
counterv=collections.Counter(x_val_filenames_d)
countert=collections.Counter(x_test_filenames_d)

print("frequency of dates in training partition: ", countertr)
print("frequency of dates in validation partition: ", counterv)
print("frequency of dates in testing partition: ", countert)

pcts_0 = []
pcts_1 = []
pcts_2 = []
pcts_3 = []
pcts_4 = []
pcts_5 = []

all_0_count, all_1_count, all_2_count, all_3_count, all_4_count, all_5_count = 0,0,0,0,0,0
all_0_count, all_1_count, all_2_count, all_3_count, all_4_count, all_5_count = get_class_counts_pcts_partitions(y_train_filenames, all_0_count, all_1_count, all_2_count, all_3_count, all_4_count, all_5_count)

df0 = pd.DataFrame(pcts_0, columns=['image', 'percent_0'])
df1 = pd.DataFrame(pcts_1, columns=['image', 'percent_1'])
df2 = pd.DataFrame(pcts_2, columns=['image', 'percent_2'])
df3 = pd.DataFrame(pcts_3, columns=['image', 'percent_3'])
df4 = pd.DataFrame(pcts_4, columns=['image', 'percent_4'])
df5 = pd.DataFrame(pcts_5, columns=['image', 'percent_5'])

df0 = df0.loc[~df0.index.duplicated(keep='first')]
df1 = df1.loc[~df1.index.duplicated(keep='first')]
df2 = df2.loc[~df2.index.duplicated(keep='first')]
df3 = df3.loc[~df3.index.duplicated(keep='first')]
df4 = df4.loc[~df4.index.duplicated(keep='first')]
df5 = df5.loc[~df5.index.duplicated(keep='first')]

print("TRAIN length of df0: ", len(df0), "length of df1: ", len(df1), "length of df2: ", len(df2), "length of df3: ", len(df3), "length of df4: ", len(df4), "length of df5: ", len(df5))

df_all = pd.concat([i.set_index('image') for i in [df0,df1,df2,df3,df4]],axis=1, join='outer')
df_all = df_all.reset_index(drop=True)

df_all.to_csv('images_class_percents_manual_qa_train.csv')

pcts_0 = []
pcts_1 = []
pcts_2 = []
pcts_3 = []
pcts_4 = []
pcts_5 = []

all_0_count, all_1_count, all_2_count, all_3_count, all_4_count, all_5_count = 0,0,0,0,0,0
all_0_count, all_1_count, all_2_count, all_3_count, all_4_count, all_5_count = get_class_counts_pcts_partitions(y_val_filenames, all_0_count, all_1_count, all_2_count, all_3_count, all_4_count, all_5_count)

df0 = pd.DataFrame(pcts_0, columns=['image', 'percent_0'])
df1 = pd.DataFrame(pcts_1, columns=['image', 'percent_1'])
df2 = pd.DataFrame(pcts_2, columns=['image', 'percent_2'])
df3 = pd.DataFrame(pcts_3, columns=['image', 'percent_3'])
df4 = pd.DataFrame(pcts_4, columns=['image', 'percent_4'])
df5 = pd.DataFrame(pcts_5, columns=['image', 'percent_5'])

df0 = df0.loc[~df0.index.duplicated(keep='first')]
df1 = df1.loc[~df1.index.duplicated(keep='first')]
df2 = df2.loc[~df2.index.duplicated(keep='first')]
df3 = df3.loc[~df3.index.duplicated(keep='first')]
df4 = df4.loc[~df4.index.duplicated(keep='first')]
df5 = df5.loc[~df5.index.duplicated(keep='first')]

print("VAL length of df0: ", len(df0), "length of df1: ", len(df1), "length of df2: ", len(df2), "length of df3: ", len(df3), "length of df4: ", len(df4), "length of df5: ", len(df5))

df_all = pd.concat([i.set_index('image') for i in [df0,df1,df2,df3,df4]],axis=1, join='outer').reset_index(drop=True) 

df_all.to_csv('images_class_percents_manual_qa_val.csv')

pcts_0 = []
pcts_1 = []
pcts_2 = []
pcts_3 = []
pcts_4 = []
pcts_5 = []

all_0_count, all_1_count, all_2_count, all_3_count, all_4_count, all_5_count = 0,0,0,0,0,0
all_0_count, all_1_count, all_2_count, all_3_count, all_4_count, all_5_count = get_class_counts_pcts_partitions(y_test_filenames, all_0_count, all_1_count, all_2_count, all_3_count, all_4_count, all_5_count)

df0 = pd.DataFrame(pcts_0, columns=['image', 'percent_0'])
df1 = pd.DataFrame(pcts_1, columns=['image', 'percent_1'])
df2 = pd.DataFrame(pcts_2, columns=['image', 'percent_2'])
df3 = pd.DataFrame(pcts_3, columns=['image', 'percent_3'])
df4 = pd.DataFrame(pcts_4, columns=['image', 'percent_4'])
df5 = pd.DataFrame(pcts_5, columns=['image', 'percent_5'])

df0 = df0.loc[~df0.index.duplicated(keep='first')]
df1 = df1.loc[~df1.index.duplicated(keep='first')]
df2 = df2.loc[~df2.index.duplicated(keep='first')]
df3 = df3.loc[~df3.index.duplicated(keep='first')]
df4 = df4.loc[~df4.index.duplicated(keep='first')]
df5 = df5.loc[~df5.index.duplicated(keep='first')]

print("TEST length of df0: ", len(df0), "length of df1: ", len(df1), "length of df2: ", len(df2), "length of df3: ", len(df3), "length of df4: ", len(df4), "length of df5: ", len(df5))

df_all = pd.concat([i.set_index('image') for i in [df0,df1,df2,df3,df4]],axis=1, join='outer').reset_index(drop=True) 
df_all.to_csv('images_class_percents_manual_qa_test.csv')


train_0_count, train_1_count, train_2_count, train_3_count, train_4_count, train_5_count = 0,0,0,0,0,0
val_0_count, val_1_count, val_2_count, val_3_count, val_4_count, val_5_count = 0,0,0,0,0,0
test_0_count, test_1_count, test_2_count, test_3_count, test_4_count, test_5_count = 0,0,0,0,0,0
train_0_count, train_1_count, train_2_count, train_3_count, train_4_count, train_5_count = get_class_counts_partitions(y_train_filenames, train_0_count, train_1_count, train_2_count, train_3_count, train_4_count, train_5_count)
val_0_count, val_1_count, val_2_count, val_3_count, val_4_count, val_5_count = get_class_counts_partitions(y_val_filenames, val_0_count, val_1_count, val_2_count, val_3_count, val_4_count, val_5_count)
test_0_count, test_1_count, test_2_count, test_3_count, test_4_count, test_5_count = get_class_counts_partitions(y_test_filenames, test_0_count, test_1_count, test_2_count, test_3_count, test_4_count, test_5_count)


r_0_vals_tr, r_1_vals_tr, r_2_vals_tr, r_3_vals_tr, r_4_vals_tr, r_5_vals_tr =  [], [], [], [], [], []
g_0_vals_tr, g_1_vals_tr, g_2_vals_tr, g_3_vals_tr, g_4_vals_tr, g_5_vals_tr =  [], [], [], [], [], []
b_0_vals_tr, b_1_vals_tr, b_2_vals_tr, b_3_vals_tr, b_4_vals_tr, b_5_vals_tr =  [], [], [], [], [], []

r_0_vals_v, r_1_vals_v, r_2_vals_v, r_3_vals_v, r_4_vals_v, r_5_vals_v =  [], [], [], [], [], []
g_0_vals_v, g_1_vals_v, g_2_vals_v, g_3_vals_v, g_4_vals_v, g_5_vals_v =  [], [], [], [], [], []
b_0_vals_v, b_1_vals_v, b_2_vals_v, b_3_vals_v, b_4_vals_v, b_5_vals_v =  [], [], [], [], [], []

r_0_vals_t, r_1_vals_t, r_2_vals_t, r_3_vals_t, r_4_vals_t, r_5_vals_t =  [], [], [], [], [], []
g_0_vals_t, g_1_vals_t, g_2_vals_t, g_3_vals_t, g_4_vals_t, g_5_vals_t =  [], [], [], [], [], []
b_0_vals_t, b_1_vals_t, b_2_vals_t, b_3_vals_t, b_4_vals_t, b_5_vals_t =  [], [], [], [], [], []

get_class_rgb_ranges_partitions(x_train_filenames, y_train_filenames, r_0_vals_tr, r_1_vals_tr, r_2_vals_tr, r_3_vals_tr, r_4_vals_tr, r_5_vals_tr , g_0_vals_tr, g_1_vals_tr, g_2_vals_tr, g_3_vals_tr, g_4_vals_tr, g_5_vals_tr, b_0_vals_tr, b_1_vals_tr, b_2_vals_tr, b_3_vals_tr, b_4_vals_tr, b_5_vals_tr)
get_class_rgb_ranges_partitions(x_val_filenames, y_val_filenames, r_0_vals_v, r_1_vals_v, r_2_vals_v, r_3_vals_v, r_4_vals_v, r_5_vals_v, g_0_vals_v, g_1_vals_v, g_2_vals_v, g_3_vals_v, g_4_vals_v, g_5_vals_v, b_0_vals_v, b_1_vals_v, b_2_vals_v, b_3_vals_v, b_4_vals_v, b_5_vals_v)
get_class_rgb_ranges_partitions(x_test_filenames, y_test_filenames, r_0_vals_t, r_1_vals_t, r_2_vals_t, r_3_vals_t, r_4_vals_t, r_5_vals_t, g_0_vals_t, g_1_vals_t, g_2_vals_t, g_3_vals_t, g_4_vals_t, g_5_vals_t, b_0_vals_t, b_1_vals_t, b_2_vals_t, b_3_vals_t, b_4_vals_t, b_5_vals_t)

rgb_avg_0_tr = [np.mean(r_0_vals_tr),np.mean(g_0_vals_tr),np.mean(b_0_vals_tr)]
rgb_avg_1_tr = [np.mean(r_1_vals_tr),np.mean(g_1_vals_tr),np.mean(b_1_vals_tr)]
rgb_avg_2_tr = [np.mean(r_2_vals_tr),np.mean(g_2_vals_tr),np.mean(b_2_vals_tr)]
rgb_avg_3_tr = [np.mean(r_3_vals_tr),np.mean(g_3_vals_tr),np.mean(b_3_vals_tr)]
rgb_avg_4_tr = [np.mean(r_4_vals_tr),np.mean(g_4_vals_tr),np.mean(b_4_vals_tr)]
rgb_avg_5_tr = [np.mean(r_5_vals_tr),np.mean(g_5_vals_tr),np.mean(b_5_vals_tr)]

rgb_avg_0_v = [np.mean(r_0_vals_v),np.mean(g_0_vals_v),np.mean(b_0_vals_v)]
rgb_avg_1_v = [np.mean(r_1_vals_v),np.mean(g_1_vals_v),np.mean(b_1_vals_v)]
rgb_avg_2_v = [np.mean(r_2_vals_v),np.mean(g_2_vals_v),np.mean(b_2_vals_v)]
rgb_avg_3_v = [np.mean(r_3_vals_v),np.mean(g_3_vals_v),np.mean(b_3_vals_v)]
rgb_avg_4_v = [np.mean(r_4_vals_v),np.mean(g_4_vals_v),np.mean(b_4_vals_v)]
rgb_avg_5_v = [np.mean(r_5_vals_v),np.mean(g_5_vals_v),np.mean(b_5_vals_v)]

rgb_avg_0_t = [np.mean(r_0_vals_t),np.mean(g_0_vals_t),np.mean(b_0_vals_t)]
rgb_avg_1_t = [np.mean(r_1_vals_t),np.mean(g_1_vals_t),np.mean(b_1_vals_t)]
rgb_avg_2_t = [np.mean(r_2_vals_t),np.mean(g_2_vals_t),np.mean(b_2_vals_t)]
rgb_avg_3_t = [np.mean(r_3_vals_t),np.mean(g_3_vals_t),np.mean(b_3_vals_t)]
rgb_avg_4_t = [np.mean(r_4_vals_t),np.mean(g_4_vals_t),np.mean(b_4_vals_t)]
rgb_avg_5_t = [np.mean(r_5_vals_t),np.mean(g_5_vals_t),np.mean(b_5_vals_t)]

rgb_std_0_tr = [np.std(r_0_vals_tr),np.std(g_0_vals_tr),np.std(b_0_vals_tr)]
rgb_std_1_tr = [np.std(r_1_vals_tr),np.std(g_1_vals_tr),np.std(b_1_vals_tr)]
rgb_std_2_tr = [np.std(r_2_vals_tr),np.std(g_2_vals_tr),np.std(b_2_vals_tr)]
rgb_std_3_tr = [np.std(r_3_vals_tr),np.std(g_3_vals_tr),np.std(b_3_vals_tr)]
rgb_std_4_tr = [np.std(r_4_vals_tr),np.std(g_4_vals_tr),np.std(b_4_vals_tr)]
rgb_std_5_tr = [np.std(r_5_vals_tr),np.std(g_5_vals_tr),np.std(b_5_vals_tr)]

rgb_std_0_v = [np.std(r_0_vals_v),np.std(g_0_vals_v),np.std(b_0_vals_v)]
rgb_std_1_v = [np.std(r_1_vals_v),np.std(g_1_vals_v),np.std(b_1_vals_v)]
rgb_std_2_v = [np.std(r_2_vals_v),np.std(g_2_vals_v),np.std(b_2_vals_v)]
rgb_std_3_v = [np.std(r_3_vals_v),np.std(g_3_vals_v),np.std(b_3_vals_v)]
rgb_std_4_v = [np.std(r_4_vals_v),np.std(g_4_vals_v),np.std(b_4_vals_v)]
rgb_std_5_v = [np.std(r_5_vals_v),np.std(g_5_vals_v),np.std(b_5_vals_v)]

rgb_std_0_t = [np.std(r_0_vals_t),np.std(g_0_vals_t),np.std(b_0_vals_t)]
rgb_std_1_t = [np.std(r_1_vals_t),np.std(g_1_vals_t),np.std(b_1_vals_t)]
rgb_std_2_t = [np.std(r_2_vals_t),np.std(g_2_vals_t),np.std(b_2_vals_t)]
rgb_std_3_t = [np.std(r_3_vals_t),np.std(g_3_vals_t),np.std(b_3_vals_t)]
rgb_std_4_t = [np.std(r_4_vals_t),np.std(g_4_vals_t),np.std(b_4_vals_t)]
rgb_std_5_t = [np.std(r_5_vals_t),np.std(g_5_vals_t),np.std(b_5_vals_t)]

print("number of pixels per class in training partition: ", "no data = ", train_0_count, "snow and bright ice = ", train_1_count, "dark and thin ice = ", train_2_count, "melt pond and submerged ice = ", train_3_count, "open water = ", train_4_count, "ridge shadows = ", train_5_count)
print("number of pixels per class in validation partition: ", "no data = ", val_0_count, "snow and bright ice = ", val_1_count, "dark and thin ice = ", val_2_count, "melt pond and submerged ice = ", val_3_count, "open water = ", val_4_count, "ridge shadows = ", val_5_count)
print("number of pixels per class in testing partition: ", "no data = ", test_0_count, "snow and bright ice = ", test_1_count, "dark and thin ice = ", test_2_count, "melt pond and submerged ice = ", test_3_count, "open water = ", test_4_count, "ridge shadows = ", test_5_count)

print("rgb averages per class in training partition: ", "no data = ", rgb_avg_0_tr, "snow and bright ice = ", rgb_avg_1_tr, "dark and thin ice = ", rgb_avg_2_tr, "melt pond and submerged ice = ", rgb_avg_3_tr, "open water = ", rgb_avg_4_tr, "ridge shadows = ", rgb_avg_5_tr)
print("rgb averages per class in validation partition: ", "no data = ", rgb_avg_0_v, "snow and bright ice = ", rgb_avg_1_v, "dark and thin ice = ", rgb_avg_2_v, "melt pond and submerged ice = ", rgb_avg_3_v, "open water = ", rgb_avg_4_v, "ridge shadows = ", rgb_avg_5_v)
print("rgb averages per class in testing partition: ", "no data = ", rgb_avg_0_t, "snow and bright ice = ", rgb_avg_1_t, "dark and thin ice = ", rgb_avg_2_t, "melt pond and submerged ice = ", rgb_avg_3_t, "open water = ", rgb_avg_4_t, "ridge shadows = ", rgb_avg_5_t)

print("rgb standard deviation per class in training partition: ", "no data = ", rgb_std_0_tr, "snow and bright ice = ", rgb_std_1_tr, "dark and thin ice = ", rgb_std_2_tr, "melt pond and submerged ice = ", rgb_std_3_tr, "open water = ", rgb_std_4_tr, "ridge shadows = ", rgb_std_5_tr)
print("rgb standard deviation per class in validation partition: ", "no data = ", rgb_std_0_v, "snow and bright ice = ", rgb_std_1_v, "dark and thin ice = ", rgb_std_2_v, "melt pond and submerged ice = ", rgb_std_3_v, "open water = ", rgb_std_4_v, "ridge shadows = ", rgb_std_5_v)
print("rgb standard deviation per class in testing partition: ", "no data = ", rgb_std_0_t, "snow and bright ice = ", rgb_std_1_t, "dark and thin ice = ", rgb_std_2_t, "melt pond and submerged ice = ", rgb_std_3_t, "open water = ", rgb_std_4_t, "ridge shadows = ", rgb_std_5_t)
