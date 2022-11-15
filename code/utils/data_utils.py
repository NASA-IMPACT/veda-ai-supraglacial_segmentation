import glob
import numpy as np
import os
import skimage.io as skio

def get_test_lists_planetscope(imdir):
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

    #print("number of images: ", len(dset_list))
    return x_filenames

def get_train_test_lists_icebridge(imdir, lbldir):
    imgs = glob.glob(os.path.join(imdir,"*.png"))
    dset_list = []
    for img in imgs:
        file_name, file_ext = os.path.splittext(img)
        basename = os.path.basename(file_name) 
        dset_list.append(basename)

    x_filenames = []
    y_filenames = []
    for img_id in dset_list:
        x_filenames.append(os.path.join(imdir, "{}.png".format(img_id)))
        y_filenames.append(os.path.join(lbldir, "RDSISC4_{}_classified{}.png".format(img_id[:-4], img_id[-4:])))

    #print("number of images: ", len(dset_list))
    return dset_list, x_filenames, y_filenames

def get_image_label_arrays(path_df):
    # reading in preds
    label_arr_lst = path_df["label_names"].apply(skio.imread)
    pred_arr_lst = path_df["pred_names"].apply(skio.imread)

    pred_arr_lst_valid,  = []
    label_arr_lst_valid = []
    for i in range(0, len(pred_arr_lst)):
        if pred_arr_lst[i].shape != label_arr_lst[i].shape:
            
            print(f"The {i}th label has an incorrect dimension, skipping.")
            #print(pred_arr_lst[i])
            #print(label_arr_lst[i])
            #print(pred_arr_lst[i].shape)
            #print(label_arr_lst[i].shape)
            
        else:
            pred_arr_lst_valid.append(pred_arr_lst[i])
            label_arr_lst_valid.append(label_arr_lst[i])
    return pred_arr_lst_valid, label_arr_lst_valid


def write_lists_train(train_list_fn, x_train_filenames_fn, y_train_filenames_fn):
    try:
        train_list = [line.strip() for line in open(train_list_fn, 'r')]
        x_train_filenames = [line.strip() for line in open(x_train_filenames_fn, 'r')]
        y_train_filenames = [line.strip() for line in open(y_train_filenames_fn, 'r')]
    except:
        train_list, x_train_filenames, y_train_filenames = read_data.get_train_test_lists_icebridge(IMG_DIR, LABEL_DIR)
        with open(os.path.join(ROOT_DIR,datasets['train_filenames']), 'w') as f:
            for item in train_list:
                f.write("%s\n" % item)

        with open(os.path.join(ROOT_DIR,datasets['x_filenames']), 'w') as f:
            for item in x_train_filenames:
                f.write("%s\n" % item)

        with open(os.path.join(ROOT_DIR,datasets['y_filenames']), 'w') as f:
            for item in y_train_filenames:
                f.write("%s\n" % item)



def read_background(train_list, ROOT_DIR, LABEL_DIR, skip, proportion_remove):
    if not skip:
        background_list_train = []
        for i in train_list: 
            # read in each labeled images
            img = np.array(Image.open(os.path.join(LABEL_DIR,"RDSISC4_{}_classified{}.png".format(i[:-4], i[-4:]))))  
            # check if no values in image are greater than zero (background value)
            if img.max()==0:
                background_list_train.append(i)

        #print("Number of background images: ", len(background_list_train))

        with open(os.path.join(ROOT_DIR,datasets['background_files']), 'w') as f:
            for item in background_list_train:
                f.write("%s\n" % item)

    else:
        background_list_train = [line.strip() for line in open(os.path.join(ROOT_DIR,datasets['background_files']), 'r')]
        #print("Number of background images: ", len(background_list_train))
    background_removal = len(background_list_train) * float(proportion_remove)
    train_list_clean = [y for y in train_list if y not in background_list_train[0:int(background_removal)]]
    return background_removal, train_list_clean


def get_lists_partitions(train_list_clean, x_train_filenames, y_train_filenames, x_train_filenames_partition_fn, y_train_filenames_partition_fn, x_val_filenames_partition_fn, y_val_filenames_partition_fn, x_test_filenames_partition_fn, y_test_filenames_partition_fn):
    for i, img_id in zip(tqdm(range(len(train_list_clean))), train_list_clean):
        pass 
        x_train_filenames.append(os.path.join(IMG_DIR, "{}.png".format(img_id)))
        y_train_filenames.append(os.path.join(LABEL_DIR, "RDSISC4_{}_classified{}.png".format(img_id[:-4], img_id[-4:])))

    #print("Number of background tiles: ", background_removal)
    #print("Remaining number of tiles after 90% background removal: ", len(train_list_clean))

    x_train_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['x_train_filenames_partition'])
    y_train_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['y_train_filenames_partition'])
    x_val_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['x_val_filenames_partition'])
    y_val_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['y_val_filenames_partition'])
    x_test_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['x_test_filenames_partition'])
    y_test_filenames_partition_fn = os.path.join(ROOT_DIR,datasets['y_test_filenames_partition'])

    try:
        x_train_filenames = [line.strip() for line in open(x_train_filenames_partition_fn, 'r')]
        y_train_filenames = [line.strip() for line in open(y_train_filenames_partition_fn, 'r')]
        x_val_filenames = [line.strip() for line in open(x_val_filenames_partition_fn, 'r')]
        y_val_filenames = [line.strip() for line in open(y_val_filenames_partition_fn, 'r')]
        x_test_filenames = [line.strip() for line in open(x_test_filenames_partition_fn, 'r')]
        y_test_filenames = [line.strip() for line in open(y_test_filenames_partition_fn, 'r')]
    except:
        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(x_train_filenames, y_train_filenames, test_size=0.3, random_state=42)
        x_val_filenames, x_test_filenames, y_val_filenames, y_test_filenames = train_test_split(x_val_filenames, y_val_filenames, test_size=0.33, random_state=42)
    return x_train_filenames, x_val_filenames, x_test_filenames, y_train_filenames, y_val_filenames, y_test_filenames


def check_write_lists_partitions(vals_train, vals_val, vals_test, train_list_clean, x_train_filenames, y_train_filenames):
    if set([0, 1, 2, 3, 4, 5]).issubset(set(flatten(vals_train))) == True and set([0, 1, 2, 3, 4, 5]).issubset(set(flatten(vals_val))) == True and set([0, 1, 2, 3, 4, 5]).issubset(set(flatten(vals_test))) == True:
        #print("each partition has all values")
    else:
        #print("re-partitioning")
        x_train_filenames = []
        y_train_filenames = []

        for i, img_id in zip(tqdm(range(len(train_list_clean))), train_list_clean):
            pass
            x_train_filenames.append(os.path.join(IMG_DIR, "{}.png".format(img_id)))
            y_train_filenames.append(os.path.join(LABEL_DIR, "RDSISC4_{}_classified{}.png".format(img_id[:-4], img_id[-4:])))
        seed = random.randint(1, 100)
        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(x_train_filenames, y_train_filenames, test_size=0.3, random_state=seed)
        x_val_filenames, x_test_filenames, y_val_filenames, y_test_filenames = train_test_split(x_val_filenames, y_val_filenames, test_size=0.33, random_state=seed)
        get_vals_in_partition(vals_train, x_train_filenames, y_train_filenames)
        get_vals_in_partition(vals_val, x_val_filenames, y_val_filenames)
        get_vals_in_partition(vals_test, x_test_filenames, y_test_filenames)
        #print("Values in training partition: ", set(flatten(vals_train)))
        #print("Values in validation partition: ", set(flatten(vals_val)))
        #print("Values in test partition: ", set(flatten(vals_test)))

    if not os.path.isfile(fn) for fn in [x_train_filenames_partition_fn, y_train_filenames_partition, x_val_filenames_partition, y_val_filenames_partition, x_test_filenames_partition, y_test_filenames_partition]:
        with open(os.path.join(ROOT_DIR,datasets['x_train_filenames_partition']), 'w') as f:
            for item in x_train_filenames:
                f.write("%s\n" % item)

        with open(os.path.join(ROOT_DIR,datasets['y_train_filenames_partition']), 'w') as f:
            for item in y_train_filenames:
                f.write("%s\n" % item)

        with open(os.path.join(ROOT_DIR,datasets['x_val_filenames_partition']), 'w') as f:
            for item in x_val_filenames:
                f.write("%s\n" % item)

        with open(os.path.join(ROOT_DIR,datasets['y_val_filenames_partition']), 'w') as f:
            for item in y_val_filenames:
                f.write("%s\n" % item)

        with open(os.path.join(ROOT_DIR,datasets['x_test_filenames_partition']), 'w') as f:
            for item in x_test_filenames:
                f.write("%s\n" % item)

        with open(os.path.join(ROOT_DIR,datasets['y_test_filenames_partition']), 'w') as f:
            for item in y_test_filenames:
                f.write("%s\n" % item)
    else:
        continue
    return x_train_filenames, x_val_filenames, x_test_filenames, y_train_filenames, y_val_filenames, y_test_filenames 

def get_vals_in_partition(partition_list, x_filenames, y_filenames):
    for x,y,i in zip(x_filenames, y_filenames, tqdm(range(len(y_filenames)))):
        pass 
        try:
            img = np.array(Image.open(y)) 
            vals = np.unique(img)
            partition_list.append(vals)
        except:
            continue
    return partition_list

def flatten(partition_list):
    return [item for sublist in partition_list for item in sublist]

def write_inferences_df(x_test_filenames, y_test_filenames, pred_paths, ROOT_DIR, BATCH_SIZE, EPOCHS):
    path_df = pd.DataFrame(list(zip(x_test_filenames, y_test_filenames, pred_paths)), columns=["img_names", "label_names", "pred_names"])
    path_df.to_csv(os.path.join(ROOT_DIR, "test_file_paths_{}_ep{}.csv".format(BATCH_SIZE, EPOCHS)))
