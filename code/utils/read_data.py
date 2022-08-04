import os, glob

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

    print("number of images: ", len(dset_list))
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

    print("number of images: ", len(dset_list))
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
            print(pred_arr_lst[i])
            print(label_arr_lst[i])
            print(pred_arr_lst[i].shape)
            print(label_arr_lst[i].shape)
            
        else:
            pred_arr_lst_valid.append(pred_arr_lst[i])
            label_arr_lst_valid.append(label_arr_lst[i])
    return pred_arr_lst_valid, label_arr_lst_valid