""" CREATE A SUBSET OF CELEBA DATASET FOR TESTING 
    TRAIN: 40 000
    VAL:   5 000
    TEST:  5 000 """

import os
import glob
import shutil

import pandas as pd

data_directory = "/home/anastasija/Documents/SBS/Seminarska/Project/CLIP/data/CelebA/CelebA/Img/img_celeba/"
eval_directory = "/home/anastasija/Documents/SBS/Seminarska/Project/CLIP/data/CelebA/CelebA/Eval/"
data_subset_directory = "/home/anastasija/Documents/SBS/Seminarska/Project/CLIP/data/CelebA_subset/Img/img_celeba"

list_eval_partition = pd.read_csv(os.path.join(eval_directory, "list_eval_partition.txt"), sep=" ", header=None)
img_filenames_all = glob.glob(data_directory + '*.jpg')

train_set = [train_img for train_img in img_filenames_all if (list_eval_partition.loc[list_eval_partition.iloc[:, 0] == os.path.basename(train_img), list_eval_partition.columns[1]].values).item() == 0]
val_set = [val_img for val_img in img_filenames_all if (list_eval_partition.loc[list_eval_partition.iloc[:, 0] == os.path.basename(val_img), list_eval_partition.columns[1]].values).item() == 1]
test_set = [test_img for test_img in img_filenames_all if (list_eval_partition.loc[list_eval_partition.iloc[:, 0] == os.path.basename(test_img), list_eval_partition.columns[1]].values).item() == 2]

print(len(train_set))
print(len(val_set))
print(len(test_set))

train_subset = train_set[:40000]
val_subset = val_set[:5000]
test_subset = test_set[:5000]

for filename in train_subset:
    destination_path = os.path.join(data_subset_directory, os.path.basename(filename))

    try:
        shutil.copy2(filename, destination_path)
        print(f"Successfully copied {filename} to {data_subset_directory}")
    except FileNotFoundError:
        print(f"File {filename} not found in {data_directory}")
    except PermissionError:
        print(f"Permission error while copying {filename}")

for filename in val_subset:
    destination_path = os.path.join(data_subset_directory, os.path.basename(filename))

    try:
        shutil.copy2(filename, destination_path)
        print(f"Successfully copied {filename} to {data_subset_directory}")
    except FileNotFoundError:
        print(f"File {filename} not found in {data_directory}")
    except PermissionError:
        print(f"Permission error while copying {filename}")

for filename in test_subset:
    destination_path = os.path.join(data_subset_directory, os.path.basename(filename))

    try:
        shutil.copy2(filename, destination_path)
        print(f"Successfully copied {filename} to {data_subset_directory}")
    except FileNotFoundError:
        print(f"File {filename} not found in {data_directory}")
    except PermissionError:
        print(f"Permission error while copying {filename}")