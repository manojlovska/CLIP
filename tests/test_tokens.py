import os
import glob
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import clip
import torch
 

captions_dir = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions"
filename = "captions_19_attr_all_images.txt"
captions_filename = os.path.join(captions_dir, filename)

# pd_captions = pd.read_csv(captions_filename, sep="\t")
# captions = pd_captions["caption"].to_list()

captions = pd.read_csv(captions_filename, sep="\t")
captions[['image_name','caption']] = captions["image_name caption"].str.split(" ", n=1, expand=True)
captions.drop(columns=captions.columns[0], axis=1,  inplace=True)
captions = captions["caption"].values.tolist()

train_captions = captions[:162770]
val_captions = captions[162771:182638]

# TRAIN SET
# Extracting the values and converting them to a set to remove duplicates
unique_values_train = set(train_captions)
# print(f"Unique_values:\n{unique_values}")

# Counting the number of unique values
num_unique_values_train = len(unique_values_train)
print(f"Number of unique values for train images: {num_unique_values_train}")

tokenized_captions_train = clip.tokenize(train_captions, truncate=True)

non_zero_values_train = torch.count_nonzero(tokenized_captions_train, dim=-1).numpy().tolist()

count_train = 0
for i, value in enumerate(non_zero_values_train):
    if value > tokenized_captions_train.shape[-1] - 1:
        count_train += 1
        print(f"Value: {value} \nToken: {tokenized_captions_train[i]}")

# VAL SET
# Extracting the values and converting them to a set to remove duplicates
unique_values_val = set(val_captions)
# print(f"Unique_values:\n{unique_values}")

# Counting the number of unique values
num_unique_values_val = len(unique_values_val)
print(f"Number of unique values for val images: {num_unique_values_val}")

tokenized_captions_val = clip.tokenize(val_captions, truncate=True)

non_zero_values_val = torch.count_nonzero(tokenized_captions_val, dim=-1).numpy().tolist()

count_val = 0
for i, value in enumerate(non_zero_values_val):
    if value > tokenized_captions_val.shape[-1] - 1:
        count_val += 1
        print(f"Value: {value} \nToken: {tokenized_captions_val[i]}")



import pdb; pdb.set_trace()