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
filename = "captions_all_attributes_new.txt"
captions_filename = os.path.join(captions_dir, filename)

captions = pd.read_csv(captions_filename, sep="\t")
captions[['image_name','caption']] = captions["image_name caption"].str.split(" ", n=1, expand=True)
captions.drop(columns=captions.columns[0], axis=1,  inplace=True)
captions = captions["caption"].values.tolist()


unique_captions = set(captions)


# Counting the number of unique values
num_unique_values_val = len(unique_captions)
print(f"Number of unique values for val images: {num_unique_values_val}")

tokenized_captions_val = clip.tokenize(captions, truncate=True)

non_zero_values_val = torch.count_nonzero(tokenized_captions_val, dim=-1).numpy().tolist()

no_zeroes_in_rows = (tokenized_captions_val != 0).all(dim=1)

count_val = 0
for i, value in enumerate(non_zero_values_val):
    if value > tokenized_captions_val.shape[-1] - 1:
        count_val += 1
        print(f"Value: {value} \nToken: {tokenized_captions_val[i]}")

import pdb; pdb.set_trace()