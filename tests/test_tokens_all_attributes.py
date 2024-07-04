import os
import glob
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import clip
import torch

captions_dir = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2"
filename = "captions_att_28052024.txt"
captions_filename = os.path.join(captions_dir, filename)

def read_captions_as_dict(captions_filename):
    image_dict = {}
    with open(captions_filename, 'r') as file:
        for line in file:
            image_name, caption = line.strip().split(' ', 1)
            image_dict[image_name] = caption
    return image_dict

captions = read_captions_as_dict(captions_filename)
captions_list = list(captions.values())

unique_captions = set(captions_list)

import pdb
pdb.set_trace()


# Counting the number of unique values
num_unique_values = len(unique_captions)
print(f"Number of unique values: {num_unique_values}")

tokenized_captions = clip.tokenize(captions_list, context_length=305, truncate=False)

non_zero_values = torch.count_nonzero(tokenized_captions, dim=-1).numpy().tolist()

no_zeroes_in_rows = (tokenized_captions != 0).all(dim=1)

count_val = 0
for i, value in enumerate(non_zero_values):
    if value > tokenized_captions.shape[-1] - 1:
        count_val += 1
        print(f"Value: {value} \nToken: {tokenized_captions[i]}")

import pdb; pdb.set_trace()