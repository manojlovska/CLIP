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

unique_captions = set(captions)

import pdb
pdb.set_trace()

# Counting the number of unique values
num_unique_values_val = len(unique_captions)
print(f"Number of unique values: {num_unique_values_val}")

count = 0
max_tokens_len = 0
for i, (image, caption) in tqdm(enumerate(captions.items())):
    tokenized_caption = clip.tokenize(caption, context_length=305, truncate=False)
    
    if tokenized_caption[:,-1].item() != 0:
        count += 1
        print("NOT OK")
        print(f"\nTokenized caption: \n {tokenized_caption}")
        print(f"Caption: \n {caption}")

    if torch.count_nonzero(tokenized_caption, dim=-1).numpy()[0] > max_tokens_len:
        max_tokens_len = torch.count_nonzero(tokenized_caption, dim=-1).numpy()[0]
        max_tokenized_caption = tokenized_caption
        max_caption = caption

    # else:
    #     # print(f"OK \n {captions[i]}")

print(count)

import pdb
pdb.set_trace()

