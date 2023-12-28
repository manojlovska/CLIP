import os
import json
from PIL import Image
import pandas as pd
import torchvision.transforms

import torch
import torch.nn as nn
import clip

""" Map the split of the data to an integer for reaading from annotatons file 
        - train: 0
        - val:   1
        - test:  2 
"""
def map_split2int(split):
    map_dict = {"train": 0,
                "val": 1,
                "test": 2}
    return map_dict[split]

class CelebADataset():
    def __init__(self, images_path, captions_path, eval_partition_path, preprocess,  name="train"):

        # Preprocessing of the images
        self.preprocess = preprocess

        # Images path
        self.images_path = images_path
        
        # Map the split to integer
        self.split = map_split2int(split=name)

        # Partitions
        # self.eval_partitions = pd.read_csv(os.path.join(eval_partition_path, "list_eval_partition.txt"), sep=" ", header=None)
        self.eval_partition_path = eval_partition_path
    
        # Images
        self.images_list = self.read_eval_partition(self.eval_partition_path)[:100]

        # All captions
        all_captions = pd.read_csv(os.path.join(captions_path, "captions_all_attributes.csv"), sep="\t")
        captions = all_captions[all_captions['image_name'].isin(self.images_list)]
        
        # Captions for the particular split
        self.captions = captions["caption"].values.tolist()

        # Tokenize text using CLIP's tokenizer
        self.tokenized_captions  = clip.tokenize(self.captions, truncate=True)

    def read_eval_partition(self, eval_partition):
        with open(os.path.join(eval_partition, "list_eval_partition.txt"), 'r') as file:
            # Initialize an empty list to store image names with 0
            image_list = []

            # Iterate through each line in the file
            for line in file:
                # Split the line into image filename and integer
                image_filename, split = line.split()

                # Check if the integer is 0
                if int(split) == self.split:
                    # Add the image filename to the list
                    image_list.append(image_filename)

        return image_list

    def __len__(self):
        return len(self.tokenized_captions)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = Image.open(os.path.join(self.images_path, self.images_list[idx]))
        caption = self.tokenized_captions[idx]
        return self.images_list[idx], self.preprocess(image), caption