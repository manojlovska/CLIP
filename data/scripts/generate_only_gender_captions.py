import os
import glob
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm
import re

data_directory = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba/"
eval_directory = "/mnt/hdd/volume1/anastasija/CelebA/Eval/"
annotations_directory = "/mnt/hdd/volume1/anastasija/CelebA/Anno/"

annotations_filename = "list_attr_celeba.txt"
annotations_path = os.path.join(annotations_directory, annotations_filename)

annotations_df = pd.read_csv(annotations_path, delimiter='\s+', skiprows=[0])
attributes = list(annotations_df.columns.values)

list_eval_partition = pd.read_csv(os.path.join(eval_directory, "list_eval_partition.txt"), sep=" ", header=None)
img_filenames_all = sorted(glob.glob(data_directory + '*.jpg'))

chosen_attributes = ["Male"]

attribute_dict = {
    "Male": {"positive": "man",
             "negative": "woman"}
             }

"""

Template:
A photo of {(a young and an attractive)/(a young)/(an attractive)/(a)} 
           {man/woman} with
           {black/blond/brown/gray},
           {straight/wavy} hair, who is
           {smiling/not smiling}. 
           {He/She} has
           {a beard/no beard}, 
           {arched eyebrows/ no arched eyebrows} and
           {eyeglasses/ no eyeglasses}.
           {He/She} is also
           {wearing hat/ not wearing hat}.

Example:
A photo of {a young} {woman} with {blond},{straight} hair, who is {not smiling}. 
{She} has {no beard}, {arched eyebrows} and {no eyeglasses}.
{She} is also {not wearing hat}.

"""

template = "A photo of a {}."

test_images = img_filenames_all
captions = {}
for image in tqdm(test_images):
    image_name = os.path.basename(image)
    attributes = annotations_df.loc[image_name, chosen_attributes]

    if attributes["Male"]  > 0:
        attribute = attribute_dict["Male"]["positive"]
    elif attributes["Male"]  < 0:
        attribute = attribute_dict["Male"]["negative"]

    captions[image_name] = template.format(attribute)

# Save the captions
output_file = 'gender_captions_all_images.txt'

# Open the file in write mode and save the data
with open(output_file, 'w') as file:
    for image_name, caption in captions.items():
        file.write(f'{image_name} {caption}\n')


    




    








