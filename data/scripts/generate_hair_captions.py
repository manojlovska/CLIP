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

# [a photo of a] {bald} [man/woman] {with} {straight/wavy} {blond/brown/black/gray} {hair}.
gender_attributes = "Male"
hair_color = ["blond", "brown", "black", "gray"]
hair_type = ["straight", "wavy"]

chosen_attributes = ["Male",
                     "Straight_Hair", "Wavy_Hair", "Bald", 
                     "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]


"""

A photo of a {man/woman}. -> if none of the attributes are annotated
A photo of a {bald} {man/woman}. -> if bald is annotated
A photo of a {man/woman} with {straight/wavy} / {blond/brown/black/gray} hair. -> If only hair type or hair color are annotated
A photo of a {man/woman} with {straight/wavy} {blond/brown/black/gray} hair.

"""

templates = [
    "A photo of a {}.",
    "A photo of a {} {}.",
    "A photo of a {} with {} hair.",
    "A photo of a {} with {} {} hair."
]

test_images = img_filenames_all
captions = {}

for image in tqdm(test_images):
    attribute_dict = {}
    image_name = os.path.basename(image)
    attributes = annotations_df.loc[image_name, chosen_attributes]
    attributes_names = list(attributes.keys())
    attributes_names.sort(key = lambda i: chosen_attributes.index(i))

    if attributes["Male"] > 0:
        gender = "man"
    elif attributes["Male"] < 0:
        gender = "woman"

    if attributes["Bald"] > 0:
        bald = "bald"
        hair_type = ""
        hair_color = ""
    elif attributes["Bald"] < 0:
        bald = ""
        if attributes["Straight_Hair"] > 0:
            hair_type = "straight"
        elif attributes["Wavy_Hair"] > 0:
            hair_type = "wavy"
        else:
            hair_type = ""
        
        if attributes["Black_Hair"] > 0:
            hair_color = "black"
        elif attributes["Blond_Hair"] > 0:
            hair_color = "blond"
        elif attributes["Brown_Hair"] > 0:
            hair_color = "brown"
        elif attributes["Gray_Hair"] > 0:
            hair_color = "gray"
        else:
            hair_color = ""
    
    attribute_dict["gender"] = gender
    attribute_dict["bald"] = bald
    attribute_dict["hair_type"] = hair_type
    attribute_dict["hair_color"] = hair_color

    if attribute_dict["bald"]:
        template = templates[1]
        caption = template.format(bald, gender)
    elif not attribute_dict["bald"] and not attribute_dict["hair_type"] and not attribute_dict["hair_color"]:
        template = templates[0]
        caption = template.format(gender)
    elif not attribute_dict["bald"] and attribute_dict["hair_type"] and attribute_dict["hair_color"]:
        template = templates[-1]
        caption = template.format(gender, hair_type, hair_color)
    elif not attribute_dict["bald"] and attribute_dict["hair_type"]:       
        template = templates[2]
        caption = template.format(gender, hair_type)
    elif not attribute_dict["bald"] and attribute_dict["hair_color"]:
        template = templates[2]
        caption = template.format(gender, hair_color)
    
    captions[image_name] = caption


# Extracting the values and converting them to a set to remove duplicates
unique_values = set(captions.values())
print(f"Unique_values:\n{unique_values}")

# Counting the number of unique values
num_unique_values = len(unique_values)
print(f"Number of unique values: {num_unique_values}")

# Save the captions
output_file = 'captions/hair_captions_all_images.txt'

# Open the file in write mode and save the data
with open(output_file, 'w') as file:
    for image_name, caption in captions.items():
        file.write(f'{image_name} {caption}\n')










