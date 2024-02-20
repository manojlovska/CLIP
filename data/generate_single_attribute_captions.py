import os
import glob
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm

data_directory = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba/"
eval_directory = "/mnt/hdd/volume1/anastasija/CelebA/Eval/"
annotations_directory = "/mnt/hdd/volume1/anastasija/CelebA/Anno/"

annotations_filename = "list_attr_celeba.txt"
annotations_path = os.path.join(annotations_directory, annotations_filename)

annotations_df = pd.read_csv(annotations_path, delimiter='\s+', skiprows=[0])
attributes = list(annotations_df.columns.values)

list_eval_partition = pd.read_csv(os.path.join(eval_directory, "list_eval_partition.txt"), sep=" ", header=None)
img_filenames_all = sorted(glob.glob(data_directory + '*.jpg'))

single_attr_path = '/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/captions_single_attribute.txt'
with open(single_attr_path, 'r') as file:
    # Read all the lines of the file into a list
    captions_single_attr = file.readlines()

attributes_dict = {
    "5_o_Clock_Shadow": "5 o'clock shadow",
    "Arched_Eyebrows": "arched eyebrows",
    "Attractive" : "attractive",
    "Bags_Under_Eyes": "bags under the eyes",
    "Bald": "bald",
    "Bangs": "bangs",
    "Big_Lips": "big lips",
    "Big_Nose": "big nose",
    "Black_Hair": "black hair",
    "Blond_Hair": "blond hair",
    "Blurry": "blurry",
    "Brown_Hair": "brown hair",
    "Bushy_Eyebrows": "bushy eyebrows",
    "Chubby": "chubby",
    "Double_Chin": "double chin",
    "Eyeglasses": "eyeglasses",
    "Goatee": "goatee",
    "Gray_Hair": "gray hair",
    "Heavy_Makeup": "heavy makeup",
    "High_Cheekbones": "high cheekbones",
    "Male": "male",
    "Mouth_Slightly_Open": "mouth slightly open",
    "Mustache": "mustache",
    "Narrow_Eyes": "narrow eyes",
    "No_Beard": "no beard",
    "Oval_Face": "oval face",
    "Pale_Skin": "pale skin",
    "Pointy_Nose": "pointy nose",
    "Receding_Hairline": "receding hairline",
    "Rosy_Cheeks": "rosy cheeks",
    "Sideburns": "sideburns",
    "Smiling": "smiling",
    "Straight_Hair": "straight hair",
    "Wavy_Hair": "wavy hair",
    "Wearing_Earrings": "wearing earrings",
    "Wearing_Hat": "wearing hat",
    "Wearing_Lipstick": "wearing lipstick",
    "Wearing_Necklace": "wearing necklace",
    "Wearing_Necktie": "wearing necktie",
    "Young": "young"
}

generated_captions_dict = {}

for image in tqdm(img_filenames_all):
    image_name = os.path.basename(image)
    captions_per_img = []

    # Present attributes in the image
    img_attr = annotations_df.loc[image_name, attributes]
    img_attr_series = annotations_df.loc[image_name, [list(img_attr)[i] > 0 for i in range(0, len(img_attr))]]
    attributes_names = list(img_attr_series.keys())

    # Get the caption
    for attr in attributes_names:
        idx = list(attributes_dict.keys()).index(attr)
        caption = captions_single_attr[idx].replace('\n', '')
        captions_per_img.append(caption)

    generated_captions_dict[image_name] = captions_per_img

import json
output_file_json = 'captions/captions_single_attr_all_imgs.json'
with open(output_file_json, 'w') as json_file:
    json.dump(generated_captions_dict, json_file, indent=4)


