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

chosen_attributes = ["Male",
                     "Straight_Hair", "Wavy_Hair", "Bald", 
                     "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", 
                     "Attractive", 
                     "Young", 
                     "Wearing_Hat", 
                     "Smiling", 
                     "Eyeglasses", 
                     "No_Beard", 
                     "Arched_Eyebrows"]

corrected_attributes = ["male",
                        "straight hair", "wavy hair", "bald",
                        "black hair", "blond hair", "brown hair", "gray hair",
                        "attractive",
                        "young",
                        "wearing hat",
                        "smiling",
                        "eyeglasses",
                        "no beard",
                        "arched eyebrows"]

attribute_dict  = {chosen_attributes[i]: corrected_attributes[i] for i in range(len(chosen_attributes))}

attribute_dict = {
    "Male": {"positive": ["man", "He"],
             "negative": ["woman", "She"]},
    "Straight_Hair": {"positive": "straight",
                      "negative": ""},
    "Wavy_Hair": {"positive": "wavy",
                  "negative": ""},
    "Bald": {"positive": "bald",
             "negative": ""},
    "Black_Hair": {"positive": "black",
                  "negative": ""},
    "Blond_Hair": {"positive": "blond",
                  "negative": ""},
    "Brown_Hair": {"positive": "brown",
                  "negative": ""},
    "Gray_Hair": {"positive": "gray",
                  "negative": ""},
    "Attractive": {"positive": "an attractive",
                  "negative": ""},
    "Young": {"positive": "a young",
              "negative": ""},
    "Wearing_Hat": {"positive": "wearing hat",
                    "negative": "not wearing hat"},
    "Smiling": {"positive": "smiling",
                "negative": "not smiling"},
    "Eyeglasses": {"positive": "eyeglasses",
                   "negative": "no eyeglasses"},
    "No_Beard": {"positive": "no beard",
                 "negative": "a beard"},
    "Arched_Eyebrows": {"positive": "arched eyebrows",
                        "negative": "no arched eyebrows"}
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

template = "A photo of {} {} with {} {} hair, who is {}. {} has {}, {} and {}. {} is also {}."

test_images = img_filenames_all
captions = {}
for image in tqdm(test_images):
    image_name = os.path.basename(image)
    attributes = annotations_df.loc[image_name, chosen_attributes]
    attributes_names = list(attributes.keys())
    attributes_names.sort(key = lambda i: chosen_attributes.index(i))

    # List of the attributes in the image
    image_attributes = []
    
    for attribute_name in attributes_names:
        if attributes[attribute_name] > 0:
            attribute = attribute_dict[attribute_name]["positive"]
        elif attributes[attribute_name] < 0:
            attribute = attribute_dict[attribute_name]["negative"]

        image_attributes.append(attribute)
    
    # Sort the attributes so that they can fit in the template
    sorted_attributes = []
    if "an attractive" and "a young" in image_attributes:
        first_attr = "a young and an attractive"
    elif "an attractive" in image_attributes and "a young" not in image_attributes:
        first_attr = "an attractive"
    elif "a young" in image_attributes and "an attractive" not in image_attributes:
        first_attr = "a young"
    else:
        first_attr = "a"
    sorted_attributes.append(first_attr)

    second_attr = image_attributes[0][0]
    sorted_attributes.append(second_attr)

    if "straight" in image_attributes:
        third_attr = "straight"
    elif "wavy" in image_attributes:
        third_attr = "wavy"
    else:
        third_attr = ""
    
    sorted_attributes.append(third_attr)

    if "blond" in image_attributes:
        fourth_attr = "blond"
    elif "brown" in image_attributes:
        fourth_attr = "brown"
    elif "black" in image_attributes:
        fourth_attr = "black"
    elif "gray" in image_attributes:
        fourth_attr = "gray"
    elif "bald" in image_attributes:
        fourth_attr = "no"
    else:
        fourth_attr = ""
    
    sorted_attributes.append(fourth_attr)

    if "smiling" in image_attributes:
        fifth_attr = "smiling"
    elif "not smiling" in image_attributes:
        fifth_attr = "not smiling"

    sorted_attributes.append(fifth_attr)

    sixth_attribute = image_attributes[0][1]
    sorted_attributes.append(sixth_attribute)

    if "beard" in image_attributes:
        seventh_attr = "beard"
    elif "no beard" in image_attributes:
        seventh_attr = "no beard"
    
    sorted_attributes.append(seventh_attr)

    if "arched eyebrows" in image_attributes:
        eighth_attr = "arched eyebrows"
    elif "no arched eyebrows" in image_attributes:
        eighth_attr = "no arched eyebrows"
    
    sorted_attributes.append(eighth_attr)

    if "eyeglasses" in image_attributes:
        ninth_attr = "eyeglasses"
    elif "no eyeglasses" in image_attributes:
        ninth_attr = "no eyeglasses"
    
    sorted_attributes.append(ninth_attr)

    tenth_attr = image_attributes[0][1]
    sorted_attributes.append(tenth_attr)

    if "wearing hat" in image_attributes:
        eleventh_attr = "wearing hat"
    elif "not wearing hat" in image_attributes:
        eleventh_attr = "not wearing hat"
    
    sorted_attributes.append(eleventh_attr)

    captions[image_name] = re.sub(r'\s+', ' ', template.format(*sorted_attributes))

# Save the captions
output_file = 'coherent_captions_all_images.txt'

# Open the file in write mode and save the data
with open(output_file, 'w') as file:
    for image_name, caption in captions.items():
        file.write(f'{image_name} {caption}\n')


    




    








