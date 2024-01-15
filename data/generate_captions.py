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

# A photo of a {man/woman} with the following attributes: {}.

# N = 10
captions = {}
for image in tqdm(img_filenames_all):
    image_name = os.path.basename(image)
    values = annotations_df.loc[image_name, :]

    if values["Male"] > 0:
        gender = "man"
    else:
        gender = "woman"

    template = "A photo of a {}".format(gender) + " with the following attributes: " + ", ".join("{}".format(attributes_dict[attribute]) for attribute in list(np.array(attributes)[np.array(values) > 0]) if attribute != "Male")
    captions[image_name] = template
    
# Save the captons
output_file = 'captions_all_attributes_new.txt'

# Open the file in write mode and save the data
with open(output_file, 'w') as file:
    for image_name, caption in captions.items():
        file.write(f'{image_name} {caption}\n')


