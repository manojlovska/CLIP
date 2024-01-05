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


# A photo of a {man/woman} with the following attributes: {}.

# N = 10
templates = []
image_names = []
for image in tqdm(img_filenames_all):
    image_name = os.path.basename(image)
    values = annotations_df.loc[image_name, :]

    if values["Male"] > 0:
        gender = "man"
    else:
        gender = "woman"

    template = "A photo of a {}".format(gender) + " with the following attributes: " + ", ".join("{}".format(attribute) for attribute in list(np.array(attributes)[np.array(values) > 0]) if attribute != "Male")
    templates.append(template)
    image_names.append(image_name)

# Creating pandas dataframe with pairs: (image_name, caption)
captions_df = pd.DataFrame(
    {'image_name': image_names,
     'caption': templates
    })

# Save the captons
captions_df.to_csv("captions_all_attributes_no_male_2.csv", sep="\t", index=False, header=None)



