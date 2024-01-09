import os
import torch
from experiments.base_exp import Exp
import wandb
from loguru import logger
from torchsummary import summary
from tqdm import tqdm
import clip
from statistics import mean
import torch.nn.functional as F
import numpy as np

from experiments.base_exp import Exp
from data.dataset import CelebADataset
import pandas as pd
import glob

device = "cuda:1"

# Check how many men and women are there in the test dataset
images_path = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba/"
eval_partition_path = "/mnt/hdd/volume1/anastasija/CelebA/Eval/"
annotations_directory = "/mnt/hdd/volume1/anastasija/CelebA/Anno/"
captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/gender_captions_all_images.txt"

annotations_filename = "list_attr_celeba.txt"
annotations_path = os.path.join(annotations_directory, annotations_filename)

annotations_df = pd.read_csv(annotations_path, delimiter='\s+', skiprows=[0])
attributes = list(annotations_df.columns.values)

list_eval_partition = pd.read_csv(os.path.join(eval_partition_path, "list_eval_partition.txt"), sep=" ", header=None)
img_filenames_all = sorted(glob.glob(images_path + '*.jpg'))

exp = Exp()
model, preprocess = exp.get_model(exp.vision_encoder)

test_dataset = CelebADataset(images_path, captions_path, eval_partition_path, preprocess,  name="test")

test_images = [os.path.join(images_path, image) for image in test_dataset.images_list]

captions = {}
count_men = 0
count_women = 0
for image in tqdm(test_images):
    image_name = os.path.basename(image)
    male_attribute = annotations_df.loc[image_name, "Male"]


    if male_attribute > 0:
        count_men += 1
    elif male_attribute  < 0:
        count_women += 1

# Check predictions
checkpoint = torch.load("/mnt/hdd/volume1/anastasija/CLIP_outputs/CLIP-fine-tuning-improved-captions/hearty-totem-6/best_ckpt.pth")

model.load_state_dict(checkpoint["model"])

captions = ["A photo of a woman.", "A photo of a man."]
texts = clip.tokenize(captions, truncate=True).to(device)

count_pred_women = 0
count_pred_men = 0
for i in tqdm(range(len(test_images))):
    test_image = test_dataset.__getitem__(i)[1].to(device)
    test_image = test_image.unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(test_image)
        text_features = model.encode_text(texts)

        logits_per_image, logits_per_text = model(test_image, texts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    if np.argmax(probs[0]) == 0:
        count_pred_women += 1
    elif np.argmax(probs[0]) == 1:
        count_pred_men += 1

    # print(f"Image name: {test_dataset.__getitem__(i)[0]}")
    # print(f"Predicted probabilities: {probs}")
    # print(f"Predicted caption: {captions[np.argmax(probs[0])]}")
    # print(f"Ground truth caption: {test_dataset.__getitem__(i)[2]}")
    # print("\n")

print(f"Number of photos of women: {count_women}")
print(f"Number of photos of men: {count_men}")
print(f"Predicted number of photos of women: {count_pred_women}")
print(f"Predicted number of photos of men: {count_pred_men}")

# Outputs
# Number of photos of women: 12247
# Number of photos of men: 7715
# Predicted number of photos of women: 12245
# Predicted number of photos of men: 7717

import pdb; pdb.set_trace()



