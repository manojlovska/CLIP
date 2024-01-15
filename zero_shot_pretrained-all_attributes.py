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
from PIL import Image
from torch.utils.data import DataLoader

def get_num_diagonal_max_values(logits, device):
    """ A function to calculate how many times the max logit is in the diagonal for given logit: 
        Inputs: 
            - logits: logits_per_image or logits_per_text 
        Returns:
            - max_value_indices:   tensor of the indices of the maximum values per batch 
            - max_values:          tensor of the maximum values per batch 
            - diagonal_max_values: tensor of the number of times the max value is in the diagonal per batch """
    
    probabilities = logits.softmax(dim=-1) # .cpu().numpy()

    max_values, max_values_indices = probabilities.max(dim=-1)
    diagonal_values = probabilities.diag()

    diagonal_max_values = max_values == diagonal_values

    labels = torch.arange(logits.shape[0]).to(device)
    mean_max_probs = max_values[max_values_indices == labels].mean()

    return max_values_indices, diagonal_max_values, diagonal_max_values.sum(), mean_max_probs

device = "cuda:0"

images_path = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba/"
eval_partition_path = "/mnt/hdd/volume1/anastasija/CelebA/Eval/"
annotations_directory = "/mnt/hdd/volume1/anastasija/CelebA/Anno/"
captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/captions_all_attributes.txt"

annotations_filename = "list_attr_celeba.txt"
annotations_path = os.path.join(annotations_directory, annotations_filename)

annotations_df = pd.read_csv(annotations_path, delimiter='\s+', skiprows=[0])
attributes = list(annotations_df.columns.values)

list_eval_partition = pd.read_csv(os.path.join(eval_partition_path, "list_eval_partition.txt"), sep=" ", header=None)
img_filenames_all = sorted(glob.glob(images_path + '*.jpg'))


model, preprocess = clip.load("ViT-B/32", device=device)

test_dataset = CelebADataset(images_path, captions_path, eval_partition_path, preprocess,  name="test")
test_images = [os.path.join(images_path, image) for image in test_dataset.images_list]

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# unique_captions = set(test_dataset.captions)

########################################################## ZERO-SHOT

# Disable gradient computation and reduce memory consumption.
test_list_num_diagonal_max_values_im_percent = []
test_list_num_diagonal_max_values_texts_percent = []
test_list_mean_max_probs_im = []
test_list_mean_max_probs_texts = []

with torch.no_grad():
    pbar = tqdm(test_dataloader, total=len(test_dataloader))
    iter = 0
    model.eval()
    for batch in pbar:
        iter = iter+1

        val_im_names, val_images, val_captions, val_texts = batch
        val_images = val_images.to(device)
        val_texts = val_texts.to(device)

        val_logits_per_image, val_logits_per_text = model(val_images, val_texts)

        # Calculate the metrics
        max_values_indices_im, diagonal_max_values_im, num_diagonal_max_values_im, mean_max_probs_im = get_num_diagonal_max_values(val_logits_per_image, device)
        max_values_indices_texts, diagonal_max_values_texts, num_diagonal_max_values_texts, mean_max_probs_texts = get_num_diagonal_max_values(val_logits_per_text, device)

        # Convert validation metrics into percentages
        num_diagonal_max_values_im_percent = num_diagonal_max_values_im / test_dataloader.batch_size
        num_diagonal_max_values_texts_percent = num_diagonal_max_values_texts / test_dataloader.batch_size

        # Add them to a list to calculate mean value
        test_list_num_diagonal_max_values_im_percent.append(num_diagonal_max_values_im_percent.item())
        test_list_num_diagonal_max_values_texts_percent.append(num_diagonal_max_values_texts_percent.item())

        test_list_mean_max_probs_im.append(mean_max_probs_im.item())
        test_list_mean_max_probs_texts.append(mean_max_probs_texts.item())


    # Mean values of all batches in epoch
    mean_num_diagonal_max_values_im_percent = mean(test_list_num_diagonal_max_values_im_percent)
    mean_num_diagonal_max_values_texts_percent = mean(test_list_num_diagonal_max_values_texts_percent)
    
    test_mean_diag_max_prob_im = mean(test_list_mean_max_probs_im)
    test_mean_diag_max_prob_texts = mean(test_list_mean_max_probs_texts)

print(f"mean_num_diagonal_max_values_im_percent: {mean_num_diagonal_max_values_im_percent}")
print(f"mean_num_diagonal_max_values_texts_percent: {mean_num_diagonal_max_values_texts_percent}")
print(f"test_mean_diag_max_prob_im: {test_mean_diag_max_prob_im}")
print(f"test_mean_diag_max_prob_texts: {test_mean_diag_max_prob_texts}")

######################################################### FINE-TUNED

checkpoint = torch.load("/mnt/hdd/volume1/anastasija/CLIP_outputs/CLIP-fine-tuning/epoch_17_ckpt.pth")

model.load_state_dict(checkpoint["model"])

# Disable gradient computation and reduce memory consumption.
test_list_num_diagonal_max_values_im_percent2 = []
test_list_num_diagonal_max_values_texts_percent2 = []
test_list_mean_max_probs_im2 = []
test_list_mean_max_probs_texts2 = []

with torch.no_grad():
    pbar = tqdm(test_dataloader, total=len(test_dataloader))
    iter = 0
    model.eval()
    for batch in pbar:
        iter = iter+1

        val_im_names, val_images, val_captions, val_texts = batch
        val_images = val_images.to(device)
        val_texts = val_texts.to(device)

        val_logits_per_image, val_logits_per_text = model(val_images, val_texts)

        # Calculate the metrics
        max_values_indices_im, diagonal_max_values_im, num_diagonal_max_values_im, mean_max_probs_im = get_num_diagonal_max_values(val_logits_per_image, device)
        max_values_indices_texts, diagonal_max_values_texts, num_diagonal_max_values_texts, mean_max_probs_texts = get_num_diagonal_max_values(val_logits_per_text, device)

        # Convert validation metrics into percentages
        num_diagonal_max_values_im_percent = num_diagonal_max_values_im / test_dataloader.batch_size
        num_diagonal_max_values_texts_percent = num_diagonal_max_values_texts / test_dataloader.batch_size

        # Add them to a list to calculate mean value
        test_list_num_diagonal_max_values_im_percent2.append(num_diagonal_max_values_im_percent.item())
        test_list_num_diagonal_max_values_texts_percent2.append(num_diagonal_max_values_texts_percent.item())

        test_list_mean_max_probs_im2.append(mean_max_probs_im.item())
        test_list_mean_max_probs_texts2.append(mean_max_probs_texts.item())


    # Mean values of all batches in epoch
    mean_num_diagonal_max_values_im_percent2 = mean(test_list_num_diagonal_max_values_im_percent2)
    mean_num_diagonal_max_values_texts_percent2 = mean(test_list_num_diagonal_max_values_texts_percent2)
    
    test_mean_diag_max_prob_im2 = mean(test_list_mean_max_probs_im2)
    test_mean_diag_max_prob_texts2 = mean(test_list_mean_max_probs_texts2)

print(f"mean_num_diagonal_max_values_im_percent2: {mean_num_diagonal_max_values_im_percent2}")
print(f"mean_num_diagonal_max_values_texts_percent2: {mean_num_diagonal_max_values_texts_percent2}")
print(f"test_mean_diag_max_prob_im2: {test_mean_diag_max_prob_im2}")
print(f"test_mean_diag_max_prob_texts2: {test_mean_diag_max_prob_texts2}")

import pdb; pdb.set_trace()




# text = clip.tokenize(unique_captions, truncate=True).to(device)
# results = []
# probabilities_results = []
# for img in test_images[:10]:
#     image = preprocess(Image.open(img)).unsqueeze(0).to(device)

#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
    
#         logits_per_image, logits_per_text = model(image, text)
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#     n_attributes = sorted(range(len(probs[0])), key=lambda i: probs[0][i], reverse=True)[:5]
#     predictions = list(map(lambda i: unique_captions[i], n_attributes))
    
#     probabilities_results.append(sorted(probs[0], reverse=True)[:5])
#     results.append(predictions)









