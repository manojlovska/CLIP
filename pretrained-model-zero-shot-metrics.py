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
import json


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

def get_num_top5_max_values(logits):
    """ A function to calculate how many times the max logit is in the diagonal for given logit: 
        Inputs: 
            - logits: logits_per_image or logits_per_text 
        Returns:
            - max_value_indices:   tensor of the indices of the maximum values per batch 
            - max_values:          tensor of the maximum values per batch 
            - diagonal_max_values: tensor of the number of times the max value is in the diagonal per batch """
    
    ground_truths = np.arange(logits.shape[0])
    probabilities = logits.softmax(dim=-1) # .cpu().numpy()
    sorted_prob, indices = probabilities.sort(descending=True, dim=-1)
    indices_top5 = indices.cpu().numpy()[:, :5]

    count = 0
    for i, ind in enumerate(indices_top5):
        if np.any(ind == ground_truths[i]):
            count += 1

    return count

def get_predicted_and_gt_caption(logits, captions, image_names, result):
    ground_truths = np.arange(logits.shape[0])
    probabilities = logits.softmax(dim=-1)

    sorted_prob, indices = probabilities.sort(descending=True, dim=-1)
    indices = indices.cpu().numpy()

    for i, ind in enumerate(indices):
        predicted = captions[ind[0]]
        gt = captions[ground_truths[i]]

        result[image_names[i]] = {"predicted": predicted,
                                  "ground_truth": gt}
        
    return result


device = "cuda:0"

images_path = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba/"
eval_partition_path = "/mnt/hdd/volume1/anastasija/CelebA/Eval/"
annotations_directory = "/mnt/hdd/volume1/anastasija/CelebA/Anno/"
captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/captions_19_attr_all_images.txt"
save_dir = "/mnt/hdd/volume1/anastasija/CLIP_outputs/results"

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

# Disable gradient computation and reduce memory consumption.
test_list_num_diagonal_max_values_im_percent = []
test_list_num_diagonal_max_values_texts_percent = []
test_list_mean_max_probs_im = []
test_list_mean_max_probs_texts = []

with torch.no_grad():
    result = {}
    pbar = tqdm(test_dataloader, total=len(test_dataloader))
    iter = 0
    count = 0
    num_diagonal_max_values_im_count = 0
    num_diagonal_max_values_texts_count = 0

    mean_max_probs_im_count = 0
    mean_max_probs_texts_count = 0
    model.eval()
    for batch in pbar:
        iter = iter+1

        im_names, images, captions, texts = batch
        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        # Results
        result = get_predicted_and_gt_caption(logits_per_image, captions, im_names, result)

        # Calculate the metrics
        _, _, num_diagonal_max_values_im, mean_max_probs_im = get_num_diagonal_max_values(logits_per_image, device)
        _, _, num_diagonal_max_values_texts, mean_max_probs_texts = get_num_diagonal_max_values(logits_per_text, device)

        num_diagonal_max_values_im_count += num_diagonal_max_values_im.item()
        num_diagonal_max_values_texts_count += num_diagonal_max_values_texts.item()

        mean_max_probs_im_count += mean_max_probs_im.item()
        mean_max_probs_texts_count += mean_max_probs_texts.item()

        batch_top5_count = get_num_top5_max_values(logits_per_image)
        count += batch_top5_count

        # # Convert validation metrics into percentages
        # num_diagonal_max_values_im_percent = num_diagonal_max_values_im / test_dataloader.batch_size
        # num_diagonal_max_values_texts_percent = num_diagonal_max_values_texts / test_dataloader.batch_size

        # # Add them to a list to calculate mean value
        # test_list_num_diagonal_max_values_im_percent.append(num_diagonal_max_values_im_percent.item())
        # test_list_num_diagonal_max_values_texts_percent.append(num_diagonal_max_values_texts_percent.item())

        test_list_mean_max_probs_im.append(mean_max_probs_im.item())
        test_list_mean_max_probs_texts.append(mean_max_probs_texts.item())

    # Mean values of all batches in epoch
    mean_num_diagonal_max_values_im_percent = num_diagonal_max_values_im_count / len(test_images)
    mean_num_diagonal_max_values_texts_percent = num_diagonal_max_values_texts_count / len(test_images)
    
    test_mean_diag_max_prob_im = mean(test_list_mean_max_probs_im)
    test_mean_diag_max_prob_texts = mean(test_list_mean_max_probs_texts)

    # Top-1 acc
    top1_acc = mean_num_diagonal_max_values_im_percent

    # Top-5 acc
    top5_acc = count / len(test_images)

# Save the results
txt_file_name = "pretrained_chosen_attributes.txt"

with open(os.path.join(save_dir, txt_file_name), 'w') as file:
     file.write(f"mean_num_diagonal_max_values_im_percent: {mean_num_diagonal_max_values_im_percent}\n")
     file.write(f"mean_num_diagonal_max_values_texts_percent: {mean_num_diagonal_max_values_texts_percent}\n")
     file.write(f"test_mean_diag_max_prob_im: {test_mean_diag_max_prob_im}\n")
     file.write(f"test_mean_diag_max_prob_texts: {test_mean_diag_max_prob_texts}\n")
     file.write(f"top5_acc: {top5_acc}\n\n")

file.close()

# Write the dictionary to a file
with open(os.path.join(save_dir, txt_file_name), 'a') as file:
    for key, value in result.items():
        file.write(f'{key}: {value}\n')

file.close()


print(f"mean_num_diagonal_max_values_im_percent: {mean_num_diagonal_max_values_im_percent}")
print(f"mean_num_diagonal_max_values_texts_percent: {mean_num_diagonal_max_values_texts_percent}")
print(f"test_mean_diag_max_prob_im: {test_mean_diag_max_prob_im}")
print(f"test_mean_diag_max_prob_texts: {test_mean_diag_max_prob_texts}")
print()
print(f"Top-5 accuracy: {top5_acc}")

import pdb; pdb.set_trace()