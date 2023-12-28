import os
import json
from PIL import Image
import pandas as pd
from tqdm import tqdm
from statistics import mean
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from transformers import CLIPProcessor, CLIPModel
from data.dataset import CelebADataset
import torch

wandb.init(project="CLIP-fine-tuning")

# Paths
images_path = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba/"
captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/"
eval_partitions_path = "/mnt/hdd/volume1/anastasija/CelebA/Eval/"

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu" 

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Dataset and dataloader
# train_dataset = CelebA_subsetDataset(images_path=images_path, captions_path=captions_path, eval_partition_path=eval_partitions_path, name="train")
train_dataset_100 = CelebADataset(images_path=images_path, captions_path=captions_path, eval_partition_path=eval_partitions_path, preprocess=preprocess, name="train")
val_dataset_100 = CelebADataset(images_path=images_path, captions_path=captions_path, eval_partition_path=eval_partitions_path, preprocess=preprocess, name="val")

train_dataloader = DataLoader(train_dataset_100, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset_100, batch_size=8, shuffle=False)

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

# TODO
def evaluate_and_save_model():
    pass

if device == "cpu":
  model.float()

# Prepare the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset

# Specify the loss function
loss = nn.MSELoss()

# Train the model
num_epochs = 3
for epoch in range(num_epochs):
    model.train(True)
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()

        image_names, images, texts = batch 
        
        images = images.to(device)
        texts = texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        # ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = loss(logits_per_image, logits_per_text)

        # Backward pass
        total_loss.backward()
        # optimizer.zero_grad() #################################################### DALI TREBA

        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    # Evaluate the model
    if (epoch + 1) % 1 == 0:
        # evaluate_and_save_model()
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        all_probabilities = []
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                val_im_names, val_images, val_texts = vdata
                val_images = val_images.to(device)
                val_texts = val_texts.to(device)

                val_logits_per_image, val_logits_per_text = model(val_images, val_texts)
                probs = val_logits_per_image.softmax(dim=-1).cpu().numpy()
                all_probabilities.append(probs)
        
        mean_values_per_batch = []
        for probs_per_batch in all_probabilities:
            max_probs_per_img = probs_per_batch.max(axis=-1)
            mean_value = max_probs_per_img.mean()

            mean_values_per_batch.append(mean_value)

        avrg = mean(mean_values_per_batch)
        print(f"Epoch {epoch+1}, average score: {avrg}")

    wandb.log({"train/loss": total_loss, "val/average_probabilities": avrg})
    pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")
    






