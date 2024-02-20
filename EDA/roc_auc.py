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
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda:1"

# Paths
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

# Model
model, preprocess = clip.load("ViT-B/32", device=device)

# Validation set
val_set = CelebADataset(images_path, captions_path, eval_partition_path, preprocess,  name="val")
val_images = [os.path.join(images_path, image) for image in val_set.images_list]

val_dataloader = DataLoader(val_set, batch_size=64, shuffle=False)

#########################################################################################################################
# Read the distinct captions
file_path = '/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/captions_single_attribute.txt'

attr_captions = []

with open(file_path, 'r') as file:
    for line in file:
        attr_captions.append(line.strip())

# Read the ground truth captions for every image
json_file_path = '/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/captions_single_attr_all_imgs.json'

# Read the JSON file into a Python dictionary
with open(json_file_path, 'r') as json_file:
    json_captions = json.load(json_file)


with torch.no_grad():
    result = {}
    pbar = tqdm(val_images, total=len(val_images))
    model.eval()
    result = np.empty(shape=(len(val_images),40))
    idx = 0
    for img in pbar:
        img_name = os.path.basename(img)
        image = preprocess(Image.open(img))
        image = image.unsqueeze(0).to(device)
        cosine_similarities = []
        for attr_caption in attr_captions:
            # attr_caption = attr_captions[0] ############################## Comment this
            tokenized_caption = clip.tokenize(attr_caption, truncate=True)

            text = tokenized_caption.to(device)

            logits_per_image, _ = model(image, text)
            cosine_sim = logits_per_image.item() / 100.
            cosine_similarities.append(cosine_sim)
        result[idx] = cosine_similarities
        idx += 1
        # print(f'logits_per_image: {logits_per_image}')
        # print(f'cosine_sim: {logits_per_image.item() / 100.}')
np.save("result", result)
import pdb
pdb.set_trace

###############################################################################################
######################### Calculate TPR, FPR, ROC, AUC

# Transform the ground truths in same format
def get_annotation(fnmtxt, columns=None, verbose=True):
    if verbose:
        print("_"*70)
        print(fnmtxt)
    
    rfile = open(fnmtxt, 'r' ) 
    texts = rfile.readlines()
    rfile.close()
    
    if not columns:
        columns = np.array(texts[1].split(" "))
        columns = columns[columns != "\n"]
        texts = texts[2:]
    
    df = []
    for txt in texts:
        txt = np.array(txt.rstrip("\n").split(" "))
        txt = txt[txt != ""]
    
        df.append(txt)
        
    df = pd.DataFrame(df)

    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"]+ list(columns)
    df.columns = columns   
    df = df.dropna()
    if verbose:
        print(" Total number of annotations {}\n".format(df.shape))
        print(df.head())
    ## cast to integer
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = pd.to_numeric(df[nm],downcast="integer")
    return(df)

# Whole dataset
annotations_path = '/mnt/hdd/volume1/anastasija/CelebA/Anno'
attr = get_annotation(os.path.join(annotations_path, 'list_attr_celeba.txt'), verbose=False)
val_attr = attr.iloc[162770:182637]

gt_all = np.empty(shape=(len(val_attr),40))
for i in range(len(val_attr)):
    gt_all[i] = list(val_attr.iloc[0][1:])

gt_all = np.where(gt_all == -1, 0, gt_all)

########################## ROC CURVES AND GTS
from sklearn.metrics import roc_curve, auc

# Calculate ROC curves
roc_curves = []
auc_scores = []
for i in range(result.shape[1]):
    fpr, tpr, _ = roc_curve(gt_all[:, i], result[:, i])
    roc_auc = auc(fpr, tpr)
    roc_curves.append((fpr, tpr))
    auc_scores.append(roc_auc)




