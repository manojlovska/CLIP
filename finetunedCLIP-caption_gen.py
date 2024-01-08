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

from experiments.base_exp import Exp
from data.dataset import CelebADataset

device = "cuda:1"

images_path = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba"
captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions_all_attributes.csv"
eval_partition_path = "/mnt/hdd/volume1/anastasija/CelebA/Eval"

exp = Exp()
model, preprocess = exp.get_model(exp.vision_encoder)

test_dataset = CelebADataset(images_path, captions_path, eval_partition_path, preprocess,  name="test")

model.load_state_dict(torch.load("/mnt/hdd/volume1/anastasija/CLIP_outputs/CLIP-fine-tuning-val_loss/best_ckpt.pth")["model"])

test_image = test_dataset.__getitem__(0)[1].to(device)
test_image = test_image.unsqueeze(0).to(device)

