from data.vgg2_dataset import VGGFace2Dataset
import clip
import os
from tqdm import tqdm

device = "cuda:1"
_, preprocess = clip.load("ViT-B/32", device=device, jit=False)

vgg_test_dataset = VGGFace2Dataset(vgg2_path="/mnt/hdd/volume1/VGGFace2", captions_path="/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_25_att_29032024.txt", preprocess=preprocess, split="test")

import pdb
pdb.set_trace()

# split = "train"
# base_path = "/mnt/hdd/volume1/VGGFace2"
# captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_25_att_29032024.txt"


# def get_images_list(base_path):
#     with open(os.path.join(base_path, f"{split}_list.txt"), 'r') as file:
#         return [line.strip() for line in file]

# def read_captions_for_images(images_list, captions_path):
#     matched_captions = {}
#     with open(captions_path, 'r') as file:
#         lines = file.readlines()
#         for line in tqdm(lines, desc="Processing captions", total=len(lines)):
#             image_name, caption = line.strip().split(' ', 1)
#             if image_name in images_list:
#                 matched_captions[image_name] = caption
#     return matched_captions

# def read_captions_for_images2(images_list, captions_path):
#     matched_captions = {}
#     images_set = set(images_list)  # Convert images_list to a set for faster lookup

#     with open(captions_path, 'r') as file:
#         for line in tqdm(file, desc="Processing captions"):
#             image_name, caption = line.strip().split(' ', 1)
#             if image_name in images_set:
#                 matched_captions[image_name] = caption

#     return matched_captions


# images_list_train = get_images_list(base_path)
# captions_train = read_captions_for_images2(images_list_train, captions_path)

# split = "test"
# images_list_test = get_images_list(base_path)
# captions_test = read_captions_for_images2(images_list_test, captions_path)
# import pdb
# pdb.set_trace()


