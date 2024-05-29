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

import torch
from accelerate import Accelerator

from clip.model import CLIP

# Initialize the accelerator
accelerator = Accelerator()

# Define the paths to the checkpoint files
model_checkpoint = "CLIP_outputs/CLIP-fine-tuning-VGGFace2/fluent-moon-10/epoch_87/model.safetensors"
optimizer_checkpoint = "CLIP_outputs/CLIP-fine-tuning-VGGFace2/fluent-moon-10/epoch_87/optimizer.bin"
random_states_checkpoint = "CLIP_outputs/CLIP-fine-tuning-VGGFace2/fluent-moon-10/epoch_87/random_states_0.pkl"

# Load the model (make sure to define the model class or use the appropriate one)
model = CLIP()  # Replace with your model class
model.load_state_dict(torch.load(model_checkpoint, map_location='cpu'))

# Load the optimizer (make sure to define the optimizer with the same parameters)
optimizer = torch.optim.Adam(model.parameters())  # Replace with your optimizer
optimizer.load_state_dict(torch.load(optimizer_checkpoint, map_location='cpu'))

# Load the random states
with open(random_states_checkpoint, 'rb') as f:
    random_states = torch.load(f, map_location='cpu')
torch.set_rng_state(random_states['torch'])
if torch.cuda.is_available():
    torch.cuda.set_rng_state(random_states['cuda'])
if 'numpy' in random_states:
    import numpy as np
    np.random.set_state(random_states['numpy'])
if 'python' in random_states:
    import random
    random.setstate(random_states['python'])

# Move the model to the CPU
model.to('cpu')

print("Done")


# attributes_caption_parts = {
#     'Gender': ['man', 'woman'],
#     'Pronoun': ['He', 'She'],
#     'PossPronoun': ['His', 'Her'],
#     'Identity': ['Young', 'Middle_Aged', 'Senior'],
#     'Ethnicity': ['Asian', 'White', 'Black']
# }

# caption_parts = {
#     'Gender': 'A photo of a {}.', # A photo of a {man/woman}
#     'Identity': 'A photo of a {} {}.', # A photo of a {young/middle aged/senior} {man/woman}
#     'Ethnicity': 'A photo of a {}'
# }

# attribute_groups = {
#         'Identity': ['Young', 'Middle_Aged', 'Senior'],
#         'Ethnicity': ['Asian', 'White', 'Black'],
#         'Appearance': ['Rosy_Cheeks', 'Shiny_Skin', 'Bald', 'Wavy_Hair', 'Receding_Hairline', 'Bangs', 
#                        'Sideburns', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'No_Beard', 
#                        'Mustache', '5_o_Clock_Shadow', 'Goatee', 'Oval_Face', 'Square_Face', 'Round_Face', 
#                        'Double_Chin', 'High_Cheekbones', 'Chubby', 'Obstructed_Forehead', 'Fully_Visible_Forehead', 
#                        'Brown_Eyes', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Arched_Eyebrows', 'Mouth_Closed', 
#                        'Smiling', 'Big_Lips', 'Big_Nose', 'Pointy_Nose', 'Heavy_Makeup', 'Attractive'],
#         'Accessories': ['Wearing_Hat', 'Wearing_Earrings', 'Wearing_Necktie', 'Wearing_Lipstick', 'No_Eyewear', 'Eyeglasses']
#     }


# attribute_groups2 = {
#     "sentence1": {
#         'Gender': ['man', 'woman'],
#         'Age': [' young', ' middle aged', ' senior'],
#         'Ethnicity': [' of asian descent', ' of white descent', ' of black descent'],
#         'Baldness': [' that is bald', ' with a receding hairline'],
#         'HairType': [ 'with wavy locks', 'with stylish bangs'],
#         'HairColor': [', sporting black hair.', ', sporting blond hair.', ', sporting brown hair.', ', sporting gray hair.'],
#         'Eyebrows': [' with arched eyebrows', ' with bushy eyebrows', 'with bushy arched eyebrows'],
#         'Lips': [', with big lips'],
#         'Nose': [', with a pointy nose', ', with a big nose', ', with a big pointy nose'],
#         'Makeup': [' and a heavy makeup.']
#     },

#     "sentence2": {
#         'Pronoun': ['He', 'She'],
#         'Smiling': [' smiles joyfully'],
#         'MouthClosed': ['with a closed mouth, '],
#         'Face': [' has an oval face', ' has a square face', ' has a round face', 
#                  ' has an oval chubby face', ' has a square chubby face', ' has a round chubby face'],
#         'DoubleChin': [' with a double chin'],
#         'Cheekbones': [' and prominent cheekbones'],
#         'Attractiveness': [' and looks pretty attractive'],
#         'EyeColour': [' with brown eyes'],
#         'BagsUnderEyes': [', but also has bags under the eyes.']
#     },

#     "sentence3": {
#         'Pronoun': ['He', 'She'],
#         'Beard': [' is clean shaven', ' has a mustache', ' has a goatee', ' has a 5 o clock shadow shave'],
#         'Earrings': [' wearing elegant earrings'],
#         'Lipstick': [' wearing a lipstick'],
#         'Hat': [' wearing a stylish hat'],
#         'Forehead': [' , with a partially obstructed forehead', ' , with a visible forehead'],
#         'Eyeglasses': ['without any eyewear', ', wearing eyeglasses'],
#         'Necktie': [' and a formal attire.'],
        

#     }

# }

# previous_caption = "A photo of a {}"






# def get_predicted_and_gt_caption(logits, captions, image_names, result):
#     ground_truths = np.arange(logits.shape[0])
#     probabilities = logits.softmax(dim=-1)

#     sorted_prob, indices = probabilities.sort(descending=True, dim=-1)
#     indices = indices.cpu().numpy()

#     for i, ind in enumerate(indices):
#         predicted = captions[ind[0]]
#         gt = captions[ground_truths[i]]

#         result[image_names[i]] = {"predicted": predicted,
#                                   "ground_truth": gt}
        
#     return result


# device = "cpu"

# images_path = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba/"
# eval_partition_path = "/mnt/hdd/volume1/anastasija/CelebA/Eval/"
# annotations_directory = "/mnt/hdd/volume1/anastasija/CelebA/Anno/"
# captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/captions_all_attributes_new.txt"
# save_dir = "/mnt/hdd/volume1/anastasija/CLIP_outputs/results"
# model_path = "/mnt/hdd/volume1/anastasija/CLIP_outputs/CLIP-fine-tuning-latest/neat-gorge-6/epoch_2_ckpt.pth"

# annotations_filename = "list_attr_celeba.txt"
# annotations_path = os.path.join(annotations_directory, annotations_filename)

# annotations_df = pd.read_csv(annotations_path, delimiter='\s+', skiprows=[0])
# attributes = list(annotations_df.columns.values)

# list_eval_partition = pd.read_csv(os.path.join(eval_partition_path, "list_eval_partition.txt"), sep=" ", header=None)
# img_filenames_all = sorted(glob.glob(images_path + '*.jpg'))


# model, preprocess = clip.load("ViT-B/32", device=device)
# model.load_state_dict(torch.load(model_path)["model"])

# test_dataset = CelebADataset(images_path, captions_path, eval_partition_path, preprocess,  name="test")
# test_images = [os.path.join(images_path, image) for image in test_dataset.images_list]

# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Disable gradient computation and reduce memory consumption.
# test_list_num_diagonal_max_values_im_percent = []
# test_list_num_diagonal_max_values_texts_percent = []
# test_list_mean_max_probs_im = []
# test_list_mean_max_probs_texts = []

# with torch.no_grad():
#     result = {}
#     pbar = tqdm(test_dataloader, total=len(test_dataloader))
#     iter = 0
#     count = 0
#     num_diagonal_max_values_im_count = 0
#     num_diagonal_max_values_texts_count = 0

#     mean_max_probs_im_count = 0
#     mean_max_probs_texts_count = 0
#     model.eval()
#     for batch in pbar:
#         iter = iter+1

#         im_names, images, captions, texts = batch
#         images = images.to(device)
#         texts = texts.to(device)

#         logits_per_image, logits_per_text = model(images, texts)

#         # Results
#         result = get_predicted_and_gt_caption(logits_per_image, captions, im_names, result)

#         # Calculate the metrics
#         _, _, num_diagonal_max_values_im, mean_max_probs_im = get_num_diagonal_max_values(logits_per_image, device)
#         _, _, num_diagonal_max_values_texts, mean_max_probs_texts = get_num_diagonal_max_values(logits_per_text, device)

#         num_diagonal_max_values_im_count += num_diagonal_max_values_im.item()
#         num_diagonal_max_values_texts_count += num_diagonal_max_values_texts.item()

#         mean_max_probs_im_count += mean_max_probs_im.item()
#         mean_max_probs_texts_count += mean_max_probs_texts.item()

#         batch_top5_count = get_num_top5_max_values(logits_per_image)
#         count += batch_top5_count

#         # # Convert validation metrics into percentages
#         # num_diagonal_max_values_im_percent = num_diagonal_max_values_im / test_dataloader.batch_size
#         # num_diagonal_max_values_texts_percent = num_diagonal_max_values_texts / test_dataloader.batch_size

#         # # Add them to a list to calculate mean value
#         # test_list_num_diagonal_max_values_im_percent.append(num_diagonal_max_values_im_percent.item())
#         # test_list_num_diagonal_max_values_texts_percent.append(num_diagonal_max_values_texts_percent.item())

#         test_list_mean_max_probs_im.append(mean_max_probs_im.item())
#         test_list_mean_max_probs_texts.append(mean_max_probs_texts.item())

#     # Mean values of all batches in epoch
#     mean_num_diagonal_max_values_im_percent = num_diagonal_max_values_im_count / len(test_images)
#     mean_num_diagonal_max_values_texts_percent = num_diagonal_max_values_texts_count / len(test_images)
    
#     test_mean_diag_max_prob_im = mean(test_list_mean_max_probs_im)
#     test_mean_diag_max_prob_texts = mean(test_list_mean_max_probs_texts)

#     # Top-1 acc
#     top1_acc = mean_num_diagonal_max_values_im_percent

#     # Top-5 acc
#     top5_acc = count / len(test_images)

# # Print
# print(f"mean_num_diagonal_max_values_im_percent: {mean_num_diagonal_max_values_im_percent}")
# print(f"mean_num_diagonal_max_values_texts_percent: {mean_num_diagonal_max_values_texts_percent}")
# print(f"test_mean_diag_max_prob_im: {test_mean_diag_max_prob_im}")
# print(f"test_mean_diag_max_prob_texts: {test_mean_diag_max_prob_texts}")
# print()
# print(f"Top-5 accuracy: {top5_acc}")

# # Save the results
# txt_file_name = "fine_tuned_all_attributes.txt"

# with open(os.path.join(save_dir, txt_file_name), 'w') as file:
#      file.write(f"mean_num_diagonal_max_values_im_percent: {mean_num_diagonal_max_values_im_percent}\n")
#      file.write(f"mean_num_diagonal_max_values_texts_percent: {mean_num_diagonal_max_values_texts_percent}\n")
#      file.write(f"test_mean_diag_max_prob_im: {test_mean_diag_max_prob_im}\n")
#      file.write(f"test_mean_diag_max_prob_texts: {test_mean_diag_max_prob_texts}\n")
#      file.write(f"top5_acc: {top5_acc}\n\n")

# file.close()

# # Write the dictionary to a file
# with open(os.path.join(save_dir, txt_file_name), 'a') as file:
#     for key, value in result.items():
#         file.write(f'{key}: {value}\n')

# file.close()

# import pdb; pdb.set_trace()
