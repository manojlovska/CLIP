import os
import glob
import shutil

import pandas as pd
import numpy as np

data_directory = "/mnt/hdd/volume1/anastasija/CelebA_subset/Img/img_celeba/"
eval_directory = "/mnt/hdd/volume1/anastasija/CelebA_subset/Eval/"
annotations_directory = "/mnt/hdd/volume1/anastasija/CelebA_subset/Anno/"

annotations_filename = "list_attr_celeba.txt"
annotations_path = os.path.join(annotations_directory, annotations_filename)

annotations_df = pd.read_csv(annotations_path, delimiter='\s+', skiprows=[0])
attributes = list(annotations_df.columns.values)


list_eval_partition = pd.read_csv(os.path.join(eval_directory, "list_eval_partition.txt"), sep=" ", header=None)
img_filenames_all = glob.glob(data_directory + '*.jpg')

train_set = [train_img for train_img in img_filenames_all if (list_eval_partition.loc[list_eval_partition.iloc[:, 0] == os.path.basename(train_img), list_eval_partition.columns[1]].values).item() == 0]
val_set = [val_img for val_img in img_filenames_all if (list_eval_partition.loc[list_eval_partition.iloc[:, 0] == os.path.basename(val_img), list_eval_partition.columns[1]].values).item() == 1]
test_set = [test_img for test_img in img_filenames_all if (list_eval_partition.loc[list_eval_partition.iloc[:, 0] == os.path.basename(test_img), list_eval_partition.columns[1]].values).item() == 2]

# """
# A photo of a {MAN/WOMAN}, 
# A photo of a {YOUNG}, {ATTRACTIVE} {MAN/WOMAN}, that is {BALD} and {BLURRY}, with {ALL OTHER ATTRIBUTES FOR THAT IMAGE}
# A photo of a {YOUNG}, {ATTRACTIVE} {MAN/WOMAN}, that is {BALD/BLURRY}, with {ALL OTHER ATTRIBUTES FOR THAT IMAGE}
# A photo of a {YOUNG/ATTRACTIVE} {MAN/WOMAN}, that is {BALD} and {BLURRY}, with {ALL OTHER ATTRIBUTES FOR THAT IMAGE}
# A photo of a {YOUNG/ATTRACTIVE} {MAN/WOMAN}, that is {BALD/BLURRY}, with {ALL OTHER ATTRIBUTES FOR THAT IMAGE}
# """
# templates = ["A photo of a {}"
#             "A photo of a {}, {} {}, that is {} and {}, with {}",
#              "A photo of a {}, {} {}, that is {}, with {}",
#              "A photo of a {} {}, that is {} and {}, with {}",
#              "A photo of a {} {}, that is {}, with {}"]


""" TEST FOR MULTIPLE IMAGES OF TEST SET """
N = 100
templates = []
for image in test_set[:N]:
    image_name = os.path.basename(image)
    values = annotations_df.loc[image_name, :]

    if values["Male"] > 0:
        gender = "man"
    else:
        gender = "woman"

    template = "A photo of a {}".format(gender) + " with the following attributes: " + ", ".join("{}".format(attribute) for attribute in list(np.array(attributes)[np.array(values) > 0]))
    templates.append(template)

import torch
import clip
from PIL import Image
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(templates, truncate=True).to(device)
idx = 0
count = 0
max_probabilities_per_image = []
results = []
for img in test_set[:N]:
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    max_prob = np.max(probs[0])
    max_probabilities_per_image.append(max_prob)

    prediction = templates[np.argmax(probs[0])]
    ground_truth = templates[idx]

    results.append([ground_truth, prediction])
    idx +=1

    if ground_truth == prediction:
        count +=1

""" DIFFERENT CAPTIONS """
N = 5
# prompts = ["A photo of a man wih {}.",
#            "A photo of a woman with {}.",
#            "A photo of a man that has {}.",
#            "A photo of a woman that has {}.",
#            "A photo of a man that is {}.",
#            "A photo of a woman that is {}."]

prompts = ["A photo of a man wih {}.",
           "A photo of a woman with {}.",
           "A photo of a man that is {}.",
           "A photo of a woman that is {}."]

templates = []
for prompt in prompts[:2]:
    template = [prompt.format(attribute) for attribute in attributes if attribute not in 
                ["Attractive", "Bald", "Blurry", "Chubby", "Male", "Smiling", "Young", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie"]]
    templates.extend(template)

for prompt in prompts[-2:]:
    template = [prompt.format(attribute) for attribute in ["Attractive", "Bald", "Blurry", "Chubby", "Smiling", "Young", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie"]]
    templates.extend(template)

import torch
import clip
from PIL import Image
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(templates, truncate=True).to(device)
results = []
probabilities_results = []
for img in test_set[:N]:
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    n_attributes = sorted(range(len(probs[0])), key=lambda i: probs[0][i], reverse=True)[:5]
    predictions = list(map(lambda i: templates[i], n_attributes))
    
    probabilities_results.append(sorted(probs[0], reverse=True)[:5])
    results.append(predictions)

# """ TEST FOR ONE IMAGE """
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open(test_set[4])).unsqueeze(0).to(device)
# text = clip.tokenize(templates).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]



