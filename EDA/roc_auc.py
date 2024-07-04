import os
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from EDA.helper_functions import get_annotation, get_gt, roc
from data.vgg2_dataset import VGGFace2Dataset
from clip.clip import load
from tqdm import tqdm
import pandas as pd

device = "cuda:1"

def get_attribute_values(annotation_path, filenames_list):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(annotation_path)
    
    # Filter the DataFrame to include only the rows with filenames in the filenames_list
    filtered_df = df[df['Filename'].isin(filenames_list)]
    
    # Extract the attribute values starting from the second column onward
    val_attr = filtered_df.iloc[:, 2:]
    
    # Convert the DataFrame to a NumPy array
    gt_all = val_attr.to_numpy()
    
    # Replace all -1 values with 0
    gt_all = np.where(gt_all == -1, 0, gt_all)
    
    return gt_all


# # Get the annotations
# annotations_path = '/mnt/hdd/volume1/anastasija/CelebA/Anno'
# attr = get_annotation(os.path.join(annotations_path, 'list_attr_celeba.txt'), verbose=False)
# val_attr = attr.iloc[162770:182637]

# # Ground truths
# gt_all = get_gt(annotations_path)
image_captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_att_07052024.txt"
vgg2_path = "/mnt/hdd/volume1/VGGFace2"
annotations_path = "/mnt/hdd/volume1/MAAD-Face/MAAD_Face.csv"
df = pd.read_csv(annotations_path)
list_attr = df.columns[2:]

_, preprocess = load("ViT-B/32",device=device,jit=False)
val_set = VGGFace2Dataset(vgg2_path=vgg2_path, captions_path=image_captions_path, preprocess=preprocess, split="test")

# Make the images list
image_names = []
for i in tqdm(range(len(val_set))):
    image_name, _, _, _ = val_set[i]
    image_names.append(image_name)

gt_all = get_attribute_values(annotations_path, image_names)

########################## ROC CURVES AND AUC SCORES
# BASIC captions and attributes
predicted_path_pretrained = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/pretrained_clip_basic_VGGFace2.npy"

# List of attributes
# list_attr = list(val_attr.columns[1:])

predicted_basic_pretrained = np.load(predicted_path_pretrained)

# Only for negatives
# if "negative" in predicted_path_pretrained:
#     gt_all = np.where(gt_all == 1, 0, 1)

roc_curves_basic, auc_scores_basic = roc(predicted_basic_pretrained, gt_all)

# Plot the ROC curves
plt.figure(figsize=(10, 8))

# Filter and plot only the curves with valid (non-NaN) AUC scores
valid_indices = [i for i, score in enumerate(auc_scores_basic) if not np.isnan(score)]
for i in valid_indices:  # Limiting to the first few valid curves for clarity
    fpr, tpr = roc_curves_basic[i]
    plt.plot(fpr, tpr, label=f'{list_attr[i]}  (AUC = {auc_scores_basic[i]:.2f})', linewidth=3)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curves Pretrained')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Average AUC
mean_auc_basic_pretrained = mean(auc_scores_basic)

# AUC scores vs. attributes
plt.figure(figsize=(10, 8))

plt.barh(valid_indices, auc_scores_basic)
plt.yticks(valid_indices, list_attr, color='black', rotation=0, fontweight='bold', fontsize='10')

plt.xlabel('AUC score')
plt.ylabel('Attribute')
plt.title('AUC scores Basic Pretrained')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Define AUC intervals based on the specified ranges
auc_intervals = [(round(x*0.1, 1), round((x+1)*0.1, 1)) for x in range(10)]

# Categorize AUC scores into intervals
categorized_scores_basic = {interval: [] for interval in auc_intervals}

for score in auc_scores_basic:
    for interval in auc_intervals:
        if interval[0] <= score < interval[1]:
            categorized_scores_basic[interval].append(score)
            break

categorized_attr_basic = {interval: [] for interval in auc_intervals}

for index, score in enumerate(auc_scores_basic):
    for interval in auc_intervals:
        if interval[0] <= score < interval[1]:
            categorized_attr_basic[interval].append(list_attr[index])
            break

# Prepare data for plotting
intervals = [f"{interval[0]}-{interval[1]}" for interval in categorized_scores_basic.keys()]
counts = [len(scores) for scores in categorized_scores_basic.values()]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(intervals, counts, color='skyblue')
plt.title('Distribution of AUC Scores Across Intervals Basic Pretrained')
plt.xlabel('AUC Score Intervals')
plt.ylabel('Count of AUC Scores')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')

# Show the plot
plt.tight_layout()
plt.show()

categorized_attr_basic
####################################################################################################################################################################
# BASIC captions and attributes
predicted_path_fine_tuned = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/fine_tuned_clip_basic_VGGFace2_con_len_77.npy"

# List of attributes
# list_attr = list(val_attr.columns[1:])

predicted_basic_fine_tuned = np.load(predicted_path_fine_tuned)

# Only for negatives
# if "negative" in predicted_path_pretrained:
#     gt_all = np.where(gt_all == 1, 0, 1)

roc_curves_basic, auc_scores_basic = roc(predicted_basic_fine_tuned, gt_all)

# Plot the ROC curves
plt.figure(figsize=(10, 8))

# Filter and plot only the curves with valid (non-NaN) AUC scores
valid_indices = [i for i, score in enumerate(auc_scores_basic) if not np.isnan(score)]
for i in valid_indices:  # Limiting to the first few valid curves for clarity
    fpr, tpr = roc_curves_basic[i]
    plt.plot(fpr, tpr, label=f'{list_attr[i]}  (AUC = {auc_scores_basic[i]:.2f})', linewidth=3)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curves Fine-Tuned')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Average AUC
mean_auc_basic_fine_tuned = mean(auc_scores_basic)

# AUC scores vs. attributes
plt.figure(figsize=(10, 8))

plt.barh(valid_indices, auc_scores_basic)
plt.yticks(valid_indices, list_attr, color='black', rotation=0, fontweight='bold', fontsize='10')

plt.xlabel('AUC score')
plt.ylabel('Attribute')
plt.title('AUC scores Basic Fine-Tuned')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Define AUC intervals based on the specified ranges
auc_intervals = [(round(x*0.1, 1), round((x+1)*0.1, 1)) for x in range(10)]

# Categorize AUC scores into intervals
categorized_scores_basic = {interval: [] for interval in auc_intervals}

for score in auc_scores_basic:
    for interval in auc_intervals:
        if interval[0] <= score < interval[1]:
            categorized_scores_basic[interval].append(score)
            break

categorized_attr_basic = {interval: [] for interval in auc_intervals}

for index, score in enumerate(auc_scores_basic):
    for interval in auc_intervals:
        if interval[0] <= score < interval[1]:
            categorized_attr_basic[interval].append(list_attr[index])
            break

# Prepare data for plotting
intervals = [f"{interval[0]}-{interval[1]}" for interval in categorized_scores_basic.keys()]
counts = [len(scores) for scores in categorized_scores_basic.values()]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(intervals, counts, color='skyblue')
plt.title('Distribution of AUC Scores Across Intervals Basic Fine-Tuned')
plt.xlabel('AUC Score Intervals')
plt.ylabel('Count of AUC Scores')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')

# Show the plot
plt.tight_layout()
plt.show()

categorized_attr_basic























































###############################################################################################################################################

# # # PARAPHRISED captions and attributes
# # predicted_path = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/results/pretrained_clip_paraphrised.npy"

# # # List of attributes
# # list_attr = list(val_attr.columns[1:])

# # predicted_paraphrised = np.load(predicted_path)

# # # Only for negatives
# # if "negative" in predicted_path:
# #     gt_all = np.where(gt_all == 1, 0, 1)

# # roc_curves_paraphrised, auc_scores_paraphrised = roc(predicted_paraphrised, gt_all)

# # # Plot the ROC curves
# # plt.figure(figsize=(10, 8))

# # # Filter and plot only the curves with valid (non-NaN) AUC scores
# # valid_indices = [i for i, score in enumerate(auc_scores_paraphrised) if not np.isnan(score)]
# # for i in valid_indices:  # Limiting to the first few valid curves for clarity
# #     fpr, tpr = roc_curves_paraphrised[i]
# #     plt.plot(fpr, tpr, label=f'{list_attr[i]}  (AUC = {auc_scores_paraphrised[i]:.2f})', linewidth=3)

# # plt.xlabel('False positive rate')
# # plt.ylabel('True positive rate')
# # plt.title('ROC Curves Paraphrised')
# # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # plt.show()

# # # Categorize AUC scores into intervals
# # categorized_scores_paraphrised = {interval: [] for interval in auc_intervals}

# # for score in auc_scores_paraphrised:
# #     for interval in auc_intervals:
# #         if interval[0] <= score < interval[1]:
# #             categorized_scores_paraphrised[interval].append(score)
# #             break

# # categorized_attr_paraphrised = {interval: [] for interval in auc_intervals}

# # for index, score in enumerate(auc_scores_paraphrised):
# #     for interval in auc_intervals:
# #         if interval[0] <= score < interval[1]:
# #             categorized_attr_paraphrised[interval].append(list_attr[index])
# #             break

# # # Prepare data for plotting
# # intervals = [f"{interval[0]}-{interval[1]}" for interval in categorized_scores_paraphrised.keys()]
# # counts = [len(scores) for scores in categorized_scores_paraphrised.values()]

# # # Create the bar chart
# # plt.figure(figsize=(10, 6))
# # plt.bar(intervals, counts, color='skyblue')
# # plt.title('Distribution of AUC Scores Across Intervals Paraphrised')
# # plt.xlabel('AUC Score Intervals')
# # plt.ylabel('Count of AUC Scores')
# # plt.xticks(rotation=45)
# # plt.grid(axis='y', linestyle='--')

# # # Show the plot
# # plt.tight_layout()
# # plt.show()

# # categorized_attr_paraphrised

# # # NEGATIVE captions and attributes
# # predicted_path = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/results/pretrained_clip_negative.npy"

# # # List of attributes
# # list_attr = list(val_attr.columns[1:])

# # predicted_negative = np.load(predicted_path)

# # # Only for negatives
# # if "negative" in predicted_path:
# #     print("yes")
# #     gt_all = np.where(gt_all == 1, 0, 1)

# # roc_curves_negative, auc_scores_negative = roc(predicted_paraphrised, gt_all)

# # # Plot the ROC curves
# # plt.figure(figsize=(10, 8))

# # # Filter and plot only the curves with valid (non-NaN) AUC scores
# # valid_indices = [i for i, score in enumerate(auc_scores_negative) if not np.isnan(score)]
# # for i in valid_indices:  # Limiting to the first few valid curves for clarity
# #     fpr, tpr = roc_curves_negative[i]
# #     plt.plot(fpr, tpr, label=f'{list_attr[i]}  (AUC = {auc_scores_negative[i]:.2f})', linewidth=3)

# # plt.xlabel('False positive rate')
# # plt.ylabel('True positive rate')
# # plt.title('ROC Curves Negative')
# # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # plt.show()

# # # Categorize AUC scores into intervals
# # categorized_scores_negative = {interval: [] for interval in auc_intervals}

# # for score in auc_scores_negative:
# #     for interval in auc_intervals:
# #         if interval[0] <= score < interval[1]:
# #             categorized_scores_negative[interval].append(score)
# #             break

# # categorized_attr_negative = {interval: [] for interval in auc_intervals}

# # for index, score in enumerate(auc_scores_negative):
# #     for interval in auc_intervals:
# #         if interval[0] <= score < interval[1]:
# #             categorized_attr_negative[interval].append(list_attr[index])
# #             break

# # # Prepare data for plotting
# # intervals = [f"{interval[0]}-{interval[1]}" for interval in categorized_scores_negative.keys()]
# # counts = [len(scores) for scores in categorized_scores_negative.values()]

# # # Create the bar chart
# # plt.figure(figsize=(10, 6))
# # plt.bar(intervals, counts, color='skyblue')
# # plt.title('Distribution of AUC Scores Across Intervals Negative')
# # plt.xlabel('AUC Score Intervals')
# # plt.ylabel('Count of AUC Scores')
# # plt.xticks(rotation=45)
# # plt.grid(axis='y', linestyle='--')

# # # Show the plot
# # plt.tight_layout()
# # plt.show()

# # categorized_attr_negative

# # # Dictionary of auc scores for every attribute in order basic, paraphrised, negative
# # auc_dict_all = {key: [round(auc_scores_basic[i], 3), round(auc_scores_paraphrised[i], 3), round(auc_scores_negative[i], 3)] for i, key in enumerate(list_attr)}

# ################################## PLOT EVERYTHING TOGETHER
# # ROC
# # Create a subplot of 1x3
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# all_auc_scores = [auc_scores_basic, auc_scores_paraphrised, auc_scores_negative]
# all_roc_curves = [roc_curves_basic, roc_curves_paraphrised, roc_curves_negative]
# gt1 = get_gt(annotations_path)
# gt2 = gt1
# gt3 = np.where(gt1 == 1, 0, 1)

# gts = [gt1, gt2, gt3]

# names = ['Basic', 'Paraphrised', 'Negative']

# for i, roc_cur in enumerate(all_roc_curves):
#     for j in valid_indices:  # Limiting to the first few valid curves for clarity
#         fpr, tpr = roc_cur[j]
#         axes[i].plot(fpr, tpr, label=f'{list_attr[j]}  (AUC = {all_auc_scores[i][j]:.2f})', linewidth=3)

#     axes[i].set_xlabel('False Positive Rate')
#     axes[i].set_ylabel('True Positive Rate')
#     axes[i].set_title(f'{names[i]}')
#     axes[i].legend(loc="lower center", bbox_to_anchor=(0.5, -2))

# # AUC scores
# # Create a figure and a set of subplots
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Plot the first bar chart
# axes[0].barh(valid_indices, auc_scores_basic)
# axes[0].set_title('Basic')
# axes[0].set_xlabel('AUC Score')
# axes[0].set_yticks(valid_indices)
# axes[0].set_yticklabels(list_attr, color='black', rotation=0, fontweight='bold', fontsize='10')

# # Plot the second bar chart
# axes[1].barh(valid_indices, auc_scores_paraphrised)
# axes[1].set_title('Paraphrised')
# axes[1].set_xlabel('AUC Score')
# axes[1].set_yticks(valid_indices)
# axes[1].set_yticklabels(list_attr, color='black', rotation=0, fontweight='bold', fontsize='10')

# # Plot the third bar chart
# axes[2].barh(valid_indices, auc_scores_negative)
# axes[2].set_title('Negative')
# axes[2].set_xlabel('AUC Score')
# axes[2].set_yticks(valid_indices)
# axes[2].set_yticklabels(list_attr, color='black', rotation=0, fontweight='bold', fontsize='10')

# # Automatically adjust subplot params for better layout
# plt.tight_layout()

# # Show the plot
# plt.show()

# # Histograms
# intervals_basic = [f"{interval[0]}-{interval[1]}" for interval in categorized_scores_basic.keys()]
# counts_basic = [len(scores) for scores in categorized_scores_basic.values()]

# intervals_paraphrised = [f"{interval[0]}-{interval[1]}" for interval in categorized_scores_paraphrised.keys()]
# counts_paraphrised = [len(scores) for scores in categorized_scores_paraphrised.values()]

# intervals_negative = [f"{interval[0]}-{interval[1]}" for interval in categorized_scores_negative.keys()]
# counts_negative = [len(scores) for scores in categorized_scores_negative.values()]

# # Create a figure and a set of subplots
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # plt.barh(valid_indices, auc_scores_basic)
# # plt.yticks(valid_indices, list_attr, color='black', rotation=0, fontweight='bold', fontsize='10')

# # Plot the first bar chart
# axes[0].bar(intervals_basic, counts_basic)
# axes[0].set_title('Basic')
# axes[0].set_xlabel('Interval')
# axes[0].set_ylabel('Counts')

# # Plot the second bar chart
# axes[1].bar(intervals_paraphrised, counts_paraphrised)
# axes[1].set_title('Paraphrised')
# axes[1].set_xlabel('Interval')
# axes[1].set_ylabel('Counts')

# # Plot the third bar chart
# axes[2].bar(intervals_negative, counts_negative)
# axes[2].set_title('Negative')
# axes[2].set_xlabel('Interval')
# axes[2].set_ylabel('Counts')

# # Automatically adjust subplot params for better layout
# plt.tight_layout()

# # Show the plot
# plt.show()

# ################# LOAD THE COMBINED ATTRIBUTES (GENDER + ANOTHER ONE)
# combined_res = np.load("results/pretrained_clip_combined.npy") #THINK ABOUT THIS


