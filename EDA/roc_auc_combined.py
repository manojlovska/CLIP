import os
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from EDA.helper_functions import get_annotation, get_gt, roc
import glob

device = "cuda:1"

def parse_attributes(filename, attribute_list):
    attributes_present = []
    indices = []
    for idx, attribute in enumerate(attribute_list):
        if attribute.lower().replace('_', '') in filename.lower().replace('_', ''):
            attributes_present.append(attribute)
            indices.append(idx)
    # Sort based on the first occurrence of each attribute in the filename
    sorted_data = sorted(zip(attributes_present, indices), key=lambda x: filename.lower().index(x[0].lower()))
    attributes_present, indices = zip(*sorted_data)
    return list(attributes_present), list(indices)

def attribute2idx(attribute, list_attr):
    idx = list_attr.index(attribute)

    return idx

def load_scores_single_att(path):
    predicted = np.load(path)

    return predicted

def get_roc_all_att(pedicted, gt):
    roc_curves, auc_scores = roc(pedicted, gt)

    return roc_curves, auc_scores

def get_gt_combined(gt_present):
    gt_combined = np.zeros((gt_present.shape[0], 2**gt_present.shape[1]))
    for i, row in enumerate(gt_present):
        row_string = ''.join(str(int(x)) for x in row)
        idx = abs(int(row_string, 2) - (2**gt_present.shape[1] - 1))
        gt_combined[i, idx] = 1

    return gt_combined


# Get the annotations
annotations_path = '/mnt/hdd/volume1/anastasija/CelebA/Anno'
attr = get_annotation(os.path.join(annotations_path, 'list_attr_celeba.txt'), verbose=False)
val_attr = attr.iloc[162770:182637]

# List of attributes
list_attr = list(val_attr.columns[1:])

# Ground truths
gt_all = get_gt(annotations_path)

########################## ROC CURVES AND AUC SCORES

# SINGLE ROC CURVE FOR EVERY ATTRIBUTE
# BASIC captions and attributes
predicted_path = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/results/pretrained_clip_basic.npy"

predicted_basic = np.load(predicted_path)

roc_curves_basic, auc_scores_basic = roc(predicted_basic, gt_all)

# BASIC captions and attributes
predictions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/results/new_results"

# Create a list to store paths of all .npy files
all_predictions_paths = []

# Use glob to find .npy files recursively
for file_path in glob.iglob(os.path.join(predictions_path, '**/*.npy'), recursive=True):
    all_predictions_paths.append(file_path)

all_predictions_paths = sorted(all_predictions_paths)
# all_predictions_paths = sorted(all_predictions_paths[8:])

corr_types = list(np.unique(np.array([predicted_path.split("/")[-2] for predicted_path in all_predictions_paths])))

auc_scores_dict = {}
for c, corr_type in enumerate(corr_types):
    predictions_paths = [pred_path for pred_path in all_predictions_paths if pred_path.split("/")[-2] == corr_type]
    # Create figure
    fig, axes = plt.subplots(len(predictions_paths), 4, figsize=(40, 20))
    for pd, predicted_path in enumerate(predictions_paths):

        predicted = np.load(predicted_path)

        attributes_present, indices = parse_attributes(os.path.basename(predicted_path), list_attr)
        
        gt_present = gt_all[:, indices]

        gt_combined = get_gt_combined(gt_present)

        max_values = np.amax(predicted, axis=2)
        max_indices = np.argmax(predicted, axis=2)

        gts = np.array([gt_combined[i, idx] for i, idx in enumerate(max_indices)])

        roc_all, auc_all = roc(max_values, gts)

        fpr, tpr = roc_all[0]

        # Get roc curves for every attribute present and save the auc scores in the dict

        # Name of the key 2
        key_name = ', '.join(attributes_present)

        # List of the auc scores
        auc_list = []
        for i, attribute in enumerate(attributes_present):
            att_idx = attribute2idx(attribute, list_attr)
            roc_attr = roc_curves_basic[att_idx]
            auc_attr = auc_scores_basic[att_idx]

            fpr_attr, tpr_attr = roc_attr

            axes[pd][i].plot(fpr_attr, tpr_attr, label=f'AUC = {auc_attr:.2f}', linewidth=3, color="red")
            axes[pd][i].set_xlabel('False Positive Rate')
            axes[pd][i].set_ylabel('True Positive Rate')
            axes[pd][i].set_title(f'ROC Curve for {attribute}')
            axes[pd][i].legend(loc="lower right")

            # Append the list with the auc scores for every attribute
            auc_list.append(round(auc_attr, 2))

        # Append the list with the auc score of the combined attribute
        auc_list.append(round(auc_all[0], 2))

        # Update the dictionary
        if corr_type in auc_scores_dict:
            auc_scores_dict[corr_type].update({key_name: auc_list})
        else:
            auc_scores_dict[corr_type] = {key_name: auc_list}
        
        axes[pd][i+1].plot(fpr, tpr, label=f'AUC = {auc_all[0]:.2f}', linewidth=3, color="red")

        axes[pd][i+1].set_xlabel('False positive rate')
        axes[pd][i+1].set_ylabel('True positive rate')
        axes[pd][i+1].set_title(f'ROC Curve for {attributes_present}')
        axes[pd][i+1].legend(loc='lower right')

        # Setting global title
        fig.suptitle(predicted_path.split("/")[-2].upper().replace('-', ' '), fontsize=20)

    # Show the plot
    plt.tight_layout()
    plt.show()


#############################################################################################
############## HEATMAP
# AUC scores for each attribute and combined attributes
for k in auc_scores_dict.keys():
    auc_scores = auc_scores_dict[k]

    # Extract attribute names and AUC scores
    keys = list(auc_scores.keys())
    attributes_list = [item.split(', ') + ['Combined'] for item in keys]

    values_list = list(auc_scores.values())
    padded_lists = [lst + [0] * (4 - len(lst)) if len(lst) == 3 else lst for lst in values_list]
    auc_values = np.array(padded_lists)

    # Define positions for each group
    positions = np.arange(len(keys))

    # Define width of bars
    bar_width = 0.15

    # Define colors for bars
    colors = ['skyblue', 'salmon', 'lightgreen', 'lightcoral', 'mediumorchid']

    # Plotting the grouped bar plot
    plt.figure(figsize=(12, 6))

    for i, attrs in enumerate(attributes_list):
        for j in range(len(attrs)):
            bar = plt.bar(positions + j * bar_width, auc_values[:, j], bar_width, label=attrs[j], color=colors[j])

            # Add labels to each bar
            plt.text(positions[i] + j * bar_width, -0.05, attrs[j], rotation=90, ha='center', va='top')

            # Add labels to each bar
            values = auc_values[:, j]
            for idx, value in enumerate(values):
                if value != 0:
                    plt.text(positions[idx] + j * bar_width, value, f'{value:.2f}', rotation=0, ha='center', va='bottom')


    # Add labels and title
    plt.xlabel('Attribute Combinations', labelpad=100)
    plt.ylabel('AUC Score')
    plt.title(k.replace('-', ' ').upper())
    plt.xticks(positions + bar_width * (len(attributes_list) - 1) / 2, [''] * len(keys))  # Hide x-axis ticks
    plt.tight_layout()
    plt.show()










