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

# Get the annotations
annotations_path = '/mnt/hdd/volume1/anastasija/CelebA/Anno'
attr = get_annotation(os.path.join(annotations_path, 'list_attr_celeba.txt'), verbose=False)
val_attr = attr.iloc[162770:182637]

# List of attributes
list_attr = list(val_attr.columns[1:])

# Ground truths
gt_all = get_gt(annotations_path)

########################## ROC CURVES AND AUC SCORES
# BASIC captions and attributes
predictions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/results"

# Create a list to store paths of all .npy files
all_predictions_paths = []

# Use glob to find .npy files recursively
for file_path in glob.iglob(os.path.join(predictions_path, '**/*.npy'), recursive=True):
    all_predictions_paths.append(file_path)

all_predictions_paths = sorted(all_predictions_paths[8:])

for predicted_path in all_predictions_paths:
    print(fr'##########     {predicted_path.split("/")[-2].upper()}     ##########')
    predicted = np.load(predicted_path)

    binary_codes_3 = {0: [1, 1, 1],
                    1: [1, 1, 0],
                    2: [1, 0, 1],
                    3: [1, 0, 0],
                    4: [0, 1, 1],
                    5: [0, 1, 0],
                    6: [0, 0, 1],
                    7: [0, 0, 0]}

    binary_codes_2 = {0: [1, 1],
                    1: [1, 0],
                    2: [0, 1],
                    3: [0, 0]}

    binary_indices = {
        3: {0: [0, 1, 2, 3],
            1: [0, 1, 4, 5],
            2: [0, 2, 4, 6]},

        2: {0: [0, 1],
            1: [0, 2]}
    }


    attributes_present, indices = parse_attributes(os.path.basename(predicted_path), list_attr)
    gt_present = gt_all[:, indices]

    # Plot ROC curves for the first set of attributes
    plt.figure(figsize=(15, 5))
    for j in range(gt_present.shape[1]):
        gt = np.tile(gt_present[:, j][:, np.newaxis], 4)

        roc_attr, auc_attr = roc(predicted[:, :, binary_indices[gt_present.shape[1]][j]].squeeze(), gt)

        if gt_present.shape[1] == 3:
            binary_codes = binary_codes_3
        elif gt_present.shape[1] == 2:
            binary_codes = binary_codes_2

        plt.subplot(1, gt_present.shape[1], j+1)
        valid_indices = [i for i, score in enumerate(auc_attr) if not np.isnan(score)]
        for i in valid_indices:
            fpr, tpr = roc_attr[i]
            plt.plot(fpr, tpr, label=f'{binary_codes[binary_indices[gt_present.shape[1]][j][i]]}  (AUC = {auc_attr[i]:.2f})', linewidth=3)

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC Curves {attributes_present[j]}')
        plt.legend(loc='lower right')

    plt.suptitle(f'ROC Curves {attributes_present}')
    plt.tight_layout()
    plt.show()
