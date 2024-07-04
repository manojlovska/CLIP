import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
# from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import pandas as pd
from EDA.helper_functions import roc
from data.vgg2_dataset import VGGFace2Dataset
from clip.clip import load

device = "cuda:0"

def load_vggface2_dataset(vgg2_path, captions_path, device):
    _, preprocess = load("ViT-B/32", device=device, jit=False)
    return VGGFace2Dataset(vgg2_path=vgg2_path, captions_path=captions_path, preprocess=preprocess, split="test")

def get_image_names(val_set):
    return [val_set[i][0] for i in tqdm(range(len(val_set)))]

def get_attribute_values(annotation_path, filenames_list):
    df = pd.read_csv(annotation_path)
    filtered_df = df[df['Filename'].isin(filenames_list)]
    val_attr = filtered_df.iloc[:, 2:]
    gt_all = val_attr.to_numpy()
    return np.where(gt_all == -1, 0, gt_all)

def plot_roc_curves(roc_curves, auc_scores, list_attr, title):
    plt.figure(figsize=(10, 8))
    valid_indices = [i for i, score in enumerate(auc_scores) if not np.isnan(score)]
    for i in valid_indices:
        fpr, tpr = roc_curves[i]
        plt.plot(fpr, tpr, label=f'{list_attr[i]}  (AUC = {auc_scores[i]:.2f})', linewidth=3)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def plot_auc_scores(auc_scores, list_attr, title):
    plt.figure(figsize=(10, 8))
    valid_indices = [i for i, score in enumerate(auc_scores) if not np.isnan(score)]
    plt.barh(valid_indices, auc_scores)
    plt.yticks(valid_indices, list_attr, color='black', rotation=0, fontweight='bold', fontsize='10')
    plt.xlabel('AUC score')
    plt.ylabel('Attribute')
    plt.title(title)
    plt.show()

def plot_auc_distribution(auc_scores, title):
    auc_intervals = [(round(x * 0.1, 1), round((x + 1) * 0.1, 1)) for x in range(10)]
    categorized_scores = {interval: [] for interval in auc_intervals}
    for score in auc_scores:
        for interval in auc_intervals:
            if interval[0] <= score < interval[1]:
                categorized_scores[interval].append(score)
                break
    intervals = [f"{interval[0]}-{interval[1]}" for interval in categorized_scores.keys()]
    counts = [len(scores) for scores in categorized_scores.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(intervals, counts, color='skyblue')
    plt.title(title)
    plt.xlabel('AUC Score Intervals')
    plt.ylabel('Count of AUC Scores')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

def process_predictions(predicted_path, gt_all, list_attr, title_roc, title_auc, title_dist):
    predicted = np.load(predicted_path)
    roc_curves, auc_scores = roc(predicted, gt_all)
    plot_roc_curves(roc_curves, auc_scores, list_attr, title_roc)
    plot_auc_scores(auc_scores, list_attr, title_auc)
    plot_auc_distribution(auc_scores, title_dist)
    return auc_scores

def main(vgg2_path, captions_path, annotations_path, predicted_path_pretrained, predicted_path_fine_tuned, predicted_path_fine_tuned_305):
    val_set = load_vggface2_dataset(vgg2_path, captions_path, device)
    image_names = get_image_names(val_set)
    gt_all = get_attribute_values(annotations_path, image_names)
    df = pd.read_csv(annotations_path)
    list_attr = df.columns[2:]

    auc_scores_pretrained = process_predictions(
        predicted_path_pretrained, gt_all, list_attr,
        'ROC Curves Pretrained', 'AUC scores Basic Pretrained', 'Distribution of AUC Scores Across Intervals Basic Pretrained'
    )
    
    auc_scores_fine_tuned = process_predictions(
        predicted_path_fine_tuned, gt_all, list_attr,
        'ROC Curves Fine-Tuned (Context length: 77)', 'AUC scores Basic Fine-Tuned (Context length: 77)', 'Distribution of AUC Scores Across Intervals Basic Fine-Tuned (Context length: 77)'
    )

    auc_scores_fine_tuned_305 = process_predictions(
        predicted_path_fine_tuned_305, gt_all, list_attr,
        'ROC Curves Fine-Tuned (Context length: 305)', 'AUC scores Basic Fine-Tuned (Context length: 305)', 'Distribution of AUC Scores Across Intervals Basic Fine-Tuned (Context length: 305)'
    )

    return auc_scores_pretrained, auc_scores_fine_tuned, auc_scores_fine_tuned_305

# Example usage
vgg2_path = "/mnt/hdd/volume1/VGGFace2"
captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_att_07052024.txt" # Ni pomembno, itak jih ne uporabljas, potrebujes samo za konstruiranje val seta (image names)
annotations_path = "/mnt/hdd/volume1/MAAD-Face/MAAD_Face.csv"
predicted_path_pretrained = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/pretrained_clip_basic_VGGFace2.npy"
predicted_path_fine_tuned = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/fine_tuned_clip_basic_VGGFace2_con_len_77.npy"
predicted_path_fine_tuned_305 = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/fine_tuned_clip_basic_VGGFace2_con_len_305.npy"


auc_scores_pretrained, auc_scores_fine_tuned, auc_scores_fine_tuned_305 = main(
    vgg2_path, captions_path, annotations_path,
    predicted_path_pretrained, predicted_path_fine_tuned,
    predicted_path_fine_tuned_305
)