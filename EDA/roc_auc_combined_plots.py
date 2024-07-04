import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import pandas as pd
from EDA.helper_functions import get_annotation, get_gt, roc
from data.vgg2_dataset import VGGFace2Dataset
from clip.clip import load

device = "cuda:1"

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
    fig, ax = plt.subplots()
    valid_indices = [i for i, score in enumerate(auc_scores) if not np.isnan(score)]
    for i in valid_indices:
        fpr, tpr = roc_curves[i]
        ax.plot(fpr, tpr, label=f'{list_attr[i]} (AUC = {auc_scores[i]:.2f})', linewidth=1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    ax.legend(loc='best', fontsize='small')
    return fig

def plot_auc_scores(auc_scores, list_attr, title):
    fig, ax = plt.subplots()
    valid_indices = [i for i, score in enumerate(auc_scores) if not np.isnan(score)]
    ax.barh(valid_indices, auc_scores)
    ax.set_yticks(valid_indices)
    ax.set_yticklabels(list_attr, color='black', rotation=0, fontweight='bold', fontsize='10')
    ax.set_xlabel('AUC score')
    ax.set_ylabel('Attribute')
    ax.set_title(title)
    return fig

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
    fig, ax = plt.subplots()
    ax.bar(intervals, counts, color='skyblue')
    ax.set_title(title)
    ax.set_xlabel('AUC Score Intervals')
    ax.set_ylabel('Count of AUC Scores')
    ax.set_xticks(rotation=45)
    ax.grid(axis='y', linestyle='--')
    return fig

def process_predictions(predicted_path, gt_all, list_attr):
    predicted = np.load(predicted_path)
    roc_curves, auc_scores = roc(predicted, gt_all)
    return roc_curves, auc_scores

def plot_combined_results(vgg2_path, captions_path, annotations_path, predicted_paths, titles):
    val_set = load_vggface2_dataset(vgg2_path, captions_path, device)
    image_names = get_image_names(val_set)
    gt_all = get_attribute_values(annotations_path, image_names)
    df = pd.read_csv(annotations_path)
    list_attr = df.columns[2:]

    roc_curves_list = []
    auc_scores_list = []
    
    for predicted_path in predicted_paths:
        roc_curves, auc_scores = process_predictions(predicted_path, gt_all, list_attr)
        roc_curves_list.append(roc_curves)
        auc_scores_list.append(auc_scores)

    roc_figs = []
    auc_figs = []
    dist_figs = []

    for roc_curves, auc_scores, title in zip(roc_curves_list, auc_scores_list, titles):
        roc_figs.append(plot_roc_curves(roc_curves, auc_scores, list_attr, title[0]))
        auc_figs.append(plot_auc_scores(auc_scores, list_attr, title[1]))
        dist_figs.append(plot_auc_distribution(auc_scores, title[2]))

    # Display combined ROC plots
    plt.figure(figsize=(20, 10))
    for idx, fig in enumerate(roc_figs):
        plt.subplot(121 + idx)
        fig.canvas.draw()
        plt.imshow(np.array(fig.canvas.renderer.buffer_rgba()))
    plt.tight_layout()
    plt.show()

    # Display combined AUC score plots
    plt.figure(figsize=(20, 10))
    for idx, fig in enumerate(auc_figs):
        plt.subplot(121 + idx)
        fig.canvas.draw()
        plt.imshow(np.array(fig.canvas.renderer.buffer_rgba()))
    plt.tight_layout()
    plt.show()

    # Display combined AUC distribution plots
    plt.figure(figsize=(20, 10))
    for idx, fig in enumerate(dist_figs):
        plt.subplot(121 + idx)
        fig.canvas.draw()
        plt.imshow(np.array(fig.canvas.renderer.buffer_rgba()))
    plt.tight_layout()
    plt.show()

def main():
    vgg2_path = "/mnt/hdd/volume1/VGGFace2"
    captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_att_07052024.txt"
    annotations_path = "/mnt/hdd/volume1/MAAD-Face/MAAD_Face.csv"
    predicted_paths = [
        "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/pretrained_clip_basic_VGGFace2.npy",
        "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/fine_tuned_clip_basic_VGGFace2_con_len_77.npy"
    ]
    titles = [
        ('ROC Curves Pretrained', 'AUC scores Basic Pretrained', 'Distribution of AUC Scores Pretrained'),
        ('ROC Curves Fine-Tuned', 'AUC scores Basic Fine-Tuned', 'Distribution of AUC Scores Fine-Tuned')
    ]

    plot_combined_results(vgg2_path, captions_path, annotations_path, predicted_paths, titles)

# if __name__ == "__main__":
main()
