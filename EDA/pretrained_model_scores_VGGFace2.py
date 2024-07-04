import os
import clip
import numpy as np
import argparse

from EDA.helper_functions import generate_zero_shot_scores_VGGFace2, generate_fine_tuned_scores_VGGFace2

if __name__ == "__main__":
    # generate_zero_shot_scores(captions_path=args.captions_path, images_path=args.images_path, eval_partition_path=args.eval_path, save_filename=args.save_filename)
    # generate_zero_shot_scores2(captions_path=args.captions_path, images_path=args.images_path, eval_partition_path=args.eval_path, save_filename=args.save_filename)
    # generate_zero_shot_scores_combined_attr(captions_path=args.captions_path, images_path=args.images_path, eval_partition_path=args.eval_path, save_filename=args.save_filename)

    # save_filename = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/pretrained_clip_basic_VGGFace2.npy"
    # save_filename2 = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/fine_tuned_clip_basic_VGGFace2_con_len_77.npy"


    
    captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/CelebA/captions_single_attribute_VGGFace2.txt"
    image_captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_att_28052024.txt"
    model_checkpoint = "CLIP_outputs/CLIP-fine-tuning-VGGFace2/scarlet-fire-11/epoch_75/model.safetensors"
    save_filename = "/home/anastasija/Documents/Projects/SBS/CLIP/EDA/VGGFace2/results/fine_tuned_clip_basic_VGGFace2_con_len_305.npy"

    # generate_zero_shot_scores_VGGFace2(captions_path=captions_path, save_filename=save_filename)
    generate_fine_tuned_scores_VGGFace2(captions_path=captions_path, image_captions_path=image_captions_path, model_checkpoint=model_checkpoint, save_filename=save_filename)
