import os
import clip
import numpy as np
import argparse

from EDA.helper_functions import generate_zero_shot_scores, generate_zero_shot_scores2, generate_zero_shot_scores_combined_attr

parser = argparse.ArgumentParser(description='Command line arguments')

parser.add_argument('--captions_path')
parser.add_argument('--images_path')
parser.add_argument('--eval_path')
parser.add_argument('--save_filename')

if __name__ == "__main__":
    args = parser.parse_args()
    # generate_zero_shot_scores(captions_path=args.captions_path, images_path=args.images_path, eval_partition_path=args.eval_path, save_filename=args.save_filename)
    # generate_zero_shot_scores2(captions_path=args.captions_path, images_path=args.images_path, eval_partition_path=args.eval_path, save_filename=args.save_filename)
    generate_zero_shot_scores_combined_attr(captions_path=args.captions_path, images_path=args.images_path, eval_partition_path=args.eval_path, save_filename=args.save_filename)

