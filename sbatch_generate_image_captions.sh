#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --partition=compute
#SBATCH --job-name=clip_generate_image_captions
#SBATCH --output=CLIP_outputs/logs/clip_generate_image_captions_%j.log
#SBATCH --export=WANDB_API_KEY
#SBATCH --export=HTTPS_PROXY
#SBATCH --export=https_proxy
#SBATCH --export=ALL
#SBATCH --chdir=/ceph/grid/home/am6417/Thesis/CLIP # Set the working directory

source /ceph/grid/home/am6417/miniconda3/etc/profile.d/conda.sh
conda activate env_clip
export NCCL_P2P_DISABLE=1

srun /ceph/grid/home/am6417/miniconda3/envs/env_clip/bin/python /ceph/grid/home/am6417/Thesis/CLIP/sbatch_generate_image_captions.sh
# accelerate launch --config_file /ceph/grid/home/am6417/Thesis/CLIP/accelerate_config_captions_gen.yaml generate_image_caption_clip.py

