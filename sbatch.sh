#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --partition=compute
#SBATCH --job-name=clip_finetuning
#SBATCH --output=CLIP_outputs/logs/clip_finetuning_%j.log
#SBATCH --export=WANDB_API_KEY
#SBATCH --export=HTTPS_PROXY
#SBATCH --export=https_proxy

source /ceph/grid/home/am6417/miniconda3/etc/profile.d/conda.sh
conda activate env_clip
export NCCL_P2P_DISABLE=1
accelerate launch --config_file /ceph/grid/home/am6417/Thesis/CLIP/accelerate_config.yaml test_fine-tuning.py

