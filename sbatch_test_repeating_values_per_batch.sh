#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --partition=compute
#SBATCH --job-name=test_repeating_values_per_batch
#SBATCH --output=tests/logs/test_repeating_values_per_batch%j.log

source /ceph/grid/home/am6417/miniconda3/etc/profile.d/conda.sh
conda activate env_clip

python -m tests.test_repeating_tokens_per_batch.py
