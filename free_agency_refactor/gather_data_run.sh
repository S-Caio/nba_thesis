#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --output=logs/gather_data/%x_%j.log       # %x = Job Name, %j = Job ID
#SBATCH --error=logs/gather_data/%x_%j.err        # %x = Job Name, %j = Job ID
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

LOG_DIR="./logs/gather_data"
# ---------------------

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

python -u "eval_scripts/gather_data.py"