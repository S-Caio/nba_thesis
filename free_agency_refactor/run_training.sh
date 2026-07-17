#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.log       # Captures standard output (stdout)
#SBATCH --error=logs/train_%j.err        # Captures ALL errors and warnings (stderr)
#SBATCH --gres=gpu:1                     # Request 1 GPU for your training
#SBATCH --cpus-per-task=4                # CPU cores for environment rollouts

# --- CONFIGURATION ---
TRAIN_SCRIPT="train.py" 
LOG_DIR="./logs"
# ---------------------

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo " Starting RLlib Training Pipeline via SLURM"
echo " Job ID:        $SLURM_JOB_ID"
echo " Node Name:     $SLURM_JOB_NODELIST"
echo " Script:        $TRAIN_SCRIPT"
echo " Time:          $(date)"
echo "=========================================================="
echo "-> Standard output is routing to: ${LOG_DIR}/train_${SLURM_JOB_ID}.log"
echo "-> Errors/Warnings are routing to: ${LOG_DIR}/train_${SLURM_JOB_ID}.err"
echo "=========================================================="

# Execute training script unbuffered
# SLURM handles the file writes entirely in the background
echo "---- nvidia-smi ----"
nvidia-smi
echo "---- CUDA_VISIBLE_DEVICES ----"
echo "$CUDA_VISIBLE_DEVICES"

python -u "check_cuda.py"
python -u "$TRAIN_SCRIPT"