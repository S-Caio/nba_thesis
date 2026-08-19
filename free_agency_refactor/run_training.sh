#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/gather_data/%x_%j.log       # %x = Job Name, %j = Job ID
#SBATCH --error=logs/gather_data/%x_%j.err        # %x = Job Name, %j = Job ID
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

# Use argument 1 if provided, otherwise default to SLURM Job Name (%x)
EXP_NAME="${1:-$SLURM_JOB_NAME}"

# --- CONFIGURATION ---
TRAIN_SCRIPT="train_parallel.py" 
LOG_DIR="./logs"
# ---------------------

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo " Starting RLlib Training Pipeline via SLURM"
echo " Experiment Name: $EXP_NAME"
echo " Job Name:        $SLURM_JOB_NAME"
echo " Job ID:          $SLURM_JOB_ID"
echo " Node Name:       $SLURM_JOB_NODELIST"
echo " Script:          $TRAIN_SCRIPT"
echo " Time:            $(date)"
echo "=========================================================="
echo "-> Log output: ${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "=========================================================="

echo "---- nvidia-smi ----"
nvidia-smi
echo "---- CUDA_VISIBLE_DEVICES ----"
echo "$CUDA_VISIBLE_DEVICES"

# python -u "check_cuda.py"

python -u "$TRAIN_SCRIPT" --exp-name "$EXP_NAME"

# python - <<'PY'
# import os
# import torch

# print("PID:", os.getpid())
# print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
# print("torch:", torch.__version__)
# print("torch CUDA:", torch.version.cuda)
# print("device count:", torch.cuda.device_count())
# print("is_available:", torch.cuda.is_available())

# try:
#     print("current device:", torch.cuda.current_device())
# except Exception as e:
#     print("current_device ERROR:", repr(e))

# try:
#     x = torch.tensor([1.0], device="cuda")
#     print("CUDA tensor:", x)
# except Exception as e:
#     print("CUDA tensor ERROR:", repr(e))
# PY

# python -m torch.utils.collect_env

# dmesg | grep -i -E 'NVRM|Xid|nvidia'