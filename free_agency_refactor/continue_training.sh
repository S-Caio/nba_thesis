#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%j_%x.log       # %x = Job Name, %j = Job ID
#SBATCH --error=logs/%j_%x.err        # %x = Job Name, %j = Job ID
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

# Use argument 1 if provided, otherwise default to SLURM Job Name (%x)
EXP_NAME="${1:-$SLURM_JOB_NAME}"

# --- CONFIGURATION ---
TRAIN_SCRIPT="continue_training.py" 
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

python -u "check_cuda.py"

# Pass experiment name to Python for MLflow / WandB / TensorBoard tagging
python -u "$TRAIN_SCRIPT" --exp-name "$EXP_NAME"