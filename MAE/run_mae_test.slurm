#!/bin/bash
#SBATCH --job-name=MAE-test
#SBATCH --partition=nopreempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1080Ti:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mae_test_%j.out
#SBATCH --error=logs/mae_test_%j.err

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail

WORKDIR="/home/a/akashsingh/FM-MAE-CL-IB"
CONDA_ENV="/home/a/akashsingh/.conda/envs/emogi-env"
PYTHON="${CONDA_ENV}/bin/python"

cd "${WORKDIR}"
mkdir -p logs MAE/outputs

echo "=============================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo "Start time  : $(date)"
echo "Working dir : ${WORKDIR}"
echo "Python      : ${PYTHON}"
echo "=============================================="

# ── MAE test run (15% masking, 10 epochs) ────────────────────────────────
# A 10-epoch run is enough to confirm:
#   - GPU is utilised
#   - loss decreases (model is learning)
#   - representations are extracted and saved
# Output goes to MAE/outputs/mask15/
echo ""
echo "[MAE] Pretraining — mask_ratio=0.15, epochs=10"
echo "----------------------------------------------"
${PYTHON} MAE/train_mae.py \
    --input_path  data/processed/raw_features_cnv.csv \
    --output_dir  MAE/outputs \
    --mask_ratio  0.15 \
    --epochs      10   \
    --batch_size  64   \
    --lr          3e-4 \
    --d_model     128  \
    --n_heads     4    \
    --n_layers    4    \
    --device      cuda

echo ""
echo "[MAE] Linear probe evaluation"
echo "----------------------------------------------"
${PYTHON} MAE/evaluate_mae.py \
    --repr_path   MAE/outputs/mask15/mae_representations.csv \
    --output_dir  MAE/outputs

echo ""
echo "=============================================="
echo "End time : $(date)"
echo "All done."
echo "=============================================="
