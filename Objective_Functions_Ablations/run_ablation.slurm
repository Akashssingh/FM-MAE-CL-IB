#!/bin/bash
#SBATCH --job-name=ablation-vae-ml
#SBATCH --partition=nopreempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1080Ti:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=Objective_Functions_Ablations/logs/ablation_%j.out
#SBATCH --error=Objective_Functions_Ablations/logs/ablation_%j.err

# ── Usage ──────────────────────────────────────────────────────────────────
# Single run with defaults (logcosh + KL):
#   sbatch Objective_Functions_Ablations/run_ablation.slurm
#
# Override via environment variables:
#   RUN_NAME=mse_mmd LOSS_FN=mse REGULARIZER=mmd MMD_WEIGHT=10.0 \
#     sbatch Objective_Functions_Ablations/run_ablation.slurm
#
# Submit all 4 ablation combinations at once:
#   bash Objective_Functions_Ablations/submit_ablations.sh

set -euo pipefail

WORKDIR="/home/a/akashsingh/FM-MAE-CL-IB"
CONDA_ENV="/home/a/akashsingh/.conda/envs/emogi-env"
PYTHON="${CONDA_ENV}/bin/python"

cd "${WORKDIR}"
mkdir -p Objective_Functions_Ablations/logs \
         Objective_Functions_Ablations/results \
         Objective_Functions_Ablations/outputs \
         Objective_Functions_Ablations/splits \
         Objective_Functions_Ablations/figures/roc \
         Objective_Functions_Ablations/figures/prc \
         Objective_Functions_Ablations/figures/confusion_matrix

# ── Ablation config (override with env vars at submission time) ────────────
RUN_NAME="${RUN_NAME:-logcosh_kl}"
LOSS_FN="${LOSS_FN:-logcosh}"
REGULARIZER="${REGULARIZER:-kl}"
KL_WEIGHT="${KL_WEIGHT:-0.001}"
MMD_WEIGHT="${MMD_WEIGHT:-10.0}"
EPOCHS="${EPOCHS:-50}"

echo "=============================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi --query-gpu=name,memory.total \
                      --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo "Start time  : $(date)"
echo "Working dir : ${WORKDIR}"
echo "=============================================="
echo ""
echo "[ABLATION CONFIG]"
echo "  run_name   : ${RUN_NAME}"
echo "  loss_fn    : ${LOSS_FN}"
echo "  regularizer: ${REGULARIZER}"
echo "  kl_weight  : ${KL_WEIGHT}"
echo "  mmd_weight : ${MMD_WEIGHT}"
echo "  epochs     : ${EPOCHS}"
echo "----------------------------------------------"

# ── Step 1: Generate splits (idempotent) ───────────────────────────────────
echo ""
echo "[1/2] Generating / verifying persistent CV splits ..."
${PYTHON} Objective_Functions_Ablations/generate_splits.py

# ── Step 2: Train VAE + run classifiers ────────────────────────────────────
echo ""
echo "[2/2] Running ablation ..."
${PYTHON} Objective_Functions_Ablations/run_ablation.py \
    --run_name   "${RUN_NAME}" \
    --loss_fn    "${LOSS_FN}" \
    --regularizer "${REGULARIZER}" \
    --kl_weight  "${KL_WEIGHT}" \
    --mmd_weight "${MMD_WEIGHT}" \
    --epochs     "${EPOCHS}" \
    --device     cuda

echo ""
echo "=============================================="
echo "End time : $(date)"
echo "Results  : ${WORKDIR}/Objective_Functions_Ablations/results/"
echo "Outputs  : ${WORKDIR}/Objective_Functions_Ablations/outputs/"
echo "All done."
echo "=============================================="
