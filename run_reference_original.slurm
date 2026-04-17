#!/bin/bash
#SBATCH --job-name=ref-original-ml
#SBATCH --partition=nopreempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1080Ti:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ref_original_%j.out
#SBATCH --error=logs/ref_original_%j.err

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail

WORKDIR="/home/a/akashsingh/FM-MAE-CL-IB"
CONDA_ENV="/home/a/akashsingh/.conda/envs/emogi-env"
PYTHON="${CONDA_ENV}/bin/python"

cd "${WORKDIR}"
mkdir -p logs results/ref_models

echo "=============================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo "Start time  : $(date)"
echo "Working dir : ${WORKDIR}"
echo "Python      : ${PYTHON}"
echo "=============================================="

echo ""
echo "[REFERENCE ORIGINAL] Replicating Arya et al. (2023) ML pipeline"
echo "  Input    : reference_files/vae_features_cnv.csv  (original authors' features)"
echo "  Modality : cnv (VAE, latent_dim=32)"
echo "  Protocol : 10-fold stratified CV + per-fold minority upsampling"
echo "  Classifiers: RBF SVM, Linear SVM, Poly SVM, Sigmoid SVM, Random Forest"
echo "  Metrics  : ROC-AUC, AUPRC, Accuracy, Precision, Sensitivity, F1, Confusion Matrix"
echo "----------------------------------------------"

${PYTHON} reference_original_model_training.py \
    --input_path   reference_files/vae_features_cnv.csv \
    --modality_name cnv \
    --results_file  results/reference_original_results.csv

echo ""
echo "=============================================="
echo "End time : $(date)"
echo "Results  : ${WORKDIR}/results/reference_original_results.csv"
echo "All done."
echo "=============================================="
