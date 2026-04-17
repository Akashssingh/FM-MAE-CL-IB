#!/bin/bash
#SBATCH --job-name=clf-ref-vae
#SBATCH --partition=nopreempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1080Ti:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/clf_ref_%j.out
#SBATCH --error=logs/clf_ref_%j.err

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail

WORKDIR="/home/a/akashsingh/FM-MAE-CL-IB"
CONDA_ENV="/home/a/akashsingh/.conda/envs/emogi-env"
PYTHON="${CONDA_ENV}/bin/python"

cd "${WORKDIR}"
mkdir -p logs results/models

echo "=============================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo "Start time  : $(date)"
echo "Working dir : ${WORKDIR}"
echo "Python      : ${PYTHON}"
echo "=============================================="

echo ""
echo "[CLASSIFIER] Reference VAE features — reference_files/vae_features_cnv.csv"
echo "  1079 patients | 32 VAE dims | label_cnv (0/1)"
echo "  Classifiers: RBF SVM, Linear SVM, Poly SVM, Sigmoid SVM, Random Forest"
echo "  Protocol: 10-fold stratified CV + per-fold minority upsampling"
echo "----------------------------------------------"

${PYTHON} train_classifier.py \
    --modality_paths reference_files/vae_features_cnv.csv \
    --modality_names cnv \
    --results_file   results/classification_results_ref_vae.csv

echo ""
echo "=============================================="
echo "End time : $(date)"
echo "All done."
echo "=============================================="
