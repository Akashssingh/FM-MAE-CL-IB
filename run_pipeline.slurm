#!/bin/bash
#SBATCH --job-name=FM-MAE-CL-IB
#SBATCH --partition=nopreempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail

WORKDIR="/home/a/akashsingh/FM-MAE-CL-IB"
CONDA_ENV="/home/a/akashsingh/.conda/envs/emogi-env"
PYTHON="${CONDA_ENV}/bin/python"

cd "${WORKDIR}"
mkdir -p logs results data/processed

echo "=============================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU(s)      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unavailable')"
echo "Start time  : $(date)"
echo "Working dir : ${WORKDIR}"
echo "Python      : ${PYTHON}"
echo "=============================================="

# ── STEP 1: Data preprocessing + 5-year OS labels ─────────────────────────
echo ""
echo "[STEP 1] Preprocessing CNV data with OS survival labels"
echo "----------------------------------------------"
${PYTHON} prepare_cnv_data.py \
    --cnv_path  data/TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes \
    --output_path data/processed/raw_features_cnv.csv \
    --survival_path survival_BRCA_survival.txt \
    --n_features 500

echo ""
echo "[STEP 1] Done — $(date)"

# ── STEP 2: Log-cosh VAE feature extraction ────────────────────────────────
echo ""
echo "[STEP 2] Training VAE and extracting latent features (GPU)"
echo "----------------------------------------------"
${PYTHON} cnv_vae_extractor.py \
    --input_path  data/processed/raw_features_cnv.csv \
    --output_path data/processed/vae_features_cnv.csv \
    --epochs 50 \
    --batch_size 32 \
    --latent_dim 32 \
    --device cuda

echo ""
echo "[STEP 2] Done — $(date)"

# ── STEP 3: SVM / RF classification (10-fold CV) ───────────────────────────
echo ""
echo "[STEP 3] Training classifiers with 10-fold stratified CV"
echo "----------------------------------------------"
${PYTHON} train_classifier.py \
    --modality_paths data/processed/vae_features_cnv.csv \
    --modality_names cnv \
    --results_file  results/classification_results.csv

echo ""
echo "[STEP 3] Done — $(date)"

echo ""
echo "=============================================="
echo "Pipeline complete — $(date)"
echo "Results : ${WORKDIR}/results/"
echo "=============================================="
