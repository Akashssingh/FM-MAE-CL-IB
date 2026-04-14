# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — CNV Log-Cosh VAE Feature Extractor
#
# Replicates the CNV unimodal preprocessing + VAE training pipeline from
# Arya et al. (2023), re-implemented in PyTorch (CPU-only).
#
# Before running, place the TCGA-CDR survival file in the project root:
#   survival_BRCA_survival.txt
#
# Build:
#   docker build -t cnv-vae .
#
# Run full pipeline (all 3 steps):
#   docker compose up
#
# Or run individual steps:
#   docker compose run --rm vae-pipeline \
#     python prepare_cnv_data.py --survival_path survival_BRCA_survival.txt
#
#   docker compose run --rm vae-pipeline python cnv_vae_extractor.py
#
#   docker compose run --rm vae-pipeline python train_classifier.py
#
# Quick CPU smoke-test (100 patients, 5 epochs, no survival file needed):
#   docker compose run --rm vae-pipeline \
#     sh -c "python prepare_cnv_data.py --test_mode && \
#            python cnv_vae_extractor.py --test_mode"
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install CPU-only PyTorch first (much smaller than CUDA build: ~200 MB vs 3 GB)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
# Exclude torch from requirements.txt since we installed it above with the
# CPU-specific index URL
RUN grep -v '^torch' requirements.txt | \
    pip install --no-cache-dir -r /dev/stdin

# ── Application code ──────────────────────────────────────────────────────────
COPY prepare_cnv_data.py .
COPY cnv_vae_extractor.py .
COPY train_classifier.py .

# ── Data and output directories ───────────────────────────────────────────────
# Raw data is expected at /app/data/ (mounted at runtime).
# Processed outputs go to /app/data/processed/.
# Classification results go to /app/results/ (mounted at runtime).
RUN mkdir -p data/processed results/models

# ── Default command: run full pipeline ───────────────────────────────────────
CMD ["sh", "-c", \
     "python prepare_cnv_data.py --survival_path survival_BRCA_survival.txt && \
      python cnv_vae_extractor.py && \
      python train_classifier.py"]
