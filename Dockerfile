# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — CNV Log-Cosh VAE Feature Extractor
#
# Replicates the CNV unimodal preprocessing + VAE training pipeline from
# Arya et al. (2023), re-implemented in PyTorch (CPU-only).
#
# Build:
#   docker build -t cnv-vae .
#
# Run pipeline with your local data directory mounted:
#   docker run --rm \
#     -v "$(pwd)/data:/app/data" \
#     cnv-vae
#
# Or run individual steps:
#   docker run --rm -v "$(pwd)/data:/app/data" cnv-vae \
#     python prepare_cnv_data.py
#
#   docker run --rm -v "$(pwd)/data:/app/data" cnv-vae \
#     python cnv_vae_extractor.py
#
# Quick CPU smoke-test (100 patients, 5 epochs):
#   docker run --rm -v "$(pwd)/data:/app/data" cnv-vae \
#     python prepare_cnv_data.py --test_mode && \
#     python cnv_vae_extractor.py --test_mode
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

# ── Data directories ──────────────────────────────────────────────────────────
# The raw data file is expected at /app/data/ (mounted at runtime).
# Processed outputs will be written to /app/data/processed/.
RUN mkdir -p data/processed

# ── Default command: run full pipeline ───────────────────────────────────────
CMD ["sh", "-c", \
     "python prepare_cnv_data.py && python cnv_vae_extractor.py"]
