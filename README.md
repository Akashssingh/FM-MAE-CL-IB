# FM-MAE-CL-IB

Research code for replicating and extending the multimodal breast cancer survival prediction pipeline from [Arya et al. (2023)](https://doi.org/10.1038/s41598-023-30143-8) — "Improving the robustness and stability of a machine learning model for breast cancer prognosis through the use of multi-modal classifiers".

Currently focused on the **CNV (copy number variation) unimodal** pipeline using a **log-cosh VAE** for feature extraction, re-implemented in **PyTorch** (original paper used Keras/TensorFlow).

---

## What this does

1. Takes raw GISTIC2-thresholded CNV data from the [TCGA-BRCA dataset on Xena Browser](https://xenabrowser.net/datapages/)
2. Preprocesses it following the paper's methodology (missing value handling, zero-variance removal, top-500 feature selection by variance)
3. Trains a Variational Autoencoder with log-cosh reconstruction loss to extract 32-dimensional latent features per patient
4. Saves the latent features as a CSV ready for downstream classification

---

## Data

Download the GISTIC2 thresholded copy number data for TCGA-BRCA from Xena Browser:

- Go to https://xenabrowser.net/datapages/
- Dataset: **TCGA Breast Cancer (BRCA)**
- File: `TCGA.BRCA.sampleMap/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes`

Place the downloaded `.tsv` file in the `data/` directory. The default expected path is:

```
data/TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.tsv
```

If you have a clinical data file (also from Xena), you can optionally pass it to get real 5-year survival labels. Otherwise a placeholder label of `-1` is used.

---

## Running locally

### Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins a CPU-only PyTorch build. No GPU required.

### Step 1 — Prepare the data

```bash
python prepare_cnv_data.py
```

Optional flags:

```
--cnv_path       Path to raw CNV TSV (default: data/TCGA.BRCA.sampleMap_...)
--output_path    Where to save the processed CSV (default: data/processed/raw_features_cnv.csv)
--clinical_path  Path to Xena clinical TSV to generate 5-year survival labels
--n_features     Number of top-variance genes to keep (default: 500)
--test_mode      Use only 100 patients — good for a quick sanity check
```

### Step 2 — Extract VAE features

```bash
python cnv_vae_extractor.py
```

Optional flags:

```
--input_path    Processed CSV from step 1 (default: data/processed/raw_features_cnv.csv)
--output_path   Where to save latent features (default: data/processed/vae_features_cnv.csv)
--epochs        Training epochs (default: 50)
--batch_size    Batch size (default: 32)
--lr            Initial learning rate (default: 1e-3)
--latent_dim    Latent space size (default: 32, matches the paper)
--test_mode     100 patients, 5 epochs — quick CPU functionality check
```

### Quick smoke-test (CPU, ~20 seconds)

```bash
python prepare_cnv_data.py --test_mode
python cnv_vae_extractor.py --test_mode
```

---

## Running with Docker

If you'd rather not deal with Python environment setup, Docker handles everything.

### Build

```bash
docker build -t cnv-vae .
```

### Run the full pipeline

```bash
docker compose up
```

This mounts your local `data/` directory into the container, runs both steps, and writes outputs back to `data/processed/`.

### Individual steps via Docker

```bash
# Data prep only
docker compose run --rm vae-pipeline python prepare_cnv_data.py

# VAE extraction only
docker compose run --rm vae-pipeline python cnv_vae_extractor.py

# Smoke-test
docker compose run --rm vae-pipeline \
  sh -c "python prepare_cnv_data.py --test_mode && python cnv_vae_extractor.py --test_mode"
```

---

## Output

After running both steps, `data/processed/` will contain:

| File | Description |
|---|---|
| `raw_features_cnv.csv` | Preprocessed CNV features (patients × 500 genes + label) |
| `vae_features_cnv.csv` | 32-dimensional VAE latent features per patient |
| `vae_features_cnv_model.pt` | Saved PyTorch model checkpoint |

The `vae_features_cnv.csv` format matches the reference paper exactly:
```
submitter_id.samples, cnv_vae_1, ..., cnv_vae_32, label_cnv
```

---

## Project structure

```
├── prepare_cnv_data.py       # Step 1: data preprocessing
├── cnv_vae_extractor.py      # Step 2: log-cosh VAE training + feature extraction
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── TCGA.BRCA.sampleMap_...tsv   # raw input (download from Xena)
│   └── processed/                   # outputs written here
└── reference_files/                 # original Keras code from the paper
```

---

## Reference

Arya, N., Saha, S., Mathur, A., & Saha, S. (2023). Improving the robustness and stability of a machine learning model for breast cancer prognosis through the use of multi-modal classifiers. *Scientific Reports*, 13, 4079. https://doi.org/10.1038/s41598-023-30143-8

Original code: https://github.com/nikhilaryan92/logcoshVAE_brca_surv
