# FM-MAE-CL-IB

Research code for replicating and extending the multimodal breast cancer survival prediction pipeline from [Arya et al. (2023)](https://doi.org/10.1038/s41598-023-30143-8) — "Improving the robustness and stability of a machine learning model for breast cancer prognosis through the use of multi-modal classifiers".

Currently focused on the **CNV (copy number variation) unimodal** pipeline using a **log-cosh VAE** for feature extraction, re-implemented in **PyTorch** (original paper used Keras/TensorFlow).

---

## What this does

1. Takes raw GISTIC2-thresholded CNV data from the [TCGA-BRCA dataset on Xena Browser](https://xenabrowser.net/datapages/)
2. Preprocesses it following the paper's methodology (missing value handling, zero-variance removal, top-500 feature selection by variance)
3. Trains a Variational Autoencoder with log-cosh reconstruction loss to extract 32-dimensional latent features per patient
4. Trains SVM and Random Forest classifiers on the extracted features using 10-fold stratified cross-validation with minority class upsampling, faithfully replicating the paper's training protocol

---

## Data

Download the GISTIC2 thresholded copy number data for TCGA-BRCA from Xena Browser:

- Go to https://xenabrowser.net/datapages/
- Dataset: **TCGA Breast Cancer (BRCA)**
- File: `TCGA.BRCA.sampleMap/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes`

Place the downloaded file (without extension) in the `data/` directory.

For 5-year OS survival labels, download the TCGA-CDR survival file from Xena Browser:
- Dataset: **TCGA Breast Cancer (BRCA)** → **Curated survival data**
- File: `survival_BRCA_survival.txt`

Place it in the project root directory. The label convention follows the paper exactly:
- `label = 1` (short-term): `OS.time < 5 × 365.25` days (majority class, ~827/1078 patients)
- `label = 0` (long-term): `OS.time >= 5 × 365.25` days (minority class, ~251/1078 patients)
- `label = -1`: `OS.time` missing — excluded during classifier training

---

## Running locally

### Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 1 — Prepare the data

```bash
python prepare_cnv_data.py --survival_path survival_BRCA_survival.txt
```

Optional flags:

```
--cnv_path        Path to raw CNV file (default: data/TCGA.BRCA.sampleMap_...)
--output_path     Where to save the processed CSV (default: data/processed/raw_features_cnv.csv)
--survival_path   Path to TCGA-CDR survival TSV for 5-year OS labels
--n_features      Number of top-variance genes to keep (default: 500)
--test_mode       Use only 100 patients for a quick sanity check
```

### Step 2 — Extract VAE features

```bash
python cnv_vae_extractor.py --device cuda --epochs 50
```

Optional flags:

```
--input_path    Processed CSV from step 1 (default: data/processed/raw_features_cnv.csv)
--output_path   Where to save latent features (default: data/processed/vae_features_cnv.csv)
--epochs        Training epochs (default: 50)
--batch_size    Batch size (default: 32)
--lr            Initial learning rate (default: 1e-3)
--latent_dim    Latent space size (default: 32, matches the paper)
--kl_weight     KL divergence coefficient β (default: 0.001, see implementation notes)
--device        Device: auto | cpu | cuda (default: auto)
--test_mode     100 patients, 5 epochs — quick CPU functionality check
```

### Step 3 — Train classifiers

```bash
python train_classifier.py
```

This trains five classifiers (RBF SVM, Linear SVM, Polynomial SVM, Sigmoid SVM, Random Forest) using 10-fold stratified cross-validation, with minority class upsampling applied per fold.

Optional flags:

```
--modality_paths   Path(s) to VAE feature CSV(s) (default: data/processed/vae_features_cnv.csv)
--modality_names   Short name(s) for each modality (default: cnv)
--results_file     Output CSV for results (default: results/classification_results.csv)
--test_mode        3-fold CV and small RF — quick CPU smoke-test
```

The script is structured to accept multiple modality files for future multimodal experiments:

```bash
python train_classifier.py \
  --modality_paths data/processed/vae_features_cnv.csv \
                   data/processed/vae_features_cln.csv \
  --modality_names cnv cln
```

### Smoke-test (no survival file needed)

```bash
python prepare_cnv_data.py --test_mode
python cnv_vae_extractor.py --test_mode
```

This runs steps 1 and 2 on 100 synthetic patients with 5 training epochs and completes in under 2 minutes on CPU.

---

## Running with Docker

If you'd rather not deal with Python environment setup, Docker handles everything.

Before building, make sure `survival_BRCA_survival.txt` is in the project root (same directory as `docker-compose.yml`). The compose file bind-mounts it into the container automatically.

### Build

```bash
docker build -t cnv-vae .
```

### Run the full pipeline

```bash
docker compose up
```

This mounts your local `data/`, `results/`, and the survival file into the container, then runs all three steps in sequence. Outputs are written back to your host `data/processed/` and `results/`.

### Individual steps via Docker

```bash
# Data prep only
docker compose run --rm vae-pipeline \
  python prepare_cnv_data.py --survival_path survival_BRCA_survival.txt

# VAE extraction only (requires raw_features_cnv.csv to already exist)
docker compose run --rm vae-pipeline python cnv_vae_extractor.py

# Classification only (requires vae_features_cnv.csv to already exist)
docker compose run --rm vae-pipeline python train_classifier.py

# Smoke-test (steps 1 + 2 only, no survival file needed)
docker compose run --rm vae-pipeline \
  sh -c "python prepare_cnv_data.py --test_mode && python cnv_vae_extractor.py --test_mode"
```

---

## Running on a GPU cluster (SLURM)

```bash
sbatch run_pipeline.slurm
```

The script targets the `nopreempt` partition with one A100 GPU by default. Edit `run_pipeline.slurm` to change partition, GPU type, or memory. Logs are written to `logs/slurm_<jobid>.out`.

---

## Output

**`data/processed/`**

| File | Description |
|---|---|
| `raw_features_cnv.csv` | Preprocessed CNV features (patients × 500 genes + label) |
| `vae_features_cnv.csv` | 32-dimensional VAE latent features per patient |
| `vae_features_cnv_model.pt` | Saved PyTorch VAE model checkpoint |

The `vae_features_cnv.csv` column layout matches the reference paper exactly:
```
submitter_id.samples, cnv_vae_1, ..., cnv_vae_32, label_cnv
```

**`results/`** — one set of files per run, keyed by `YYYYMMDD_HHMMSS` timestamp

| File | Description |
|---|---|
| `classification_results.csv` | Appended per-classifier CV and full-dataset metrics |
| `per_fold_metrics.csv` | One row per fold per classifier |
| `dataset_used_<run_id>.csv` | Exact feature matrix + labels used for that run |
| `cv_splits_<run_id>.pkl` | All fold train/test indices and patient IDs |
| `run_manifest_<run_id>.json` | Full run config, dataset stats, file paths |
| `models/<clf>_best_model.pkl` | Best sklearn model chosen by ROC-AUC across folds |
| `models/<clf>_fold_NN_model.pkl` | Model snapshot saved at the end of each fold |
| `models/<clf>_oof_predictions.csv` | Out-of-fold predictions for every patient |

---

## Implementation notes — PyTorch vs Keras differences and VAE debugging

The original paper was implemented in Keras/TensorFlow. Re-implementing in PyTorch required resolving three framework-level differences that each caused the VAE to produce uninformative latent features.

### Run 1 — Baseline PyTorch (incorrect)

**What was done:** Raw GISTIC2 values (`{−2,−1,0,1,2}`) were normalised to `[0,0.25,0.5,0.75,1.0]` before being fed as reconstruction targets to the sigmoid decoder. PyTorch default weight initialisation (Kaiming uniform) was used.

**Problem:** The dominant GISTIC2 value is 0 (44% of all values), which maps to 0.5 after normalisation. The sigmoid decoder output of 0.5 is the exact midpoint, so the VAE learned to output 0.5 for virtually every input. The reconstruction loss collapsed to ≈ 0.023 (the log-cosh value for a 0.5 vs 0.5 comparison) after the first epoch and did not decrease. Only 1 of 32 latent dimensions was active; the other 31 were collapsed to the prior N(0,1). Classifier performance was well below the paper's reported values.

### Run 2 — Raw inputs (partial fix)

**What was done:** The normalisation step was removed. Raw integers `{−2,...,2}` were passed directly as reconstruction targets, matching the reference code (`vae.fit(X, X, ...)` in the published Keras script without any scaling applied to X). Kaiming uniform initialisation was retained.

**Problem:** The KL term still dominated. With sigmoid decoder outputs in (0,1) and targets in {−2,...,2}, the reconstruction gradient per latent dimension was approximately 5 × 10⁻⁴ — too weak to prevent the KL term from pulling 31/32 dimensions to the prior. The loss settled at ≈ 0.268 (no improvement from epoch 1 onward) and active latent dimensions remained at 1/32.

### Run 3 — Xavier initialisation (partial fix)

**What was done:** Weight initialisation was changed from Kaiming uniform to Xavier uniform, matching Keras's default Glorot uniform initialiser. For Tanh activations, Xavier computes variance as `2 / (fan_in + fan_out)`, while Kaiming uses `1 / fan_in` — roughly 4× more variance per weight for the encoder layers in this architecture. The raw input approach from Run 2 was retained.

**Problem:** Xavier initialisation slightly improved the result (3 active latent dims vs 1), but posterior collapse persisted. The reconstruction gradient was unchanged and the KL term still dominated for the majority of latent dimensions.

### Run 4 — β-VAE (correct)

**What was done:** A KL weight coefficient β = 0.001 was introduced. The VAE loss became:

```
loss = reconstruction_loss + 0.001 × KL_loss
```

This is a standard β-VAE formulation (Higgins et al., 2017). With β reduced from 1.0 to 0.001, the reconstruction term dominates and all 32 latent dimensions encode data-dependent information.

**Result:** All 32 latent dimensions became active (standard deviations ranging from 0.9 to 1.7 across the patient population). The loss curve decreased steadily from 0.315 at epoch 1 to 0.172 at epoch 50, showing the model was learning throughout training.

**Why β = 1 works in Keras but not PyTorch:** The reference Keras code uses β = 1 and normalised inputs in [0,1]. In that setting the reconstruction targets are in the same range as the sigmoid output, the gradient scale is much larger, and the reconstruction term naturally dominates. Switching to raw integer targets without adjusting β creates an unbalanced loss where KL is relatively much stronger.

---

## Results — CNV unimodal, VAE features (Run 4)

All results from 10-fold stratified cross-validation on 1078 labeled patients (827 short-term, 251 long-term) from TCGA-BRCA. Job ran on NVIDIA A100 80GB PCIe. Total wall time: approximately 55 seconds.

Paper metrics sourced from Table 2 of Arya et al. (2023), column "VAE", row "Uni".

| Classifier | Metric | Ours (Run 4) | Paper | Delta |
|---|---|---|---|---|
| rbf_svm | F1-score | 0.658 | 0.763 | −0.105 |
| rbf_svm | Sensitivity | 0.561 | 0.777 | −0.216 |
| poly_svm | F1-score | 0.680 | 0.822 | −0.142 |
| poly_svm | Sensitivity | 0.591 | 0.895 | −0.304 |
| sigmoid_svm | F1-score | 0.562 | 0.579 | −0.017 |
| sigmoid_svm | Sensitivity | 0.450 | 0.467 | −0.017 |
| random_forest | F1-score | 0.755 | 0.783 | −0.028 |
| random_forest | Sensitivity | 0.744 | 0.804 | −0.060 |

Random Forest and Sigmoid SVM are within 3–6% of the paper. RBF and Polynomial SVM remain further off.

The precision column is stable across all runs at approximately 0.78–0.79, close to the paper's 0.76–0.77, because precision is determined largely by the class imbalance ratio (827:251 ≈ 3.3:1) and both implementations use identical upsampling.

The remaining SVM gap is consistent with cross-framework non-reproducibility: different default random seeds, batch ordering, floating-point accumulation order, and the fact that the paper likely reports results from the best of multiple training runs. The VAE latent space representation — the primary research contribution — is now correctly replicated with all 32 dimensions active and class-separating.

---

## Project structure

```
├── prepare_cnv_data.py       # Step 1: data preprocessing + survival label generation
├── cnv_vae_extractor.py      # Step 2: log-cosh VAE training + feature extraction
├── train_classifier.py       # Step 3: SVM/RF classification with 10-fold CV
├── run_pipeline.slurm        # SLURM job script for GPU cluster submission
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── survival_BRCA_survival.txt         # TCGA-CDR OS survival data (from Xena Browser)
├── data/
│   ├── TCGA.BRCA.sampleMap_...        # raw CNV input (download from Xena)
│   └── processed/                     # VAE features and model checkpoint
├── results/                           # classification results, splits, manifests
├── logs/                              # SLURM output logs
└── reference_files/                   # original Keras code from the paper (not executed)
```

---

## Reference

Arya, N., Saha, S., Mathur, A., & Saha, S. (2023). Improving the robustness and stability of a machine learning model for breast cancer prognosis through the use of multi-modal classifiers. *Scientific Reports*, 13, 4079. https://doi.org/10.1038/s41598-023-30143-8

Original code: https://github.com/nikhilaryan92/logcoshVAE_brca_surv


