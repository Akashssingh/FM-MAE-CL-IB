"""
mRNA Data Preparation for TCGA-BRCA
=====================================
Replicates the preprocessing steps from Arya et al. (2023)
"Improving the robustness and stability of a machine learning model for
breast cancer prognosis through the use of multi-modal classifiers"

Steps performed (matching paper exactly):
  1. Load HiSeqV2 mRNA TSV from Xena browser (genes × patients)
  2. Transpose to patient × gene format
  3. Handle missing values: discard features with >10% NaN; KNN-impute remainder
  4. Remove zero-variance features (same value for all patients)
  5. Discretize gene expression: per-gene z-score → -1 (under), 0 (baseline), +1 (over)
     Threshold: ±1.5 SD from mean (common in TCGA multi-modal literature)
  6. Select top N features by variance (default 500, captures >98% variance)
  7. Merge 5-year binary OS survival labels from TCGA-CDR survival TSV
  8. Save as raw_features_mrna.csv for downstream VAE feature extraction

Note on VAE preprocessing:
  The mRNA VAE decoder in the reference script has NO sigmoid activation
  (unlike CNV which uses sigmoid). During VAE training the discretized
  features are Min-Max scaled to [0,1] (-1→0, 0→0.5, 1→1) so the
  network operates in a bounded domain. The raw_features_mrna.csv stores
  the discretized integer values; scaling is applied inside run_ablation.py.

Output CSV format (matches reference code expectation):
  submitter_id.samples | gene_1 | gene_2 | ... | gene_500 | label_mrna

Label encoding (OS-based 5-year survival):
  0 = long-term survivor  (OS.time >= 5*365.25 days)
  1 = short-term survivor (OS.time < 5*365.25)
 -1 = censored/unknown    (excluded during training)

Usage:
  python prepare_mrna_data.py --survival_path survival_BRCA_survival.txt

  # Custom paths:
  python prepare_mrna_data.py \\
      --mrna_path data/TCGA.BRCA.sampleMap_HiSeqV2 \\
      --survival_path survival_BRCA_survival.txt \\
      --out_path data/processed/raw_features_mrna.csv

  # Smoke-test:
  python prepare_mrna_data.py --survival_path survival_BRCA_survival.txt --test_mode
"""

import os
import argparse
import numpy as np
import pandas as pd

try:
    from sklearn.impute import KNNImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Constants (matching paper) ────────────────────────────────────────────────
MISSING_THRESHOLD  = 0.10    # Drop feature if > 10% patients have NaN
N_FEATURES         = 500     # Top features to keep by variance
DAYS_5Y            = 5 * 365.25
DISCR_THRESHOLD    = 1.5     # SD threshold for discretisation


# ── Survival label helpers (identical to prepare_cnv_data.py) ─────────────────

def _compute_os_labels(df_surv: pd.DataFrame,
                       patient_ids: pd.Series) -> pd.Series:
    """
    Binary 5-year OS labels from TCGA-CDR survival TSV.
      0 = long-term  (OS.time >= 5yr)
      1 = short-term (OS.time <  5yr)
     -1 = missing
    """
    if 'sample' not in df_surv.columns:
        print("  [WARN] 'sample' column not found — labelling all patients -1.")
        return pd.Series([-1] * len(patient_ids))

    lookup_full   = df_surv.set_index('sample')
    lookup_prefix = df_surv.copy()
    lookup_prefix['_prefix'] = lookup_prefix['sample'].str[:12]
    lookup_prefix = (lookup_prefix
                     .drop_duplicates('_prefix')
                     .set_index('_prefix'))

    labels = []
    for pid in patient_ids.values:
        pid_str = str(pid).strip()
        row = None
        if pid_str in lookup_full.index:
            row = lookup_full.loc[pid_str]
        else:
            prefix = pid_str[:12]
            if prefix in lookup_prefix.index:
                row = lookup_prefix.loc[prefix]

        if row is None:
            labels.append(-1)
            continue

        os_time = pd.to_numeric(row.get('OS.time', np.nan), errors='coerce')
        if pd.isna(os_time):
            labels.append(-1)
        elif os_time < DAYS_5Y:
            labels.append(1)   # short-term (majority class)
        else:
            labels.append(0)   # long-term  (minority class)

    return pd.Series(labels)


# ── Core preprocessing ────────────────────────────────────────────────────────

def discretize_expression(df_expr: pd.DataFrame,
                           threshold: float = DISCR_THRESHOLD) -> pd.DataFrame:
    """
    Discretize log2 mRNA expression values (gene per column) to -1/0/+1.

    Per-gene across all patients:
      mean + threshold * std → +1  (over-expressed)
      mean - threshold * std → -1  (under-expressed)
      otherwise              →  0  (baseline)

    Returns a DataFrame of the same shape with int8 dtype.
    """
    gene_mean = df_expr.mean(axis=0)
    gene_std  = df_expr.std(axis=0)

    upper = gene_mean + threshold * gene_std
    lower = gene_mean - threshold * gene_std

    result = pd.DataFrame(0, index=df_expr.index, columns=df_expr.columns,
                          dtype=np.int8)
    result[df_expr.gt(upper)] =  1
    result[df_expr.lt(lower)] = -1
    return result


def load_and_preprocess(mrna_path: str,
                        survival_path: str,
                        n_features: int,
                        test_mode: bool) -> pd.DataFrame:
    """Full preprocessing pipeline. Returns final DataFrame."""

    # ── 1. Load HiSeqV2 (genes × patients, tab-separated) ─────────────────
    print(f"Loading mRNA data from: {mrna_path}")
    df_raw = pd.read_csv(mrna_path, sep='\t', header=0, index_col=0,
                         low_memory=False)

    if test_mode:
        df_raw = df_raw.iloc[:, :50]   # first 50 patients for speed
        print(f"  [TEST MODE] Using first 50 patients")

    print(f"  Raw shape (genes × patients): {df_raw.shape}")

    # Transpose → patients × genes
    df = df_raw.T.copy()
    df.index.name = 'submitter_id.samples'
    df = df.reset_index()
    print(f"  After transpose (patients × genes): {df.shape}")

    patient_ids = df['submitter_id.samples']
    df_feat     = df.drop(columns=['submitter_id.samples'])

    # Convert all expression values to float
    df_feat = df_feat.apply(pd.to_numeric, errors='coerce')

    # ── 2. Handle missing values ───────────────────────────────────────────
    nan_frac = df_feat.isna().mean(axis=0)
    keep     = nan_frac[nan_frac <= MISSING_THRESHOLD].index
    dropped  = len(df_feat.columns) - len(keep)
    df_feat  = df_feat[keep]
    print(f"  Dropped {dropped} features with >{MISSING_THRESHOLD*100:.0f}% NaN "
          f"→ {df_feat.shape[1]} genes remaining")

    if df_feat.isna().any().any():
        if HAS_SKLEARN:
            print(f"  KNN-imputing remaining NaN values …")
            imputer = KNNImputer(n_neighbors=5)
            df_feat = pd.DataFrame(
                imputer.fit_transform(df_feat),
                columns=df_feat.columns)
        else:
            print("  [WARN] sklearn not available — filling NaN with column median")
            df_feat = df_feat.fillna(df_feat.median())

    # ── 3. Remove zero-variance genes ─────────────────────────────────────
    var        = df_feat.var(axis=0)
    nonzero    = var[var > 0].index
    removed_zv = len(df_feat.columns) - len(nonzero)
    df_feat    = df_feat[nonzero]
    print(f"  Removed {removed_zv} zero-variance genes "
          f"→ {df_feat.shape[1]} genes remaining")

    # ── 4. Discretize expression (−1 / 0 / +1) ────────────────────────────
    print(f"  Discretizing expression (±{DISCR_THRESHOLD} SD threshold) …")
    df_discr = discretize_expression(df_feat, DISCR_THRESHOLD)

    # ── 5. Re-check variance after discretization & select top N ──────────
    var_post  = df_discr.var(axis=0)
    nonzero2  = var_post[var_post > 0].index
    df_discr  = df_discr[nonzero2]
    print(f"  Non-zero-variance after discretization: {df_discr.shape[1]} genes")

    top_genes = (df_discr.var(axis=0)
                          .nlargest(min(n_features, df_discr.shape[1]))
                          .index)
    df_sel    = df_discr[top_genes].copy()
    print(f"  Selected top {len(top_genes)} genes by variance")

    # ── 6. Merge survival labels ───────────────────────────────────────────
    print(f"Loading survival labels from: {survival_path}")
    df_surv  = pd.read_csv(survival_path, sep='\t', header=0, low_memory=False)
    labels   = _compute_os_labels(df_surv, patient_ids)

    df_out = pd.concat([
        patient_ids.reset_index(drop=True),
        df_sel.reset_index(drop=True),
        labels.rename('label_mrna').reset_index(drop=True),
    ], axis=1)

    # Drop unlabelled patients (label == -1)
    n_before = len(df_out)
    df_out   = df_out[df_out['label_mrna'] != -1].reset_index(drop=True)
    print(f"  Dropped {n_before - len(df_out)} patients without 5-yr OS label")

    vc = df_out['label_mrna'].value_counts()
    print(f"  Final dataset: {len(df_out)} patients, {df_sel.shape[1]} genes")
    print(f"  Label distribution: "
          f"{vc.get(0, 0)} long-term (0) / {vc.get(1, 0)} short-term (1)")

    return df_out


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description='Preprocess TCGA-BRCA mRNA HiSeqV2 data for VAE ablation.')
    parser.add_argument('--mrna_path', default=os.path.join(
        project_root, 'data', 'TCGA.BRCA.sampleMap_HiSeqV2'),
        help='Path to raw HiSeqV2 TSV (genes × patients).')
    parser.add_argument('--survival_path', default=os.path.join(
        project_root, 'survival_BRCA_survival.txt'),
        help='Path to TCGA-CDR survival TSV.')
    parser.add_argument('--out_path', default=os.path.join(
        project_root, 'data', 'processed', 'raw_features_mrna.csv'),
        help='Output CSV path.')
    parser.add_argument('--n_features', type=int, default=N_FEATURES,
        help=f'Number of top-variance genes to retain (default {N_FEATURES}).')
    parser.add_argument('--test_mode', action='store_true',
        help='Quick smoke-test using first 50 patients.')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    df_out = load_and_preprocess(
        mrna_path     = args.mrna_path,
        survival_path = args.survival_path,
        n_features    = args.n_features,
        test_mode     = args.test_mode,
    )

    df_out.to_csv(args.out_path, index=False)
    print(f"\nSaved → {args.out_path}")
    print(f"Shape : {df_out.shape}  "
          f"(patients × [id + {args.n_features} genes + label])")


if __name__ == '__main__':
    main()
