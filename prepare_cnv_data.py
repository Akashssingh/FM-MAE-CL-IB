"""
CNV Data Preparation for TCGA-BRCA
====================================
Replicates the preprocessing steps from Arya et al. (2023)
"Improving the robustness and stability of a machine learning model for
breast cancer prognosis through the use of multi-modal classifiers"

Steps performed:
  1. Load GISTIC2-thresholded CNV TSV from Xena browser (genes x patients)
  2. Transpose to patient x gene format
  3. Handle missing values: discard features with >10% NaN; KNN-impute remainder
  4. Remove zero-variance features (same value for all patients)
  5. Select top N features by variance (default 500, captures >98% variance)
  6. Optionally merge 5-year survival labels from a clinical TSV
  7. Save as raw_features_cnv.csv for downstream VAE feature extraction

Output CSV format (matches reference code expectation):
  submitter_id.samples | gene_1 | gene_2 | ... | gene_500 | label_cnv

Usage:
  # Basic (no labels):
  python prepare_cnv_data.py

  # With survival labels from Xena clinical file:
  python prepare_cnv_data.py --clinical_path data/TCGA.BRCA.sampleMap_BRCA_clinicalMatrix

  # Quick smoke-test (first 100 patients, 3-epoch VAE):
  python prepare_cnv_data.py --test_mode
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


# ── Constants ────────────────────────────────────────────────────────────────
MISSING_THRESHOLD = 0.10   # Drop feature if > 10% of patients have NaN
N_FEATURES = 500           # Top features to keep by variance
DAYS_5Y = 5 * 365.25       # 5-year survival cutoff in days


# ── Survival label helpers ────────────────────────────────────────────────────

def _compute_survival_labels(df_clin: pd.DataFrame,
                              patient_ids: pd.Series) -> pd.Series:
    """
    Derive binary survival labels from Xena clinical data.
      0 = long-term survivor  (survived >= 5 years)
      1 = short-term survivor (survived < 5 years)
     -1 = unknown / missing
    """
    labels = []
    # Xena clinical columns (common names; may vary by download)
    VITAL_COL = next(
        (c for c in df_clin.columns if 'vital_status' in c.lower()), None)
    DEATH_COL = next(
        (c for c in df_clin.columns if 'days_to_death' in c.lower()), None)
    FOLLOW_COL = next(
        (c for c in df_clin.columns if 'days_to_last_follow' in c.lower()), None)

    if not (VITAL_COL and (DEATH_COL or FOLLOW_COL)):
        print("  [WARN] Could not find required clinical columns. "
              "Using placeholder label -1 for all patients.")
        return pd.Series([-1] * len(patient_ids))

    for pid in patient_ids.values:
        if pid not in df_clin.index:
            labels.append(-1)
            continue
        row = df_clin.loc[pid]
        vital = str(row.get(VITAL_COL, '')).strip().lower()
        if vital in ('dead', 'deceased'):
            days = pd.to_numeric(
                row.get(DEATH_COL, np.nan) if DEATH_COL else np.nan,
                errors='coerce')
        else:
            days = pd.to_numeric(
                row.get(FOLLOW_COL, np.nan) if FOLLOW_COL else np.nan,
                errors='coerce')
        if pd.isna(days):
            labels.append(-1)
        elif days < DAYS_5Y:
            labels.append(1)   # short-term survivor
        else:
            labels.append(0)   # long-term survivor

    return pd.Series(labels)


# ── Main preprocessing ────────────────────────────────────────────────────────

def prepare_cnv_data(cnv_path: str,
                     output_path: str,
                     clinical_path: str = None,
                     n_features: int = N_FEATURES,
                     test_mode: bool = False) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Returns the saved DataFrame.
    """
    # ── 1. Load raw data ────────────────────────────────────────────────────
    print(f"[1/5] Loading CNV data from:\n      {cnv_path}")
    df_raw = pd.read_csv(cnv_path, sep='\t', index_col=0, low_memory=False)
    print(f"      Raw shape (genes x patients): {df_raw.shape}")

    # ── 2. Transpose ────────────────────────────────────────────────────────
    print("[2/5] Transposing to (patients x genes) ...")
    df = df_raw.T.copy()
    df.index.name = 'submitter_id.samples'
    df.reset_index(inplace=True)
    df.columns = df.columns.astype(str)

    if test_mode:
        df = df.iloc[:100].copy()
        print(f"      [TEST MODE] Limiting to 100 patients.")

    patient_ids = df['submitter_id.samples'].copy()
    feature_cols = [c for c in df.columns if c != 'submitter_id.samples']
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    print(f"      Shape after transpose: {X.shape}")

    # ── 3. Handle missing values ─────────────────────────────────────────────
    print("[3/5] Handling missing values ...")
    n_patients = len(X)
    missing_frac = X.isnull().mean(axis=0)
    cols_to_keep = missing_frac[missing_frac <= MISSING_THRESHOLD].index
    cols_dropped = X.shape[1] - len(cols_to_keep)
    X = X[cols_to_keep]
    print(f"      Dropped {cols_dropped} features with >{MISSING_THRESHOLD*100:.0f}% missing values")

    if X.isnull().any().any():
        if HAS_SKLEARN:
            print(f"      KNN-imputing remaining missing values ...")
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            X_arr = imputer.fit_transform(X)
            X = pd.DataFrame(X_arr, columns=X.columns)
        else:
            print("      [WARN] scikit-learn not available; filling NaN with 0 (diploid normal).")
            X = X.fillna(0)

    # ── 4. Remove zero-variance features ────────────────────────────────────
    print("[4/5] Removing zero-variance features ...")
    variances = X.var(axis=0)
    pre_filter = X.shape[1]
    X = X.loc[:, variances > 0]
    print(f"      Removed {pre_filter - X.shape[1]} zero-variance features; "
          f"{X.shape[1]} remain.")

    # ── 5. Select top N features by variance ────────────────────────────────
    actual_n = min(n_features, X.shape[1])
    if actual_n < n_features:
        print(f"[5/5] Only {X.shape[1]} features available; "
              f"selecting all of them.")
    else:
        print(f"[5/5] Selecting top {actual_n} features by variance ...")

    top_features = X.var(axis=0).nlargest(actual_n).index
    X_selected = X[top_features].copy()

    # Explained variance check
    total_var = variances.sum()
    selected_var = variances[top_features].sum()
    explained = selected_var / total_var * 100
    print(f"      Top {actual_n} features explain {explained:.1f}% of total variance")

    # ── 6. Merge survival labels ─────────────────────────────────────────────
    if clinical_path and os.path.exists(clinical_path):
        print(f"      Loading clinical data from {clinical_path} ...")
        try:
            df_clin = pd.read_csv(
                clinical_path, sep='\t', index_col=0, low_memory=False)
            labels = _compute_survival_labels(df_clin, patient_ids)
            n_labeled = (labels >= 0).sum()
            print(f"      Labeled {n_labeled}/{len(labels)} patients "
                  f"({labels.value_counts().to_dict()})")
        except Exception as exc:
            print(f"      [WARN] Clinical data error: {exc}. "
                  "Using placeholder label -1.")
            labels = pd.Series([-1] * len(patient_ids))
    else:
        if clinical_path:
            print(f"      [WARN] Clinical file not found at {clinical_path}.")
        print("      No clinical data → using placeholder label -1. "
              "Pass --clinical_path to add real 5-year survival labels.")
        labels = pd.Series([-1] * len(patient_ids))

    # ── 7. Build and save output ─────────────────────────────────────────────
    id_df    = pd.DataFrame({'submitter_id.samples': patient_ids.values})
    label_df = pd.DataFrame({'label_cnv': labels.values})
    result   = pd.concat(
        [id_df, X_selected.reset_index(drop=True), label_df], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"\nDone. Saved to: {output_path}")
    print(f"  Patients : {result.shape[0]}")
    print(f"  Features : {result.shape[1] - 2}  "
          f"(excluding patient ID and label columns)")
    print(f"  Label distribution: "
          f"{result['label_cnv'].value_counts().to_dict()}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare TCGA-BRCA CNV data for log-cosh VAE feature extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--cnv_path', type=str,
        default=os.path.join(
            'data',
            'TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.tsv'),
        help='Path to raw GISTIC2-thresholded CNV TSV (genes x patients).')
    parser.add_argument(
        '--output_path', type=str,
        default=os.path.join('data', 'processed', 'raw_features_cnv.csv'),
        help='Output path for the processed raw_features CSV.')
    parser.add_argument(
        '--clinical_path', type=str, default=None,
        help='(Optional) Path to Xena clinical TSV for 5-year survival labels.')
    parser.add_argument(
        '--n_features', type=int, default=N_FEATURES,
        help='Number of top-variance features to select.')
    parser.add_argument(
        '--test_mode', action='store_true',
        help='Limit to 100 patients for quick CPU functionality check.')
    args = parser.parse_args()

    prepare_cnv_data(
        cnv_path=args.cnv_path,
        output_path=args.output_path,
        clinical_path=args.clinical_path,
        n_features=args.n_features,
        test_mode=args.test_mode)


if __name__ == '__main__':
    main()
