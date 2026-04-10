"""
Breast Cancer Survival Classifier
===================================
Trains SVM and Random Forest classifiers on VAE-extracted features following
the methodology from Arya et al. (2023).

Training protocol (faithful to the paper):
  - 10-fold stratified cross-validation
  - Per-fold minority class upsampling applied to TRAINING splits only
  - Best model tracked by ROC-AUC across folds
  - Soft-margin SVM (C=1) with gamma inversely proportional to input features
  - Classifiers: RBF SVM, Linear SVM, Polynomial SVM, Sigmoid SVM, Random Forest

Metrics reported per classifier:
  Accuracy, Precision, Sensitivity (Recall), F1-score, ROC-AUC (CV avg + full-set)

Currently supports: unimodal CNV (VAE features)
Extensible to:       any number of modalities via --modality_paths

Usage:
  # Unimodal CNV (default):
  python train_classifier.py

  # Custom feature file:
  python train_classifier.py \\
      --modality_paths data/processed/vae_features_cnv.csv \\
      --modality_names cnv

  # Quick smoke-test (fewer folds, small RF):
  python train_classifier.py --test_mode

  # Future multimodal example (when other modalities are ready):
  python train_classifier.py \\
      --modality_paths data/processed/vae_features_cnv.csv \\
                       data/processed/vae_features_cln.csv \\
      --modality_names cnv cln
"""

import os
import csv
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, f1_score, recall_score
)
from sklearn.utils import resample

warnings.filterwarnings('ignore')   # suppress convergence/UndefinedMetric warnings

# ── Constants ─────────────────────────────────────────────────────────────────
N_FOLDS       = 10
RANDOM_STATE  = 123
RESULTS_FILE  = 'results/classification_results.csv'
BEST_MODEL_PATH = 'results/best_model.pkl'

# CSV header for results file
RESULTS_HEADER = [
    'modality', 'classifier',
    'cv_roc_auc',
    'cv_tn', 'cv_fp', 'cv_fn', 'cv_tp',
    'full_roc_auc',
    'full_tn', 'full_fp', 'full_fn', 'full_tp',
    'cv_accuracy', 'cv_precision', 'cv_sensitivity', 'cv_f1'
]


# ── Data loading & merging ────────────────────────────────────────────────────

def load_modality(path: str) -> pd.DataFrame:
    """Load a single modality feature CSV (format: id | features... | label)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run prepare_cnv_data.py followed by cnv_vae_extractor.py first.")
    df = pd.read_csv(path, header=0, index_col=None, low_memory=False)
    return df


def merge_modalities(dfs: list, names: list) -> pd.DataFrame:
    """
    Inner-join multiple modality DataFrames on patient ID (first column).
    Drops all per-modality label columns except the last one, renaming it 'label'.
    This mirrors the reference multi-modal merge logic exactly.
    """
    if len(dfs) == 1:
        df = dfs[0].copy()
        df = df.rename(columns={df.columns[-1]: 'label'})
        return df

    merged = dfs[0].copy()
    id_col = merged.columns[0]

    for i, (df, name) in enumerate(zip(dfs[1:], names[1:]), start=1):
        merged = pd.merge(
            merged, df,
            how='inner',
            left_on=merged.columns[0],
            right_on=df.columns[0]
        )
        # Drop duplicate patient-ID column introduced by merge
        if merged.columns[0] != df.columns[0] and df.columns[0] in merged.columns:
            merged = merged.drop(columns=[df.columns[0]])

    # Drop all intermediate label columns; keep only the last one
    label_cols = [c for c in merged.columns if c.startswith('label_')]
    if label_cols:
        merged = merged.drop(columns=label_cols[:-1])
        merged = merged.rename(columns={label_cols[-1]: 'label'})

    print(f"  Merged shape ({' + '.join(names)}): "
          f"{merged.shape[0]} patients × {merged.shape[1]-2} features")
    return merged


# ── Upsampling ────────────────────────────────────────────────────────────────

def upsample_minority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversample the minority class to match the majority class size.
    Applied per training fold (never on test data).
    Label column must be the last column.
    """
    label_col = df.columns[-1]
    counts = df[label_col].value_counts()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    majority_n = counts[majority_class]

    df_majority = df[df[label_col] == majority_class]
    df_minority = df[df[label_col] == minority_class]

    df_minority_up = resample(
        df_minority,
        replace=True,
        n_samples=majority_n,
        random_state=RANDOM_STATE
    )
    return pd.concat([df_majority, df_minority_up]).reset_index(drop=True)


# ── Core CV training loop ─────────────────────────────────────────────────────

def run_cv(model, df: pd.DataFrame, n_folds: int = N_FOLDS) -> dict:
    """
    10-fold stratified CV with per-fold upsampling.
    Tracks the best model (by ROC-AUC) across folds.
    Returns a dict with CV-averaged metrics and full-dataset metrics.

    Faithfully replicates the reference model_run() logic including best-model
    tracking and reload before each subsequent fold.
    """
    os.makedirs(os.path.dirname(os.path.abspath(BEST_MODEL_PATH)), exist_ok=True)

    # Separate features from id/label
    feature_df = df.drop(columns=[df.columns[0], df.columns[-1]])
    X = feature_df.values.astype(np.float32)
    y = df[df.columns[-1]].values.astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)

    cum_cm   = np.zeros(4)       # tn fp fn tp  (cumulative across folds)
    cum_roc  = 0.0
    best_roc = 0.0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Upsample minority in training split only
        train_df = pd.DataFrame(X_train, columns=feature_df.columns)
        train_df['label'] = y_train
        train_df_up = upsample_minority(train_df)
        X_train_up = train_df_up.drop(columns=['label']).values
        y_train_up = train_df_up['label'].values

        model.fit(X_train_up, y_train_up)

        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred).ravel()  # tn fp fn tp
        roc    = roc_auc_score(y_test, y_pred)

        cum_cm  += cm
        cum_roc += roc

        print(f"    Fold {fold_idx:2d}/{n_folds} | "
              f"ROC-AUC={roc:.4f} | "
              f"TN={cm[0]:.0f} FP={cm[1]:.0f} FN={cm[2]:.0f} TP={cm[3]:.0f}")

        # Track best model
        if roc > best_roc:
            best_roc = roc
            pickle.dump(model, open(BEST_MODEL_PATH, 'wb'))

        # Reload best-so-far model for next fold (matches reference behaviour)
        model = pickle.load(open(BEST_MODEL_PATH, 'rb'))

    # CV averages
    avg_cm  = np.round(cum_cm / n_folds, 4)
    avg_roc = cum_roc / n_folds

    tn, fp, fn, tp = avg_cm
    eps = 1e-9
    cv_accuracy    = (tp + tn) / (tp + tn + fp + fn + eps)
    cv_precision   = tp / (tp + fp + eps)
    cv_sensitivity = tp / (tp + fn + eps)
    cv_f1          = (2 * cv_precision * cv_sensitivity /
                      (cv_precision + cv_sensitivity + eps))

    # Full-dataset evaluation with best model
    best_model = pickle.load(open(BEST_MODEL_PATH, 'rb'))
    y_full_pred = best_model.predict(X)
    full_cm     = confusion_matrix(y, y_full_pred).ravel()
    full_roc    = roc_auc_score(y, y_full_pred)

    return {
        'cv_roc_auc':    avg_roc,
        'cv_tn':         avg_cm[0],
        'cv_fp':         avg_cm[1],
        'cv_fn':         avg_cm[2],
        'cv_tp':         avg_cm[3],
        'cv_accuracy':   round(cv_accuracy, 4),
        'cv_precision':  round(cv_precision, 4),
        'cv_sensitivity':round(cv_sensitivity, 4),
        'cv_f1':         round(cv_f1, 4),
        'full_roc_auc':  full_roc,
        'full_tn':       full_cm[0],
        'full_fp':       full_cm[1],
        'full_fn':       full_cm[2],
        'full_tp':       full_cm[3],
    }


# ── Classifier definitions ────────────────────────────────────────────────────

def get_classifiers(n_features: int, n_estimators: int = 10) -> list:
    """
    Returns list of (name, sklearn_model) pairs to evaluate.

    SVM hyperparameters follow the paper's unimodal setup:
      - C=1  (soft-margin, small regularisation)
      - gamma='scale'  (1 / (n_features * X.var())) — inversely proportional
                        to n_features, controls per-sample influence radius

    RF uses n_estimators=10 (paper's uni-modal setting) to keep CPU runs fast.
    Increase to 1000 for final results (matches multi-modal setting in the paper).
    """
    return [
        ('rbf_svm',       SVC(kernel='rbf',     C=1, gamma='scale')),
        ('linear_svm',    SVC(kernel='linear',  C=1, gamma='scale')),
        ('poly_svm',      SVC(kernel='poly',    C=1, gamma='scale')),
        ('sigmoid_svm',   SVC(kernel='sigmoid', C=1, gamma='scale')),
        ('random_forest', RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='gini',
            max_features=None,
            random_state=RANDOM_STATE,
        )),
    ]


# ── Results I/O ───────────────────────────────────────────────────────────────

def init_results_file(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(RESULTS_HEADER)


def append_result(path: str, modality_name: str,
                  clf_name: str, metrics: dict):
    row = [
        modality_name, clf_name,
        metrics['cv_roc_auc'],
        metrics['cv_tn'], metrics['cv_fp'],
        metrics['cv_fn'], metrics['cv_tp'],
        metrics['full_roc_auc'],
        metrics['full_tn'], metrics['full_fp'],
        metrics['full_fn'], metrics['full_tp'],
        metrics['cv_accuracy'], metrics['cv_precision'],
        metrics['cv_sensitivity'], metrics['cv_f1'],
    ]
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def print_summary(modality_name: str, clf_name: str, metrics: dict):
    print(f"\n  ── {modality_name} | {clf_name} ──────────────────────────────")
    print(f"    CV  ROC-AUC   : {metrics['cv_roc_auc']:.4f}")
    print(f"    CV  Accuracy  : {metrics['cv_accuracy']:.4f}")
    print(f"    CV  Precision : {metrics['cv_precision']:.4f}")
    print(f"    CV  Sensitivity: {metrics['cv_sensitivity']:.4f}")
    print(f"    CV  F1-score  : {metrics['cv_f1']:.4f}")
    print(f"    Full ROC-AUC  : {metrics['full_roc_auc']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs('results', exist_ok=True)
    init_results_file(RESULTS_FILE)

    # ── Load and merge modalities ─────────────────────────────────────────────
    print(f"\nLoading {len(args.modality_paths)} modality file(s) ...")
    dfs = []
    for path, name in zip(args.modality_paths, args.modality_names):
        print(f"  {name}: {path}")
        df = load_modality(path)
        dfs.append(df)

    df_merged = merge_modalities(dfs, args.modality_names)
    modality_tag = '_'.join(args.modality_names)

    # Check labels
    label_vals = df_merged['label'].unique()
    if set(label_vals) == {-1}:
        print("\n[WARNING] All labels are -1 (placeholder). "
              "Classification requires real 0/1 survival labels.\n"
              "Re-run prepare_cnv_data.py with --clinical_path to generate them.")
        return

    # Drop rows with unknown labels
    df_merged = df_merged[df_merged['label'].isin([0, 1])].reset_index(drop=True)
    n_total  = len(df_merged)
    n_short  = (df_merged['label'] == 1).sum()
    n_long   = (df_merged['label'] == 0).sum()
    n_feat   = df_merged.shape[1] - 2

    print(f"\nDataset : {n_total} patients | {n_feat} features")
    print(f"  Short-term survivors (label=1) : {n_short}")
    print(f"  Long-term survivors  (label=0) : {n_long}")
    print(f"  Class imbalance ratio : {max(n_short,n_long)/min(n_short,n_long):.2f}")

    # ── Run classifiers ───────────────────────────────────────────────────────
    n_est = 5 if args.test_mode else 10
    n_folds = 3 if args.test_mode else N_FOLDS
    classifiers = get_classifiers(n_features=n_feat, n_estimators=n_est)

    print(f"\nRunning {len(classifiers)} classifiers with "
          f"{n_folds}-fold stratified CV ...\n")

    for clf_name, clf in classifiers:
        print(f"\n{'='*60}")
        print(f"  Classifier : {clf_name}")
        print(f"{'='*60}")
        metrics = run_cv(clf, df_merged, n_folds=n_folds)
        print_summary(modality_tag, clf_name, metrics)
        append_result(RESULTS_FILE, modality_tag, clf_name, metrics)

    print(f"\n{'='*60}")
    print(f"All results saved to: {RESULTS_FILE}")
    print(f"Best model saved to : {BEST_MODEL_PATH}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train SVM/RF classifiers on VAE-extracted features.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--modality_paths', nargs='+',
        default=[os.path.join('data', 'processed', 'vae_features_cnv.csv')],
        help='Path(s) to VAE feature CSV file(s). '
             'Pass multiple paths for multimodal training (future use).')

    parser.add_argument(
        '--modality_names', nargs='+',
        default=['cnv'],
        help='Short names for each modality (must match order of --modality_paths).')

    parser.add_argument(
        '--test_mode', action='store_true',
        help='Use 3-fold CV and small RF for a quick CPU smoke-test.')

    parser.add_argument(
        '--results_file', type=str,
        default=RESULTS_FILE,
        help='Output CSV path for classification results.')

    args = parser.parse_args()

    # Validate
    if len(args.modality_paths) != len(args.modality_names):
        parser.error('--modality_paths and --modality_names must have the same length.')

    # Allow overriding global results file from CLI
    RESULTS_FILE = args.results_file

    main(args)
