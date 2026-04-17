"""
Reference Original Model Training
===================================
Exact replication of Arya et al. (2023) ML classifier pipeline on CNV VAE features.

Reference: ML_multimodal_train.py (author's Keras-based implementation)

This script is a faithful Python/scikit-learn port of the original:
  - 10-fold StratifiedKFold (no shuffle, matching reference)
  - Per-fold minority class upsampling (training split only)
  - Best model tracked by ROC-AUC across folds and reloaded each fold
  - Classifiers: RBF SVM, Linear SVM, Poly SVM, Sigmoid SVM, Random Forest
  - Metrics: ROC-AUC, AUROC, AUPRC (average precision), Accuracy,
             Precision, Sensitivity (Recall), F1, Confusion Matrix
  - Works on the reference VAE feature CSV from the original paper

Input CSV format (reference_files/vae_features_cnv.csv):
  submitter_id.samples | cnv_vae_1 ... cnv_vae_32 | label_cnv

Usage:
  python reference_original_model_training.py

  # Custom feature file or output:
  python reference_original_model_training.py \\
      --input_path reference_files/vae_features_cnv.csv \\
      --results_file results/reference_original_results.csv

  # Quick smoke-test (3-fold, small RF):
  python reference_original_model_training.py --test_mode
"""

import os
import csv
import json
import pickle
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings('ignore')

# ── Constants (matching reference implementation) ─────────────────────────────
N_FOLDS      = 10
RANDOM_STATE = 123
RESULTS_DIR  = 'results'
MODELS_DIR   = 'results/ref_models'

RESULTS_HEADER = [
    'modality', 'classifier',
    'cv_roc_auc', 'cv_auprc',
    'cv_tn', 'cv_fp', 'cv_fn', 'cv_tp',
    'full_roc_auc', 'full_auprc',
    'full_tn', 'full_fp', 'full_fn', 'full_tp',
    'cv_accuracy', 'cv_precision', 'cv_sensitivity', 'cv_f1',
]

FOLD_HEADER = [
    'run_id', 'modality', 'classifier', 'fold',
    'roc_auc', 'auprc',
    'tn', 'fp', 'fn', 'tp',
    'accuracy', 'precision', 'sensitivity', 'f1',
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_feature_csv(path: str) -> pd.DataFrame:
    """Load VAE feature CSV.  Expected format: id | features | label."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, header=0, index_col=None, low_memory=False)
    print(f"Loaded: {path}  →  {df.shape[0]} rows × {df.shape[1]} cols")
    return df


# ── Upsampling (exact replica of reference upsample()) ───────────────────────

def upsample_minority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversample the minority class to match the majority class.
    Label must be the last column.
    Replicates reference upsample() exactly (replace=True, random_state=123).
    """
    label_col = df.columns[-1]
    counts = df[label_col].value_counts()
    majority_cls = counts.idxmax()
    minority_cls = counts.idxmin()
    n_majority = counts[majority_cls]

    df_maj = df[df[label_col] == majority_cls]
    df_min = df[df[label_col] == minority_cls]

    df_min_up = resample(df_min, replace=True,
                         n_samples=n_majority,
                         random_state=RANDOM_STATE)
    return pd.concat([df_maj, df_min_up]).reset_index(drop=True)


# ── Classifier definitions ────────────────────────────────────────────────────

def get_classifiers(n_estimators: int = 10) -> list:
    """
    Returns list of (name, model) pairs.

    SVM: C=1 as in reference uni-modal setup; gamma='scale' (scikit-learn
    equivalent of Keras default for SVMs with n_features features).
    RF: n_estimators=10 for uni-modal (reference uses 10 for uni, 1000 for multi).
    """
    return [
        ('rbf_svm',       SVC(kernel='rbf',     C=1, gamma='scale', probability=True)),
        ('linear_svm',    SVC(kernel='linear',  C=1, gamma='scale', probability=True)),
        ('poly_svm',      SVC(kernel='poly',    C=1, gamma='scale', probability=True)),
        ('sigmoid_svm',   SVC(kernel='sigmoid', C=1, gamma='scale', probability=True)),
        ('random_forest', RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='gini',
            max_features=None,
            random_state=RANDOM_STATE,
        )),
    ]


# ── Core CV loop (mirrors reference model_run()) ──────────────────────────────

def run_cv(model, df: pd.DataFrame, n_folds: int,
           run_id: str, modality: str, clf_name: str,
           fold_metrics_path: str) -> dict:
    """
    10-fold stratified CV with per-fold minority upsampling.

    Matches reference model_run() behaviour:
      - Best model tracked by ROC-AUC and saved to disk
      - Best model reloaded before each subsequent fold
      - Final full-dataset evaluation uses the best-ever model

    Extends the reference by additionally computing AUPRC per fold/overall.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, f'{clf_name}_best_model.pkl')
    oof_path = os.path.join(MODELS_DIR, f'{clf_name}_oof_predictions.csv')

    # Separate patient IDs, features, labels
    feature_df = df.drop(columns=[df.columns[0], df.columns[-1]])
    X = feature_df.values.astype(np.float32)
    y = df[df.columns[-1]].values.astype(int)
    patient_ids = df[df.columns[0]].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)

    cum_cm   = np.zeros(4, dtype=np.float64)   # tn fp fn tp
    cum_roc  = 0.0
    cum_prc  = 0.0
    best_roc = 0.0

    oof_ids, oof_true, oof_pred, oof_prob, oof_folds = [], [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Per-fold upsampling on training split only (mirrors reference exactly)
        train_df = pd.DataFrame(X_train, columns=feature_df.columns)
        train_df['label'] = y_train
        train_df_up = upsample_minority(train_df)
        X_train_up = train_df_up.drop(columns=['label']).values
        y_train_up = train_df_up['label'].values

        model.fit(X_train_up, y_train_up)

        y_pred = model.predict(X_test)

        # Probability scores for AUROC/AUPRC
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        cm  = confusion_matrix(y_test, y_pred).ravel()   # tn fp fn tp
        roc = roc_auc_score(y_test, y_score)
        prc = average_precision_score(y_test, y_score)

        cum_cm  += cm
        cum_roc += roc
        cum_prc += prc

        oof_ids.extend(patient_ids[test_idx])
        oof_true.extend(y_test)
        oof_pred.extend(y_pred)
        oof_prob.extend(y_score)
        oof_folds.extend([fold_idx] * len(test_idx))

        eps = 1e-9
        tn, fp, fn, tp = cm
        fold_acc  = (tp + tn) / (tp + tn + fp + fn + eps)
        fold_prec = tp / (tp + fp + eps)
        fold_sens = tp / (tp + fn + eps)
        fold_f1   = 2 * fold_prec * fold_sens / (fold_prec + fold_sens + eps)

        print(f"    Fold {fold_idx:2d}/{n_folds} | "
              f"ROC-AUC={roc:.4f} | AUPRC={prc:.4f} | "
              f"TN={tn:.0f} FP={fp:.0f} FN={fn:.0f} TP={tp:.0f}")

        with open(fold_metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                run_id, modality, clf_name, fold_idx,
                round(roc, 4), round(prc, 4),
                int(tn), int(fp), int(fn), int(tp),
                round(fold_acc, 4), round(fold_prec, 4),
                round(fold_sens, 4), round(fold_f1, 4),
            ])

        # Save fold snapshot and track best model
        fold_path = os.path.join(MODELS_DIR, f'{clf_name}_fold_{fold_idx:02d}.pkl')
        pickle.dump(model, open(fold_path, 'wb'))

        if roc > best_roc:
            best_roc = roc
            pickle.dump(model, open(best_model_path, 'wb'))

        # Reload best-so-far model for next fold (matches reference logic)
        model = pickle.load(open(best_model_path, 'rb'))

    # Save out-of-fold predictions
    pd.DataFrame({
        'patient_id': oof_ids,
        'fold': oof_folds,
        'true_label': oof_true,
        'predicted_label': oof_pred,
        'prob_score': oof_prob,
    }).to_csv(oof_path, index=False)
    print(f"    OOF predictions → {oof_path}")

    # CV averages
    avg_cm  = np.round(cum_cm / n_folds, 4)
    avg_roc = cum_roc / n_folds
    avg_prc = cum_prc / n_folds

    tn, fp, fn, tp = avg_cm
    eps = 1e-9
    cv_acc  = (tp + tn) / (tp + tn + fp + fn + eps)
    cv_prec = tp / (tp + fp + eps)
    cv_sens = tp / (tp + fn + eps)
    cv_f1   = 2 * cv_prec * cv_sens / (cv_prec + cv_sens + eps)

    # Full-dataset evaluation with best model (mirrors reference)
    best_model = pickle.load(open(best_model_path, 'rb'))
    y_full_pred = best_model.predict(X)

    if hasattr(best_model, 'predict_proba'):
        y_full_score = best_model.predict_proba(X)[:, 1]
    else:
        y_full_score = best_model.decision_function(X)

    full_cm  = confusion_matrix(y, y_full_pred).ravel()
    full_roc = roc_auc_score(y, y_full_score)
    full_prc = average_precision_score(y, y_full_score)

    return {
        'cv_roc_auc':    round(avg_roc, 4),
        'cv_auprc':      round(avg_prc, 4),
        'cv_tn':         avg_cm[0],
        'cv_fp':         avg_cm[1],
        'cv_fn':         avg_cm[2],
        'cv_tp':         avg_cm[3],
        'cv_accuracy':   round(cv_acc, 4),
        'cv_precision':  round(cv_prec, 4),
        'cv_sensitivity':round(cv_sens, 4),
        'cv_f1':         round(cv_f1, 4),
        'full_roc_auc':  round(full_roc, 4),
        'full_auprc':    round(full_prc, 4),
        'full_tn':       full_cm[0],
        'full_fp':       full_cm[1],
        'full_fn':       full_cm[2],
        'full_tp':       full_cm[3],
    }


# ── Results I/O ───────────────────────────────────────────────────────────────

def _init_csv(path: str, header: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)


def append_result(path: str, modality: str, clf_name: str, metrics: dict):
    row = [
        modality, clf_name,
        metrics['cv_roc_auc'], metrics['cv_auprc'],
        metrics['cv_tn'],  metrics['cv_fp'],
        metrics['cv_fn'],  metrics['cv_tp'],
        metrics['full_roc_auc'], metrics['full_auprc'],
        metrics['full_tn'], metrics['full_fp'],
        metrics['full_fn'], metrics['full_tp'],
        metrics['cv_accuracy'], metrics['cv_precision'],
        metrics['cv_sensitivity'], metrics['cv_f1'],
    ]
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def print_summary(modality: str, clf_name: str, metrics: dict):
    print(f"\n  ── {modality} | {clf_name} ──────────────────────────────────")
    print(f"    CV  ROC-AUC   : {metrics['cv_roc_auc']:.4f}")
    print(f"    CV  AUPRC     : {metrics['cv_auprc']:.4f}")
    print(f"    CV  Accuracy  : {metrics['cv_accuracy']:.4f}")
    print(f"    CV  Precision : {metrics['cv_precision']:.4f}")
    print(f"    CV  Sensitivity:{metrics['cv_sensitivity']:.4f}")
    print(f"    CV  F1-score  : {metrics['cv_f1']:.4f}")
    print(f"    Full ROC-AUC  : {metrics['full_roc_auc']:.4f}")
    print(f"    Full AUPRC    : {metrics['full_auprc']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    results_file   = args.results_file
    fold_file      = os.path.join(RESULTS_DIR, f'ref_fold_metrics_{run_id}.csv')

    _init_csv(results_file, RESULTS_HEADER)
    _init_csv(fold_file,    FOLD_HEADER)

    # Load data
    df = load_feature_csv(args.input_path)

    # Rename last column to 'label' for uniformity
    df = df.rename(columns={df.columns[-1]: 'label'})

    # Validate labels
    label_vals = df['label'].unique()
    print(f"Label values: {sorted(label_vals)}")
    df = df[df['label'].isin([0, 1])].reset_index(drop=True)

    n_total  = len(df)
    n_pos    = (df['label'] == 1).sum()
    n_neg    = (df['label'] == 0).sum()
    n_feat   = df.shape[1] - 2   # minus id and label

    print(f"\nDataset : {n_total} patients | {n_feat} features")
    print(f"  Positive (label=1) : {n_pos}")
    print(f"  Negative (label=0) : {n_neg}")
    print(f"  Imbalance ratio    : {max(n_pos, n_neg) / min(n_pos, n_neg):.2f}")

    modality = args.modality_name
    n_folds  = 3 if args.test_mode else N_FOLDS
    n_est    = 5 if args.test_mode else 10

    classifiers = get_classifiers(n_estimators=n_est)

    print(f"\nRunning {len(classifiers)} classifiers  |  {n_folds}-fold stratified CV\n")

    # Save run manifest
    manifest = {
        'run_id':      run_id,
        'timestamp':   datetime.datetime.now().isoformat(),
        'input_path':  args.input_path,
        'modality':    modality,
        'n_patients':  int(n_total),
        'n_features':  int(n_feat),
        'n_positive':  int(n_pos),
        'n_negative':  int(n_neg),
        'n_folds':     n_folds,
        'random_state':RANDOM_STATE,
        'test_mode':   args.test_mode,
        'results_file':results_file,
        'fold_file':   fold_file,
    }
    manifest_path = os.path.join(RESULTS_DIR, f'ref_manifest_{run_id}.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Run manifest → {manifest_path}\n")

    for clf_name, clf in classifiers:
        print(f"\n{'='*60}")
        print(f"  Classifier : {clf_name}")
        print(f"{'='*60}")
        metrics = run_cv(clf, df, n_folds=n_folds,
                         run_id=run_id, modality=modality,
                         clf_name=clf_name,
                         fold_metrics_path=fold_file)
        print_summary(modality, clf_name, metrics)
        append_result(results_file, modality, clf_name, metrics)

    print(f"\n{'='*60}")
    print(f"Results saved to    : {results_file}")
    print(f"Per-fold metrics    : {fold_file}")
    print(f"Best models         : {MODELS_DIR}/")
    print(f"Run manifest        : {manifest_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replicate reference ML classifier pipeline on CNV VAE features.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_path', type=str,
        default='reference_files/vae_features_cnv.csv',
        help='Path to the reference VAE feature CSV.')

    parser.add_argument(
        '--modality_name', type=str,
        default='cnv',
        help='Short name for the modality (used in result rows).')

    parser.add_argument(
        '--results_file', type=str,
        default='results/reference_original_results.csv',
        help='Output CSV for classification results.')

    parser.add_argument(
        '--test_mode', action='store_true',
        help='Use 3-fold CV and small RF for a quick smoke-test.')

    args = parser.parse_args()
    main(args)
