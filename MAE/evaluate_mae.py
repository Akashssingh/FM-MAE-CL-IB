"""
Linear probe evaluation of MAE representations.

Loads mae_representations.csv output(s) from train_mae.py and evaluates
how well the learned representations support 5-year OS survival prediction
using a logistic regression classifier (linear probe) with 10-fold
stratified cross-validation.

The linear probe is intentionally simple: if the MAE learned meaningful
gene-correlation structure, it will be encoded in the representation even
without task-specific fine-tuning.

Usage — evaluate one mask ratio:
    python MAE/evaluate_mae.py \\
        --repr_path MAE/outputs/mask15/mae_representations.csv

Usage — compare across all ablation mask ratios:
    python MAE/evaluate_mae.py --ablation_dir MAE/outputs

Results are printed to stdout and written to
MAE/outputs/probe_results_<timestamp>.csv.
"""

import argparse
import datetime
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTDIR  = 'MAE/outputs'
DEFAULT_N_FOLDS = 10
DEFAULT_SEED    = 42


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def linear_probe(
    representations_path: str,
    n_folds: int = DEFAULT_N_FOLDS,
    seed:    int = DEFAULT_SEED,
) -> dict:
    """
    Run a logistic regression linear probe on one mae_representations.csv.

    Returns a dict with mean ± std of accuracy, f1, sensitivity,
    precision, and roc_auc across folds.
    """
    df = pd.read_csv(representations_path)

    # Filter out patients with missing labels.
    df = df[df['label'] != -1].reset_index(drop=True)

    repr_cols = [c for c in df.columns if c.startswith('mae_repr_')]
    X = df[repr_cols].values.astype(np.float32)
    y = df['label'].values.astype(int)

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    print(f"  Patients : {len(y)}  |  label=1 (short-term): {n_pos}  "
          f"|  label=0 (long-term): {n_neg}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics = {
        'accuracy': [], 'f1': [], 'sensitivity': [],
        'precision': [], 'roc_auc': [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Standardise inside the fold to prevent leakage.
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        clf = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        fold_metrics['accuracy'].append(accuracy_score(y_te, y_pred))
        fold_metrics['f1'].append(f1_score(y_te, y_pred, zero_division=0))
        fold_metrics['sensitivity'].append(
            recall_score(y_te, y_pred, zero_division=0)
        )
        fold_metrics['precision'].append(
            precision_score(y_te, y_pred, zero_division=0)
        )
        fold_metrics['roc_auc'].append(roc_auc_score(y_te, y_prob))

    result = {'repr_path': representations_path}
    for metric, values in fold_metrics.items():
        result[f'{metric}_mean'] = float(np.mean(values))
        result[f'{metric}_std']  = float(np.std(values))

    return result


def _fmt(result: dict, metric: str) -> str:
    return (
        f"{result[f'{metric}_mean']:.3f} "
        f"± {result[f'{metric}_std']:.3f}"
    )


def print_result(result: dict, label: str = ''):
    prefix = f"[{label}] " if label else ''
    print(
        f"{prefix}"
        f"F1={_fmt(result, 'f1')}  "
        f"Acc={_fmt(result, 'accuracy')}  "
        f"Sens={_fmt(result, 'sensitivity')}  "
        f"Prec={_fmt(result, 'precision')}  "
        f"AUC={_fmt(result, 'roc_auc')}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Linear probe evaluation of MAE representations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--repr_path',
        help='Path to a single mae_representations.csv',
    )
    group.add_argument(
        '--ablation_dir',
        help='Root MAE output directory; discovers all mae_representations.csv '
             'files across mask subdirectories and compares them',
    )
    parser.add_argument('--n_folds',    type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument('--seed',       type=int, default=DEFAULT_SEED)
    parser.add_argument(
        '--output_dir', default=DEFAULT_OUTDIR,
        help='Where to write probe_results_<timestamp>.csv',
    )
    args = parser.parse_args()

    if args.repr_path:
        paths = [args.repr_path]
    else:
        pattern = os.path.join(args.ablation_dir, '*', 'mae_representations.csv')
        paths   = sorted(glob.glob(pattern))
        if not paths:
            print(f"No mae_representations.csv files found under {args.ablation_dir}")
            sys.exit(1)
        print(f"Found {len(paths)} representation file(s).\n")

    all_results = []
    for path in paths:
        # Infer mask ratio from parent directory name (e.g. mask15 -> 15%).
        mask_tag = os.path.basename(os.path.dirname(path))
        print(f"{'─'*64}")
        print(f"  {mask_tag}  |  {path}")
        result = linear_probe(path, n_folds=args.n_folds, seed=args.seed)
        result['mask_tag'] = mask_tag
        print_result(result, label=mask_tag)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*64}")
        print("Ablation summary — sorted by F1:")
        df_res = pd.DataFrame(all_results).sort_values(
            'f1_mean', ascending=False
        )
        cols = ['mask_tag', 'f1_mean', 'f1_std', 'roc_auc_mean',
                'sensitivity_mean', 'accuracy_mean']
        print(df_res[cols].to_string(index=False))

    # Save results CSV.
    os.makedirs(args.output_dir, exist_ok=True)
    ts          = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path    = os.path.join(args.output_dir, f'probe_results_{ts}.csv')
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nProbe results written to {out_path}")


if __name__ == '__main__':
    main()
