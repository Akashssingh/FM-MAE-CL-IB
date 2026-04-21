"""
Generate Persistent 10-Fold Stratified CV Splits — mRNA modality
=================================================================
Run this script ONCE before starting any mRNA ablation experiments.

Splits are keyed by patient ID, not row index, so they remain valid
regardless of the order or subset of patients in each ablation's feature CSV.

Source data: data/processed/raw_features_mrna.csv
  - Our own preprocessed 500-gene mRNA features with survival labels.

Usage:
  cd /home/a/akashsingh/FM-MAE-CL-IB
  python Objective_Functions_Ablations/mRNA/generate_splits.py

Output:
  Objective_Functions_Ablations/mRNA/splits/splits_10fold.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(SCRIPT_DIR, 'splits')

DEFAULT_REF = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'processed',
                            'raw_features_mrna.csv')


def main(args):
    os.makedirs(SPLITS_DIR, exist_ok=True)

    out_path = os.path.join(SPLITS_DIR, args.output_name)
    if os.path.exists(out_path) and not args.overwrite:
        print(f"Splits file already exists: {out_path}")
        print("Use --overwrite to regenerate.")
        return

    print(f"Loading data from: {args.input_path}")
    df = pd.read_csv(args.input_path, header=0, index_col=None, low_memory=False)

    patient_ids = df.iloc[:, 0].values
    labels      = df.iloc[:, -1].values

    valid_mask = np.isin(labels, [0, 1])
    n_dropped  = (~valid_mask).sum()
    if n_dropped:
        print(f"  Dropped {n_dropped} patients with label not in {{0,1}}")
    patient_ids = patient_ids[valid_mask]
    labels      = labels[valid_mask].astype(int)

    n_total = len(patient_ids)
    counts  = np.bincount(labels)
    print(f"  Patients : {n_total}")
    print(f"  Label=0  : {counts[0]}  (long-term survival / negative)")
    print(f"  Label=1  : {counts[1]}  (short-term survival / positive)")
    print(f"  Imbalance: {max(counts)/min(counts):.2f}:1")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=False)

    folds = {}
    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(patient_ids, labels), start=1):
        folds[f"fold_{fold_idx}"] = {
            "train_ids":    patient_ids[train_idx].tolist(),
            "test_ids":     patient_ids[test_idx].tolist(),
            "train_labels": labels[train_idx].tolist(),
            "test_labels":  labels[test_idx].tolist(),
            "n_train":      int(len(train_idx)),
            "n_test":       int(len(test_idx)),
            "n_train_pos":  int(labels[train_idx].sum()),
            "n_test_pos":   int(labels[test_idx].sum()),
        }
        print(f"  Fold {fold_idx:2d}: "
              f"train={len(train_idx)} (pos={labels[train_idx].sum()})  "
              f"test={len(test_idx)}  (pos={labels[test_idx].sum()})")

    payload = {
        "n_folds":    args.n_folds,
        "n_patients": int(n_total),
        "n_label_0":  int(counts[0]),
        "n_label_1":  int(counts[1]),
        "source_file": os.path.abspath(args.input_path),
        "shuffle":    False,
        "note": (
            "Splits indexed by patient ID. "
            "ALL mRNA ablation runs must use these same splits. "
            "StratifiedKFold(shuffle=False) matches the reference ML pipeline."
        ),
        "folds": folds,
    }

    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"\nSplits saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate persistent 10-fold stratified CV splits for mRNA ablations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=str, default=DEFAULT_REF,
                        help='CSV with patient IDs (col 0) and labels (last col).')
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--output_name', type=str, default='splits_10fold.json')
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
