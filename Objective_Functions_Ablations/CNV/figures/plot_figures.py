"""
Figure Generation — AUROC, AUPRC, Confusion Matrix
====================================================
Loads per-fold curve data saved by run_ablation.py and generates
publication-quality figures.

Outputs:
  figures/roc/             — AUROC curve PNGs
  figures/prc/             — AUPRC curve PNGs
  figures/confusion_matrix/— Confusion matrix heatmap PNGs

Usage:
  # Single run (all classifiers):
  python Objective_Functions_Ablations/figures/plot_figures.py \\
      --run_dir Objective_Functions_Ablations/outputs/logcosh_kl_20260420_...

  # Compare all 4 ablation runs (overlay on one plot per classifier):
  python Objective_Functions_Ablations/figures/plot_figures.py \\
      --compare \\
      --run_dirs \\
          Objective_Functions_Ablations/outputs/logcosh_kl_... \\
          Objective_Functions_Ablations/outputs/mse_kl_...    \\
          Objective_Functions_Ablations/outputs/logcosh_mmd_... \\
          Objective_Functions_Ablations/outputs/mse_mmd_...

  # Compare, specifying a single classifier only:
  python Objective_Functions_Ablations/figures/plot_figures.py \\
      --compare --classifier rbf_svm \\
      --run_dirs ...
"""

import os
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for cluster nodes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────────
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))   # figures/

ALL_CLASSIFIERS = [
    'rbf_svm', 'linear_svm', 'poly_svm', 'sigmoid_svm', 'random_forest',
]

# Colour palette — distinct enough for 4–5 overlapping curves
PALETTE = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#9467bd',  # purple
]


# =============================================================================
#  Helpers
# =============================================================================

def load_manifest(run_dir: str) -> dict:
    path = os.path.join(run_dir, 'manifest.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")
    with open(path) as f:
        return json.load(f)


def run_label(manifest: dict) -> str:
    """Short descriptive label for plot legends."""
    loss = manifest.get('loss_fn', '?')
    reg  = manifest.get('regularizer', '?')
    if reg == 'kl':
        w = manifest.get('kl_weight', '')
        return f"{loss} + KL (β={w})"
    else:
        w = manifest.get('mmd_weight', '')
        return f"{loss} + MMD (λ={w})"


def _load_npz(path: str) -> dict:
    return dict(np.load(path, allow_pickle=True))


def _n_folds_in_npz(data: dict, prefix: str) -> int:
    return sum(1 for k in data if k.startswith(prefix))


def _prevalence_from_manifest(manifest: dict) -> float:
    n0 = manifest.get('n_label_0', 0)
    n1 = manifest.get('n_label_1', 0)
    if (n0 + n1) > 0:
        return n1 / (n0 + n1)
    return 0.77  # fallback: approximate BRCA class balance


def _fig_path(subdir: str, fname: str) -> str:
    d = os.path.join(FIGURES_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


# =============================================================================
#  AUROC curve — single run
# =============================================================================

def plot_roc_single(run_dir: str, clf_name: str, manifest: dict,
                    out_path: str) -> None:
    data_path = os.path.join(run_dir, f'{clf_name}_roc_curves.npz')
    if not os.path.exists(data_path):
        print(f"    [skip] {clf_name}_roc_curves.npz not found")
        return

    data     = _load_npz(data_path)
    n_folds  = _n_folds_in_npz(data, 'fpr_fold')
    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []

    for i in range(1, n_folds + 1):
        fpr = data.get(f'fpr_fold{i}')
        tpr = data.get(f'tpr_fold{i}')
        auc = data.get(f'auc_fold{i}')
        if fpr is None:
            continue
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
        aucs.append(float(auc[0]))

    if not tprs:
        return

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs,  axis=0)
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_fpr, mean_tpr, color='steelblue', lw=2,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.2, color='steelblue', label='±1 std (10 folds)')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.50)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'AUROC — {clf_name}\n{run_label(manifest)}', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")


# =============================================================================
#  AUPRC curve — single run
# =============================================================================

def plot_prc_single(run_dir: str, clf_name: str, manifest: dict,
                    out_path: str) -> None:
    data_path = os.path.join(run_dir, f'{clf_name}_prc_curves.npz')
    if not os.path.exists(data_path):
        print(f"    [skip] {clf_name}_prc_curves.npz not found")
        return

    data        = _load_npz(data_path)
    n_folds     = _n_folds_in_npz(data, 'precision_fold')
    mean_recall = np.linspace(0, 1, 200)
    precs, aps  = [], []

    for i in range(1, n_folds + 1):
        prec = data.get(f'precision_fold{i}')
        rec  = data.get(f'recall_fold{i}')
        ap   = data.get(f'ap_fold{i}')
        if prec is None:
            continue
        # sklearn returns recall in decreasing order; sort ascending
        idx    = np.argsort(rec)
        interp = np.interp(mean_recall, rec[idx], prec[idx])
        precs.append(interp)
        aps.append(float(ap[0]))

    if not precs:
        return

    mean_prec  = np.mean(precs, axis=0)
    std_prec   = np.std(precs,  axis=0)
    mean_ap    = np.mean(aps)
    std_ap     = np.std(aps)
    prevalence = _prevalence_from_manifest(manifest)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_recall, mean_prec, color='darkorange', lw=2,
            label=f'Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})')
    ax.fill_between(mean_recall,
                    np.clip(mean_prec - std_prec, 0, 1),
                    np.clip(mean_prec + std_prec, 0, 1),
                    alpha=0.2, color='darkorange', label='±1 std (10 folds)')
    ax.axhline(prevalence, color='k', ls='--', lw=1,
               label=f'No-skill (prevalence={prevalence:.2f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'AUPRC — {clf_name}\n{run_label(manifest)}', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")


# =============================================================================
#  Confusion matrix heatmap — single run
# =============================================================================

def plot_cm_single(run_dir: str, clf_name: str, manifest: dict,
                   out_path: str) -> None:
    cm_path = os.path.join(run_dir, f'{clf_name}_confusion_matrices.npy')
    if not os.path.exists(cm_path):
        print(f"    [skip] {clf_name}_confusion_matrices.npy not found")
        return

    cms     = np.load(cm_path)          # (n_folds, 2, 2)
    mean_cm = cms.mean(axis=0)
    std_cm  = cms.std(axis=0)

    labels = ['Long-term\n(label=0)', 'Short-term\n(label=1)']
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mean_cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted label', fontsize=11)
    ax.set_ylabel('True label', fontsize=11)

    thresh = mean_cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f'{mean_cm[i, j]:.1f}\n±{std_cm[i, j]:.1f}',
                    ha='center', va='center', fontsize=10,
                    color='white' if mean_cm[i, j] > thresh else 'black')

    ax.set_title(
        f'Confusion Matrix (mean ± std, {cms.shape[0]} folds)\n'
        f'{clf_name} | {run_label(manifest)}', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")


# =============================================================================
#  Comparison plots — overlay mean curves from multiple runs
# =============================================================================

def _interp_roc(run_dir: str, clf_name: str, mean_fpr: np.ndarray):
    """Return (mean_tpr, mean_auc) interpolated at mean_fpr, or None."""
    data_path = os.path.join(run_dir, f'{clf_name}_roc_curves.npz')
    if not os.path.exists(data_path):
        return None, None
    data    = _load_npz(data_path)
    n_folds = _n_folds_in_npz(data, 'fpr_fold')
    tprs, aucs = [], []
    for i in range(1, n_folds + 1):
        fpr = data.get(f'fpr_fold{i}')
        tpr = data.get(f'tpr_fold{i}')
        auc = data.get(f'auc_fold{i}')
        if fpr is None:
            continue
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
        aucs.append(float(auc[0]))
    if not tprs:
        return None, None
    return np.mean(tprs, axis=0), np.mean(aucs)


def _interp_prc(run_dir: str, clf_name: str, mean_recall: np.ndarray):
    """Return (mean_prec, mean_ap) interpolated at mean_recall, or None."""
    data_path = os.path.join(run_dir, f'{clf_name}_prc_curves.npz')
    if not os.path.exists(data_path):
        return None, None
    data    = _load_npz(data_path)
    n_folds = _n_folds_in_npz(data, 'precision_fold')
    precs, aps = [], []
    for i in range(1, n_folds + 1):
        prec = data.get(f'precision_fold{i}')
        rec  = data.get(f'recall_fold{i}')
        ap   = data.get(f'ap_fold{i}')
        if prec is None:
            continue
        idx = np.argsort(rec)
        precs.append(np.interp(mean_recall, rec[idx], prec[idx]))
        aps.append(float(ap[0]))
    if not precs:
        return None, None
    return np.mean(precs, axis=0), np.mean(aps)


def plot_roc_compare(run_dirs: list, clf_name: str, out_path: str) -> None:
    mean_fpr = np.linspace(0, 1, 200)
    fig, ax  = plt.subplots(figsize=(7, 6))

    for idx, run_dir in enumerate(run_dirs):
        try:
            manifest = load_manifest(run_dir)
        except FileNotFoundError:
            print(f"  [skip] manifest not found in {run_dir}")
            continue
        mean_tpr, mean_auc = _interp_roc(run_dir, clf_name, mean_fpr)
        if mean_tpr is None:
            print(f"  [skip] no ROC data for {clf_name} in {run_dir}")
            continue
        label = f"{run_label(manifest)} | AUC={mean_auc:.3f}"
        ax.plot(mean_fpr, mean_tpr,
                color=PALETTE[idx % len(PALETTE)], lw=2, label=label)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'AUROC Comparison — {clf_name}', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_prc_compare(run_dirs: list, clf_name: str, out_path: str) -> None:
    mean_recall = np.linspace(0, 1, 200)
    fig, ax     = plt.subplots(figsize=(7, 6))

    # Compute prevalence from first run with a valid manifest
    prevalence = 0.77
    for run_dir in run_dirs:
        try:
            prevalence = _prevalence_from_manifest(load_manifest(run_dir))
            break
        except FileNotFoundError:
            pass

    for idx, run_dir in enumerate(run_dirs):
        try:
            manifest = load_manifest(run_dir)
        except FileNotFoundError:
            continue
        mean_prec, mean_ap = _interp_prc(run_dir, clf_name, mean_recall)
        if mean_prec is None:
            print(f"  [skip] no PRC data for {clf_name} in {run_dir}")
            continue
        label = f"{run_label(manifest)} | AP={mean_ap:.3f}"
        ax.plot(mean_recall, mean_prec,
                color=PALETTE[idx % len(PALETTE)], lw=2, label=label)

    ax.axhline(prevalence, color='k', ls='--', lw=1,
               label=f'No-skill (prevalence={prevalence:.2f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'AUPRC Comparison — {clf_name}', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cm_compare(run_dirs: list, clf_name: str, out_dir: str) -> None:
    """Side-by-side confusion matrices for all runs."""
    manifests, cms_list = [], []
    for run_dir in run_dirs:
        try:
            manifest = load_manifest(run_dir)
        except FileNotFoundError:
            continue
        cm_path = os.path.join(run_dir, f'{clf_name}_confusion_matrices.npy')
        if not os.path.exists(cm_path):
            continue
        manifests.append(manifest)
        cms_list.append(np.load(cm_path))

    if not manifests:
        return

    n = len(manifests)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    labels = ['Long-term\n(0)', 'Short-term\n(1)']
    for ax, manifest, cms in zip(axes, manifests, cms_list):
        mean_cm = cms.mean(axis=0)
        std_cm  = cms.std(axis=0)
        im = ax.imshow(mean_cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        thresh = mean_cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{mean_cm[i, j]:.1f}\n±{std_cm[i, j]:.1f}',
                        ha='center', va='center', fontsize=9,
                        color='white' if mean_cm[i, j] > thresh else 'black')
        ax.set_title(run_label(manifest), fontsize=9)

    fig.suptitle(f'Confusion Matrices — {clf_name}', fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'compare_cm_{clf_name}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
#  Main
# =============================================================================

def main(args):
    classifiers = [args.classifier] if args.classifier else ALL_CLASSIFIERS

    if args.compare and args.run_dirs:
        # ── Comparison mode ───────────────────────────────────────────────────
        print(f"\nComparison mode: {len(args.run_dirs)} runs | "
              f"{len(classifiers)} classifiers")
        for clf in classifiers:
            print(f"\n  Classifier: {clf}")
            plot_roc_compare(
                args.run_dirs, clf,
                _fig_path('roc', f'compare_roc_{clf}.png'))
            plot_prc_compare(
                args.run_dirs, clf,
                _fig_path('prc', f'compare_prc_{clf}.png'))
            plot_cm_compare(
                args.run_dirs, clf,
                _fig_path('confusion_matrix', ''))

    elif args.run_dir:
        # ── Single-run mode ───────────────────────────────────────────────────
        manifest = load_manifest(args.run_dir)
        run_tag  = manifest.get('run_tag', os.path.basename(args.run_dir))
        print(f"\nSingle run: {run_tag}")
        print(f"  {run_label(manifest)}")

        for clf in classifiers:
            print(f"\n  Classifier: {clf}")
            plot_roc_single(
                args.run_dir, clf, manifest,
                _fig_path('roc', f'{run_tag}_{clf}_roc.png'))
            plot_prc_single(
                args.run_dir, clf, manifest,
                _fig_path('prc', f'{run_tag}_{clf}_prc.png'))
            plot_cm_single(
                args.run_dir, clf, manifest,
                _fig_path('confusion_matrix', f'{run_tag}_{clf}_cm.png'))

    else:
        print("Provide --run_dir for a single run or --compare --run_dirs ... "
              "for cross-run comparison.")
        print("Use --classifier to restrict to one classifier.")


# =============================================================================
#  CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate AUROC, AUPRC, and confusion matrix figures.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--run_dir', type=str, default=None,
                        help='Path to a single ablation run output directory.')
    parser.add_argument('--run_dirs', type=str, nargs='+', default=None,
                        help='Paths to multiple run dirs for comparison plots.')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison overlay plots across runs.')
    parser.add_argument('--classifier', type=str, default=None,
                        choices=ALL_CLASSIFIERS,
                        help='Restrict figures to one classifier.')

    args = parser.parse_args()
    main(args)
