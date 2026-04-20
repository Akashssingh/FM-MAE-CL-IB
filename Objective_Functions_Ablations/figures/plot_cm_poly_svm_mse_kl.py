#!/usr/bin/env python3
"""
plot_cm_poly_svm_mse_kl.py
===========================
Research-quality per-fold confusion-matrix figure.

  Classifier : Poly SVM
  Objective  : MSE (L2) + KL divergence
  Folds      : all 10 (or however many exist in the CSV)

Layout per fold row (3-column inner grid — zero overlaps)
---------------------------------------------------------
  [Fold label col]  |  2×2 CM (pink→red)  |  1×5 metrics (white→blue)
                                              Accuracy · Precision · Recall · AUROC · AUPRC

Usage (from project root)
--------------------------
    python Objective_Functions_Ablations/figures/plot_cm_poly_svm_mse_kl.py
    python Objective_Functions_Ablations/figures/plot_cm_poly_svm_mse_kl.py \\
        --csv Objective_Functions_Ablations/results/betavalue=0.001/fold_metrics_mse_kl_<ts>.csv
    python Objective_Functions_Ablations/figures/plot_cm_poly_svm_mse_kl.py --out my_fig.png

Output
------
    figures/confusion_matrix/poly_svm_mse_kl_all_folds.png  (300 DPI)
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── Colour maps ───────────────────────────────────────────────────────────────

CM_CMAP = LinearSegmentedColormap.from_list(
    "cm_pink_red",
    [(0.00, "#FFFFFF"),
     (0.20, "#FFCDD2"),
     (0.50, "#EF9A9A"),
     (0.75, "#E53935"),
     (1.00, "#7B0000")],
)

MET_CMAP = LinearSegmentedColormap.from_list(
    "met_blue",
    [(0.00, "#FFFFFF"),
     (0.20, "#BBDEFB"),
     (0.50, "#64B5F6"),
     (0.75, "#1565C0"),
     (1.00, "#0D2358")],
)

MET_COLS    = ["accuracy", "precision", "recall", "f1", "roc_auc", "auprc"]
MET_HDRS    = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"]
CLASS_NAMES = ["Long-term", "Short-term"]
FONT        = 12          # universal label/text font size
FONT_KW     = "bold"      # universal font weight


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cm_text_color(value: float, vmax: float, threshold: float = 0.52) -> str:
    """White text on dark CM cells, black on light ones."""
    return "white" if (value / (vmax + 1e-9)) > threshold else "black"


def _draw_fold_row(fig, gs_row, fold_num, tn, fp, fn, tp,
                   metrics_1d, show_headers):
    """
    Three-column inner grid per fold row:
      [blank fold-label col]  |  [2×2 CM heatmap]  |  [partial-fill metric bars]

    The blank label column keeps the 'Fold N' text inside its own axes so it
    never clips over the CM axes.  The metric panel uses individually-drawn
    Rectangle patches so each cell is filled only to the fraction of its value
    (0-to-1), like a horizontal progress bar.
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_row,
        width_ratios=[0.75, 2.0, 4.6],
        wspace=0.45,
    )

    # ── 1. Fold label (invisible axes) ───────────────────────────────────
    ax_lbl = fig.add_subplot(inner[0])
    ax_lbl.axis("off")
    ax_lbl.text(
        1.0, 0.5, f"Fold {fold_num}",
        ha="right", va="center",
        fontsize=FONT, fontweight=FONT_KW,
        transform=ax_lbl.transAxes,
    )

    # ── 2. Confusion matrix (full-cell heatmap) ───────────────────────────
    cm      = np.array([[tn, fp], [fn, tp]], dtype=float)
    cm_vmax = max(cm.max(), 1.0)

    ax_cm = fig.add_subplot(inner[1])
    ax_cm.imshow(cm, cmap=CM_CMAP, vmin=0, vmax=cm_vmax, aspect="auto")

    for r in range(2):
        for c in range(2):
            tc = _cm_text_color(cm[r, c], cm_vmax)
            ax_cm.text(c, r, str(int(cm[r, c])),
                       ha="center", va="center",
                       fontsize=FONT, fontweight=FONT_KW, color=tc)

    ax_cm.set_yticks([0, 1])
    ax_cm.set_yticklabels(CLASS_NAMES, fontsize=FONT, fontweight=FONT_KW)
    ax_cm.tick_params(axis="y", length=0, pad=4)

    ax_cm.set_xticks([0, 1])
    ax_cm.xaxis.tick_top()
    ax_cm.tick_params(axis="x", length=0, pad=3)
    if show_headers:
        ax_cm.set_xticklabels(CLASS_NAMES, fontsize=FONT, fontweight=FONT_KW)
        ax_cm.xaxis.set_label_position("top")
        ax_cm.set_xlabel("Predicted", fontsize=FONT, labelpad=6,
                         fontweight=FONT_KW)
    else:
        ax_cm.set_xticklabels(["", ""])

    for sp in ax_cm.spines.values():
        sp.set_linewidth(0.5)

    # ── 3. Metric partial-fill bars (1 row × 5 cells) ────────────────────
    N = len(metrics_1d)   # 5

    ax_met = fig.add_subplot(inner[2])
    ax_met.set_xlim(0, N)
    ax_met.set_ylim(-0.5, 0.5)
    ax_met.set_xticks([])
    ax_met.set_yticks([])
    for sp in ax_met.spines.values():
        sp.set_visible(False)

    for c, val in enumerate(metrics_1d):
        val = float(np.clip(val, 0.0, 1.0))

        # white full-cell background + border
        ax_met.add_patch(plt.Rectangle(
            (c, -0.5), 1.0, 1.0,
            facecolor="white", edgecolor="#999999", linewidth=0.7,
            zorder=1,
        ))
        # partial fill — width proportional to metric value
        if val > 0.0:
            ax_met.add_patch(plt.Rectangle(
                (c, -0.5), val, 1.0,
                facecolor=MET_CMAP(val), edgecolor="none",
                zorder=2,
            ))
        # value text — white when fill passes cell centre, else black
        tc = "white" if val > 0.65 else "black"
        ax_met.text(
            c + 0.5, 0.0, f"{val:.4f}",
            ha="center", va="center",
            fontsize=FONT, fontweight=FONT_KW, color=tc,
            zorder=3,
        )

    # metric column headers — drawn above the axes in axes-fraction coords
    # (clip_on=False so they appear outside the ylim boundary)
    if show_headers:
        for c, hdr in enumerate(MET_HDRS):
            ax_met.text(
                (c + 0.5) / N, 1.06, hdr,
                ha="center", va="bottom",
                fontsize=FONT, fontweight=FONT_KW, color="black",
                transform=ax_met.transAxes,
                clip_on=False,
            )


# ── CSV auto-detection ────────────────────────────────────────────────────────

def _find_latest_csv(results_root):
    pattern = os.path.join(results_root, "**/fold_metrics_mse_kl_*.csv")
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[-1] if matches else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    here         = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(here, "..", "results")

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    csv_path = args.csv or _find_latest_csv(results_root)
    if csv_path is None:
        sys.exit("ERROR: No fold_metrics_mse_kl_*.csv found. Pass --csv.")
    print(f"CSV: {csv_path}")

    df = (pd.read_csv(csv_path)
            .query("classifier == 'poly_svm' and loss_fn == 'mse' and regularizer == 'kl'")
            .sort_values("fold")
            .reset_index(drop=True))

    if df.empty:
        sys.exit("ERROR: No poly_svm / mse / kl rows found.")

    n_folds = len(df)

    # ── figure layout ─────────────────────────────────────────────────────
    ROW_H = 1.55
    FIG_W = 15.0
    FIG_H = ROW_H * n_folds + 0.50   # extra headroom for top metric headers

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(
        n_folds, 1, figure=fig,
        hspace=0.30,
        left=0.01, right=0.99,
        top=0.985, bottom=0.01,
    )

    for i, row in df.iterrows():
        metrics_1d = np.array([row[c] for c in MET_COLS], dtype=float)
        _draw_fold_row(
            fig         = fig,
            gs_row      = outer[i],
            fold_num    = int(row["fold"]),
            tn          = int(row["tn"]),
            fp          = int(row["fp"]),
            fn          = int(row["fn"]),
            tp          = int(row["tp"]),
            metrics_1d  = metrics_1d,
            show_headers= (i == 0),
        )

    # ── save ──────────────────────────────────────────────────────────────
    out_path = args.out
    if out_path is None:
        cm_dir = os.path.join(here, "confusion_matrix")
        os.makedirs(cm_dir, exist_ok=True)
        out_path = os.path.join(cm_dir, "poly_svm_mse_kl_all_folds.png")

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
