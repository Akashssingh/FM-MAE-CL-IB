"""
Dataset and masking utilities for CNV MAE pretraining.

The dataset loads raw_features_cnv.csv (output of prepare_cnv_data.py),
strips the label column, and returns gene value tensors as LongTensors.
Labels are retained as a convenience for downstream evaluation but are
not used during MAE pretraining.

Masking strategy
----------------
make_mask() performs exact random masking: for each sample in a batch,
exactly round(n_genes * mask_ratio) gene positions are selected uniformly
at random using a vectorised argsort. This guarantees a fixed number of
masked genes per sample (unlike Bernoulli sampling which has variance),
making per-epoch loss values directly comparable across mask ratios.

For CNV data, random masking is the appropriate starting point: it forces
the model to predict masked gene states from the full neighbourhood of
unmasked genes, and the Transformer's attention mechanism can discover
co-amplification/co-deletion structure from the resulting gradient signal
without any prior assumptions about chromosomal organisation.

Future masking strategies to consider:
  - Chromosomal block masking: mask all genes in a contiguous chromosomal
    segment to test whether inter-arm correlations are learned.
  - Correlation-grouped masking: mask genes from the same PCA component
    to force learning from weaker signals.
These can be swapped in by replacing make_mask() without touching the model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Columns that are not gene features.
_NON_GENE_COLS = {'submitter_id.samples', 'label', 'label_cnv'}
GISTIC_MIN, GISTIC_MAX = -2, 2


class CNVMAEDataset(Dataset):
    """
    PyTorch Dataset for CNV MAE pretraining.

    Parameters
    ----------
    csv_path  : str   Path to raw_features_cnv.csv.
    label_col : str   Name of the survival label column (default: 'label_cnv').
                      Patients with label == -1 are included in pretraining
                      because MAE is self-supervised and does not need labels.
    """

    def __init__(self, csv_path: str, label_col: str = 'label_cnv'):
        df = pd.read_csv(csv_path, index_col=0)

        gene_cols = [c for c in df.columns if c not in _NON_GENE_COLS]
        self.gene_cols = gene_cols
        self.n_genes   = len(gene_cols)
        self.patient_ids = df.index.tolist()

        # Clip to valid GISTIC2 range and store as int8 to save memory.
        X = df[gene_cols].values.astype(np.int8)
        X = np.clip(X, GISTIC_MIN, GISTIC_MAX)
        self.X = X

        # Labels retained for evaluation convenience.
        if label_col in df.columns:
            self.labels = df[label_col].values.astype(np.int32)
        else:
            self.labels = np.full(len(df), -1, dtype=np.int32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        genes = torch.tensor(self.X[idx], dtype=torch.long)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return genes, label


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def make_mask(
    batch_size: int,
    n_genes:    int,
    mask_ratio: float,
    device:     torch.device,
) -> torch.Tensor:
    """
    Generate an exact random boolean mask.

    Exactly round(n_genes * mask_ratio) gene positions are masked per sample.
    Uses a vectorised argsort over uniform noise so there is no Python loop.

    Parameters
    ----------
    batch_size : int
    n_genes    : int
    mask_ratio : float   Fraction of genes to mask, e.g. 0.15 for 15%.
    device     : torch.device

    Returns
    -------
    mask : BoolTensor [batch_size, n_genes]
        True at positions to be masked (hidden from value embedding).
    """
    n_mask = max(1, round(n_genes * mask_ratio))
    # Draw uniform noise, sort indices, take the first n_mask per sample.
    noise   = torch.rand(batch_size, n_genes, device=device)
    ids     = noise.argsort(dim=1)[:, :n_mask]              # [B, n_mask]
    mask    = torch.zeros(batch_size, n_genes, dtype=torch.bool, device=device)
    mask.scatter_(1, ids, True)
    return mask
