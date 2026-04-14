"""
MAE pretraining script for CNV tabular genomic data.

Usage — single mask ratio (default 15%):
    python MAE/train_mae.py

    python MAE/train_mae.py --mask_ratio 0.30 --epochs 100

Usage — ablation sweep (10% to 90% in 5% increments):
    python MAE/train_mae.py --ablation --epochs 50

All outputs are written under MAE/outputs/<mask_tag>/ where
<mask_tag> is e.g. mask15, mask30, etc.

For each mask ratio the script saves:
  - best_model.pt            : model state dict at best validation loss
  - train_metrics.csv        : per-epoch train/val loss and accuracy
  - mae_representations.csv  : d_model-dim patient representations extracted
                               with the best checkpoint (no masking applied)

After all runs a summary JSON is written to MAE/outputs/summary_<run_id>.json.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Allow running as `python MAE/train_mae.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mae_model   import TabularMAE
from mae_dataset import CNVMAEDataset, make_mask

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT    = 'data/processed/raw_features_cnv.csv'
DEFAULT_OUTDIR   = 'MAE/outputs'
DEFAULT_MASK     = 0.15
DEFAULT_EPOCHS   = 100
DEFAULT_BATCH    = 64
DEFAULT_LR       = 3e-4
DEFAULT_D_MODEL      = 128
DEFAULT_N_HEADS      = 4
DEFAULT_N_LAYERS     = 4
DEFAULT_DEC_D_MODEL  = 64
DEFAULT_DEC_N_HEADS  = 2
DEFAULT_DEC_N_LAYERS = 2
DEFAULT_DROPOUT      = 0.1
DEFAULT_VAL_FRAC = 0.10

# Ablation ratios: 10% to 90% in 5% steps (17 values).
ABLATION_RATIOS = [round(r, 2) for r in np.arange(0.10, 0.95, 0.05)]


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    logits:  torch.Tensor,   # [B, G, 5]
    targets: torch.Tensor,   # [B, G]  shifted to [0,4]
    mask:    torch.Tensor,   # [B, G]  BoolTensor, True = masked
) -> torch.Tensor:
    """Cross-entropy computed only at masked positions."""
    B, G, C = logits.shape
    logits_flat  = logits.view(B * G, C)
    targets_flat = targets.view(B * G)
    mask_flat    = mask.view(B * G)
    return nn.functional.cross_entropy(
        logits_flat[mask_flat], targets_flat[mask_flat]
    )


def masked_accuracy(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    mask:    torch.Tensor,
) -> float:
    """Fraction of masked positions predicted correctly."""
    preds   = logits.argmax(dim=-1)         # [B, G]
    correct = (preds == targets) & mask
    return correct.sum().item() / max(mask.sum().item(), 1)


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model:      TabularMAE,
    loader:     DataLoader,
    mask_ratio: float,
    optimizer,
    device:     torch.device,
    training:   bool,
):
    model.train(training)
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(training):
        for genes, _ in loader:
            genes = genes.to(device)                         # [B, G]
            mask  = make_mask(genes.size(0), genes.size(1), mask_ratio, device)

            logits  = model(genes, mask)                     # [B, G, 5]
            targets = genes + 2                              # shift to [0,4]

            loss = masked_cross_entropy(logits, targets, mask)
            acc  = masked_accuracy(logits, targets, mask)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_acc  += acc
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_single(
    args,
    mask_ratio: float,
    dataset:    CNVMAEDataset,
    outdir:     str,
) -> dict:
    """Train one MAE with the given mask_ratio. Returns final metrics dict."""

    device = (
        torch.device('cuda') if args.device == 'cuda'
        else torch.device('cpu') if args.device == 'cpu'
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # ── Train / val split ─────────────────────────────────────────────
    n_val   = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = TabularMAE(
        n_genes      = dataset.n_genes,
        d_model      = args.d_model,
        n_heads      = args.n_heads,
        n_layers     = args.n_layers,
        dec_d_model  = args.dec_d_model,
        dec_n_heads  = args.dec_n_heads,
        dec_n_layers = args.dec_n_layers,
        dropout      = args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Output directory for this mask ratio ──────────────────────────
    mask_tag   = f"mask{int(mask_ratio * 100):02d}"
    run_outdir = os.path.join(outdir, mask_tag)
    os.makedirs(run_outdir, exist_ok=True)

    metrics_path   = os.path.join(run_outdir, 'train_metrics.csv')
    best_ckpt_path = os.path.join(run_outdir, 'best_model.pt')

    with open(metrics_path, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')

    n_masked = round(dataset.n_genes * mask_ratio)
    print(f"\n{'='*64}")
    print(f"  Mask ratio : {mask_ratio:.0%}  "
          f"({n_masked} / {dataset.n_genes} genes masked per sample)")
    print(f"  Device     : {device}  |  Params: {model.n_params():,}")
    print(f"  Epochs     : {args.epochs}  |  Batch: {args.batch_size}")
    print(f"  encoder    : d_model={args.d_model}  heads={args.n_heads}  layers={args.n_layers}")
    print(f"  decoder    : d_model={args.dec_d_model}  heads={args.dec_n_heads}  layers={args.dec_n_layers}")
    print(f"  Train/Val  : {n_train} / {n_val} patients")
    print(f"  Output     : {run_outdir}")
    print(f"{'='*64}")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, mask_ratio, optimizer, device, training=True
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, mask_ratio, optimizer, device, training=False
        )
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{args.epochs}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}"
            )

        with open(metrics_path, 'a') as f:
            f.write(
                f"{epoch},{tr_loss:.6f},{tr_acc:.6f},"
                f"{va_loss:.6f},{va_acc:.6f}\n"
            )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_ckpt_path)

    print(f"  Best val loss : {best_val_loss:.4f}  "
          f"checkpoint : {best_ckpt_path}")

    # ── Extract representations with the best checkpoint ──────────────
    model.load_state_dict(
        torch.load(best_ckpt_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_reprs  = []
    all_labels = []
    full_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    with torch.no_grad():
        for genes, labels in full_loader:
            rep = model.encode(genes.to(device))
            all_reprs.append(rep.cpu().numpy())
            all_labels.append(labels.numpy())

    reprs  = np.vstack(all_reprs)
    labels = np.concatenate(all_labels)

    repr_cols = [f'mae_repr_{i + 1}' for i in range(reprs.shape[1])]
    df_repr   = pd.DataFrame(reprs, columns=repr_cols)
    df_repr.insert(0, 'patient_id', dataset.patient_ids)
    df_repr.insert(1, 'label', labels)

    repr_path = os.path.join(run_outdir, 'mae_representations.csv')
    df_repr.to_csv(repr_path, index=False)
    print(f"  Representations : {repr_path}  shape={reprs.shape}")

    return {
        'mask_ratio':    mask_ratio,
        'mask_tag':      mask_tag,
        'best_val_loss': best_val_loss,
        'repr_path':     repr_path,
        'checkpoint':    best_ckpt_path,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MAE pretraining for CNV tabular genomic data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input_path', default=DEFAULT_INPUT,
        help='Path to raw_features_cnv.csv',
    )
    parser.add_argument(
        '--output_dir', default=DEFAULT_OUTDIR,
        help='Root directory for all MAE outputs',
    )
    parser.add_argument(
        '--mask_ratio', type=float, default=DEFAULT_MASK,
        help='Fraction of genes to mask (e.g. 0.15 for 15%%)',
    )
    parser.add_argument(
        '--ablation', action='store_true',
        help='Sweep mask ratios 10%%-90%% in 5%% increments',
    )
    parser.add_argument('--epochs',     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int,   default=DEFAULT_BATCH)
    parser.add_argument('--lr',         type=float, default=DEFAULT_LR)
    parser.add_argument('--d_model',      type=int,   default=DEFAULT_D_MODEL,      help='Encoder hidden dim')
    parser.add_argument('--n_heads',      type=int,   default=DEFAULT_N_HEADS,      help='Encoder attention heads')
    parser.add_argument('--n_layers',     type=int,   default=DEFAULT_N_LAYERS,     help='Encoder Transformer layers')
    parser.add_argument('--dec_d_model',  type=int,   default=DEFAULT_DEC_D_MODEL,  help='Decoder hidden dim')
    parser.add_argument('--dec_n_heads',  type=int,   default=DEFAULT_DEC_N_HEADS,  help='Decoder attention heads')
    parser.add_argument('--dec_n_layers', type=int,   default=DEFAULT_DEC_N_LAYERS, help='Decoder Transformer layers')
    parser.add_argument('--dropout',      type=float, default=DEFAULT_DROPOUT)
    parser.add_argument(
        '--val_frac', type=float, default=DEFAULT_VAL_FRAC,
        help='Fraction of data held out for validation',
    )
    parser.add_argument(
        '--device', default='auto', choices=['auto', 'cpu', 'cuda'],
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────
    print(f"Loading dataset from {args.input_path} ...")
    dataset = CNVMAEDataset(args.input_path)
    print(f"  {len(dataset)} patients  |  {dataset.n_genes} genes")

    # ── Determine mask ratios to run ──────────────────────────────────
    ratios = ABLATION_RATIOS if args.ablation else [args.mask_ratio]

    run_id      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    for ratio in ratios:
        result = train_single(args, ratio, dataset, args.output_dir)
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        'run_id':      run_id,
        'input_path':  args.input_path,
        'n_patients':  len(dataset),
        'n_genes':     dataset.n_genes,
        'epochs':      args.epochs,
        'd_model':     args.d_model,
        'n_heads':     args.n_heads,
        'n_layers':    args.n_layers,
        'ablation':    args.ablation,
        'results':     all_results,
    }
    summary_path = os.path.join(args.output_dir, f'summary_{run_id}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    if args.ablation:
        print("\nAblation summary:")
        print(f"  {'mask_ratio':>12}  {'best_val_loss':>14}")
        for r in all_results:
            print(f"  {r['mask_ratio']:>12.0%}  {r['best_val_loss']:>14.4f}")


if __name__ == '__main__':
    main()
