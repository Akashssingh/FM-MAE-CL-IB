"""
Ablation Training Script — PyTorch VAE + ML Classifiers (mRNA modality)
========================================================================
Full pipeline for mRNA objective-function ablation experiments:
  1. Load preprocessed mRNA features  (data/processed/raw_features_mrna.csv)
  2. Min-Max scale features to [0,1]  (discretized -1/0/+1 → 0/0.5/1)
  3. Train PyTorch VAE  (architecture = Arya et al. 2023)
  4. Extract 32-dim latent features  (reparameterized z, matching author)
  5. Run 5 ML classifiers using pre-generated persistent 10-fold splits
  6. Record all metrics + save all data needed for figure regeneration

Key difference vs CNV VAE:
  The mRNA decoder output has NO sigmoid activation (reference script:
  layers.Dense(original_dim) — no activation). This is because mRNA
  features after discretization + MinMaxScaling are not strictly bounded
  like CNV GISTIC2 scores.

Ablation dimensions:
  Reconstruction loss : logcosh (reference) | mse | mae
  Latent regularizer  : kl (reference) | mmd (WAE-MMD)

Metrics recorded per fold and aggregated:
  Accuracy, F1, AUROC, AUPRC, Precision, Recall, Confusion matrix

Saved for figure regeneration (per run output dir):
  vae_training_log.csv          — epoch, total/recon/reg loss, LR
  vae_checkpoint.pt             — full model checkpoint
  vae_latent_features.csv       — extracted 32-dim latent features
  manifest.json                 — complete run configuration
  {clf}_oof.csv                 — out-of-fold predictions + probabilities
  {clf}_roc_curves.npz          — (fpr, tpr, auc) per fold
  {clf}_prc_curves.npz          — (precision, recall, ap) per fold
  {clf}_confusion_matrices.npy  — per-fold 2×2 confusion matrices

Usage:
  # Generate splits once:
  python Objective_Functions_Ablations/mRNA/generate_splits.py

  # logcosh + KL (reference objective):
  python Objective_Functions_Ablations/mRNA/run_ablation.py \\
      --run_name logcosh_kl --loss_fn logcosh --regularizer kl

  # MSE + MMD:
  python Objective_Functions_Ablations/mRNA/run_ablation.py \\
      --run_name mse_mmd --loss_fn mse --regularizer mmd

  # Quick smoke-test:
  python Objective_Functions_Ablations/mRNA/run_ablation.py \\
      --run_name smoke_test --test_mode
"""

import os
import sys
import csv
import json
import pickle
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

warnings.filterwarnings('ignore')

# ── Directory layout ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

SPLITS_FILE = os.path.join(SCRIPT_DIR, 'splits', 'splits_10fold.json')
RAW_DATA    = os.path.join(PROJECT_ROOT, 'data', 'processed',
                            'raw_features_mrna.csv')

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
LOGS_DIR    = os.path.join(SCRIPT_DIR, 'logs')
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, 'outputs')

# ── VAE hyperparameters (mirror reference Keras implementation) ───────────────
LATENT_DIM     = 32
BATCH_SIZE     = 32
EPOCHS         = 50
LEARNING_RATE  = 1e-3
LOSS_THRESHOLD = 0.01
DECAY_RATE     = 0.96
DECAY_STEPS    = 100_000

# ── ML hyperparameters ────────────────────────────────────────────────────────
N_FOLDS      = 10
RANDOM_STATE = 123

RESULTS_HEADER = [
    'run_name', 'loss_fn', 'regularizer', 'kl_weight', 'mmd_weight',
    'modality', 'classifier',
    'cv_roc_auc', 'cv_auprc', 'cv_accuracy',
    'cv_precision', 'cv_recall', 'cv_f1',
    'cv_tn', 'cv_fp', 'cv_fn', 'cv_tp',
    'full_roc_auc', 'full_auprc',
    'full_tn', 'full_fp', 'full_fn', 'full_tp',
]

FOLD_HEADER = [
    'run_name', 'loss_fn', 'regularizer', 'kl_weight', 'mmd_weight',
    'modality', 'classifier', 'fold',
    'roc_auc', 'auprc', 'accuracy', 'precision', 'recall', 'f1',
    'tn', 'fp', 'fn', 'tp',
]


# =============================================================================
#  VAE architecture  (exact match to reference Keras VAE)
#  IMPORTANT: decoder output has NO sigmoid — matches mrna_pca_vae_logcosh_
#             feature_extractor.py where the last Dense has no activation.
# =============================================================================

class VAE_mRNA(nn.Module):
    """
    VAE for mRNA data.  Architecture matches Arya et al. (2023):
      Encoder: input → Linear(256,Tanh) → Linear(128,Tanh) → Linear(64,Tanh)
               → [z_mean(32), z_log_var(32)]
      Decoder: z → Linear(64,Tanh) → Linear(128,Tanh) → Linear(256,Tanh)
               → Linear(input_dim)  ← NO sigmoid (unlike CNV)
    Xavier uniform init (matches Keras Glorot default for Tanh).
    """

    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(),
            nn.Linear(256, 128),       nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
        )
        self.fc_mu      = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.Tanh(),
            nn.Linear(64, 128),         nn.Tanh(),
            nn.Linear(128, 256),        nn.Tanh(),
            nn.Linear(256, input_dim),  # NO sigmoid — mRNA decoder
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z


# =============================================================================
#  Reconstruction loss functions  (ablation dimension 1)
# =============================================================================

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Numerically-stable log-cosh, matching tf.keras.losses.LogCosh()."""
    diff = (y_pred - y_true).clamp(-50, 50)
    return (diff.abs()
            + F.softplus(-2.0 * diff.abs())
            - torch.log(torch.tensor(2.0, device=y_pred.device))).mean()


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_pred, y_true)


def mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(y_pred, y_true)


LOSS_FNS = {'logcosh': log_cosh_loss, 'mse': mse_loss, 'mae': mae_loss}


# =============================================================================
#  Latent regularizers  (ablation dimension 2)
# =============================================================================

def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(log_var - mu.pow(2) - log_var.exp() + 1)


def mmd_rbf(z: torch.Tensor, z_prior: torch.Tensor,
            bandwidth: float = None) -> torch.Tensor:
    if bandwidth is None:
        bandwidth = float(z.shape[1])
    n = z.shape[0]

    def rbf(a, b):
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return torch.exp(-diff.pow(2).sum(-1) / (2.0 * bandwidth))

    kzz = rbf(z, z)
    kpp = rbf(z_prior, z_prior)
    kzp = rbf(z, z_prior)
    denom = max(float(n * (n - 1)), 1.0)
    mmd = ((kzz.sum() - kzz.trace()) / denom
           + (kpp.sum() - kpp.trace()) / denom
           - 2.0 * kzp.mean())
    return mmd.clamp(min=0.0)


def vae_loss(recon_x, x, mu, log_var, z,
             recon_fn, regularizer: str,
             kl_weight: float, mmd_weight: float):
    recon = recon_fn(recon_x, x)
    if regularizer == 'kl':
        reg = kl_weight * kl_divergence(mu, log_var)
    else:
        z_prior = torch.randn_like(z)
        reg = mmd_weight * mmd_rbf(z, z_prior)
    total = recon + reg
    return total, recon.item(), reg.item()


# =============================================================================
#  Lookahead optimizer  (mirrors tfa.optimizers.Lookahead from reference)
# =============================================================================

class Lookahead:
    def __init__(self, base_optimizer, k: int = 6, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        self._slow_weights = [
            [p.clone().detach() for p in group['params']]
            for group in base_optimizer.param_groups
        ]

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self):
        self.base_optimizer.step()
        self._step_count += 1
        if self._step_count % self.k == 0:
            for g_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, fast in enumerate(group['params']):
                    slow = self._slow_weights[g_idx][p_idx]
                    slow.data.add_(self.alpha * (fast.data - slow.data))
                    fast.data.copy_(slow.data)


# =============================================================================
#  VAE training
# =============================================================================

def train_vae(X_scaled: np.ndarray,
              loss_fn_name: str,
              regularizer: str,
              kl_weight: float,
              mmd_weight: float,
              epochs: int,
              device: torch.device,
              outputs_dir: str) -> np.ndarray:
    """
    Train mRNA VAE on MinMax-scaled features.
    Returns reparameterized z latent (N × LATENT_DIM).
    """
    recon_fn = LOSS_FNS[loss_fn_name]
    X_tensor = torch.FloatTensor(X_scaled)
    loader   = DataLoader(TensorDataset(X_tensor),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = VAE_mRNA(input_dim=X_scaled.shape[1], latent_dim=LATENT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  VAE_mRNA: input={X_scaled.shape[1]} → latent={LATENT_DIM} "
          f"| {n_params:,} params | recon={loss_fn_name} | reg={regularizer}")
    if regularizer == 'kl':
        print(f"       KL weight β={kl_weight}")
    else:
        print(f"       MMD weight λ={mmd_weight}")

    adam_base = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gamma     = DECAY_RATE ** (1.0 / DECAY_STEPS)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(adam_base, gamma=gamma)
    optimizer = Lookahead(adam_base)

    log_path = os.path.join(outputs_dir, 'vae_training_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'total_loss', 'recon_loss', 'reg_loss', 'lr'])

    final_epoch, avg_total = 1, float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        sum_total = sum_recon = sum_reg = 0.0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x, mu, lv, z = model(batch_x)
            loss, r_val, g_val = vae_loss(
                recon_x, batch_x, mu, lv, z,
                recon_fn, regularizer, kl_weight, mmd_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()
            sum_total += loss.item()
            sum_recon += r_val
            sum_reg   += g_val

        n_b       = len(loader)
        avg_total = sum_total / n_b
        avg_recon = sum_recon / n_b
        avg_reg   = sum_reg   / n_b
        lr        = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"total={avg_total:.5f}  recon={avg_recon:.5f}  "
              f"reg={avg_reg:.5f} | lr={lr:.2e}")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch, round(avg_total, 6), round(avg_recon, 6),
                 round(avg_reg, 6), f'{lr:.4e}'])

        final_epoch = epoch
        if avg_total < LOSS_THRESHOLD:
            print(f"  Early stop: loss {avg_total:.6f} < {LOSS_THRESHOLD}")
            break

    # Reparameterized z (matches author's encoder.predict(X))
    model.eval()
    with torch.no_grad():
        mu_all, log_var_all = model.encode(X_tensor.to(device))
        latent = model.reparameterize(mu_all, log_var_all).cpu().numpy()
    print(f"  Latent features: {latent.shape}  (epochs run: {final_epoch})")

    ckpt_path = os.path.join(outputs_dir, 'vae_checkpoint.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim':    X_scaled.shape[1],
        'latent_dim':   LATENT_DIM,
        'loss_fn':      loss_fn_name,
        'regularizer':  regularizer,
        'kl_weight':    kl_weight,
        'mmd_weight':   mmd_weight,
        'epochs_run':   final_epoch,
        'final_loss':   avg_total,
        'n_patients':   X_scaled.shape[0],
    }, ckpt_path)
    print(f"  Checkpoint   → {ckpt_path}")
    print(f"  Training log → {log_path}")

    return latent


# =============================================================================
#  ML helpers  (mirrors reference ML_multimodal_train.py exactly)
# =============================================================================

def upsample_minority(df: pd.DataFrame) -> pd.DataFrame:
    label_col = df.columns[-1]
    counts    = df[label_col].value_counts()
    maj_cls   = counts.idxmax()
    min_cls   = counts.idxmin()
    df_maj    = df[df[label_col] == maj_cls]
    df_min    = df[df[label_col] == min_cls]
    df_min_up = resample(df_min, replace=True,
                         n_samples=counts[maj_cls],
                         random_state=RANDOM_STATE)
    return pd.concat([df_maj, df_min_up]).reset_index(drop=True)


def get_classifiers(n_estimators: int = 10):
    return [
        ('rbf_svm',       SVC(kernel='rbf',     C=1, gamma='scale', probability=True)),
        ('linear_svm',    SVC(kernel='linear',  C=1, gamma='scale', probability=True)),
        ('poly_svm',      SVC(kernel='poly',    C=1, gamma='scale', probability=True)),
        ('sigmoid_svm',   SVC(kernel='sigmoid', C=1, gamma='scale', probability=True)),
        ('random_forest', RandomForestClassifier(
            n_estimators=n_estimators, criterion='gini',
            max_features=None, random_state=RANDOM_STATE)),
    ]


def _safe_metrics(y_true, y_pred, y_score):
    eps = 1e-9
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = cm
    acc  = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = float('nan')
    try:
        prc = average_precision_score(y_true, y_score)
    except Exception:
        prc = float('nan')
    return dict(roc=roc, prc=prc, acc=acc, prec=prec, rec=rec, f1=f1,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


# =============================================================================
#  CV loop — pre-fixed patient-ID splits
# =============================================================================

def run_cv_with_splits(clf_name, model, latent_df, splits,
                       run_name, loss_fn, regularizer,
                       kl_weight, mmd_weight,
                       fold_csv, oof_dir):
    id_col    = latent_df.columns[0]
    label_col = latent_df.columns[-1]
    feat_cols = [c for c in latent_df.columns if c != id_col and c != label_col]
    latent_df = latent_df.set_index(id_col)

    n_folds  = splits['n_folds']
    cum      = {k: 0.0 for k in ['roc', 'prc', 'acc', 'prec', 'rec', 'f1',
                                   'tn', 'fp', 'fn', 'tp']}
    best_roc = -1.0
    best_model_path = os.path.join(oof_dir, f'{clf_name}_best.pkl')

    oof_rows = []
    roc_save = {}
    prc_save = {}
    cms      = []

    for fold_idx in range(1, n_folds + 1):
        fold_info = splits['folds'][f'fold_{fold_idx}']
        train_ids = [p for p in fold_info['train_ids'] if p in latent_df.index]
        test_ids  = [p for p in fold_info['test_ids']  if p in latent_df.index]

        if len(test_ids) == 0:
            print(f"    Fold {fold_idx}: no test patients found — skipping")
            continue

        X_train = latent_df.loc[train_ids, feat_cols].values.astype(np.float32)
        y_train = latent_df.loc[train_ids, label_col].values.astype(int)
        X_test  = latent_df.loc[test_ids,  feat_cols].values.astype(np.float32)
        y_test  = latent_df.loc[test_ids,  label_col].values.astype(int)

        # Per-fold minority upsampling on training split only
        train_df  = pd.DataFrame(X_train, columns=feat_cols)
        train_df['label'] = y_train
        train_up  = upsample_minority(train_df)
        X_train_up = train_up[feat_cols].values
        y_train_up = train_up['label'].values

        model.fit(X_train_up, y_train_up)

        y_pred  = model.predict(X_test)
        y_score = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, 'predict_proba')
                   else model.decision_function(X_test))

        m = _safe_metrics(y_test, y_pred, y_score)
        for k in cum:
            cum[k] += m[k]

        print(f"    Fold {fold_idx:2d}/{n_folds} | "
              f"ROC={m['roc']:.4f}  AUPRC={m['prc']:.4f}  "
              f"Acc={m['acc']:.4f}  F1={m['f1']:.4f} | "
              f"TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

        with open(fold_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                run_name, loss_fn, regularizer, kl_weight, mmd_weight,
                'mrna', clf_name, fold_idx,
                round(m['roc'], 4), round(m['prc'], 4),
                round(m['acc'], 4), round(m['prec'], 4),
                round(m['rec'], 4), round(m['f1'], 4),
                m['tn'], m['fp'], m['fn'], m['tp'],
            ])

        fpr, tpr, _      = roc_curve(y_test, y_score)
        prec_a, rec_a, _ = precision_recall_curve(y_test, y_score)

        roc_save[f'fpr_fold{fold_idx}']       = fpr
        roc_save[f'tpr_fold{fold_idx}']       = tpr
        roc_save[f'auc_fold{fold_idx}']       = np.array([m['roc']])
        prc_save[f'precision_fold{fold_idx}'] = prec_a
        prc_save[f'recall_fold{fold_idx}']    = rec_a
        prc_save[f'ap_fold{fold_idx}']        = np.array([m['prc']])
        cms.append(confusion_matrix(y_test, y_pred, labels=[0, 1]))

        for pid, yt, yp, ys in zip(test_ids, y_test, y_pred, y_score):
            oof_rows.append({
                'patient_id': pid, 'fold': fold_idx,
                'true_label': int(yt), 'pred_label': int(yp),
                'prob_score': round(float(ys), 6),
            })

        if m['roc'] > best_roc:
            best_roc = m['roc']
            pickle.dump(model, open(best_model_path, 'wb'))
        model = pickle.load(open(best_model_path, 'rb'))

    np.savez(os.path.join(oof_dir, f'{clf_name}_roc_curves.npz'), **roc_save)
    np.savez(os.path.join(oof_dir, f'{clf_name}_prc_curves.npz'), **prc_save)
    if cms:
        np.save(os.path.join(oof_dir, f'{clf_name}_confusion_matrices.npy'),
                np.stack(cms))
    pd.DataFrame(oof_rows).to_csv(
        os.path.join(oof_dir, f'{clf_name}_oof.csv'), index=False)

    n_done = len(cms)
    avg    = {k: cum[k] / n_done for k in cum}

    X_all      = latent_df[feat_cols].values.astype(np.float32)
    y_all      = latent_df[label_col].values.astype(int)
    best_model = pickle.load(open(best_model_path, 'rb'))
    yf_pred    = best_model.predict(X_all)
    yf_score   = (best_model.predict_proba(X_all)[:, 1]
                  if hasattr(best_model, 'predict_proba')
                  else best_model.decision_function(X_all))
    mf = _safe_metrics(y_all, yf_pred, yf_score)

    return {
        'cv_roc_auc':   round(avg['roc'], 4),
        'cv_auprc':     round(avg['prc'], 4),
        'cv_accuracy':  round(avg['acc'], 4),
        'cv_precision': round(avg['prec'], 4),
        'cv_recall':    round(avg['rec'], 4),
        'cv_f1':        round(avg['f1'], 4),
        'cv_tn':        round(avg['tn'], 2),
        'cv_fp':        round(avg['fp'], 2),
        'cv_fn':        round(avg['fn'], 2),
        'cv_tp':        round(avg['tp'], 2),
        'full_roc_auc': round(mf['roc'], 4),
        'full_auprc':   round(mf['prc'], 4),
        'full_tn': mf['tn'], 'full_fp': mf['fp'],
        'full_fn': mf['fn'], 'full_tp': mf['tp'],
    }


# =============================================================================
#  Results I/O
# =============================================================================

def _init_csv(path, header):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)


def append_result(path, run_name, loss_fn, regularizer,
                  kl_weight, mmd_weight, clf_name, m):
    row = [
        run_name, loss_fn, regularizer, kl_weight, mmd_weight, 'mrna', clf_name,
        m['cv_roc_auc'], m['cv_auprc'], m['cv_accuracy'],
        m['cv_precision'], m['cv_recall'], m['cv_f1'],
        m['cv_tn'], m['cv_fp'], m['cv_fn'], m['cv_tp'],
        m['full_roc_auc'], m['full_auprc'],
        m['full_tn'], m['full_fp'], m['full_fn'], m['full_tp'],
    ]
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def print_summary(clf_name, m):
    print(f"\n  ── {clf_name} ────────────────────────────────────────")
    print(f"    CV  AUROC     : {m['cv_roc_auc']:.4f}")
    print(f"    CV  AUPRC     : {m['cv_auprc']:.4f}")
    print(f"    CV  Accuracy  : {m['cv_accuracy']:.4f}")
    print(f"    CV  Precision : {m['cv_precision']:.4f}")
    print(f"    CV  Recall    : {m['cv_recall']:.4f}")
    print(f"    CV  F1        : {m['cv_f1']:.4f}")
    print(f"    Full AUROC    : {m['full_roc_auc']:.4f}")
    print(f"    Full AUPRC    : {m['full_auprc']:.4f}")


# =============================================================================
#  Main
# =============================================================================

def main(args):
    run_id  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_tag = f"{args.run_name}_{run_id}"

    # Group all runs under betavalue= folder regardless of regularizer
    subdir = f"betavalue={args.kl_weight}"

    run_outputs_dir = os.path.join(OUTPUTS_DIR, subdir, run_tag)
    results_subdir  = os.path.join(RESULTS_DIR, subdir)
    logs_subdir     = os.path.join(LOGS_DIR,    subdir)

    os.makedirs(run_outputs_dir, exist_ok=True)
    os.makedirs(results_subdir,  exist_ok=True)
    os.makedirs(logs_subdir,     exist_ok=True)

    results_csv = os.path.join(results_subdir, 'ablation_results.csv')
    fold_csv    = os.path.join(results_subdir, f'fold_metrics_{run_tag}.csv')

    _init_csv(results_csv, RESULTS_HEADER)
    _init_csv(fold_csv,    FOLD_HEADER)

    if not os.path.exists(args.splits_file):
        print(f"ERROR: splits file not found: {args.splits_file}")
        print("Run: python Objective_Functions_Ablations/mRNA/generate_splits.py")
        sys.exit(1)

    with open(args.splits_file) as f:
        splits = json.load(f)
    print(f"Splits: {splits['n_folds']} folds | {splits['n_patients']} patients")

    print(f"\nLoading: {args.input_path}")
    df = pd.read_csv(args.input_path, header=0, index_col=None, low_memory=False)

    patient_ids  = df.iloc[:, 0].values
    class_labels = df.iloc[:, -1].values
    X_raw        = df.iloc[:, 1:-1].values.astype(np.float32)

    valid_mask   = np.isin(class_labels, [0, 1])
    patient_ids  = patient_ids[valid_mask]
    class_labels = class_labels[valid_mask].astype(int)
    X_raw        = X_raw[valid_mask]

    # MinMax scale: discretized -1/0/+1 → 0/0.5/1
    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    if args.test_mode:
        args.epochs = 3
        print(f"[TEST MODE] {X_scaled.shape[0]} patients, {args.epochs} epochs")

    print(f"  {X_scaled.shape[0]} patients × {X_scaled.shape[1]} features (MinMax-scaled)")
    print(f"  Label 0 (long-term): {(class_labels==0).sum()}  "
          f"Label 1 (short-term): {(class_labels==1).sum()}")

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"        {torch.cuda.get_device_name(device)}")

    print(f"\n{'='*60}")
    print(f"  Training VAE_mRNA | recon={args.loss_fn} | reg={args.regularizer}")
    print(f"{'='*60}")

    latent = train_vae(
        X_scaled, args.loss_fn, args.regularizer,
        args.kl_weight, args.mmd_weight,
        args.epochs, device, run_outputs_dir,
    )

    feat_names = [f'mrna_vae_{i}' for i in range(1, LATENT_DIM + 1)]
    latent_df  = pd.DataFrame(latent, columns=feat_names)
    latent_df.insert(0, 'submitter_id.samples', patient_ids)
    latent_df['label_mrna'] = class_labels

    latent_csv = os.path.join(run_outputs_dir, 'vae_latent_features.csv')
    latent_df.to_csv(latent_csv, index=False)
    print(f"\nLatent features → {latent_csv}")

    manifest = {
        'run_id':          run_id,
        'run_name':        args.run_name,
        'run_tag':         run_tag,
        'modality':        'mrna',
        'loss_fn':         args.loss_fn,
        'regularizer':     args.regularizer,
        'kl_weight':       args.kl_weight,
        'mmd_weight':      args.mmd_weight,
        'epochs':          args.epochs,
        'latent_dim':      LATENT_DIM,
        'n_patients':      int(X_scaled.shape[0]),
        'n_raw_features':  int(X_scaled.shape[1]),
        'n_label_0':       int((class_labels == 0).sum()),
        'n_label_1':       int((class_labels == 1).sum()),
        'device':          str(device),
        'splits_file':     args.splits_file,
        'input_path':      args.input_path,
        'results_csv':     results_csv,
        'fold_csv':        fold_csv,
        'latent_csv':      latent_csv,
        'outputs_dir':     run_outputs_dir,
        'test_mode':       args.test_mode,
    }
    manifest_path = os.path.join(run_outputs_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest → {manifest_path}")

    n_est       = 3 if args.test_mode else 10
    classifiers = get_classifiers(n_estimators=n_est)

    print(f"\n{'='*60}")
    print(f"  {len(classifiers)} classifiers | {splits['n_folds']}-fold pre-fixed splits")
    print(f"{'='*60}")

    for clf_name, clf in classifiers:
        print(f"\n{'─'*60}")
        print(f"  Classifier: {clf_name}")
        print(f"{'─'*60}")
        m = run_cv_with_splits(
            clf_name, clf, latent_df.copy(), splits,
            args.run_name, args.loss_fn, args.regularizer,
            args.kl_weight, args.mmd_weight,
            fold_csv, run_outputs_dir,
        )
        print_summary(clf_name, m)
        append_result(results_csv, args.run_name, args.loss_fn, args.regularizer,
                      args.kl_weight, args.mmd_weight, clf_name, m)

    print(f"\n{'='*60}")
    print(f"Run complete.")
    print(f"  Aggregate results → {results_csv}")
    print(f"  Per-fold metrics  → {fold_csv}")
    print(f"  Run outputs       → {run_outputs_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='mRNA ablation: train PyTorch VAE_mRNA + run ML classifiers.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--input_path', type=str, default=RAW_DATA)
    parser.add_argument('--splits_file', type=str, default=SPLITS_FILE)
    parser.add_argument('--loss_fn', type=str, default='logcosh',
                        choices=list(LOSS_FNS.keys()))
    parser.add_argument('--regularizer', type=str, default='kl',
                        choices=['kl', 'mmd'])
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='β for KL divergence.')
    parser.add_argument('--mmd_weight', type=float, default=10.0,
                        help='λ for MMD.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--test_mode', action='store_true')

    main(parser.parse_args())
