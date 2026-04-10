"""
CNV Log-Cosh VAE Feature Extractor (PyTorch)
=============================================
Replicates VAEcnv from Arya et al. (2023) using PyTorch instead of Keras/TensorFlow.

Architecture (matches original):
  Encoder : input → Dense(256,tanh) → Dense(128,tanh) → Dense(64,tanh)
              → [z_mean(32), z_log_var(32)]
  Sampling: z = z_mean + eps * exp(0.5 * z_log_var)
  Decoder : z → Dense(64,tanh) → Dense(128,tanh) → Dense(256,tanh)
              → Dense(input_dim, sigmoid)

Loss:
  L = LogCosh(x_recon, x) + KL(q(z|x) || N(0,I))
  KL = -0.5 * mean(z_log_var - z_mean^2 - exp(z_log_var) + 1)

Optimizer: Adam (lr=1e-3, ExponentialDecay) wrapped with Lookahead

Output CSV format (matches reference):
  submitter_id.samples | cnv_vae_1 | ... | cnv_vae_32 | label_cnv

Usage:
  # Full run (input must already exist from prepare_cnv_data.py):
  python cnv_vae_extractor.py

  # Quick CPU smoke-test (5 epochs, 100 patients):
  python cnv_vae_extractor.py --test_mode

  # Custom paths / hyperparameters:
  python cnv_vae_extractor.py \\
      --input_path  data/processed/raw_features_cnv.csv \\
      --output_path data/processed/vae_features_cnv.csv \\
      --epochs 50 --latent_dim 32 --batch_size 32
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── Defaults (mirror the reference Keras script) ─────────────────────────────
LATENT_DIM      = 32
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 1e-3
LOSS_THRESHOLD  = 0.01   # early-stop when avg epoch loss drops below this
DECAY_RATE      = 0.96   # LR multiplied by this every epoch (approximates TF's
DECAY_STEPS     = 100_000  # ExponentialDecay with decay_steps=100k; see note below)


# ── Log-Cosh loss ─────────────────────────────────────────────────────────────

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Numerically-stable log-cosh loss (a=1), matching tf.keras.losses.LogCosh().

    log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)

    This is smoother than L1 for small errors and less aggressively penalising
    than L2 for large errors — ideal for balancing VAE reconstruction vs KL.
    """
    diff = y_pred - y_true
    # Clamp to avoid exp overflow on very large diffs
    diff = diff.clamp(-50, 50)
    return (diff.abs()
            + F.softplus(-2.0 * diff.abs())
            - torch.log(torch.tensor(2.0, device=y_pred.device))).mean()


# ── Lookahead optimizer wrapper ───────────────────────────────────────────────

class Lookahead:
    """
    Lookahead (Zhang et al. 2019) wrapping any PyTorch optimizer.
    Every k steps the slow weights are interpolated toward the fast weights.
    Replicates tfa.optimizers.Lookahead used in the reference Keras code.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer,
                 k: int = 6, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        # Initialise slow-weight copies
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
                    # slow_w += alpha * (fast_w - slow_w)
                    slow.data.add_(self.alpha * (fast.data - slow.data))
                    fast.data.copy_(slow.data)


# ── VAE model ─────────────────────────────────────────────────────────────────

class VAE_CNV(nn.Module):
    """
    Variational Autoencoder for CNV data.
    Architecture mirrors the Keras VAE in the reference codebase.
    """

    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(),
            nn.Linear(256, 128),       nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
        )
        self.fc_mu      = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.Tanh(),
            nn.Linear(64, 128),         nn.Tanh(),
            nn.Linear(128, 256),        nn.Tanh(),
            nn.Linear(256, input_dim),  nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# ── Training helpers ──────────────────────────────────────────────────────────

def vae_loss(recon_x: torch.Tensor,
             x: torch.Tensor,
             mu: torch.Tensor,
             log_var: torch.Tensor) -> torch.Tensor:
    """
    Total VAE loss = LogCosh reconstruction + KL divergence.
    KL formula from the paper: -0.5 * mean(log_var - mu^2 - exp(log_var) + 1)
    """
    recon = log_cosh_loss(recon_x, x)
    kl    = -0.5 * torch.mean(log_var - mu.pow(2) - log_var.exp() + 1)
    return recon + kl


def run_epoch(model: VAE_CNV,
              loader: DataLoader,
              optimizer: Lookahead,
              scheduler: torch.optim.lr_scheduler._LRScheduler,
              device: torch.device) -> float:
    """One training epoch. Returns mean loss over all batches."""
    model.train()
    total_loss = 0.0
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        recon_x, mu, log_var = model(batch_x)
        loss = vae_loss(recon_x, batch_x, mu, log_var)
        loss.backward()
        optimizer.step()
        # Per-step LR decay (approximates TF's step-based ExponentialDecay)
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading data from: {args.input_path}")
    df = pd.read_csv(args.input_path, header=0, index_col=None, low_memory=False)

    if df.shape[0] == 0:
        raise ValueError("Input CSV is empty. Run prepare_cnv_data.py first.")

    patient_ids  = df.iloc[:, 0].copy()          # first column
    class_labels = df.iloc[:, -1].copy()          # last column
    X_df = df.iloc[:, 1:-1]                        # gene feature columns
    X = X_df.values.astype(np.float32)

    if args.test_mode:
        X           = X[:100]
        patient_ids  = patient_ids.iloc[:100]
        class_labels = class_labels.iloc[:100]
        args.epochs  = 5
        print(f"[TEST MODE] Using 100 patients, {args.epochs} epochs.")

    print(f"Data  : {X.shape[0]} patients × {X.shape[1]} features")

    # ── Normalise to [0, 1] for sigmoid decoder ───────────────────────────────
    # GISTIC2 values are integers in {{-2,-1,0,1,2}} → map to {{0,0.25,0.5,0.75,1}}
    X_norm = (X - X.min()) / (X.max() - X.min() + 1e-8)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cpu')
    print(f"Device: {device}  (GPU support can be added via --device cuda)")

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset = TensorDataset(torch.FloatTensor(X_norm))
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         drop_last=False)

    # ── Model + Optimizer ─────────────────────────────────────────────────────
    model = VAE_CNV(input_dim=X.shape[1], latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : {n_params:,} trainable parameters  |  latent_dim={args.latent_dim}")

    adam_base = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Approximate TF ExponentialDecay(decay_rate=0.96, decay_steps=100k)
    # per step: lr_t = lr_0 * gamma^step  where gamma = decay_rate^(1/decay_steps)
    gamma_per_step = DECAY_RATE ** (1.0 / DECAY_STEPS)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        adam_base, gamma=gamma_per_step)

    optimizer = Lookahead(adam_base)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for up to {args.epochs} epochs "
          f"(early stop if avg loss < {LOSS_THRESHOLD}) ...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = run_epoch(model, loader, optimizer, scheduler, device)
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.6f} | "
              f"lr={current_lr:.2e}")
        if avg_loss < LOSS_THRESHOLD:
            print(f"  Early stop: loss {avg_loss:.6f} < {LOSS_THRESHOLD}")
            break

    # ── Extract latent features (use z_mean, not sampled z) ──────────────────
    print("\nExtracting latent features ...")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_norm).to(device)
        mu, _ = model.encode(X_tensor)
        latent_np = mu.cpu().numpy()

    print(f"Latent feature shape: {latent_np.shape}")

    # ── Build output DataFrame (matching reference format) ────────────────────
    col_names = [f'cnv_vae_{i}' for i in range(1, latent_np.shape[1] + 1)]
    out_df = pd.DataFrame(latent_np, columns=col_names)
    out_df.insert(0, 'submitter_id.samples', patient_ids.values)
    out_df['label_cnv'] = class_labels.values

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"VAE features saved to: {args.output_path}")

    # ── Save model checkpoint ─────────────────────────────────────────────────
    ckpt_path = args.output_path.replace('.csv', '_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim':  X.shape[1],
        'latent_dim': args.latent_dim,
        'epochs_run': epoch,
        'final_loss': avg_loss,
    }, ckpt_path)
    print(f"Model checkpoint  : {ckpt_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train log-cosh VAE and extract CNV latent features.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=str,
                        default=os.path.join('data', 'processed', 'raw_features_cnv.csv'),
                        help='Processed raw features CSV from prepare_cnv_data.py')
    parser.add_argument('--output_path', type=str,
                        default=os.path.join('data', 'processed', 'vae_features_cnv.csv'),
                        help='Output path for VAE latent features CSV')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Initial Adam learning rate')
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM,
                        help='VAE latent space dimension (32 per paper)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Use 100 patients and 5 epochs for a quick CPU smoke-test')
    args = parser.parse_args()

    main(args)
