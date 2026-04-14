"""
Masked Autoencoder for tabular genomic (CNV) data.

Architecture — Asymmetric MAE (He et al., 2022)
------------------------------------------------
The encoder operates only on unmasked gene tokens.
The decoder is a lightweight Transformer that receives both the encoder
outputs (at unmasked positions) and learnable [MASK] tokens (at masked
positions), then predicts the original GISTIC2 value for every position.
Loss is computed only at masked positions.

This asymmetric design means:
  - The encoder representation is never polluted by mask tokens. It learns
    a richer, more transferable representation of observable CNV patterns.
  - The encoder processes fewer tokens per forward pass
    (e.g. 425 instead of 500 at 15% masking), which reduces compute.
  - The heavy burden of reconstruction is placed on the lightweight decoder,
    freeing the encoder to focus on representation quality.

Encoder choice — Transformer with learned gene-position embeddings
------------------------------------------------------------------
Each gene is treated as a token. The encoder is a full Transformer with:
  - Value embedding: maps each GISTIC2 integer to d_model.
  - Position embedding: one learned vector per gene index (0..499),
    encoding gene identity independently of copy-number value.
  - Multi-head self-attention across all unmasked gene positions.

Transformer self-attention is the right inductive bias here because
co-amplification and co-deletion patterns in CNV data involve arbitrary
pairs of gene loci (not just adjacent ones). Unlike a 1D-CNN, which would
only model local structure, a Transformer can directly attend from one
amplified region to another regardless of their distance in the feature
vector. The gradient signal from reconstructing masked genes then teaches
the attention weights which gene pairs co-vary.

Decoder choice — Lightweight Transformer
-----------------------------------------
Two-layer Transformer (dec_d_model=64) that operates on the full position
sequence. Encoder outputs are projected to dec_d_model and placed at
unmasked positions; learnable [MASK] tokens are placed at masked positions.
All positions carry positional embeddings so the decoder knows which gene
each token corresponds to.

Gene encoding
-------------
GISTIC2 integers {-2,-1,0,1,2} are shifted by +2 to {0,1,2,3,4} for
embedding lookup. The decoder head outputs 5-class logits per position.
"""

import torch
import torch.nn as nn

GISTIC_OFFSET = 2   # shift {-2,...,2} -> {0,...,4}
GISTIC_VOCAB  = 5   # number of distinct GISTIC2 values


class TabularMAE(nn.Module):
    """
    Asymmetric Masked Autoencoder for tabular CNV genomic data.

    Parameters
    ----------
    n_genes      : int   Number of input gene features (e.g. 500).
    d_model      : int   Encoder hidden dimension.
    n_heads      : int   Encoder attention heads (must divide d_model).
    n_layers     : int   Number of encoder Transformer layers.
    ffn_dim      : int   Encoder feed-forward dimension (default 2×d_model).
    dec_d_model  : int   Decoder hidden dimension (should be < d_model).
    dec_n_heads  : int   Decoder attention heads.
    dec_n_layers : int   Number of decoder Transformer layers (lightweight).
    dropout      : float Dropout applied in both encoder and decoder.
    """

    def __init__(
        self,
        n_genes:      int   = 500,
        d_model:      int   = 128,
        n_heads:      int   = 4,
        n_layers:     int   = 4,
        ffn_dim:      int   = 256,
        dec_d_model:  int   = 64,
        dec_n_heads:  int   = 2,
        dec_n_layers: int   = 2,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.n_genes     = n_genes
        self.d_model     = d_model
        self.dec_d_model = dec_d_model

        # ── Encoder embeddings ────────────────────────────────────────────
        # Value embedding: GISTIC2 integer → d_model.
        self.enc_value_emb = nn.Embedding(GISTIC_VOCAB, d_model)
        # Position embedding: gene index → d_model (per-gene identity).
        self.enc_pos_emb   = nn.Embedding(n_genes, d_model)

        # ── Encoder (heavy, processes only unmasked tokens) ───────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim if ffn_dim else 2 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN for stability
        )
        self.encoder  = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.enc_norm = nn.LayerNorm(d_model)

        # ── Encoder → Decoder projection ─────────────────────────────────
        # Maps encoder output (d_model) to decoder width (dec_d_model).
        self.enc_to_dec = nn.Linear(d_model, dec_d_model, bias=False)

        # ── Decoder inputs ────────────────────────────────────────────────
        # Learnable [MASK] token placed at every masked position in the decoder.
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, dec_d_model))
        # Positional embedding shared across all positions in the decoder.
        self.dec_pos_emb = nn.Embedding(n_genes, dec_d_model)

        # ── Decoder (lightweight, processes full sequence) ────────────────
        dec_layer = nn.TransformerEncoderLayer(
            d_model=dec_d_model,
            nhead=dec_n_heads,
            dim_feedforward=dec_d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder  = nn.TransformerEncoder(
            dec_layer, num_layers=dec_n_layers, enable_nested_tensor=False
        )
        self.dec_norm = nn.LayerNorm(dec_d_model)

        # ── Prediction head ───────────────────────────────────────────────
        self.decoder_head = nn.Linear(dec_d_model, GISTIC_VOCAB)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # Internal: split indices into unmasked / masked
    # ------------------------------------------------------------------

    def _split_indices(self, mask: torch.Tensor):
        """
        Sort positions so unmasked (False=0) come before masked (True=1).

        Returns
        -------
        ids_unmask : LongTensor [B, n_unmask]
        ids_mask   : LongTensor [B, n_mask]
        """
        # argsort on {0,1} puts all unmasked indices first
        ids      = mask.long().argsort(dim=1)   # stable False-first ordering
        n_mask   = int(mask[0].sum().item())
        n_unmask = mask.shape[1] - n_mask
        return ids[:, :n_unmask], ids[:, n_unmask:]

    # ------------------------------------------------------------------
    # Forward pass (used during training)
    # ------------------------------------------------------------------

    def forward(
        self,
        x:    torch.Tensor,  # [B, n_genes]  LongTensor of raw GISTIC2 integers
        mask: torch.Tensor,  # [B, n_genes]  BoolTensor, True = masked position
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : LongTensor [B, n_genes]   Raw GISTIC2 values {-2,...,2}.
        mask : BoolTensor [B, n_genes]   True at positions to reconstruct.

        Returns
        -------
        logits : FloatTensor [B, n_genes, 5]
            Reconstruction logits at every position.
            Caller computes cross-entropy loss only where mask == True.
        """
        B, G = x.shape
        device = x.device
        x_shifted = (x + GISTIC_OFFSET).clamp(0, GISTIC_VOCAB - 1)  # [B, G]

        ids_unmask, ids_mask = self._split_indices(mask)
        n_unmask = ids_unmask.shape[1]
        n_mask   = ids_mask.shape[1]

        # ── Encoder: process only unmasked tokens ─────────────────────────
        # Gather unmasked values and their gene-position indices.
        x_unmask   = x_shifted.gather(1, ids_unmask)             # [B, n_unmask]
        val_emb    = self.enc_value_emb(x_unmask)                # [B, n_unmask, d]
        pos_emb    = self.enc_pos_emb(ids_unmask)                # [B, n_unmask, d]
        enc_tokens = val_emb + pos_emb                           # [B, n_unmask, d]

        enc_out  = self.encoder(enc_tokens)                      # [B, n_unmask, d]
        enc_out  = self.enc_norm(enc_out)
        enc_proj = self.enc_to_dec(enc_out)                      # [B, n_unmask, d_dec]

        # ── Decoder: reconstruct full sequence ────────────────────────────
        # Allocate full-sequence buffer in decoder dimension.
        dec_tokens = torch.zeros(B, G, self.dec_d_model, device=device)

        # Place projected encoder outputs at unmasked positions.
        idx_u = ids_unmask.unsqueeze(-1).expand(-1, -1, self.dec_d_model)
        dec_tokens.scatter_(1, idx_u, enc_proj + self.dec_pos_emb(ids_unmask))

        # Place [MASK] token + positional embedding at masked positions.
        mask_toks = self.mask_token.expand(B, n_mask, -1)        # [B, n_mask, d_dec]
        idx_m = ids_mask.unsqueeze(-1).expand(-1, -1, self.dec_d_model)
        dec_tokens.scatter_(1, idx_m, mask_toks + self.dec_pos_emb(ids_mask))

        dec_out = self.decoder(dec_tokens)                       # [B, G, d_dec]
        dec_out = self.dec_norm(dec_out)
        logits  = self.decoder_head(dec_out)                     # [B, G, 5]
        return logits

    # ------------------------------------------------------------------
    # Representation extraction (used after training)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract a fixed-length patient representation.

        Passes the full (unmasked) sequence through the encoder only.
        The representation is the mean-pooled encoder output.

        Parameters
        ----------
        x : LongTensor [B, n_genes]   Raw GISTIC2 values {-2,...,2}.

        Returns
        -------
        repr : FloatTensor [B, d_model]
        """
        B, G   = x.shape
        device = x.device
        x_shifted = (x + GISTIC_OFFSET).clamp(0, GISTIC_VOCAB - 1)
        pos_ids   = torch.arange(G, device=device).unsqueeze(0).expand(B, -1)
        tokens    = self.enc_value_emb(x_shifted) + self.enc_pos_emb(pos_ids)
        encoded   = self.encoder(tokens)
        encoded   = self.enc_norm(encoded)
        return encoded.mean(dim=1)                               # [B, d_model]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict:
        """Return parameter counts by sub-module."""
        enc_params = sum(
            p.numel() for p in list(self.enc_value_emb.parameters()) +
            list(self.enc_pos_emb.parameters()) +
            list(self.encoder.parameters()) +
            list(self.enc_norm.parameters()) +
            list(self.enc_to_dec.parameters())
            if p.requires_grad
        )
        dec_params = sum(
            p.numel() for p in list(self.decoder.parameters()) +
            list(self.dec_norm.parameters()) +
            list(self.dec_pos_emb.parameters()) +
            [self.mask_token] +
            list(self.decoder_head.parameters())
            if p.requires_grad
        )
        return {'encoder': enc_params, 'decoder': dec_params, 'total': enc_params + dec_params}
