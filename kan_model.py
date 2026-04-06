# kan_model.py — Improved Temporal KAN Forecaster
#
# Key improvements over v1:
#   1. Adaptive per-feature grid (fitted to actual input distribution, not fixed [-1,1])
#   2. Per-output-neuron learnable scale (instead of one global scalar)
#   3. Temporal attention pooling before flattening (preserves sequence structure)
#   4. Residual connections between KAN blocks (stabilise deep gradient flow)
#   5. Dropout only after deeper layers, not input layer
#   6. Tanh-scaled output head (soft-clips predictions to realistic return range)
#   7. Backward-compatible: TemporalKANForecaster signature unchanged

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Improved KAN Layer ─────────────────────────────────────────────────────

class ReLUKANLayer(nn.Module):
    """
    ReLU-basis KAN layer with:
    - Adaptive grid updated from data statistics (call update_grid before training)
    - Per-neuron learnable spline scale instead of one global float
    - SiLU base activation (smoother gradient than raw linear)
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 20):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size

        # Base (linear + SiLU) path
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        # Spline path
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))

        # Per-output-neuron learnable scale for spline contribution
        self.spline_scale = nn.Parameter(torch.ones(out_features, 1, 1) * 0.1)

        # Grid: initially uniform [-2, 2], updated from real data in update_grid()
        grid = torch.linspace(-2.0, 2.0, steps=grid_size)
        self.register_buffer("grid", grid)  # (G,)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.05)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Fit grid knots to the percentile distribution of x.
        Call once on a representative batch before training begins.
        x: (B, in_features)
        """
        # Use per-feature percentiles so every feature has active knots
        q = torch.linspace(margin, 1.0 - margin, self.grid_size,
                           device=x.device, dtype=x.dtype)
        # Flatten batch dimension, shape (in_features, B)
        x_flat = x.detach().T  # (in_features, B)
        # Take mean percentile across features for shared grid
        sorted_x, _ = x_flat.sort(dim=1)
        idx = (q * (sorted_x.shape[1] - 1)).long().clamp(0, sorted_x.shape[1] - 1)
        percentiles = sorted_x[:, idx]          # (in_features, G)
        grid = percentiles.mean(dim=0)          # (G,) — shared across features
        self.grid.copy_(grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)

        # Base path: SiLU(x) @ W^T
        base_out = F.linear(F.silu(x), self.base_weight)           # (B, out)

        # Spline path: relu(x_i - knot_k) basis
        x_exp    = x.unsqueeze(-1)                                  # (B, in, 1)
        grid_exp = self.grid.view(1, 1, -1)                         # (1, 1, G)
        basis    = F.relu(x_exp - grid_exp)                         # (B, in, G)

        # Weighted sum over in_features and grid_size → (B, out)
        # spline_weight: (out, in, G), spline_scale: (out, 1, 1)
        scaled_w  = self.spline_weight * self.spline_scale          # (out, in, G)
        spline_out = torch.einsum('oig,big->bo', scaled_w, basis)   # (B, out)

        return base_out + spline_out


# ── 2. KAN Block with residual connection ────────────────────────────────────

class KANBlock(nn.Module):
    """
    KAN layer + LayerNorm + optional residual projection.
    Residual is added when in_features == out_features (or via a 1x1 projection).
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 20, dropout: float = 0.0):
        super().__init__()
        self.kan  = ReLUKANLayer(in_features, out_features, grid_size)
        self.norm = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual projection if dims differ
        self.residual_proj = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features else nn.Identity()
        )

    def update_grid(self, x: torch.Tensor):
        self.kan.update_grid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out      = self.kan(x)
        out      = self.drop(out)
        out      = self.norm(out + residual)   # pre-norm residual
        return out


# ── 3. Temporal Attention Pooling ────────────────────────────────────────────

class TemporalAttentionPool(nn.Module):
    """
    Learns a soft attention mask over the T time steps so that recent
    and regime-relevant days get higher weight before flattening.
    Input:  (B, T, D)
    Output: (B, T*D) — attention-weighted then flattened
    """

    def __init__(self, seq_len: int, feat_dim: int):
        super().__init__()
        # Small MLP over time dimension → scalar weight per step
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),          # (B, T, 1)
        )
        self.seq_len  = seq_len
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  — D = n_features per time step
        weights = self.attn(x)                      # (B, T, 1)
        weights = torch.softmax(weights, dim=1)     # normalise over T
        x_w     = x * weights                       # (B, T, D)
        return x_w.reshape(x.size(0), -1)           # (B, T*D)


# ── 4. Main Forecaster ───────────────────────────────────────────────────────

class TemporalKANForecaster(nn.Module):
    """
    Temporal KAN Forecaster — drop-in replacement for the v1 model.

    Architecture:
        TemporalAttentionPool  →  KANBlock × N  →  Linear head  →  Tanh scale

    Args:
        input_dim   : T * n_features  (flattened sequence length, same as v1)
        hidden_dims : list of hidden widths, e.g. [256, 128]
        output_dim  : number of ETFs to score
        grid_size   : number of knots per KAN spline (20 default)
        seq_len     : time-steps in the lookback window (used for attention pool)
                      Set to None to disable attention pooling (pure flat mode).
        dropout     : dropout rate applied in deeper KAN blocks only
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list  = None,
        output_dim:  int   = None,
        grid_size:   int   = 20,
        seq_len:     int   = None,   # NEW — pass SEQ_LEN from app/train
        dropout:     float = 0.15,   # slightly higher than v1's 0.1
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # ── Temporal attention pool (optional) ───────────────────────────
        if seq_len is not None and input_dim % seq_len == 0:
            feat_dim        = input_dim // seq_len
            self.attn_pool  = TemporalAttentionPool(seq_len, feat_dim)
        else:
            self.attn_pool  = None   # fallback: plain flatten

        # ── KAN blocks ───────────────────────────────────────────────────
        self.blocks = nn.ModuleList()
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            # Only apply dropout from the second block onward
            block_dropout = dropout if i > 0 else 0.0
            self.blocks.append(KANBlock(prev, h, grid_size, block_dropout))
            prev = h

        # ── Output head ──────────────────────────────────────────────────
        # Tanh scales output to (-output_scale, +output_scale).
        # 0.15 ≈ ±15% daily return — generous enough for real signals,
        # tight enough to prevent exploding predictions.
        self.head         = nn.Linear(prev, output_dim)
        self.output_scale = 0.15

    # ── Grid initialisation (call before training on a data batch) ───────────
    @torch.no_grad()
    def init_grids(self, x: torch.Tensor):
        """
        Fit all KAN layer grids to real data distribution.
        x: (B, input_dim) — a representative batch (e.g. full training set).
        Call once before the first training epoch.
        """
        h = x.view(x.size(0), -1) if x.dim() > 2 else x
        if self.attn_pool is not None:
            h = self.attn_pool(
                h.view(h.size(0), -1,
                       self.attn_pool.feat_dim)
            )
            h = h.view(x.size(0), -1)  # re-flatten after weighted pool
        for block in self.blocks:
            block.update_grid(h)
            h = block(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, T, F) or (B, T*F) — same as v1
        h = x.view(x.size(0), -1) if x.dim() == 3 else x

        # Temporal attention pooling
        if self.attn_pool is not None:
            h_3d = h.view(h.size(0), self.attn_pool.seq_len,
                          self.attn_pool.feat_dim)
            h    = self.attn_pool(h_3d)

        # KAN blocks with residuals
        for block in self.blocks:
            h = block(h)

        # Soft-clipped output
        return torch.tanh(self.head(h)) * self.output_scale
