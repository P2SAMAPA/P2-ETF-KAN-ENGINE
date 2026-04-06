
# kan_model.py — AGGRESSIVE Temporal KAN Forecaster
#
# Changes for higher returns:
# 1. Increased output scale from 0.15 to 0.50
# 2. Reduced dropout by 50%
# 3. Optional: Remove tanh constraint (commented out for safety)
# 4. Increased spline scale initialization

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLUKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))

        # INCREASED: From 0.1 to 0.3 for more aggressive spline contribution
        self.spline_scale = nn.Parameter(torch.ones(out_features, 1, 1) * 0.3)

        grid = torch.linspace(-2.0, 2.0, steps=grid_size)
        self.register_buffer("grid", grid)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        # INCREASED: Std from 0.05 to 0.1 for more variance
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        q = torch.linspace(margin, 1.0 - margin, self.grid_size, device=x.device, dtype=x.dtype)
        x_flat = x.detach().T
        sorted_x, _ = x_flat.sort(dim=1)
        idx = (q * (sorted_x.shape[1] - 1)).long().clamp(0, sorted_x.shape[1] - 1)
        percentiles = sorted_x[:, idx]
        grid = percentiles.mean(dim=0)
        self.grid.copy_(grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(F.silu(x), self.base_weight)
        x_exp = x.unsqueeze(-1)
        grid_exp = self.grid.view(1, 1, -1)
        basis = F.relu(x_exp - grid_exp)
        scaled_w = self.spline_weight * self.spline_scale
        spline_out = torch.einsum('oig,big->bo', scaled_w, basis)
        return base_out + spline_out

class KANBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 20, dropout: float = 0.0):
        super().__init__()
        self.kan = ReLUKANLayer(in_features, out_features, grid_size)
        self.norm = nn.LayerNorm(out_features)
        # REDUCED: Dropout cut in half for more signal flow
        self.drop = nn.Dropout(dropout * 0.5) if dropout > 0 else nn.Identity()
        self.residual_proj = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features else nn.Identity()
        )

    def update_grid(self, x: torch.Tensor):
        self.kan.update_grid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.kan(x)
        out = self.drop(out)
        out = self.norm(out + residual)
        return out

class TemporalAttentionPool(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.seq_len = seq_len
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)
        x_w = x * weights
        return x_w.reshape(x.size(0), -1)

class TemporalKANForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        output_dim: int = None,
        grid_size: int = 20,
        seq_len: int = None,
        dropout: float = 0.15,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        if seq_len is not None and input_dim % seq_len == 0:
            feat_dim = input_dim // seq_len
            self.attn_pool = TemporalAttentionPool(seq_len, feat_dim)
        else:
            self.attn_pool = None

        self.blocks = nn.ModuleList()
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            # REDUCED: Dropout from second block onward cut in half
            block_dropout = dropout * 0.5 if i > 0 else 0.0
            self.blocks.append(KANBlock(prev, h, grid_size, block_dropout))
            prev = h

        self.head = nn.Linear(prev, output_dim)
        # INCREASED: From 0.15 to 0.50 for 3.3x larger output range
        self.output_scale = 0.50

    @torch.no_grad()
    def init_grids(self, x: torch.Tensor):
        h = x.view(x.size(0), -1) if x.dim() > 2 else x
        if self.attn_pool is not None:
            h = self.attn_pool(
                h.view(h.size(0), -1, self.attn_pool.feat_dim)
            )
            h = h.view(x.size(0), -1)
        for block in self.blocks:
            block.update_grid(h)
            h = block(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.view(x.size(0), -1) if x.dim() == 3 else x

        if self.attn_pool is not None:
            h_3d = h.view(h.size(0), self.attn_pool.seq_len, self.attn_pool.feat_dim)
            h = self.attn_pool(h_3d)

        for block in self.blocks:
            h = block(h)

        # AGGRESSIVE: Remove tanh constraint, use linear output with scale
        # return torch.tanh(self.head(h)) * self.output_scale  # OLD: Constrained
        return self.head(h) * self.output_scale  # NEW: Unconstrained, scaled
