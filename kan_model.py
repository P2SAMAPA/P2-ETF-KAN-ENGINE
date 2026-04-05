import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLayer(nn.Module):
    """
    KAN layer from original paper (simplified).
    Each weight is a linear combination of B-spline basis functions.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid points (fixed, not learnable)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(grid_range[0] - spline_order * h,
                              grid_range[1] + spline_order * h,
                              steps=grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)
        
        # Base weight (linear part)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Spline coefficients (for each basis function)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        # Scale parameters for numerical stability
        self.scale_base = 1.0
        self.scale_spline = 1.0
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)
    
    def b_spline(self, x, grid, k=3):
        """
        Compute B-spline basis functions for input x.
        x: (batch, in)
        grid: (num_knots,)
        returns: (batch, in, num_basis) where num_basis = len(grid)-k-1
        """
        # Add batch and feature dimensions
        x = x.unsqueeze(-1)  # (batch, in, 1)
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1,1,num_knots)
        
        # Compute basis recursively
        basis = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).float()
        for _ in range(1, k+1):
            # Compute denominators (avoid division by zero)
            left_denom = grid[..., 1:-1] - grid[..., :-2]
            right_denom = grid[..., 2:] - grid[..., 1:-1]
            left_denom = torch.where(left_denom > 0, left_denom, torch.ones_like(left_denom))
            right_denom = torch.where(right_denom > 0, right_denom, torch.ones_like(right_denom))
            
            left = (x - grid[..., :-2]) / left_denom * basis[..., :-1]
            right = (grid[..., 2:] - x) / right_denom * basis[..., 1:]
            basis = left + right
        return basis
    
    def forward(self, x):
        batch = x.shape[0]
        # Base linear part
        base_out = F.linear(x, self.base_weight)  # (batch, out)
        
        # Spline part
        # Compute B-spline basis for each input feature
        basis = self.b_spline(x, self.grid, self.spline_order)  # (batch, in, basis_len)
        # Spline output: sum over input features and basis functions
        # spline_weight: (out, in, basis_len)
        spline_out = torch.einsum('b i l, o i l -> b o', basis, self.spline_weight)
        
        return self.scale_base * base_out + self.scale_spline * spline_out

class TemporalKANForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, grid_size=5, spline_order=3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(KANLayer(prev, h, grid_size, spline_order))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.net(x)
