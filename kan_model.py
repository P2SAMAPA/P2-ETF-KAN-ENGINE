import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_spline=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # grid: (out_features, in_features, grid_size + 2*spline_order?) Actually standard KAN uses grid size + spline_order
        # We'll create a grid for each (out, in) pair
        h = 2.0 / grid_size
        grid = torch.linspace(-1.0, 1.0, steps=grid_size + 1)
        # Expand to (out, in, grid_size+1)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(out_features, in_features, -1)
        self.register_buffer("grid", grid)
        # Spline coefficients: (out, in, grid_size + spline_order)
        self.spline_weight = nn.Parameter(torch.Tensor(
            out_features, in_features, grid_size + spline_order
        ))
        self.scale_spline = scale_spline
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)
    
    def b_spline(self, x, grid, k=3):
        """
        x: (batch, 1, 1, in_features) - expanded for broadcasting
        grid: (out_features, in_features, grid_size + 1)
        Returns: (batch, out_features, in_features, grid_size + k) basis
        """
        # Add extra dimensions for broadcasting
        x = x.unsqueeze(-1)  # (batch, 1, 1, in_features, 1)
        grid = grid.unsqueeze(0)  # (1, out, in, grid_len)
        
        # Initialize basis: 1 if x in [grid[i], grid[i+1]) else 0
        basis = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).float()
        
        for _ in range(1, k+1):
            # Compute denominator
            left_denom = grid[..., 1:-1] - grid[..., :-2]
            right_denom = grid[..., 2:] - grid[..., 1:-1]
            
            # Avoid division by zero
            left_denom = torch.where(left_denom > 0, left_denom, torch.ones_like(left_denom))
            right_denom = torch.where(right_denom > 0, right_denom, torch.ones_like(right_denom))
            
            # Compute left and right contributions
            left = (x - grid[..., :-2]) / left_denom * basis[..., :-1]
            right = (grid[..., 2:] - x) / right_denom * basis[..., 1:]
            basis = left + right
        return basis
    
    def forward(self, x):
        batch_size = x.shape[0]
        # Base linear part
        base_output = F.linear(x, self.base_weight)  # (batch, out)
        
        # Spline part
        x_expanded = x.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, in_features)
        spline_basis = self.b_spline(x_expanded, self.grid, self.spline_order)  # (batch, out, in, basis_len)
        # Weighted sum over basis functions
        spline_output = torch.einsum('b o i b_len, o i b_len -> b o', spline_basis, self.spline_weight)
        
        return base_output + self.scale_spline * spline_output
    
    def get_feature_importance(self):
        # Importance = L2 norm of spline coefficients across output and basis dimensions
        return torch.norm(self.spline_weight, dim=(0, 2))

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
