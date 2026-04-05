import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLayer(nn.Module):
    """
    Simplified KAN layer using linear splines (piecewise linear) + base linear.
    Avoids complex B-spline recursion.
    """
    def __init__(self, in_features, out_features, grid_size=5, scale_spline=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Base linear weight
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Grid points (learnable) – shape (out_features, in_features, grid_size+1)
        # Initialize evenly spaced between -1 and 1
        grid = torch.linspace(-1, 1, steps=grid_size+1)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(out_features, in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        
        # Spline coefficients for each piecewise linear segment (grid_size segments)
        # shape (out_features, in_features, grid_size)
        self.spline_coeff = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        
        self.scale_spline = scale_spline
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_coeff, mean=0.0, std=0.1)
    
    def forward(self, x):
        """
        x: (batch, in_features)
        Returns: (batch, out_features)
        """
        batch_size = x.shape[0]
        
        # Base linear part
        base_out = F.linear(x, self.base_weight)  # (batch, out)
        
        # Spline part: piecewise linear interpolation
        # For each input feature (in), we compute the output contribution per output neuron.
        # Expand x to (batch, out, in) for broadcasting with grid (out, in, grid+1)
        x_exp = x.unsqueeze(1)  # (batch, 1, in) -> will broadcast to out
        x_exp = x_exp.expand(-1, self.out_features, -1)  # (batch, out, in)
        
        # Clamp to grid range [-1, 1]
        x_clamped = torch.clamp(x_exp, -1.0, 1.0)
        
        # Find the interval index for each x
        # grid: (out, in, grid+1) -> need to compare with x_exp (batch, out, in)
        # We'll compute left index
        grid_expanded = self.grid.unsqueeze(0)  # (1, out, in, grid+1)
        # Find where x_clamped >= grid point
        mask = (x_clamped.unsqueeze(-1) >= grid_expanded)  # (batch, out, in, grid+1)
        # The index of the last True is the left segment index (0 to grid_size-1)
        left_idx = mask.sum(dim=-1) - 1  # (batch, out, in)
        left_idx = torch.clamp(left_idx, 0, self.grid_size - 1)  # clamp to valid
        
        # Get grid points for left and right
        # left grid value: grid[:, :, left_idx] ; right: grid[:, :, left_idx+1]
        # We need to gather from grid (out, in, grid+1)
        # Use index_select or gather
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).unsqueeze(2)
        out_indices = torch.arange(self.out_features, device=x.device).unsqueeze(0).unsqueeze(2)
        in_indices = torch.arange(self.in_features, device=x.device).unsqueeze(0).unsqueeze(1)
        
        left_grid = self.grid[out_indices, in_indices, left_idx]  # (batch, out, in)
        right_grid = self.grid[out_indices, in_indices, left_idx + 1]
        
        # Compute interpolation weight
        # Avoid division by zero
        denom = right_grid - left_grid
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        weight = (x_clamped - left_grid) / denom
        
        # Spline coefficient for the segment: (out, in, grid_size) -> (batch, out, in) via gather
        # left_idx is (batch, out, in) -> need to select along dim=2
        coeff = torch.gather(self.spline_coeff, 2, left_idx.unsqueeze(-1)).squeeze(-1)  # (batch, out, in)
        
        # Spline output: sum over in features
        spline_out = (coeff * weight).sum(dim=-1)  # (batch, out)
        
        # Combine
        return base_out + self.scale_spline * spline_out

class TemporalKANForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, grid_size=5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(KANLayer(prev, h, grid_size))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.net(x)
