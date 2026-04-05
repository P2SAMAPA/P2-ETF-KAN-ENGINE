import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ReLUKANLayer(nn.Module):
    """
    ReLU-KAN: using ReLU as base activation and learnable piecewise linear functions.
    More stable than Fourier and better for noisy financial data.
    """
    def __init__(self, in_features, out_features, grid_size=20, scale=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        # Fixed grid points between -1 and 1
        grid = torch.linspace(-1, 1, steps=grid_size)
        self.register_buffer("grid", grid)
        self.scale = scale
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)
    
    def forward(self, x):
        # Base linear
        base_out = F.linear(x, self.base_weight)
        # Spline: ReLU(x - knot) for all knots
        x_exp = x.unsqueeze(-1)  # (batch, in, 1)
        knots_exp = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, G)
        relu = F.relu(x_exp - knots_exp)  # (batch, in, G)
        spline_out = torch.einsum('o i g, b i g -> b o', self.spline_weight, relu)
        return base_out + self.scale * spline_out

class TemporalKANForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, grid_size=20):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(ReLUKANLayer(prev, h, grid_size))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.net(x)
