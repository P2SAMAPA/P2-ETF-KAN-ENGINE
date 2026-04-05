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
        h = 1.0 / grid_size
        grid = torch.linspace(-1.0, 1.0, steps=grid_size + 1)
        grid = grid.unsqueeze(0).repeat(out_features, in_features, 1)
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.Tensor(
            out_features, in_features, grid_size + spline_order
        ))
        self.scale_spline = scale_spline
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)
    
    def b_spline(self, x, grid, k=3):
        x = x.unsqueeze(-1)
        grid = grid.unsqueeze(0)
        basis = (x >= grid[..., :-1]) * (x < grid[..., 1:]).float()
        for _ in range(k):
            basis = (basis * (x - grid[..., :-1-k]) / (grid[..., k:-1] - grid[..., :-1-k]) +
                    basis[..., 1:] * (grid[..., k+1:] - x) / (grid[..., k+1:] - grid[..., 1:-k]))
        return basis[..., 0]
    
    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        x_expanded = x.unsqueeze(1).unsqueeze(2)
        spline_basis = self.b_spline(x_expanded, self.grid)
        spline_output = torch.einsum('b i j k, o i j k -> b o', spline_basis, self.spline_weight)
        return base_output + self.scale_spline * spline_output

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
