import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLayer(nn.Module):
    """
    KAN layer using linear spline with ReLU basis functions.
    For each input feature j and output neuron i:
        output_i = sum_j ( w_base_ij * x_j + sum_{k=1}^{G} w_spline_ijk * max(0, x_j - knot_k) )
    where knots are fixed (e.g., evenly spaced between -1 and 1).
    This is efficient, stable, and standard for KAN approximations.
    """
    def __init__(self, in_features, out_features, grid_size=5, scale_spline=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Base linear weights (bias is incorporated in the spline part via zero knot?)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Spline weights: (out_features, in_features, grid_size)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        
        # Fixed knots: grid_size points between -1 and 1 (excluding -1? we can include)
        # We'll use grid_size knots, e.g., -1 + 2*k/(G+1) to avoid boundaries?
        # Better: use interior knots, but ReLU will handle.
        knots = torch.linspace(-1, 1, steps=grid_size+2)[1:-1]  # exclude -1 and 1? actually include
        # But we want knots within [-1,1]. Let's use evenly spaced.
        knots = torch.linspace(-1, 1, steps=grid_size)
        self.register_buffer("knots", knots)  # (grid_size,)
        
        self.scale_spline = scale_spline
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)
    
    def forward(self, x):
        """
        x: (batch, in_features)
        returns: (batch, out_features)
        """
        batch = x.shape[0]
        # Base linear: (batch, out)
        base_out = F.linear(x, self.base_weight)  # (batch, out)
        
        # Spline part: for each input feature, compute ReLU(x - knot) for all knots
        # x: (batch, in) -> expand to (batch, in, grid_size)
        x_exp = x.unsqueeze(-1)  # (batch, in, 1)
        knots_exp = self.knots.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size)
        relu = F.relu(x_exp - knots_exp)  # (batch, in, grid_size)
        
        # Spline contribution: sum over knots and input features
        # spline_weight: (out, in, grid_size)
        # relu: (batch, in, grid_size)
        # We want (batch, out) = sum_{in, grid} spline_weight[out, in, grid] * relu[batch, in, grid]
        spline_out = torch.einsum('o i g, b i g -> b o', self.spline_weight, relu)
        
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
