import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierKANLayer(nn.Module):
    """
    KAN layer using Fourier basis (sin/cos) – matches training.
    """
    def __init__(self, in_features, out_features, num_frequencies=10, scale=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.fourier_coeff = nn.Parameter(torch.Tensor(out_features, in_features, num_frequencies * 2))
        freqs = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)
        self.scale = scale
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.fourier_coeff, mean=0.0, std=0.1)
    
    def forward(self, x):
        base_out = F.linear(x, self.base_weight)
        x_scaled = x * math.pi
        x_exp = x_scaled.unsqueeze(-1)
        freqs_exp = self.freqs.unsqueeze(0).unsqueeze(0)
        angles = x_exp * freqs_exp
        sin_vals = torch.sin(angles)
        cos_vals = torch.cos(angles)
        fourier_basis = torch.cat([sin_vals, cos_vals], dim=-1)
        fourier_out = torch.einsum('o i f, b i f -> b o', self.fourier_coeff, fourier_basis)
        return base_out + self.scale * fourier_out

class TemporalKANForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_frequencies=30):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(FourierKANLayer(prev, h, num_frequencies))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.net(x)
