import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierKANLayer(nn.Module):
    """
    KAN layer using Fourier basis (sin/cos) – more expressive for periodic patterns.
    """
    def __init__(self, in_features, out_features, num_frequencies=10, scale=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        
        # Base linear weights (for residual)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Fourier coefficients: for each frequency and each (out, in) pair
        # We'll use sin and cos separately: shape (out, in, num_frequencies*2)
        self.fourier_coeff = nn.Parameter(torch.Tensor(out_features, in_features, num_frequencies * 2))
        # Frequencies (fixed, from 1 to num_frequencies)
        freqs = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)  # (num_frequencies,)
        self.scale = scale
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.fourier_coeff, mean=0.0, std=0.1)
    
    def forward(self, x):
        # x: (batch, in_features)
        # Base linear part
        base_out = F.linear(x, self.base_weight)  # (batch, out)
        
        # Fourier features: for each input feature, compute sin(k*pi*x) and cos(k*pi*x)
        # Scale x to [-pi, pi] (assuming x is roughly in [-1,1] after scaling)
        x_scaled = x * math.pi  # (batch, in)
        
        # Expand to (batch, in, num_frequencies)
        x_exp = x_scaled.unsqueeze(-1)  # (batch, in, 1)
        freqs_exp = self.freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, num_freq)
        angles = x_exp * freqs_exp  # (batch, in, num_freq)
        
        sin_vals = torch.sin(angles)
        cos_vals = torch.cos(angles)
        # Concatenate sin and cos: (batch, in, 2*num_freq)
        fourier_basis = torch.cat([sin_vals, cos_vals], dim=-1)
        
        # Fourier output: sum over input features and frequencies
        # fourier_coeff: (out, in, 2*num_freq)
        fourier_out = torch.einsum('o i f, b i f -> b o', self.fourier_coeff, fourier_basis)
        
        return base_out + self.scale * fourier_out

class TemporalKANForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_frequencies=20):
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
