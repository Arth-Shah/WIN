import torch
import torch.nn as nn
import math


class BumpWavelet(nn.Module):
    """
    Learnable Bump wavelet layer for waveform-based feature extraction.

    This module implements a differentiable Bump wavelet with learnable
    scale parameters and integrates it directly into the network,
    enabling end-to-end optimization.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Linear(in_channels, out_channels)

        # Learnable log-scale for numerical stability
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C)

        Returns:
            Tensor of shape (B, T, out_channels)
        """
        z = self.proj(x)

        scale = torch.exp(self.log_scale) + 1e-6
        u = z / scale

        # Bump wavelet
        wavelet = torch.exp(-1.0 / (1 - u ** 2).clamp(min=1e-6))
        wavelet = wavelet * (u.abs() < 1).float()

        return wavelet
