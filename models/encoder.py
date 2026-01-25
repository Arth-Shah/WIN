# models/encoder.py

import torch
import torch.nn as nn


# ============================================================
# Positional Aggregator
# ============================================================

class PositionalAggregator1D(nn.Module):
    """
    Flattens CNN feature maps and adds sinusoidal positional encoding.

    Input : (B, C, F, T)
    Output: (B, F*T, C)
    """

    def __init__(
        self,
        max_C: int,
        max_ft: int,
    ):

        super().__init__()

        self.max_C = max_C
        self.max_ft = max_ft

        self.flattener = nn.Flatten(start_dim=-2, end_dim=-1)

        self._init_encoding()


    def _init_encoding(self):

        # Create on CPU first (will move with model.to())
        pos = torch.arange(
            1, self.max_ft - 1
        ).float().unsqueeze(1)          # (L-2, 1)

        dim = torch.arange(
            0, self.max_C, step=2
        ).float().unsqueeze(0)          # (1, D/2)

        enc = torch.zeros(
            self.max_ft,
            self.max_C,
        )

        enc[1:-1, 0::2] = torch.sin(
            pos / (10000 ** (dim / self.max_C))
        )

        enc[1:-1, 1::2] = torch.cos(
            pos / (10000 ** (dim / self.max_C))
        )

        # Register as buffer (moves automatically to device)
        self.register_buffer("encoding", enc)


    def forward(self, HFM):

        """
        Args:
            HFM: (B, C, F, T)

        Returns:
            (B, F*T, C)
        """

        B, C, f, t = HFM.shape

        ft = f * t

        out = self.flattener(HFM)      # (B, C, F*T)

        out = out.transpose(1, 2)      # (B, F*T, C)

        out = out + self.encoding[:ft, :C]

        return out
