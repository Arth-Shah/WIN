# models/WIN_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Wavelet Feature Map (Bump Wavelet)
# ============================================================

class WaveletFeatureMap(nn.Module):
    """
    Bump Wavelet Feature Mapping (SAFE version)
    """

    def __init__(
        self,
        d_head: int,
        d_phi: int = None,
    ):

        super().__init__()

        if d_phi is None:
            d_phi = d_head

        assert d_phi % 2 == 0, "d_phi must be even"

        self.linear_p = nn.Linear(d_head, d_phi // 2)
        self.linear_g = nn.Linear(d_head, d_phi // 2)

        self.act = nn.GELU()

        self.log_scale = nn.Parameter(torch.zeros(1))
        self.gate = nn.Parameter(torch.randn(1))


    def forward(self, x: torch.Tensor):

        """
        Args:
            x: (B, S, d_head)

        Returns:
            (B, S, d_phi)
        """

        z = self.linear_p(x)

        g = self.act(self.linear_g(x))

        # Stable scale
        s = torch.exp(self.log_scale) + 1e-4

        u = z / s


        # -------- Bump Wavelet -------- #

        wavelet = torch.zeros_like(u)

        mask = u.abs() < 1.0

        wavelet[mask] = torch.exp(
            -1.0 / (1.0 - u[mask] ** 2)
        )

        # Normalize
        wavelet = wavelet / (
            wavelet.std(dim=-1, keepdim=True) + 1e-5
        )


        gate = torch.sigmoid(self.gate)

        out = torch.cat(
            [
                gate * wavelet,
                (1.0 - gate) * g,
            ],
            dim=-1,
        )

        return out


# ============================================================
# Multi-Head Wavelet Module
# ============================================================

class MultiHeadWavelet(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_head: int = 8,
        d_head: int = None,
    ):

        super().__init__()

        self.n_head = n_head

        if d_head is None:
            assert d_model % n_head == 0
            d_head = d_model // n_head

        self.d_head = d_head


        self.heads = nn.ModuleList(
            [
                WaveletFeatureMap(d_head)
                for _ in range(n_head)
            ]
        )


        self.out_proj = nn.Linear(
            n_head * d_head,
            d_model,
        )


    def forward(self, x: torch.Tensor):

        """
        Args:
            x: (B, S, D)

        Returns:
            (B, S, D)
        """

        B, S, D = x.shape

        x = (
            x.view(B, S, self.n_head, self.d_head)
             .permute(0, 2, 1, 3)
        )  # (B, H, S, d_head)


        head_outs = []

        for h in range(self.n_head):

            head_out = self.heads[h](x[:, h])

            head_outs.append(head_out)


        out = torch.cat(head_outs, dim=-1)

        return self.out_proj(out)


# ============================================================
# Custom LayerNorm
# ============================================================

class LayerNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-12,
    ):

        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        self.eps = eps


    def forward(self, x: torch.Tensor):

        mean = x.mean(-1, keepdim=True)

        var = x.var(-1, unbiased=False, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x_hat + self.beta


# ============================================================
# Feed-Forward Network
# ============================================================

class FFN(nn.Module):

    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        drop_prob: float = 0.1,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),

            nn.Dropout(drop_prob),

            nn.Linear(ffn_hidden, d_model),

        )


    def forward(self, x: torch.Tensor):

        return self.net(x)


# ============================================================
# Wavelet Transformer Encoder Layer
# ============================================================

class TransformerEncoderLayerWavelet(nn.Module):

    def __init__(
        self,
        d_model: int = 64,
        n_head: int = 8,
        ffn_hidden: int = 2048,
        drop_prob: float = 0.1,
    ):

        super().__init__()


        # Attention
        self.attn = MultiHeadWavelet(d_model, n_head)

        self.dropout1 = nn.Dropout(drop_prob)

        self.norm1 = LayerNorm(d_model)


        # Feedforward
        self.ffn = FFN(d_model, ffn_hidden, drop_prob)

        self.dropout2 = nn.Dropout(drop_prob)

        self.norm2 = LayerNorm(d_model)


    def forward(self, x: torch.Tensor):

        # Attention block
        residual = x

        x = self.attn(x)

        x = self.dropout1(x)

        x = self.norm1(x + residual)


        # FFN block
        residual = x

        x = self.ffn(x)

        x = self.dropout2(x)

        x = self.norm2(x + residual)


        return x
