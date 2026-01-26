# models/WIN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Local imports
from models.preprocess import PreEmphasis
from models.frontend import Frontend_S
from models.encoder import PositionalAggregator1D
from models.WIN_classifier import TransformerEncoderLayerWavelet


# ============================================================
# Sequence Pooling
# ============================================================

class SequencePooling(nn.Module):
    """
    Attention-style weighted pooling.

    Input : (B, S, C)
    Output: (B, C)
    """

    def __init__(self, d_model: int):

        super().__init__()

        self.linear = nn.Linear(d_model, 1)


    def forward(self, x):

        # x: (B, S, C)

        w = self.linear(x)                 # (B, S, 1)

        w = F.softmax(
            w.transpose(1, 2),
            dim=-1,
        )                                  # (B, 1, S)

        out = torch.matmul(w, x)           # (B, 1, C)

        return out.squeeze(1)              # (B, C)


# ============================================================
# WIN Classifier
# ============================================================

class WIN_c(nn.Module):

    def __init__(
        self,
        C,
        n_encoder,
        transformer_hidden,
        wavelet_type="bump",
    ):

        super().__init__()


        self.encoders = nn.Sequential(
            OrderedDict([
                (
                    f"encoder{i}",
                    TransformerEncoderLayerWavelet(
                        d_model=C,
                        n_head=8,
                        ffn_hidden=transformer_hidden,
                        wavelet_type=wavelet_type,
                    )
                )
                for i in range(n_encoder)
            ])
        )


        self.seq_pool = SequencePooling(d_model=C)

        self.fc = nn.Linear(C, 1)


    def forward(self, x):

        x = self.encoders(x)

        x = self.seq_pool(x)

        x = self.fc(x)

        return torch.sigmoid(x).squeeze(-1)


# ============================================================
# Full WIN Model
# ============================================================

class WIN(nn.Module):
    """
    Wavelet Interface Network (WIN)
    """
    
    def __init__(
        self,
        sample_rate=16000,
        pre_emphasis=0.97,
        transformer_hidden=64,
        n_encoder=2,
        C=64,
        wavelet_type="bump",   # NEW
    ):


        super().__init__()


        # ---------- Preprocessing ----------
        self.pre_emphasis = PreEmphasis(pre_emphasis)


        # ---------- Frontend ----------
        self.front_end = Frontend_S(
            sinc_kernel_size=128,
            sample_rate=sample_rate,
        )


        # ---------- Positional Encoding ----------
        self.positional_embedding = PositionalAggregator1D(
            max_C=C,
            max_ft=23 * 16,   # from frontend geometry
        )


        # ---------- Classifier ----------
        self.classifier = WIN_c(
            C=C,
            n_encoder=n_encoder,
            transformer_hidden=transformer_hidden,
            wavelet_type=wavelet_type,
        )


    def forward(self, x):

        """
        Args:
            x: (B, T) waveform

        Returns:
            (B,) probability
        """

        # Pre-emphasis
        x = self.pre_emphasis(x)

        # Frontend
        x = self.front_end(x)

        # Tokenization + PosEnc
        x = self.positional_embedding(x)

        # Classifier
        x = self.classifier(x)

        return x
