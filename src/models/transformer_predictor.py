"""
transformer_predictor.py
────────────────────────
Transformer-encoder-based model for multivariate time-series
failure detection and RUL estimation.

Architecture
────────────
  Input  (batch, seq_len, n_features)
    │
    ▼
  Linear projection → d_model
    +
  Sinusoidal positional encoding
    │
    ▼
  TransformerEncoder (N layers of MultiHeadSelfAttention + FFN)
    │
    ▼  (mean-pool across time)
  LayerNorm
    │
    ▼
  FC(d_model → fc_hidden) → ReLU → Dropout
    │
  ┌─┴──────────────────┐
  ▼                    ▼
  cls head           rul head
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer encoder with dual output head.

    Outputs
    -------
    failure_prob : (batch,)
    rul_pred     : (batch,)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        fc_hidden: int = 64,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.fc_shared = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(fc_hidden, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x : (batch, seq_len, n_features)
        """
        x = self.input_proj(x)          # (batch, seq, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)             # (batch, seq, d_model)
        x = x.mean(dim=1)              # mean-pool over time
        x = self.norm(x)

        shared = self.fc_shared(x)

        failure_prob = torch.sigmoid(self.cls_head(shared)).squeeze(-1)
        rul_pred = self.rul_head(shared).squeeze(-1)

        return failure_prob, rul_pred

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
