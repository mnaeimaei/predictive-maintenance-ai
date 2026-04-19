"""
lstm_predictor.py
─────────────────
Bidirectional stacked LSTM for multivariate time-series failure prediction.

Architecture
────────────
  Input  (batch, seq_len, n_features)
    │
    ▼
  Bi-LSTM × num_layers  ──→  (batch, seq_len, 2 × hidden_size)
    │
    ▼  (take last timestep)
  LayerNorm
    │
    ▼
  FC(2×hidden → fc_hidden) → ReLU → Dropout
    │
  ┌─┴──────────────────────┐
  ▼                        ▼
  FC → sigmoid           FC → ReLU
  failure_prob           RUL estimate
"""

import torch
import torch.nn as nn
from torch import Tensor


class LSTMPredictor(nn.Module):
    """
    Bidirectional LSTM with dual output head.

    Outputs
    -------
    failure_prob : (batch,)   probability of imminent failure
    rul_pred     : (batch,)   estimated remaining useful life
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        fc_hidden: int = 64,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1
        lstm_out = hidden_size * directions

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.norm = nn.LayerNorm(lstm_out)
        self.dropout = nn.Dropout(dropout)

        # Shared trunk
        self.fc_shared = nn.Sequential(
            nn.Linear(lstm_out, fc_hidden),
            nn.ReLU(),
            self.dropout,
        )

        # Head 1: binary classification
        self.cls_head = nn.Linear(fc_hidden, 1)

        # Head 2: RUL regression
        self.rul_head = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1),
            nn.ReLU(),   # RUL ≥ 0
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget-gate bias = 1 (helps long-range memory)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x : (batch, seq_len, n_features)

        Returns:
            failure_prob : (batch,)
            rul_pred     : (batch,)
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq, lstm_out)
        last = lstm_out[:, -1, :]           # take last timestep
        last = self.norm(last)

        shared = self.fc_shared(last)       # (batch, fc_hidden)

        failure_prob = torch.sigmoid(self.cls_head(shared)).squeeze(-1)
        rul_pred = self.rul_head(shared).squeeze(-1)

        return failure_prob, rul_pred

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
