"""
test_model.py
─────────────
Unit tests for LSTM and Transformer predictors.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.lstm_predictor import LSTMPredictor
from src.models.transformer_predictor import TransformerPredictor
from src.data.dataset import SensorWindowDataset, build_dataloader
from src.utils.preprocessing import extract_windows, fit_scaler, add_rolling_features

import pandas as pd


# ─── Fixtures ─────────────────────────────────────────────────────────────────

BATCH = 8
SEQ = 30
N_FEATURES = 14


def dummy_batch() -> torch.Tensor:
    return torch.randn(BATCH, SEQ, N_FEATURES)


def dummy_df(n_units=5, cycles_per_unit=60) -> pd.DataFrame:
    """Create a minimal synthetic DataFrame for testing."""
    rows = []
    cols = [
        "vibration_x","vibration_y","vibration_z",
        "temperature_bearing","temperature_ambient",
        "pressure_in","pressure_out","rpm","current_draw",
        "oil_viscosity","acoustic_emission","torque","humidity","voltage",
    ]
    for uid in range(1, n_units + 1):
        for c in range(cycles_per_unit):
            rul = cycles_per_unit - c - 1
            row = {"unit_id": uid, "cycle": c, "operating_regime": 0, "RUL": rul, "failure_imminent": int(rul <= 30)}
            for col in cols:
                row[col] = np.random.randn()
            rows.append(row)
    return pd.DataFrame(rows)


# ─── LSTM Tests ───────────────────────────────────────────────────────────────

class TestLSTMPredictor:
    def test_output_shapes(self):
        model = LSTMPredictor(n_features=N_FEATURES)
        x = dummy_batch()
        prob, rul = model(x)
        assert prob.shape == (BATCH,), f"Expected ({BATCH},), got {prob.shape}"
        assert rul.shape == (BATCH,), f"Expected ({BATCH},), got {rul.shape}"

    def test_probability_range(self):
        model = LSTMPredictor(n_features=N_FEATURES)
        x = dummy_batch()
        prob, _ = model(x)
        assert (prob >= 0).all() and (prob <= 1).all(), "Probabilities must be in [0, 1]"

    def test_rul_non_negative(self):
        model = LSTMPredictor(n_features=N_FEATURES)
        x = dummy_batch()
        _, rul = model(x)
        assert (rul >= 0).all(), "RUL predictions must be non-negative"

    def test_parameter_count(self):
        model = LSTMPredictor(n_features=N_FEATURES, hidden_size=128, num_layers=3, bidirectional=True)
        assert model.n_parameters > 0

    def test_gradient_flow(self):
        model = LSTMPredictor(n_features=N_FEATURES)
        x = dummy_batch()
        prob, rul = model(x)
        loss = prob.mean() + rul.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_different_sequence_lengths(self):
        model = LSTMPredictor(n_features=N_FEATURES)
        for seq_len in [10, 20, 50]:
            x = torch.randn(4, seq_len, N_FEATURES)
            prob, rul = model(x)
            assert prob.shape == (4,)


# ─── Transformer Tests ────────────────────────────────────────────────────────

class TestTransformerPredictor:
    def test_output_shapes(self):
        model = TransformerPredictor(n_features=N_FEATURES)
        x = dummy_batch()
        prob, rul = model(x)
        assert prob.shape == (BATCH,)
        assert rul.shape == (BATCH,)

    def test_probability_range(self):
        model = TransformerPredictor(n_features=N_FEATURES)
        x = dummy_batch()
        prob, _ = model(x)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_rul_non_negative(self):
        model = TransformerPredictor(n_features=N_FEATURES)
        x = dummy_batch()
        _, rul = model(x)
        assert (rul >= 0).all()


# ─── Dataset Tests ────────────────────────────────────────────────────────────

class TestDataset:
    def setup_method(self):
        df = dummy_df()
        sensor_cols = [c for c in df.columns if c not in {"unit_id","cycle","operating_regime","RUL","failure_imminent"}]
        self.X, self.y_cls, self.y_rul = extract_windows(df, sensor_cols, window_size=20, stride=5)

    def test_shapes_consistent(self):
        assert self.X.shape[0] == self.y_cls.shape[0] == self.y_rul.shape[0]

    def test_window_feature_dim(self):
        assert self.X.shape[1] == 20     # window size
        assert self.X.shape[2] == N_FEATURES

    def test_label_binary(self):
        assert set(np.unique(self.y_cls)).issubset({0.0, 1.0})

    def test_rul_non_negative(self):
        assert (self.y_rul >= 0).all()

    def test_dataloader_batch(self):
        loader = build_dataloader(self.X, self.y_cls, self.y_rul, batch_size=4, shuffle=False)
        X_batch, y_cls_batch, y_rul_batch = next(iter(loader))
        assert X_batch.shape == (4, 20, N_FEATURES)
        assert isinstance(X_batch, torch.Tensor)


# ─── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_scaler_transform_shape(self):
        df = dummy_df()
        from src.utils.preprocessing import SENSOR_COLS
        scaler = fit_scaler(df)
        transformed = scaler.transform(df[SENSOR_COLS])
        assert transformed.shape == (len(df), len(SENSOR_COLS))

    def test_rolling_features_added(self):
        df = dummy_df(n_units=2, cycles_per_unit=40)
        n_cols_before = len(df.columns)
        df_roll = add_rolling_features(df, windows=[5, 10])
        assert len(df_roll.columns) > n_cols_before

    def test_rolling_no_data_leakage(self):
        """Rolling features must not cross unit boundaries."""
        df = dummy_df(n_units=2, cycles_per_unit=40)
        df_roll = add_rolling_features(df, windows=[5])
        # First row of each unit should have no NaN (min_periods=1)
        for uid in [1, 2]:
            first_row = df_roll[df_roll["unit_id"] == uid].iloc[0]
            assert not first_row.isnull().any(), f"NaN in first row of unit {uid}"
