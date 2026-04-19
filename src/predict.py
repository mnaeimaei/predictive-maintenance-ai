"""
predict.py
──────────
Convenience script for single or batch inference from CSV.

Usage
-----
    python src/predict.py --checkpoint outputs/best_model.pt --input data/raw/test.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.lstm_predictor import LSTMPredictor
from src.models.transformer_predictor import TransformerPredictor
from src.utils.preprocessing import add_rolling_features, extract_windows, load_scaler, SENSOR_COLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to CSV file with sensor data")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    df = pd.read_csv(args.input)
    if cfg["preprocessing"]["add_rolling_features"]:
        df = add_rolling_features(df, cfg["preprocessing"]["rolling_windows"])

    scaler = load_scaler(f"{cfg['output']['dir']}/scaler.pkl")
    df[SENSOR_COLS] = scaler.transform(df[SENSOR_COLS])

    feature_cols = [c for c in df.columns if c not in {"unit_id","cycle","operating_regime","RUL","failure_imminent"}]
    X, y_cls, y_rul = extract_windows(df, feature_cols, cfg["data"]["window_size"], cfg["data"]["stride"], cfg["data"]["rul_clip"])

    model_type = ckpt["model_type"]
    n_features = ckpt["n_features"]

    if model_type == "lstm":
        mc = cfg["model"]["lstm"]
        model = LSTMPredictor(n_features=n_features, hidden_size=mc["hidden_size"],
                              num_layers=mc["num_layers"], dropout=mc["dropout"],
                              bidirectional=mc["bidirectional"], fc_hidden=cfg["model"]["fc_hidden"])
    else:
        mc = cfg["model"]["transformer"]
        model = TransformerPredictor(n_features=n_features, d_model=mc["d_model"],
                                     nhead=mc["nhead"], num_encoder_layers=mc["num_encoder_layers"])

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X)
        probs, ruls = [], []
        for i in range(0, len(X_t), 256):
            p, r = model(X_t[i:i+256])
            probs.extend(p.numpy())
            ruls.extend(r.numpy() * cfg["data"]["rul_clip"])

    results = pd.DataFrame({
        "sample_idx": range(len(probs)),
        "failure_probability": np.round(probs, 4),
        "failure_predicted": (np.array(probs) >= cfg["evaluation"]["threshold"]).astype(int),
        "rul_predicted": np.round(ruls, 1),
        "true_label": y_cls.astype(int),
        "true_rul": y_rul,
    })
    results.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output} ({len(results)} rows)")


if __name__ == "__main__":
    main()
