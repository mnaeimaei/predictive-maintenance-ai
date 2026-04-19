"""
evaluate.py
───────────
Full evaluation of a trained checkpoint on the test set.

Outputs
───────
  - Classification report (per-class precision / recall / F1)
  - Confusion matrix (saved to PNG)
  - ROC curve + AUC
  - RUL regression metrics (MAE, RMSE)
  - Threshold sweep (optimal operating point)

Usage
-----
    python src/evaluate.py --checkpoint outputs/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import build_dataloader
from src.models.lstm_predictor import LSTMPredictor
from src.models.transformer_predictor import TransformerPredictor
from src.utils.preprocessing import (
    add_rolling_features,
    extract_windows,
    load_scaler,
    SENSOR_COLS,
)


def load_checkpoint(path: str, device: torch.device) -> tuple:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    n_features = ckpt["n_features"]
    model_type = ckpt["model_type"]

    if model_type == "lstm":
        mc = cfg["model"]["lstm"]
        model = LSTMPredictor(
            n_features=n_features,
            hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"],
            dropout=mc["dropout"],
            bidirectional=mc["bidirectional"],
            fc_hidden=cfg["model"]["fc_hidden"],
        )
    else:
        mc = cfg["model"]["transformer"]
        model = TransformerPredictor(
            n_features=n_features,
            d_model=mc["d_model"],
            nhead=mc["nhead"],
            num_encoder_layers=mc["num_encoder_layers"],
            dim_feedforward=mc["dim_feedforward"],
            dropout=mc["dropout"],
            max_seq_len=mc["max_seq_len"],
            fc_hidden=cfg["model"]["fc_hidden"],
        )

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def run_inference(model, loader, device, rul_clip: float):
    probs, labels, rul_preds, rul_true = [], [], [], []
    model = model.to(device)
    for X, y_cls, y_rul in loader:
        X = X.to(device)
        prob, rul = model(X)
        probs.extend(prob.cpu().numpy())
        labels.extend(y_cls.numpy())
        rul_preds.extend(rul.cpu().numpy() * rul_clip)
        rul_true.extend(y_rul.numpy())
    return (
        np.array(probs),
        np.array(labels, dtype=int),
        np.array(rul_preds),
        np.array(rul_true),
    )


def plot_confusion_matrix(cm, save_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Failure"]);
    ax.set_yticklabels(["Normal", "Failure"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


def plot_roc(fpr, tpr, auc, save_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Failure Detection")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved to {save_path}")


def threshold_sweep(probs, labels):
    """Find the threshold that maximises F1."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Model: {ckpt['model_type'].upper()}  |  Epoch: {ckpt['epoch']}  |  Val F1: {ckpt['val_f1']:.4f}")

    # ── Load test data ───────────────────────────────────────────────────────
    data_dir = cfg["data"]["raw_dir"]
    df_test = pd.read_csv(f"{data_dir}/test.csv")

    if cfg["preprocessing"]["add_rolling_features"]:
        df_test = add_rolling_features(df_test, cfg["preprocessing"]["rolling_windows"])

    scaler = load_scaler(f"{cfg['output']['dir']}/scaler.pkl")
    df_test[SENSOR_COLS] = scaler.transform(df_test[SENSOR_COLS])

    feature_cols = [c for c in df_test.columns
                    if c not in {"unit_id", "cycle", "operating_regime", "RUL", "failure_imminent"}]

    X_test, y_cls_test, y_rul_test = extract_windows(
        df_test, feature_cols,
        cfg["data"]["window_size"],
        cfg["data"]["stride"],
        cfg["data"]["rul_clip"],
    )
    test_loader = build_dataloader(X_test, y_cls_test, y_rul_test, batch_size=256, shuffle=False)

    # ── Inference ────────────────────────────────────────────────────────────
    print("Running inference on test set...")
    probs, labels, rul_preds, rul_true = run_inference(model, test_loader, device, cfg["data"]["rul_clip"])

    # ── Threshold ────────────────────────────────────────────────────────────
    opt_thresh, opt_f1 = threshold_sweep(probs, labels)
    preds = (probs >= opt_thresh).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(labels, preds, target_names=["Normal", "Failure"], digits=4))

    auc = roc_auc_score(labels, probs)
    rul_mae = np.mean(np.abs(rul_preds - rul_true))
    rul_rmse = np.sqrt(np.mean((rul_preds - rul_true) ** 2))

    print(f"ROC-AUC            : {auc:.4f}")
    print(f"Optimal threshold  : {opt_thresh:.2f}  (F1={opt_f1:.4f})")
    print(f"RUL MAE            : {rul_mae:.2f} cycles")
    print(f"RUL RMSE           : {rul_rmse:.2f} cycles")

    # ── Plots ────────────────────────────────────────────────────────────────
    out_dir = cfg["output"]["dir"]
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, f"{out_dir}/confusion_matrix.png")
    fpr, tpr, _ = roc_curve(labels, probs)
    plot_roc(fpr, tpr, auc, f"{out_dir}/roc_curve.png")

    print(f"\n✅ Evaluation complete. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
