"""
train.py
────────
Main training script for predictive maintenance models.

Features
────────
  - Multi-task loss: BCE (failure detection) + MSE (RUL regression)
  - Cosine LR schedule with warmup
  - Early stopping on validation F1
  - Gradient clipping
  - Optional MLflow experiment tracking
  - Saves best checkpoint + scaler

Usage
-----
    python src/train.py
    python src/train.py --model transformer --epochs 100
    python src/train.py --mlflow
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import build_dataloader
from src.models.lstm_predictor import LSTMPredictor
from src.models.transformer_predictor import TransformerPredictor
from src.utils.preprocessing import (
    SENSOR_COLS,
    add_rolling_features,
    apply_scaler,
    extract_windows,
    fit_scaler,
    save_scaler,
)


# ─── Config & CLI ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["lstm", "transformer"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(cfg: dict, n_features: int, model_type: str) -> nn.Module:
    if model_type == "lstm":
        mc = cfg["model"]["lstm"]
        return LSTMPredictor(
            n_features=n_features,
            hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"],
            dropout=mc["dropout"],
            bidirectional=mc["bidirectional"],
            fc_hidden=cfg["model"]["fc_hidden"],
        )
    else:
        mc = cfg["model"]["transformer"]
        return TransformerPredictor(
            n_features=n_features,
            d_model=mc["d_model"],
            nhead=mc["nhead"],
            num_encoder_layers=mc["num_encoder_layers"],
            dim_feedforward=mc["dim_feedforward"],
            dropout=mc["dropout"],
            max_seq_len=mc["max_seq_len"],
            fc_hidden=cfg["model"]["fc_hidden"],
        )


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
) -> dict:
    model.train()
    cls_w = cfg["training"]["cls_loss_weight"]
    rul_w = cfg["training"]["rul_loss_weight"]
    pos_w = torch.tensor(cfg["training"]["pos_weight"]).to(device)
    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss()
    grad_clip = cfg["training"]["grad_clip"]

    total_loss, total_cls, total_rul = 0.0, 0.0, 0.0
    all_probs, all_labels = [], []

    for X, y_cls, y_rul in loader:
        X = X.to(device)
        y_cls = y_cls.to(device)
        y_rul = y_rul.to(device) / cfg["data"]["rul_clip"]  # normalise 0–1

        optimizer.zero_grad()
        prob, rul = model(X)

        # Weighted BCE (upweight positive class)
        weights = torch.where(y_cls == 1, pos_w, torch.ones_like(y_cls))
        cls_loss = (bce(prob, y_cls) * weights).mean()
        rul_loss = mse(rul, y_rul)

        loss = cls_w * cls_loss + rul_w * rul_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_rul += rul_loss.item()
        all_probs.extend(prob.detach().cpu().numpy())
        all_labels.extend(y_cls.cpu().numpy())

    n = len(loader)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls / n,
        "rul_loss": total_rul / n,
        "f1": f1,
    }


@torch.no_grad()
def validate(model: nn.Module, loader, device: torch.device, cfg: dict) -> dict:
    model.eval()
    pos_w = torch.tensor(cfg["training"]["pos_weight"]).to(device)
    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss()

    total_loss = 0.0
    all_probs, all_labels, all_rul_pred, all_rul_true = [], [], [], []

    for X, y_cls, y_rul in loader:
        X = X.to(device)
        y_cls = y_cls.to(device)
        y_rul_norm = y_rul.to(device) / cfg["data"]["rul_clip"]

        prob, rul = model(X)
        weights = torch.where(y_cls == 1, pos_w, torch.ones_like(y_cls))
        cls_loss = (bce(prob, y_cls) * weights).mean()
        rul_loss = mse(rul, y_rul_norm)
        total_loss += (cfg["training"]["cls_loss_weight"] * cls_loss +
                       cfg["training"]["rul_loss_weight"] * rul_loss).item()

        all_probs.extend(prob.cpu().numpy())
        all_labels.extend(y_cls.cpu().numpy())
        all_rul_pred.extend(rul.cpu().numpy() * cfg["data"]["rul_clip"])
        all_rul_true.extend(y_rul.numpy())

    preds = (np.array(all_probs) >= cfg["evaluation"]["threshold"]).astype(int)
    labels = np.array(all_labels)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, all_probs) if len(np.unique(labels)) > 1 else 0.0
    rul_mae = np.mean(np.abs(np.array(all_rul_pred) - np.array(all_rul_true)))

    return {
        "val_loss": total_loss / len(loader),
        "val_f1": f1,
        "val_auc": auc,
        "val_rul_mae": rul_mae,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    model_type = args.model or cfg["model"]["type"]
    epochs = args.epochs or cfg["training"]["epochs"]
    lr = args.lr or cfg["training"]["learning_rate"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}")

    # ── Load data ───────────────────────────────────────────────────────────
    data_dir = cfg["data"]["raw_dir"]
    df_train = pd.read_csv(f"{data_dir}/train.csv")
    df_val = pd.read_csv(f"{data_dir}/val.csv")

    # ── Rolling features ────────────────────────────────────────────────────
    if cfg["preprocessing"]["add_rolling_features"]:
        print("Adding rolling features...")
        df_train = add_rolling_features(df_train, cfg["preprocessing"]["rolling_windows"])
        df_val = add_rolling_features(df_val, cfg["preprocessing"]["rolling_windows"])

    feature_cols = [c for c in df_train.columns
                    if c not in {"unit_id", "cycle", "operating_regime", "RUL", "failure_imminent"}]

    # ── Scaler ──────────────────────────────────────────────────────────────
    print("Fitting scaler...")
    scaler = fit_scaler(df_train, method=cfg["preprocessing"]["method"])
    df_train[SENSOR_COLS] = scaler.transform(df_train[SENSOR_COLS])
    df_val[SENSOR_COLS] = scaler.transform(df_val[SENSOR_COLS])
    out_dir = cfg["output"]["dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_scaler(scaler, f"{out_dir}/scaler.pkl")

    # ── Sliding windows ─────────────────────────────────────────────────────
    ws = cfg["data"]["window_size"]
    print(f"Extracting windows (size={ws})...")
    X_train, y_cls_train, y_rul_train = extract_windows(
        df_train, feature_cols, ws, cfg["data"]["stride"], cfg["data"]["rul_clip"]
    )
    X_val, y_cls_val, y_rul_val = extract_windows(
        df_val, feature_cols, ws, cfg["data"]["stride"], cfg["data"]["rul_clip"]
    )

    n_features = X_train.shape[2]
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Features: {n_features}")
    print(f"Failure rate train: {y_cls_train.mean():.2%}  val: {y_cls_val.mean():.2%}")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_loader = build_dataloader(
        X_train, y_cls_train, y_rul_train,
        batch_size=batch_size,
        shuffle=True,
        use_weighted_sampler=cfg["training"]["use_weighted_sampler"],
    )
    val_loader = build_dataloader(X_val, y_cls_val, y_rul_val, batch_size=batch_size, shuffle=False)

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(cfg, n_features, model_type).to(device)
    print(f"Model: {model_type.upper()}  |  Parameters: {model.n_parameters:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    # ── Optional MLflow ─────────────────────────────────────────────────────
    mlflow_run = None
    if args.mlflow:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        mlflow_run = mlflow.start_run()
        mlflow.log_params({
            "model": model_type, "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "window_size": ws, "n_features": n_features,
        })

    # ── Training loop ────────────────────────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]
    ckpt_path = f"{out_dir}/{cfg['output']['checkpoint_name']}"

    print(f"\n{'Epoch':>6} | {'Loss':>8} | {'F1':>6} | {'ValLoss':>8} | {'ValF1':>6} | {'AUC':>6} | {'RUL_MAE':>8}")
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_m = validate(model, val_loader, device, cfg)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"{epoch:>6} | {train_m['loss']:>8.4f} | {train_m['f1']:>6.4f} | "
            f"{val_m['val_loss']:>8.4f} | {val_m['val_f1']:>6.4f} | "
            f"{val_m['val_auc']:>6.4f} | {val_m['val_rul_mae']:>8.2f}  [{elapsed:.1f}s]"
        )

        if args.mlflow:
            import mlflow
            mlflow.log_metrics({**train_m, **val_m}, step=epoch)

        # ── Checkpoint ──────────────────────────────────────────────────────
        if val_m["val_f1"] > best_f1:
            best_f1 = val_m["val_f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": best_f1,
                "val_auc": val_m["val_auc"],
                "n_features": n_features,
                "model_type": model_type,
                "config": cfg,
            }, ckpt_path)
            print(f"  ✓ Saved best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if mlflow_run:
        import mlflow
        mlflow.log_artifact(ckpt_path)
        mlflow.end_run()

    print(f"\n✅ Training complete. Best val F1: {best_f1:.4f}")
    print(f"   Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
