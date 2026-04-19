"""
generate_dataset.py
───────────────────
Generates a realistic synthetic multivariate sensor dataset for
predictive maintenance, inspired by the NASA CMAPSS benchmark.

Each "unit" represents one piece of industrial equipment that
degrades over time until failure. Sensor readings exhibit:
  - gradual degradation trends
  - multi-modal operation (3 operating regimes)
  - additive Gaussian noise
  - sensor drift / bias faults
  - occasional transient spikes

Usage
-----
    python src/data/generate_dataset.py
    python src/data/generate_dataset.py --n-units 200 --seed 7
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path


# ─── Operating regime definitions ────────────────────────────────────────────
REGIMES = {
    0: {"rpm_base": 1200, "pressure_in": 0.90, "torque": 210},
    1: {"rpm_base": 1450, "pressure_in": 1.02, "torque": 245},
    2: {"rpm_base": 1700, "pressure_in": 1.18, "torque": 285},
}

SENSOR_NOISE = {
    "vibration_x": 0.015,
    "vibration_y": 0.012,
    "vibration_z": 0.010,
    "temperature_bearing": 0.8,
    "temperature_ambient": 0.5,
    "pressure_in": 0.008,
    "pressure_out": 0.007,
    "rpm": 8.0,
    "current_draw": 0.15,
    "oil_viscosity": 0.3,
    "acoustic_emission": 0.6,
    "torque": 2.0,
    "humidity": 1.2,
    "voltage": 0.4,
}


def degradation_curve(t: np.ndarray, max_life: int, mode: str = "exponential") -> np.ndarray:
    """
    Returns a degradation index in [0, 1] where 1 = imminent failure.

    Args:
        t:         time steps (0 … max_life)
        max_life:  total life length
        mode:      'exponential' | 'linear' | 'sigmoid'
    """
    x = t / max_life
    if mode == "exponential":
        return np.exp(3 * x) / np.exp(3) * x
    elif mode == "linear":
        return x
    elif mode == "sigmoid":
        return 1 / (1 + np.exp(-10 * (x - 0.7)))
    return x


def generate_unit(
    unit_id: int,
    rng: np.random.Generator,
    min_life: int = 100,
    max_life: int = 350,
) -> pd.DataFrame:
    """Generate sensor readings for one equipment unit."""

    life = rng.integers(min_life, max_life)
    times = np.arange(life)
    deg_mode = rng.choice(["exponential", "linear", "sigmoid"])
    degradation = degradation_curve(times, life, mode=deg_mode)

    # Assign operating regime per time step (random switches)
    regime_seq = rng.integers(0, 3, size=life)

    rows = []
    for i, t in enumerate(times):
        regime = REGIMES[regime_seq[i]]
        d = degradation[i]  # 0 = healthy, 1 = failure

        # ── Baseline sensor values (healthy, regime-adjusted) ──────────────
        vib_base = 0.10 + regime_seq[i] * 0.03
        temp_bear = 55.0 + regime_seq[i] * 8.0
        temp_amb = 22.0 + rng.normal(0, 0.3)
        press_in = regime["pressure_in"]
        press_out = press_in - 0.05
        rpm = regime["rpm_base"]
        current = 7.5 + regime_seq[i] * 0.6
        oil_visc = 46.0
        acoustic = 85.0 + regime_seq[i] * 2.0
        torque = regime["torque"]
        humidity = 40.0 + rng.normal(0, 2)
        voltage = 400.0

        # ── Degradation effects ────────────────────────────────────────────
        vib_x = vib_base + d * 0.45 + rng.normal(0, SENSOR_NOISE["vibration_x"])
        vib_y = vib_base + d * 0.38 + rng.normal(0, SENSOR_NOISE["vibration_y"])
        vib_z = vib_base * 0.7 + d * 0.22 + rng.normal(0, SENSOR_NOISE["vibration_z"])
        temp_bear += d * 35.0 + rng.normal(0, SENSOR_NOISE["temperature_bearing"])
        temp_amb += rng.normal(0, SENSOR_NOISE["temperature_ambient"])
        press_in += rng.normal(0, SENSOR_NOISE["pressure_in"])
        press_out = press_in - 0.05 - d * 0.12 + rng.normal(0, SENSOR_NOISE["pressure_out"])
        rpm += -d * 60 + rng.normal(0, SENSOR_NOISE["rpm"])
        current += d * 1.8 + rng.normal(0, SENSOR_NOISE["current_draw"])
        oil_visc -= d * 12.0 + rng.normal(0, SENSOR_NOISE["oil_viscosity"])
        acoustic += d * 18.0 + rng.normal(0, SENSOR_NOISE["acoustic_emission"])
        torque += d * 30.0 + rng.normal(0, SENSOR_NOISE["torque"])
        humidity += rng.normal(0, SENSOR_NOISE["humidity"])
        voltage -= d * 8.0 + rng.normal(0, SENSOR_NOISE["voltage"])

        # ── Rare transient spike (2% chance) ──────────────────────────────
        if rng.random() < 0.02:
            vib_x *= rng.uniform(2.5, 4.0)
            acoustic += rng.uniform(10, 25)

        # ── Labels ─────────────────────────────────────────────────────────
        rul = life - t - 1
        failure_imminent = int(rul <= 30)

        rows.append({
            "unit_id": unit_id,
            "cycle": t,
            "operating_regime": regime_seq[i],
            "vibration_x": round(vib_x, 4),
            "vibration_y": round(vib_y, 4),
            "vibration_z": round(vib_z, 4),
            "temperature_bearing": round(temp_bear, 2),
            "temperature_ambient": round(temp_amb, 2),
            "pressure_in": round(press_in, 4),
            "pressure_out": round(press_out, 4),
            "rpm": round(rpm, 1),
            "current_draw": round(current, 3),
            "oil_viscosity": round(oil_visc, 2),
            "acoustic_emission": round(acoustic, 2),
            "torque": round(torque, 2),
            "humidity": round(humidity, 1),
            "voltage": round(voltage, 1),
            "RUL": rul,
            "failure_imminent": failure_imminent,
        })

    return pd.DataFrame(rows)


def generate_dataset(n_units: int, seed: int, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    units = []
    for uid in range(1, n_units + 1):
        df_unit = generate_unit(unit_id=uid, rng=rng)
        units.append(df_unit)
        if uid % 10 == 0:
            print(f"  Generated unit {uid}/{n_units}")

    df = pd.concat(units, ignore_index=True)

    # ── Train / val / test split by unit_id ────────────────────────────────
    all_ids = np.arange(1, n_units + 1)
    rng.shuffle(all_ids)
    n_train = int(0.70 * n_units)
    n_val = int(0.15 * n_units)

    train_ids = set(all_ids[:n_train])
    val_ids = set(all_ids[n_train: n_train + n_val])
    test_ids = set(all_ids[n_train + n_val:])

    df_train = df[df["unit_id"].isin(train_ids)].reset_index(drop=True)
    df_val = df[df["unit_id"].isin(val_ids)].reset_index(drop=True)
    df_test = df[df["unit_id"].isin(test_ids)].reset_index(drop=True)

    df_train.to_csv(f"{output_dir}/train.csv", index=False)
    df_val.to_csv(f"{output_dir}/val.csv", index=False)
    df_test.to_csv(f"{output_dir}/test.csv", index=False)
    df.to_csv(f"{output_dir}/full_dataset.csv", index=False)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n✅ Dataset generation complete")
    print(f"   Total records  : {len(df):,}")
    print(f"   Total units    : {n_units}")
    print(f"   Train rows     : {len(df_train):,}  ({len(train_ids)} units)")
    print(f"   Val rows       : {len(df_val):,}  ({len(val_ids)} units)")
    print(f"   Test rows      : {len(df_test):,}  ({len(test_ids)} units)")
    print(f"   Failure ratio  : {df['failure_imminent'].mean():.1%}")
    print(f"   Avg unit life  : {df.groupby('unit_id')['cycle'].max().mean():.1f} cycles")
    print(f"   Saved to       : {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PM dataset")
    parser.add_argument("--n-units", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    args = parser.parse_args()

    print(f"Generating dataset: {args.n_units} units, seed={args.seed}")
    generate_dataset(args.n_units, args.seed, args.output_dir)
