# Predictive Maintenance AI — Industrial Sensor Failure Detection

[](https://github.com/yourusername/predictive-maintenance-ai/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project uses **LSTM** and **Transformer** models for predictive maintenance on industrial machinery. Given multivariate sensor time-series data, the models perform two tasks simultaneously: **Remaining Useful Life (RUL) estimation** and **failure prediction**, helping maintenance teams prevent costly breakdowns and optimize repair schedules.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model and Architecture Design](#model-and-architecture-design)
- [LSTM Architecture](#lstm-architecture)
- [Transformer Architecture](#transformer-architecture)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [API](#api)
- [Docker](#docker)
- [Results](#results)

---

## Overview

Industrial machinery — such as jet engines, turbines, pumps, and motors — degrades over time. The challenge is not only detecting failures but also predicting them before they occur. This project addresses two of the most critical questions in industrial operations:

- **How long until this machine fails?** *(Remaining Useful Life estimation)*
- **Is this machine about to fail?** *(Anomaly and fault detection)*

Accurately answering these questions is the foundation of **predictive maintenance** — a strategy that enables maintenance to be performed just before equipment failure, avoiding both unnecessary scheduled replacements and costly unexpected breakdowns.

Some applications of this project across different domains include:

| Domain | Application |
| --- | --- |
| Wind turbines | Gearbox and bearing failure prediction |
| Water infrastructure | Pump cavitation and seal degradation detection |
| Rail & transport | Wheel bearing and axle fatigue monitoring |
| Oil & gas | Compressor anomaly detection |
| Manufacturing | CNC spindle predictive maintenance |

## Dataset

To study this problem, a synthetic multivariate sensor dataset was created, inspired by NASA's CMAPSS benchmark — a widely used public dataset of simulated turbofan engine runs that gradually degrade until failure. CMAPSS is commonly regarded as a standard benchmark for evaluating failure prediction and remaining useful life (RUL) estimation models.

The dataset simulates a fleet of machines operating under realistic conditions. Each machine has 14 sensors that measure variables such as vibration, temperature, pressure, rotational speed, and electrical current. As the machine ages, these sensor readings gradually change to reflect degradation. The dataset also incorporates realistic measurement noise, multiple operating conditions, and occasional sensor spikes.

The dataset contains **22,000 records** across **18 columns**, grouped into four categories.

### 1. Identifiers (2 columns)

These columns are used for bookkeeping and are **not provided to the model as input features**.

- **unit_id** — Identifies the machine to which a row belongs (1, 2, 3, …). It is used to keep machine cycles together and to perform train/validation/test splits by machine. The dataset contains **100 unique machines**.

- **cycle** — The time step of a machine, starting from 0. The first cycle represents a new machine, while the final cycle represents the last observation before failure. This serves as the time axis.

### 2. Operating Context (1 column)

- **operating_regime** — Indicates one of three operating conditions:
  - 0 = Light load
  - 1 = Normal load
  - 2 = Heavy load

This variable shifts sensor baselines independently of machine degradation. For example, vibration may be higher because the machine is operating under heavy load rather than because it is approaching failure. The model must learn to distinguish between operating-condition effects and actual degradation.

### 3. Sensor Measurements (14 columns)

These are the model input features (**X**). Each sensor value is generated from a baseline signal, a degradation component, and random noise.

| Sensor | Unit | Description |
| --- | --- | --- |
| `vibration_x/y/z` | g | Tri-axial accelerometer measurements |
| `temperature_bearing` | °C | Bearing surface temperature |
| `temperature_ambient` | °C | Ambient temperature |
| `pressure_in/out` | bar | Inlet and outlet pressure |
| `rpm` | RPM | Rotational speed |
| `current_draw` | A | Motor current consumption |
| `oil_viscosity` | cSt | Lubrication quality |
| `acoustic_emission` | dB | High-frequency acoustic emission |
| `torque` | Nm | Load torque |
| `humidity` | % | Ambient humidity |
| `voltage` | V | Supply voltage |

- **vibration_x, vibration_y, vibration_z** — Vibration measurements along three axes. These generally increase as components wear and become one of the strongest indicators of failure.

- **temperature_bearing** — Bearing temperature. Increases significantly with wear due to increased friction.

- **temperature_ambient** — Ambient temperature surrounding the machine. Primarily a noise feature with little predictive value.

- **pressure_in, pressure_out** — Inlet and outlet pressure measurements. Outlet pressure tends to decrease as the machine degrades.

- **rpm** — Rotational speed. Typically decreases as wear progresses.

- **current_draw** — Electrical current consumption. Increases as the machine requires more effort to operate.

- **oil_viscosity** — Lubricant viscosity. Decreases as the lubricant degrades over time.

- **acoustic_emission** — Acoustic or ultrasonic emission level. Generally increases with wear and during occasional anomaly spikes.

- **torque** — Rotational force. Increases as degradation progresses.

- **humidity** — Ambient humidity. Similar to ambient temperature, it serves mainly as a distractor feature.

- **voltage** — Supply voltage. Slightly decreases as the machine approaches failure.

Most sensor variables exhibit meaningful degradation trends as failure approaches, while a small number of sensors (such as ambient temperature and humidity) are intentionally uninformative, forcing the model to learn which signals are truly predictive.

### 4. Target Labels (2 columns)

These columns are prediction targets and are **not used as model inputs**.

- **RUL (Remaining Useful Life)** — The number of cycles remaining until machine failure. This value decreases from the beginning of the machine's life to zero and serves as the regression target (**y_rul**).

- **failure_imminent** — A binary indicator equal to 1 when **RUL ≤ 30**, and 0 otherwise. It identifies machines in their final 30 operating cycles and serves as the classification target (**y_cls**).

## Model and Architecture Design

The model performs two tasks simultaneously through a **multi-task learning** framework, using a window of recent sensor readings as input:

1. **Regression** — *"How long until it fails?"* → Predict **Remaining Useful Life (RUL)** as the number of cycles remaining before failure. This output supports maintenance planning by estimating the available time before intervention is required.

2. **Classification** — *"Is this machine about to fail?"* → Predict **failure_imminent** (yes/no). This output serves as an early warning signal, indicating whether immediate maintenance action may be necessary.

LSTM and Transformer architectures were selected because predictive maintenance is fundamentally a **multivariate time-series problem**. The input consists not of a single set of sensor values but of a sequence of recent sensor observations. Temporal changes in sensor behavior must therefore be learned in order to capture machine degradation patterns.

The **LSTM** architecture is well suited for this task because sequences are processed step by step, enabling temporal degradation patterns to be captured through recurrent memory mechanisms.

The **Transformer** architecture is also well suited for this task because its attention mechanism allows each timestep to be compared directly with all other timesteps within the input window. This enables important degradation signals, anomalous spikes, and long-range temporal dependencies to be identified effectively.

Other approaches — including Random Forest, XGBoost, CNN, GRU, and autoencoder-based models — could also be employed as baseline methods or future extensions. However, LSTM and Transformer architectures were selected to compare two powerful sequence-modeling paradigms: **recurrent memory-based learning** and **attention-based learning**.

```text
Raw Sensor Stream
       │
       ▼
┌─────────────────┐
│  Preprocessing    │  Normalization, sliding window, missing value imputation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature Eng.     │  Rolling statistics, FFT features, RUL labeling
└────────┬────────┘
         │
    ┌────┴────┐
    │          │
    ▼          ▼
┌───────┐ ┌──────────────┐
│ LSTM  │ │ Transformer    │  Two model variants selected via configuration
└───┬───┘ └──────┬───────┘
    └─────┬──────┘
          │
          ▼
┌─────────────────┐
│ Dual Head Output │  Binary failure prediction + RUL regression
└────────┬────────┘
         │
         ▼
   FastAPI /predict
```

The project follows a modular structure that separates data processing, model development, training, evaluation, deployment, and testing. This organization improves maintainability, scalability, and reproducibility.

```text
predictive-maintenance-ai/
├── configs/
│   └── config.yaml                    # Hyperparameters and configuration settings
├── src/
│   ├── data/
│   │   ├── generate_dataset.py        # Synthetic dataset generation
│   │   └── dataset.py                 # PyTorch Dataset and DataLoader implementation
│   ├── models/
│   │   ├── lstm_predictor.py          # LSTM-based predictive maintenance model
│   │   └── transformer_predictor.py   # Transformer-based predictive maintenance model
│   ├── utils/
│   │   ├── preprocessing.py           # Normalization, windowing, and FFT feature extraction
│   │   └── visualization.py           # Training curves, confusion matrix, and RUL visualization
│   ├── api/
│   │   └── app.py                     # FastAPI prediction service
│   ├── train.py                       # Model training pipeline
│   ├── evaluate.py                    # Model evaluation and performance metrics
│   └── predict.py                     # Batch and single-sample inference
├── tests/
│   ├── test_dataset.py                # Dataset unit tests
│   ├── test_model.py                  # Model unit tests
│   └── test_api.py                    # API unit tests
├── notebooks/
│   └── exploratory_analysis.ipynb     # Exploratory data analysis
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions continuous integration workflow
├── Dockerfile                         # Docker image definition
├── docker-compose.yml                 # Multi-container deployment configuration
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## LSTM Architecture

The predictive maintenance model is implemented as a **Stacked Bidirectional LSTM (Bi-LSTM)** operating on multivariate sensor time-series. It follows a **multi-task learning** paradigm, sharing a single temporal feature extractor across two prediction objectives:

1. **Remaining Useful Life (RUL) Estimation** — a regression task producing a continuous estimate of cycles remaining before failure
2. **Failure Imminence Detection** — a binary classification task producing a probability of imminent failure

This joint formulation allows the model to learn degradation dynamics that generalize across both tasks, rather than optimizing each in isolation.

---

### Design Choices

**Stacked LSTM (3 layers).** Hierarchical stacking enables the network to capture temporal patterns at multiple timescales — from cycle-level fluctuations in the lower layers to longer-term degradation trends in the upper layers.

**Bidirectional processing.** Each layer processes the input window in both the forward and backward directions. This gives every timestep access to the full local context within the observation window, which improves representation quality for fixed-length sequences.

**Sequence-to-one readout.** Only the hidden state at the final timestep ($h_{T}^{(3)}$) is passed to the output heads. This distills the entire sequence history into a single representation before prediction.

**Shared feature layer.** A fully connected layer with ReLU activation and dropout sits between the LSTM stack and both output heads. This shared bottleneck encourages the model to learn representations that are jointly useful for regression and classification.

**LayerNorm.** Applied to the final LSTM hidden state before the dense layer to stabilize training and reduce sensitivity to the scale of learned features.

---

### Architecture

```
Input:  (B × T × F)
        B = batch size
        T = 30 timesteps
        F = 14 sensor features

         │
         ▼
┌─────────────────────────────────┐
│         Bi-LSTM Layer 1            │
│   hidden_size = 128  (× 2 dir)     │  → output: B × T × 256
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│         Bi-LSTM Layer 2            │
│   hidden_size = 128  (× 2 dir)     │  → output: B × T × 256
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│         Bi-LSTM Layer 3            │
│   hidden_size = 128  (× 2 dir)     │  → output: B × T × 256
└────────────────┬────────────────┘
                 │
         Select h_T (last timestep)
                 │
                 ▼
┌─────────────────────────────────┐
│           LayerNorm                │  → B × 256
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Linear(256 → 64) + ReLU         │
│           + Dropout                │  → B × 64  (shared representation)
└────────────────┬────────────────┘
                  │
        ┌────────┴────────────────────┐
        ▼                             ▼
┌───────────────────┐   ┌─────────────────────────┐
│ Classification      │   │      Regression Head      │
│ Head                │   │                           │
│                     │   │  Linear(64 → 32) + ReLU   │
│ Linear(64 → 1)      │   │  Linear(32 → 1)  + ReLU   │
│ + Sigmoid           │   │                           │
│                     │   │  Output: RUL (cycles)     │
│ Output: P(failure)  │   │                           │
└───────────────────┘   └─────────────────────────┘
```

---

### Configuration Summary

| Component | Configuration |
| --- | --- |
| Input window | 30 timesteps × 14 features |
| LSTM layers | 3, bidirectional |
| LSTM hidden size | 128 per direction (256 total) |
| Shared FC layer | 256 → 64, ReLU, Dropout |
| Classification head | 64 → 1, Sigmoid |
| Regression head | 64 → 32 → 1, ReLU |
| Readout | Last hidden state, $h_T^{(3)}$ |
| Normalization | LayerNorm (post-LSTM) |

---

### What the Model Learns

The Bi-LSTM stack learns to encode the trajectory of sensor readings over a 30-cycle window into a compact latent vector that captures both the current degradation state and its rate of change. The shared representation then supports two complementary views of machine health: a continuous estimate of remaining life, and a risk probability for operators who need a simple threshold-based alert. Because both heads are trained jointly, the shared features must remain informative for both objectives throughout training.

## Transformer Architecture

The predictive maintenance model is implemented as an **encoder-only Transformer** operating on multivariate sensor time-series. It follows the same **multi-task learning** paradigm as the Bi-LSTM model, sharing a single temporal feature extractor across two prediction objectives:

1. **Remaining Useful Life (RUL) Estimation** — a regression task producing a continuous estimate of cycles remaining before failure
2. **Failure Imminence Detection** — a binary classification task producing a probability of imminent failure

The key distinction from the recurrent model is the use of **multi-head self-attention** in place of sequential state propagation. Rather than accumulating information step by step, the Transformer processes all timesteps in parallel and learns pairwise relevance across the entire observation window — making it better suited for capturing long-range dependencies and non-local degradation patterns.

---

### Design Choices

**Linear input projection.** Raw sensor vectors (14-dimensional) are projected into a higher-dimensional embedding space (128-dimensional) before encoding. This allows the model to learn cross-sensor interactions — for example, the joint signature of rising temperature and increasing vibration — rather than treating each sensor independently.

**Sinusoidal positional encoding.** Because self-attention is permutation-invariant, temporal order must be injected explicitly. Fixed sinusoidal encodings are added to each timestep embedding, preserving positional information without introducing additional learned parameters.

**Stacked encoder layers (4 layers).** Each encoder layer refines the contextual representation of every timestep by attending to all others. Stacking four layers allows the model to compose increasingly abstract temporal relationships, analogous to how depth enables hierarchical feature learning in feedforward networks.

**Multi-head self-attention (8 heads).** Splitting attention across 8 independent heads (each operating in 16-dimensional subspaces) allows the model to simultaneously track distinct temporal phenomena — short-term sensor spikes, long-term degradation trends, cross-sensor correlations, and periodic patterns — without these signals interfering with one another.

**Mean pooling readout.** The per-timestep encoder outputs are averaged across the time dimension to produce a single sequence-level representation. Unlike the Bi-LSTM's last-timestep readout, mean pooling ensures that information from all 30 timesteps contributes equally to the prediction, which is desirable when degradation signals may appear at any point in the window.

**Shared dense bottleneck.** A GELU-activated fully connected layer with dropout compresses the pooled representation from 128 to 64 dimensions before branching into the two output heads. Joint optimization through this shared layer encourages features that are simultaneously informative for both regression and classification.

---

### Architecture

```
Input:  (B × T × F)
        B = batch size
        T = 30 timesteps
        F = 14 sensor features

         │
         ▼
┌─────────────────────────────────┐
│    Linear Projection               │
│    14 → 128                        │  → B × T × 128
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Sinusoidal Positional            │
│   Encoding (fixed)                 │  → B × T × 128
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Transformer Encoder Layer 1      │
│   d_model=128, nhead=8, FFN        │
│   dim=256, residual+LayerNorm      │  → B × T × 128
└────────────────┬────────────────┘
                 │  (× 3 more layers)
                 ▼
┌─────────────────────────────────┐
│   Transformer Encoder Layer 4      │  → B × T × 128
└────────────────┬────────────────┘
                 │
         Mean Pooling over T
                 │
                 ▼
┌─────────────────────────────────┐
│           LayerNorm                │  → B × 128
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Linear(128 → 64) + GELU           │
│         + Dropout                  │  → B × 64  (shared representation)
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼

┌────────────────────┐   ┌─────────────────────────┐
│ Classification       │   │      Regression Head      │
│ Head                 │   │                           │
│                      │   │  Linear(64 → 32) + ReLU   │
│ Linear(64 → 1)       │   │  Linear(32 → 1)  + ReLU   │
│ + Sigmoid            │   │                           │
│                      │   │  Output: RUL (cycles)     │
│ Output: P(failure)   │   │                           │
└────────────────────┘   └─────────────────────────┘
```

Each Transformer encoder layer has the internal structure:

```
Input x (B × T × 128)
   │
   ├──→ Multi-Head Self-Attention → + x → LayerNorm → y
   │
   └──→ y → FFN (128 → 256 → 128) → + y → LayerNorm → output
```

The 30×30 attention score matrix computed at each layer encodes how strongly each timestep should attend to every other — allowing early-window anomalies to directly influence the representation of later timesteps, and vice versa.

---

### Configuration Summary

| Component | Configuration |
| --- | --- |
| Input window | 30 timesteps × 14 features |
| Input projection | 14 → 128 |
| Positional encoding | Sinusoidal (fixed) |
| Encoder layers | 4 |
| Attention heads | 8 (16 dims each) |
| FFN hidden dim | 256 |
| Readout | Mean pooling over time |
| Normalization | LayerNorm (post-pool) |
| Shared FC layer | 128 → 64, GELU, Dropout |
| Classification head | 64 → 1, Sigmoid |
| Regression head | 64 → 32 → 1, ReLU |

---

### What the Model Learns

Each encoder layer refines a contextual representation of every timestep by gathering information from the rest of the window through self-attention. After four such refinement passes, mean pooling distills this enriched sequence into a single vector that summarizes the machine's degradation state over the full 30-cycle window. The shared dense layer then projects this summary into a space that jointly supports a continuous RUL estimate and a binary failure-risk score — two complementary views of the same underlying health trajectory.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/predictive-maintenance-ai.git
cd predictive-maintenance-ai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic dataset
python src/data/generate_dataset.py

# 5. Train the model
python src/train.py --config configs/config.yaml

# 6. Evaluate
python src/evaluate.py --checkpoint outputs/best_model.pt

# 7. Start the API
uvicorn src.api.app:app --reload --port 8000
```

---

## Training

```bash
# Train LSTM (default)
python src/train.py --config configs/config.yaml

# Train Transformer variant
python src/train.py --config configs/config.yaml --model transformer

# Override hyperparameters inline
python src/train.py --config configs/config.yaml \
    --epochs 100 --lr 0.0005 --batch-size 64

# With MLflow tracking (start server first: mlflow ui)
python src/train.py --config configs/config.yaml --mlflow
```

Training produces:

- `outputs/best_model.pt` — best checkpoint (by validation F1)
- `outputs/training_curves.png` — loss and metric plots
- MLflow run with all hyperparameters and artifacts

---

## Evaluation

```bash
python src/evaluate.py --checkpoint outputs/best_model.pt
```

Outputs:

- Classification report (precision, recall, F1 per class)
- Confusion matrix
- RUL regression MAE and RMSE
- ROC-AUC curve
- Feature importance via gradient-based saliency

---

## API

Start the service:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

### Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `POST` | `/predict` | Single-window prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/model/info` | Model metadata |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_window": [[0.12, 72.3, 24.1, 1.02, 0.98, 1450, 8.4, 42.1, 88.2, 0.21, -0.15, 0.09, 245.3, 18.7]],
    "unit_id": "pump-001"
  }'
```

### Example Response

```json
{
  "unit_id": "pump-001",
  "failure_probability": 0.84,
  "failure_predicted": true,
  "remaining_useful_life_cycles": 12,
  "alert_level": "CRITICAL",
  "confidence": 0.91,
  "timestamp": "2025-11-10T14:32:00Z"
}
```

---

## Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run directly
docker build -t pm-ai .
docker run -p 8000:8000 pm-ai
```

---

## Results

Evaluated on a held-out test set of 20 equipment units not seen during training:

| Metric | LSTM | Transformer |
| --- | --- | --- |
| F1-score (failure class) | **0.89** | **0.91** |
| ROC-AUC | 0.94 | 0.96 |
| RUL MAE (cycles) | 8.3 | 7.1 |
| RUL RMSE (cycles) | 11.2 | 9.8 |
| Inference latency | 4 ms | 9 ms |

The system detects failures an average of **18 cycles** before occurrence.

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built as a portfolio project demonstrating production-ready ML engineering for industrial predictive maintenance.*
