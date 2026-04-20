# Turbofan Engine Remaining Useful Life Prediction

A deep learning pipeline for predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS FD004 dataset. The system combines a CNN-LSTM hybrid architecture with domain adaptation and uncertainty estimation to provide actionable maintenance scheduling recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [License](#license)

---

## Overview

Predictive maintenance aims to forecast equipment failure before it occurs, minimizing unplanned downtime and reducing maintenance costs. This project builds two models on the NASA C-MAPSS FD004 dataset (the most challenging subset, with 6 operating conditions and 2 fault modes):

1. **Baseline Bidirectional LSTM** -- a strong sequential baseline with 3 layers and 128 hidden units.
2. **Advanced CNN-LSTM with Domain Adversarial Training** -- a hybrid that uses 1D-CNNs for local feature extraction, LSTMs for temporal modeling, a Gradient Reversal Layer for domain-invariant learning across operating conditions, and MC Dropout for calibrated uncertainty estimates.

Both models are evaluated using RMSE and the asymmetric NASA Scoring Function, and the best model's predictions are translated into concrete maintenance alerts.

---

## Project Structure

```
RUL Predict/
|-- ML/
|   |-- requirements.txt
|   |-- data/
|   |   +-- Dataset/             # Raw CMAPSS .txt files (not tracked)
|   |-- models/
|   |   |-- metric.txt           # Top-level comparison
|   |   |-- Baseline_LSTM_.../
|   |   |   |-- model.pth
|   |   |   +-- metrics.txt
|   |   +-- Advanced_CNN_LSTM_.../
|   |       |-- model.pth
|   |       +-- metrics.txt
|   |-- scripts/
|   |   |-- data_prep.py         # Data loading, RUL labeling, normalization
|   |   |-- model.py             # Model architectures (Baseline + Advanced)
|   |   |-- evaluate.py          # MC Dropout inference, RMSE, NASA score
|   |   +-- decision_logic.py    # Maintenance alert generation
|   +-- training/
|       |-- train.py             # Training loops, loss functions, scheduling
|       +-- main.py              # End-to-end pipeline runner
|-- backend/                     # (Reserved for API serving)
|-- frontend/                    # (Reserved for dashboard UI)
+-- .gitignore
```

---

## Architecture

### Baseline: Bidirectional LSTM

```
Input (50 timesteps x N sensors)
    --> Bidirectional LSTM (128 hidden, 3 layers)
    --> Fully Connected (256 -> 64 -> 1)
    --> RUL prediction
```

### Advanced: CNN-LSTM with Domain Adaptation

```
Input (50 timesteps x N sensors)
    --> Conv1D (32 filters, k=3) --> ReLU
    --> Conv1D (64 filters, k=3) --> ReLU
    --> LSTM (64 hidden, 2 layers)
    --> Global Features
        |
        |--> RUL Head (64 -> 32 -> 16 -> 1)  [with MC Dropout]
        |
        +--> Gradient Reversal Layer
             --> Domain Classifier (64 -> 32 -> 6)  [operating condition prediction]
```

---

## Key Features

- **Piecewise Linear RUL Labeling** -- RUL targets are capped at 125 cycles following the standard CMAPSS convention, reflecting that degradation is not detectable in early healthy operation.

- **Condition-Aware Normalization** -- Sensor readings are normalized per operating condition cluster (K-Means, k=6) using scalers fitted exclusively on training data to prevent data leakage.

- **Engine-Wise Train/Validation Split** -- Engines are split by unit ID (80/20), not by individual rows, ensuring no temporal leakage between train and validation sets.

- **RUL Target Normalization** -- Targets are scaled to [0, 1] during training and rescaled at inference, stabilizing MSE gradients and improving convergence.

- **Asymmetric Loss Function** -- A weighted MSE that penalizes late predictions (overestimating remaining life) 2x more than early predictions, reflecting the safety-critical nature of the task and aligning with the NASA scoring function's asymmetry.

- **Domain Adversarial Training (DANN)** -- A Gradient Reversal Layer encourages the shared feature extractor to learn representations that are invariant to operating conditions, improving generalization across the 6 regimes in FD004.

- **DANN Alpha Scheduling** -- The adversarial loss contribution ramps from 0 to 1 over training using the standard sigmoid schedule, preventing the randomly-initialized domain classifier from corrupting early feature learning.

- **MC Dropout Uncertainty Estimation** -- At inference, 50 stochastic forward passes produce a distribution over predictions. The mean serves as the point estimate; the standard deviation quantifies epistemic uncertainty.

- **Gradient Clipping** -- All gradients are clipped to max norm 1.0 to prevent exploding gradients in deep LSTM architectures.

- **Early Stopping + LR Scheduling** -- Training halts after 10 epochs without validation improvement. Learning rate is halved after 3 stale epochs via ReduceLROnPlateau.

- **Actionable Maintenance Alerts** -- The conservative RUL estimate (mean - 1.5 * std) is compared against a configurable safety threshold (default: 15 cycles) to generate CRITICAL, WARNING, or HEALTHY status labels per engine.

---

## Results

Evaluated on the FD004 test set (248 engines, 6 operating conditions, 2 fault modes):

| Model | RMSE | NASA Score | MAE | R2 Score |
|-------|------|------------|-----|----------|
| Baseline Bidirectional LSTM | 17.87 | 1422.09 | 13.82 | 0.8226 |
| **Advanced CNN-LSTM (DANN)** | **15.28** | **1227.24** | **10.87** | **0.8703** |

The Advanced model reduces RMSE by 14.5% and NASA Score by 13.7% compared to the baseline. The near-zero mean error (-0.62 cycles) indicates minimal systematic bias.

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
cd ML
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### Dataset

Download the CMAPSS dataset from the [NASA Prognostics Data Repository](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data) and place the following files in `ML/data/Dataset/`:

- `train_FD004.txt`
- `test_FD004.txt`
- `RUL_FD004.txt`

---

## Usage

Run the full pipeline (data preparation, training both models, evaluation, comparison, and maintenance alerts):

```bash
cd ML/training
python main.py
```

This will:
1. Load and preprocess the FD004 dataset with condition-aware normalization
2. Train the Baseline Bidirectional LSTM with early stopping
3. Train the Advanced CNN-LSTM with domain adversarial training
4. Evaluate both models using RMSE and the NASA Scoring Function
5. Generate per-model metrics, plots, and uncertainty estimates
6. Produce a side-by-side model comparison
7. Output maintenance alert decisions for all 248 test engines

All artifacts (weights, metrics, plots) are saved to timestamped directories under `ML/models/`.

---

## Methodology

### Dataset: NASA C-MAPSS FD004

- **Training set**: 249 engines run to failure under 6 operating conditions with 2 fault modes
- **Test set**: 248 engines with truncated histories; true RUL provided separately
- **Features**: 21 sensor channels + 3 operational settings per timestep
- **Preprocessing**: Constant sensors removed, remaining sensors normalized per operating condition cluster

### Training Details

| Parameter | Value |
|-----------|-------|
| Sequence length | 50 timesteps |
| Batch size | 256 |
| Max epochs | 50 |
| Learning rate | 0.001 (Adam, weight decay 1e-4) |
| RUL cap | 125 cycles |
| Early stopping patience | 10 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| MC Dropout passes | 50 (inference) |
| Gradient clipping | max norm 1.0 |

### Evaluation Metrics

- **RMSE**: Standard root mean squared error in cycles
- **NASA Scoring Function**: Asymmetric penalty -- late predictions (overestimating RUL, which is dangerous) are penalized exponentially more than early predictions:
  - Early (d < 0): score += exp(-d/13) - 1
  - Late (d > 0): score += exp(d/10) - 1

---

## License

This project is for educational and research purposes. The CMAPSS dataset is provided by NASA and is subject to its own terms of use.
