import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from data_prep import prepare_data, load_data, SETTING_NAMES, RUL_CAP
from model import BaselineLSTM, AdvancedCNNLSTM
from train import train_baseline, train_advanced
from evaluate import evaluate_model
from decision_logic import generate_maintenance_alerts


# ─────────────────────────────────────────────
#  HELPER: Create unique timestamped model dir
# ─────────────────────────────────────────────
def create_model_dir(save_dir: str, model_name: str) -> str:
    """Return a new flat directory for one model run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace(" ", "_")
    model_dir = os.path.join(save_dir, f"{safe_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


# ─────────────────────────────────────────────
#  HELPER: Save config.json
# ─────────────────────────────────────────────
def save_config(model_dir: str, model_name: str, epochs: int,
                batch_size: int, lr: float, seq_length: int, device: str) -> None:
    config = {
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "sequence_length": seq_length,
        "device": device,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


# ─────────────────────────────────────────────
#  HELPER: Save predictions / true / uncertainty
# ─────────────────────────────────────────────
def save_predictions(model_dir: str, pred_mean: np.ndarray,
                     true_rul: np.ndarray, pred_std: np.ndarray) -> None:
    np.save(os.path.join(model_dir, "predictions.npy"), pred_mean)
    np.save(os.path.join(model_dir, "true_values.npy"), true_rul)
    np.save(os.path.join(model_dir, "uncertainty.npy"), pred_std)


# ─────────────────────────────────────────────
#  HELPER: Compute & save metrics.txt
# ─────────────────────────────────────────────
def save_metrics(model_dir: str, model_name: str,
                 pred_mean: np.ndarray, true_rul: np.ndarray,
                 pred_std: np.ndarray, rmse: float, nasa_score: float) -> dict:

    errors = pred_mean - true_rul
    abs_errors = np.abs(errors)

    # Core
    mae = float(mean_absolute_error(true_rul, pred_mean))
    nonzero = true_rul != 0
    mape = float(np.mean(np.abs(errors[nonzero] / true_rul[nonzero])) * 100) if nonzero.any() else float('nan')
    r2 = float(r2_score(true_rul, pred_mean))

    # Uncertainty
    mean_unc = float(np.mean(pred_std))
    max_unc  = float(np.max(pred_std))
    min_unc  = float(np.min(pred_std))

    # Confidence  (higher = more certain)
    confidence = 1.0 / (1.0 + pred_std)
    mean_conf = float(np.mean(confidence))
    min_conf  = float(np.min(confidence))

    # Error analysis
    mean_err    = float(np.mean(errors))
    std_err     = float(np.std(errors))
    max_abs_err = float(np.max(abs_errors))

    lines = [
        f"MODEL: {model_name}\n",
        "================= CORE METRICS =================",
        f"RMSE:            {rmse:.4f}",
        f"NASA Score:      {nasa_score:.4f}",
        "",
        "================= REGRESSION =================",
        f"MAE:             {mae:.4f}",
        f"MAPE:            {mape:.4f} %",
        f"R2 Score:        {r2:.4f}",
        "",
        "================= UNCERTAINTY =================",
        f"Mean Uncertainty: {mean_unc:.4f}",
        f"Max Uncertainty:  {max_unc:.4f}",
        f"Min Uncertainty:  {min_unc:.4f}",
        "",
        "================= CONFIDENCE =================",
        f"Mean Confidence: {mean_conf:.4f}",
        f"Min Confidence:  {min_conf:.4f}",
        "",
        "================= ERROR ANALYSIS =================",
        f"Mean Error:      {mean_err:.4f}",
        f"Std Error:       {std_err:.4f}",
        f"Max Abs Error:   {max_abs_err:.4f}",
    ]
    with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(lines))

    return {
        "model": model_name,
        "rmse": rmse,
        "nasa_score": nasa_score,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "mean_uncertainty": mean_unc,
        "mean_confidence": mean_conf,
    }


# ─────────────────────────────────────────────
#  HELPER: All per-model plots
# ─────────────────────────────────────────────
def save_plots(model_dir: str, model_name: str,
               pred_mean: np.ndarray, true_rul: np.ndarray,
               pred_std: np.ndarray,
               train_losses: list, val_losses: list) -> None:

    idx = np.arange(len(pred_mean))
    errors = pred_mean - true_rul
    confidence = 1.0 / (1.0 + pred_std)

    plt.style.use('seaborn-v0_8-darkgrid')
    TITLE_SIZE, LABEL_SIZE = 13, 11

    def _save(fig, name: str):
        fig.tight_layout()
        fig.savefig(os.path.join(model_dir, name), dpi=120)
        plt.close(fig)

    # (1) Training loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, color='steelblue', linewidth=1.5)
    ax.set_title(f"{model_name} — Training Loss", fontsize=TITLE_SIZE)
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("MSE Loss", fontsize=LABEL_SIZE)
    _save(fig, "loss_curve.png")

    # (2) Validation loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(val_losses, color='darkorange', linewidth=1.5)
    ax.set_title(f"{model_name} — Validation Loss", fontsize=TITLE_SIZE)
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("MSE Loss", fontsize=LABEL_SIZE)
    _save(fig, "val_loss_curve.png")

    # (3) Prediction vs True
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, true_rul,  label="True RUL",      color='navy',       linewidth=1.2, alpha=0.85)
    ax.plot(idx, pred_mean, label="Predicted RUL",  color='crimson',    linewidth=1.2, alpha=0.85)
    ax.set_title(f"{model_name} — Prediction vs True RUL", fontsize=TITLE_SIZE)
    ax.set_xlabel("Sample Index", fontsize=LABEL_SIZE)
    ax.set_ylabel("RUL (cycles)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LABEL_SIZE)
    _save(fig, "prediction_vs_true.png")

    # (4) Residuals
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(idx, errors, s=4, alpha=0.5, color='darkorange')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"{model_name} — Residuals", fontsize=TITLE_SIZE)
    ax.set_xlabel("Sample Index", fontsize=LABEL_SIZE)
    ax.set_ylabel("Prediction − True", fontsize=LABEL_SIZE)
    _save(fig, "residuals.png")

    # (5) Residual distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=50, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"{model_name} — Error Distribution", fontsize=TITLE_SIZE)
    ax.set_xlabel("Residual (cycles)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)
    _save(fig, "residual_distribution.png")

    # (6) |Error| vs True RUL
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(true_rul, np.abs(errors), s=4, alpha=0.45, color='purple')
    ax.set_title(f"{model_name} — |Error| vs True RUL", fontsize=TITLE_SIZE)
    ax.set_xlabel("True RUL (cycles)", fontsize=LABEL_SIZE)
    ax.set_ylabel("|Error| (cycles)", fontsize=LABEL_SIZE)
    _save(fig, "error_vs_true.png")

    # (7) Uncertainty plot (shaded prediction interval)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, pred_mean, color='crimson', linewidth=1.2, label="Mean Prediction")
    ax.fill_between(idx,
                    pred_mean - pred_std,
                    pred_mean + pred_std,
                    alpha=0.25, color='crimson', label="± 1 std")
    ax.plot(idx, true_rul, color='navy', linewidth=1.0, alpha=0.7, label="True RUL")
    ax.set_title(f"{model_name} — Uncertainty Band", fontsize=TITLE_SIZE)
    ax.set_xlabel("Sample Index", fontsize=LABEL_SIZE)
    ax.set_ylabel("RUL (cycles)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LABEL_SIZE)
    _save(fig, "uncertainty_plot.png")

    # (8) Confidence vs sample index
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, confidence, color='teal', linewidth=1.0)
    ax.set_title(f"{model_name} — Confidence (1 / (1 + σ))", fontsize=TITLE_SIZE)
    ax.set_xlabel("Sample Index", fontsize=LABEL_SIZE)
    ax.set_ylabel("Confidence", fontsize=LABEL_SIZE)
    _save(fig, "confidence_plot.png")


# ─────────────────────────────────────────────
#  HELPER: Comparison plots + summary
# ─────────────────────────────────────────────
def generate_comparison(save_dir: str, all_metrics: list,
                         all_predictions: dict, true_rul: np.ndarray) -> None:
    comparison_dir = os.path.join(save_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    names   = [m["model"]           for m in all_metrics]
    rmses   = [m["rmse"]            for m in all_metrics]
    nasas   = [m["nasa_score"]      for m in all_metrics]
    confs   = [m["mean_confidence"] for m in all_metrics]

    plt.style.use('seaborn-v0_8-darkgrid')
    COLORS = ['steelblue', 'darkorange', 'seagreen', 'crimson']

    def _bar(values, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(max(5, len(names) * 2), 4))
        bars = ax.bar(names, values, color=COLORS[:len(names)], edgecolor='white', width=0.5)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Model", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(comparison_dir, fname), dpi=120)
        plt.close(fig)

    # (1) RMSE comparison
    _bar(rmses, "RMSE (cycles)", "RMSE Comparison", "rmse_comparison.png")

    # (2) NASA score comparison
    _bar(nasas, "NASA Score", "NASA Score Comparison", "nasa_score_comparison.png")

    # (3) Confidence comparison
    _bar(confs, "Mean Confidence", "Confidence Comparison", "confidence_comparison.png")

    # (4) Overlay: True RUL + all model predictions
    fig, ax = plt.subplots(figsize=(12, 5))
    idx = np.arange(len(true_rul))
    ax.plot(idx, true_rul, color='black', linewidth=1.5, label="True RUL", alpha=0.8)
    for i, (name, preds) in enumerate(all_predictions.items()):
        ax.plot(idx, preds, linewidth=1.1, alpha=0.75,
                color=COLORS[i % len(COLORS)], label=name)
    ax.set_title("Overlay — All Model Predictions", fontsize=13)
    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("RUL (cycles)", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(comparison_dir, "overlay_predictions.png"), dpi=120)
    plt.close(fig)

    # summary.txt — rank by RMSE (lower = better), then NASA (lower = better)
    ranked = sorted(all_metrics, key=lambda m: (m["rmse"], m["nasa_score"]))
    lines = ["MODEL RANKING\n" + "=" * 40]
    for rank, m in enumerate(ranked, start=1):
        lines += [
            f"\n{rank}. {m['model']}",
            f"   RMSE:             {m['rmse']:.4f}",
            f"   NASA Score:       {m['nasa_score']:.4f}",
            f"   MAE:              {m['mae']:.4f}",
            f"   R2 Score:         {m['r2']:.4f}",
            f"   Mean Confidence:  {m['mean_confidence']:.4f}",
        ]
    with open(os.path.join(comparison_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))

    print(f"[*] Comparison artefacts saved → {comparison_dir}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # ── Configuration ──────────────────────────
    base_dir   = r"c:\Users\rudra\Desktop\RUL Predict\ML\data\Dataset"
    save_dir   = r"c:\Users\rudra\Desktop\RUL Predict\ML\models"
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    seq_length = 50
    batch_size = 256
    epochs     = 50        # increased from 20; early stopping still guards overfitting
    lr         = 0.001

    print(f"[*] Running on device: {device}")

    # ── Data Preparation ───────────────────────
    print("\n[====== PHASE 1: DATA PREP ======]")
    train_ds, val_ds, test_ds, n_feats = prepare_data(base_dir, sequence_length=seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Load raw test data for maintenance decision engine IDs
    test_path = os.path.join(base_dir, 'test_FD004.txt')
    raw_test  = load_data(test_path)
    engine_ids = raw_test['unit_nr'].unique()

    # Accumulators for comparison
    all_metrics:     list = []
    all_predictions: dict = {}

    # ── Phase 2: Baseline LSTM ─────────────────
    print("\n[====== PHASE 2: BASELINE LSTM ======]")
    baseline_model = BaselineLSTM(input_size=n_feats, hidden_size=128, num_layers=3)

    b_dir = create_model_dir(save_dir, "Baseline_LSTM")
    save_config(b_dir, "Baseline LSTM", epochs, batch_size, lr, seq_length, device)

    baseline_model, b_train_losses, b_val_losses = train_baseline(
        baseline_model, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device, save_dir=b_dir
    )

    print("Evaluating Baseline LSTM …")
    b_mean, b_std, true_rul, b_rmse, b_nasa = evaluate_model(
        baseline_model, test_loader,
        model_name="Baseline LSTM", device=device,
        mc_passes=1, out_dir=b_dir, rul_cap=RUL_CAP
    )
    b_std = np.zeros_like(b_mean)   # no MC dropout for baseline

    # Rescale from [0,1] back to real cycles
    b_mean   = np.maximum(b_mean * RUL_CAP, 0)   # clamp negatives to 0
    b_std    = b_std    * RUL_CAP
    true_rul = true_rul * RUL_CAP

    torch.save(baseline_model.state_dict(), os.path.join(b_dir, "model.pth"))
    save_predictions(b_dir, b_mean, true_rul, b_std)
    b_metrics = save_metrics(b_dir, "Baseline LSTM", b_mean, true_rul, b_std, b_rmse, b_nasa)
    save_plots(b_dir, "Baseline LSTM", b_mean, true_rul, b_std, b_train_losses, b_val_losses)

    all_metrics.append(b_metrics)
    all_predictions["Baseline LSTM"] = b_mean
    print(f"[✓] Baseline LSTM artefacts saved → {b_dir}")

    # ── Phase 3: Advanced CNN-LSTM ─────────────
    print("\n[====== PHASE 3: ADVANCED CNN-LSTM ======]")
    advanced_model = AdvancedCNNLSTM(
        input_size=n_feats, seq_len=seq_length, num_domains=6, dropout_rate=0.3
    )

    a_dir = create_model_dir(save_dir, "Advanced_CNN_LSTM")
    save_config(a_dir, "Advanced CNN-LSTM", epochs, batch_size, lr, seq_length, device)

    advanced_model, a_train_losses, a_val_losses = train_advanced(
        advanced_model, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device, save_dir=a_dir
    )

    print("Evaluating Advanced CNN-LSTM …")
    a_mean, a_std, true_rul_norm, a_rmse, a_nasa = evaluate_model(
        advanced_model, test_loader,
        model_name="Advanced CNN LSTM", device=device,
        mc_passes=50, out_dir=a_dir, rul_cap=RUL_CAP
    )

    # Rescale from [0,1] back to real cycles
    a_mean   = np.maximum(a_mean * RUL_CAP, 0)   # clamp negatives to 0
    a_std    = a_std         * RUL_CAP
    true_rul = true_rul_norm * RUL_CAP

    torch.save(advanced_model.state_dict(), os.path.join(a_dir, "model.pth"))
    save_predictions(a_dir, a_mean, true_rul, a_std)
    a_metrics = save_metrics(a_dir, "Advanced CNN-LSTM", a_mean, true_rul, a_std, a_rmse, a_nasa)
    save_plots(a_dir, "Advanced CNN-LSTM", a_mean, true_rul, a_std, a_train_losses, a_val_losses)

    all_metrics.append(a_metrics)
    all_predictions["Advanced CNN-LSTM"] = a_mean
    print(f"[✓] Advanced CNN-LSTM artefacts saved → {a_dir}")

    # ── Phase 4: Comparison ────────────────────
    print("\n[====== PHASE 4: COMPARISON ======]")
    generate_comparison(save_dir, all_metrics, all_predictions, true_rul)

    # ── Phase 5: Maintenance Decisions ────────
    print("\n[====== PHASE 5: MAINTENANCE DECISIONS ======]")
    # Pass a_mean directly — decision_logic.py computes conservative_rul internally
    decisions = generate_maintenance_alerts(a_mean, a_std, engine_ids, threshold=15)

    # ── Final metrics file at save_dir root ───
    metrics_path = os.path.join(save_dir, "metric.txt")
    with open(metrics_path, "w") as f:
        f.write("=== FINAL MODEL METRICS ===\n")
        f.write(f"Baseline LSTM     RMSE: {b_rmse:.2f}  |  NASA Score: {b_nasa:.2f}\n")
        f.write(f"Advanced CNN-LSTM RMSE: {a_rmse:.2f}  |  NASA Score: {a_nasa:.2f}\n")
    print(f"[*] Top-level metrics → {metrics_path}")

    print("\n[====== EXECUTION COMPLETE ======]")


if __name__ == "__main__":
    main()