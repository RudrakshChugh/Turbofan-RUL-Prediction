import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error


def compute_nasa_score(y_true, y_pred):
    """Asymmetric NASA scoring function (computed on real-cycle values)."""
    d = y_pred - y_true
    score = 0.0
    for i in range(len(d)):
        if d[i] < 0:
            score += np.exp(-d[i] / 13.0) - 1.0
        else:
            score += np.exp(d[i] / 10.0) - 1.0
    return score


def mc_dropout_inference(model, data_loader, num_passes=50, device='cpu'):
    """Run MC-Dropout stochastic forward passes and return raw model outputs."""
    model.to(device)
    model.train()   # keep dropout active

    all_preds = []
    trues = []

    for i in range(num_passes):
        pass_preds = []
        with torch.no_grad():
            for seqs, ruls, _ in data_loader:
                seqs = seqs.to(device)

                out = model(seqs)
                if isinstance(out, tuple):
                    rul_pred = out[0]
                else:
                    rul_pred = out

                rul_pred = rul_pred.squeeze(-1).cpu().numpy()
                pass_preds.extend(rul_pred)

                if i == 0:
                    trues.extend(ruls.numpy())

        all_preds.append(pass_preds)

    all_preds_np = np.array(all_preds)

    mean_preds = np.mean(all_preds_np, axis=0)
    std_preds  = np.std(all_preds_np, axis=0)

    return mean_preds, std_preds, np.array(trues)


def evaluate_model(model, data_loader, model_name="Model", device='cpu',
                   mc_passes=50, out_dir="ML/models", rul_cap=125):
    """
    Evaluate a model with MC-Dropout inference.

    Parameters
    ----------
    rul_cap : int
        The RUL cap used during target normalisation.  Predictions and
        true labels are in [0, 1]; they are rescaled by `rul_cap` before
        computing RMSE, NASA score, and drawing plots so that all reported
        numbers are in **real cycle units**.

    Returns
    -------
    mean_preds  : ndarray  — normalised [0,1] mean predictions
    std_preds   : ndarray  — normalised [0,1] std predictions
    trues       : ndarray  — normalised [0,1] true RUL
    rmse        : float    — RMSE in real cycles
    nasa_score  : float    — NASA score in real cycles
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Evaluating {model_name} ---")

    # Raw normalised predictions
    mean_preds, std_preds, trues = mc_dropout_inference(
        model, data_loader, num_passes=mc_passes, device=device
    )

    # Rescale to real cycles for metrics & plots
    mean_real = mean_preds * rul_cap
    std_real  = std_preds  * rul_cap
    true_real = trues      * rul_cap

    rmse       = np.sqrt(mean_squared_error(true_real, mean_real))
    nasa_score = compute_nasa_score(true_real, mean_real)

    print(f"RMSE: {rmse:.2f}")
    print(f"NASA Score: {nasa_score:.2f}")

    # ── Evaluation plot (real-cycle units) ──────────
    plt.figure(figsize=(15, 6))
    n_show  = min(50, len(true_real))
    indices = np.arange(n_show)

    plt.plot(indices, true_real[:n_show], label='True RUL', color='blue', marker='o')
    plt.plot(indices, mean_real[:n_show], label='Predicted RUL (Mean)', color='red', marker='x')

    lower = mean_real[:n_show] - 2 * std_real[:n_show]
    upper = mean_real[:n_show] + 2 * std_real[:n_show]
    plt.fill_between(indices, lower, upper, color='red', alpha=0.2,
                     label='95% Confidence Bound (MC Dropout)')

    plt.title(f'{model_name} — True vs Predicted RUL (First {n_show} Test Engines)')
    plt.xlabel('Test Engine Index')
    plt.ylabel('RUL (Cycles)')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(out_dir,
                             f"{model_name.replace(' ', '_').lower()}_evaluation.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved evaluation plot to {plot_path}")

    # Return normalised values so the caller can do its own rescaling
    return mean_preds, std_preds, trues, rmse, nasa_score
