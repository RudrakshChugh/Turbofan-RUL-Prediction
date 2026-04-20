import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


# ─────────────────────────────────────────────
#  Asymmetric NASA-Inspired Loss  (Fix 3)
# ─────────────────────────────────────────────
class AsymmetricRULLoss(nn.Module):
    """
    Weighted MSE that penalises **late** predictions (pred > true, i.e.
    the model thinks the engine has MORE life left than it really does)
    more heavily than early predictions.  This mirrors the NASA scoring
    function's asymmetry:
        late  → exp(d/10)   (steep)
        early → exp(-d/13)  (gentler)
    We approximate the ratio as a simple multiplier:
        late  weight = `late_weight`  (default 2.0)
        early weight = 1.0
    """
    def __init__(self, late_weight: float = 2.0):
        super().__init__()
        self.late_weight = late_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = pred - target              # positive = late (dangerous)
        weights = torch.where(errors > 0,
                              self.late_weight,
                              1.0)
        return (weights * errors ** 2).mean()


# ─────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter   = 0


# ─────────────────────────────────────────────
#  Loss curve helper (kept for standalone use)
# ─────────────────────────────────────────────
def plot_losses(train_losses: list, val_losses: list,
                title: str, save_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ─────────────────────────────────────────────
#  Baseline LSTM training
# ─────────────────────────────────────────────
def train_baseline(model, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 50, lr: float = 0.001,
                   device: str = 'cpu', save_dir: str = 'ML/models'):
    """
    Returns
    -------
    model            : best-weight model
    train_history    : list of per-epoch train losses
    val_history      : list of per-epoch val losses
    """
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt = os.path.join(save_dir, "model.pth")

    criterion     = nn.MSELoss()
    optimizer     = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=10)

    model.to(device)
    train_history, val_history = [], []
    best_val_loss = float('inf')

    print("--- Starting Training: Baseline LSTM ---")
    for epoch in range(epochs):

        # ── Training ──────────────────────────
        model.train()
        running_loss = 0.0
        for seqs, ruls, _ in train_loader:
            seqs = seqs.to(device)
            ruls = ruls.to(device)

            optimizer.zero_grad()
            preds = model(seqs).squeeze(-1)
            loss  = criterion(preds, ruls)
            loss.backward()

            # Gradient clipping — prevents LSTM exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ── Validation ────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, ruls, _ in val_loader:
                seqs = seqs.to(device)
                ruls = ruls.to(device)
                preds = model(seqs).squeeze(-1)
                val_loss += criterion(preds, ruls).item()
        val_loss /= len(val_loader)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch+1:>3}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Reload best weights before returning
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Optional standalone combined loss plot (main.py generates individual plots)
    plot_losses(train_history, val_history,
                'Baseline LSTM Learning Curves',
                os.path.join(save_dir, 'combined_loss_curve.png'))

    return model, train_history, val_history


# ─────────────────────────────────────────────
#  Advanced CNN-LSTM (DANN) training
# ─────────────────────────────────────────────
def train_advanced(model, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 50, lr: float = 0.001,
                   device: str = 'cpu', save_dir: str = 'ML/models'):
    """
    Returns
    -------
    model            : best-weight model
    train_history    : list of per-epoch RUL losses
    val_history      : list of per-epoch val RUL losses
    """
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt = os.path.join(save_dir, "model.pth")

    rul_criterion = AsymmetricRULLoss(late_weight=2.0)
    dom_criterion = nn.CrossEntropyLoss()

    optimizer      = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler      = optim.lr_scheduler.ReduceLROnPlateau(
                         optimizer, mode='min', factor=0.5, patience=3)
    early_stopping  = EarlyStopping(patience=10)

    model.to(device)
    train_history, val_history = [], []
    best_val_loss = float('inf')

    print("--- Starting Training: Advanced CNN-LSTM (Domain Adversarial) ---")
    for epoch in range(epochs):

        # GRL alpha: ramps 0 → 1 so domain pressure builds gradually
        p     = epoch / max(epochs - 1, 1)
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0   # standard DANN schedule

        # ── Training ──────────────────────────
        model.train()
        running_rul_loss = 0.0
        running_dom_loss = 0.0

        for seqs, ruls, doms in train_loader:
            # FIX: all tensors must be on the same device
            seqs = seqs.to(device)
            ruls = ruls.to(device)
            doms = doms.to(device)

            optimizer.zero_grad()
            rul_preds, dom_preds = model(seqs, alpha)
            rul_preds = rul_preds.squeeze(-1)

            loss_rul = rul_criterion(rul_preds, ruls)
            loss_dom = dom_criterion(dom_preds, doms)

            # Fix 4: domain loss scale raised to 1.0 so DANN actually
            # enforces meaningful domain invariance.  Alpha still ramps
            # from 0 → 1 so the model isn't disturbed early on.
            total_loss = loss_rul + (alpha * loss_dom * 1.0)

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_rul_loss += loss_rul.item()
            running_dom_loss += loss_dom.item()

        train_rul = running_rul_loss / len(train_loader)
        train_dom = running_dom_loss / len(train_loader)

        # ── Validation (RUL only) ─────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, ruls, _ in val_loader:   # FIX: domain labels not needed in val
                seqs = seqs.to(device)
                ruls = ruls.to(device)
                rul_preds, _ = model(seqs, alpha=1.0)
                rul_preds    = rul_preds.squeeze(-1)
                val_loss    += rul_criterion(rul_preds, ruls).item()
        val_loss /= len(val_loader)

        train_history.append(train_rul)
        val_history.append(val_loss)

        print(f"Epoch {epoch+1:>3}/{epochs} | α={alpha:.3f} | "
              f"RUL Loss: {train_rul:.4f} | Dom Loss: {train_dom:.4f} | "
              f"Val RUL Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # FIX: Reload best weights before returning (not the last epoch)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    plot_losses(train_history, val_history,
                'Advanced CNN-LSTM Learning Curves',
                os.path.join(save_dir, 'combined_loss_curve.png'))

    return model, train_history, val_history