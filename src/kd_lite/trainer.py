"""Unified training loop covering base, fine-tune, and from-scratch runs."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .losses import feature_sparsity_loss
from .utils import ensure_dir, plot_curves


@dataclass
class TrainConfig:
    epochs: int = 1500
    lr: float | None = None  # None -> Adam default
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5
    min_lr: float = 1e-4
    use_scheduler: bool = True

    # Sparsity regularization
    sparse_lambda: float = 0.0
    sparse_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    inception_first_layer_filters: int | None = 96  # for LITE

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """Train a classifier with optional feature-sparsity regularization."""

    def __init__(self, model: nn.Module, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            **({"lr": config.lr} if config.lr is not None else {}),
        )
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                min_lr=config.min_lr,
            )
            if config.use_scheduler
            else None
        )

    # ------------------------------------------------------------------ epoch
    def _compute_loss(self, outputs: torch.Tensor, features: list[torch.Tensor] | None, targets: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(outputs, targets)
        if features is not None and self.config.sparse_lambda > 0:
            loss = loss + feature_sparsity_loss(
                features,
                weights=self.config.sparse_weights,
                lambda_reg=self.config.sparse_lambda,
                inception_first_layer_filters=self.config.inception_first_layer_filters,
            )
        return loss

    def _forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Use ``forward_with_features`` if regularization is active, else plain forward."""
        if self.config.sparse_lambda > 0 and hasattr(self.model, "forward_with_features"):
            return self.model.forward_with_features(inputs)
        return self.model(inputs), None

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs, features = self._forward(inputs)
            loss = self._compute_loss(outputs, features, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        n_batches = max(1, len(loader))
        return running_loss / n_batches, 100.0 * correct / max(1, total)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        n_batches = max(1, len(loader))
        return running_loss / n_batches, 100.0 * correct / max(1, total)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        output_dir: str | Path,
        eval_each_epoch: bool = False,
    ) -> dict:
        """Train, save best/last checkpoints, return summary metrics."""
        output_dir = ensure_dir(output_dir)
        torch.save(self.model.state_dict(), output_dir / "first_model.pt")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        best_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())

        start = time.time()
        for epoch in range(self.config.epochs):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if eval_each_epoch:
                v_loss, v_acc = self.evaluate(test_loader)
                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)

            if tr_loss < best_loss:
                best_loss = tr_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())

            if self.scheduler is not None:
                self.scheduler.step(tr_loss)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(
                    f"epoch {epoch + 1}/{self.config.epochs} | "
                    f"loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

        duration = time.time() - start
        torch.save(best_state, output_dir / "best_model.pt")
        torch.save(self.model.state_dict(), output_dir / "last_model.pt")

        plot_curves(
            history["train_loss"], history["val_loss"],
            history["train_acc"], history["val_acc"],
            output_dir,
        )

        # Save history
        hist_df = pd.DataFrame({"loss": history["train_loss"], "learning_rate": history["lr"]})
        hist_df.to_csv(output_dir / "history.csv", index=False)

        # Reload best and evaluate
        self.model.load_state_dict(best_state)
        t0 = time.time()
        test_loss, test_acc = self.evaluate(test_loader)
        test_duration = time.time() - t0

        metrics = {
            "best_train_loss": best_loss,
            "best_epoch": best_epoch + 1,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "train_duration_sec": duration,
            "test_duration_sec": test_duration,
        }
        pd.DataFrame([metrics]).to_csv(output_dir / "df_metrics.csv", index=False)
        print(f"done. best epoch={metrics['best_epoch']} test_acc={test_acc:.2f}%")
        return metrics
