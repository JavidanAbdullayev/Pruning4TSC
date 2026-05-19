"""General training utilities: seeding, IO, plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Set numpy + torch seeds and enable deterministic cuDNN."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_curves(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    train_acc: Sequence[float],
    val_acc: Sequence[float],
    out_dir: str | Path,
) -> None:
    """Save loss/accuracy curve PNGs to ``out_dir``."""
    out_dir = Path(out_dir)

    plt.figure()
    plt.plot(train_loss, label="train_loss")
    if len(val_loss) > 0:
        plt.plot(val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(out_dir / "losses.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_acc, label="train_acc")
    if len(val_acc) > 0:
        plt.plot(val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.legend()
    plt.savefig(out_dir / "accuracies.png", bbox_inches="tight")
    plt.close()


def reinitialize_weights(model: torch.nn.Module, seed: int, skip: str = "hybrid_block") -> torch.nn.Module:
    """Reset learnable weights for lottery-ticket-style retraining.

    Modules whose qualified name contains ``skip`` are left untouched
    (LITE's hand-designed hybrid filters are non-trainable).
    """
    torch.manual_seed(seed)
    for name, module in model.named_modules():
        if hasattr(module, "reset_parameters") and skip not in name:
            module.reset_parameters()
    return model
