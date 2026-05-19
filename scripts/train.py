"""Unified training entry point.

Replaces the family of ``base_*.py``, ``finetune_*.py``, and
``scratch_training*.py`` scripts. Behavior is fully driven by a YAML config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make ``src/`` importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kd_lite.data import load_ucr_dataset, make_dataloader  # noqa: E402
from kd_lite.datasets import DEFAULT_SEEDS, UCR_2018_DATASETS  # noqa: E402
from kd_lite.models import LITE, InceptionTime  # noqa: E402
from kd_lite.trainer import Trainer, TrainConfig  # noqa: E402
from kd_lite.utils import ensure_dir, reinitialize_weights, set_seeds  # noqa: E402


def build_model(cfg: dict, length_TS: int, n_classes: int) -> torch.nn.Module:
    name = cfg["model"].lower()
    if name == "lite":
        return LITE(
            length_TS=length_TS,
            n_classes=n_classes,
            n_filters=cfg.get("n_filters", [[32, 32, 32], 32, 32]),
        )
    if name == "inception":
        return InceptionTime(
            input_shape=(length_TS,),
            n_classes=n_classes,
            n_filters=cfg.get("n_filters", 32),
            depth=cfg.get("depth", 6),
        )
    raise ValueError(f"Unknown model: {cfg['model']}")


def load_checkpoint(path: Path) -> torch.nn.Module:
    """Load a full pickled model (used for the 'init_from' pruned-model workflow)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def run_one(cfg: dict, dataset: str, seed: int, output_root: Path) -> None:
    print(f"\n=== {dataset} | seed={seed} ===")

    x_train, y_train, x_test, y_test = load_ucr_dataset(dataset)
    train_loader = make_dataloader(x_train, y_train, shuffle=True, seed=seed)
    test_loader = make_dataloader(x_test, y_test, shuffle=False, seed=seed)
    n_classes = len(np.unique(y_train))
    length_TS = int(x_train.shape[1])

    set_seeds(seed)

    init_from = cfg.get("init_from")
    if init_from:
        ckpt_path = Path(init_from.format(seed=seed, dataset=dataset))
        print(f"loading pruned model from {ckpt_path}")
        model = load_checkpoint(ckpt_path)
        if cfg.get("reinit_weights", False):
            print("reinitializing weights (scratch-training mode)")
            model = reinitialize_weights(model, seed=seed)
    else:
        model = build_model(cfg, length_TS=length_TS, n_classes=n_classes)

    train_cfg = TrainConfig(
        epochs=cfg.get("epochs", 1500),
        lr=cfg.get("lr"),
        scheduler_patience=cfg.get("scheduler_patience", 50),
        scheduler_factor=cfg.get("scheduler_factor", 0.5),
        min_lr=cfg.get("min_lr", 1e-4),
        use_scheduler=cfg.get("use_scheduler", True),
        sparse_lambda=cfg.get("sparse_lambda", 0.0),
        sparse_weights=cfg.get("sparse_weights", [1.0, 1.0, 1.0]),
        inception_first_layer_filters=cfg.get("inception_first_layer_filters", 96),
    )

    run_dir = output_root / f"seed_{seed}" / dataset
    ensure_dir(run_dir)
    Trainer(model, train_cfg).fit(train_loader, test_loader, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str, help="Single UCR dataset name")
    group.add_argument("--all-datasets", action="store_true", help="Loop over all UCR datasets")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seeds")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output root")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    datasets = UCR_2018_DATASETS if args.all_datasets else [args.dataset]
    seeds = args.seeds or cfg.get("seeds", DEFAULT_SEEDS)
    output_root = args.output_dir or Path("results") / cfg.get("name", args.config.stem)

    for dataset in datasets:
        for seed in seeds:
            run_one(cfg, dataset, seed, output_root)


if __name__ == "__main__":
    main()
