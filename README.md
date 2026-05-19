# Feature-Sparse Pruning for LITE / Inception Time-Series Classifiers

PyTorch implementation of feature-sparsity-driven structured pruning for
[LITE](https://arxiv.org/abs/2409.02869) and InceptionTime(https://arxiv.org/abs/1909.04939), evaluated on the
[UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

The pipeline has three stages:

1. **Base training** — train a LITE/Inception model with an instance-wise
   L₂,₁ feature-sparsity penalty so a small subset of filters carries most
   of the signal.
2. **Pruning** — drop filters whose feature maps stay near-zero across the
   training set, producing a smaller model.
3. **Fine-tune / retrain-from-scratch** — recover any accuracy lost to
   pruning. Both are supported for ablation.

## Repository layout

```
.
├── src/kd_lite/           # Library code
│   ├── models/            # LITE, Inception architectures
│   ├── losses.py          # InstanceFeatureSparseLoss, etc.
│   ├── data.py            # UCR loaders, preprocessing
│   └── utils.py           # IO, plotting, seeding
├── scripts/
│   ├── train.py           # Unified training entry point
│   └── analyze.py         # 1-vs-1 perf plots, Wilcoxon tests
├── configs/               # YAML configs for each experiment
├── legacy/                # Original scripts kept for reference
└── requirements.txt
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch 2.x.

## Data

Download the [UCR Archive 2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
and set the path:

```bash
export UCR_ARCHIVE_DIR=/path/to/UCRArchive_2018
```

Each dataset folder must contain `<NAME>_TRAIN.tsv` and `<NAME>_TEST.tsv`.

## Usage

All experiments are driven by `scripts/train.py` + a YAML config.

```bash
# Train a base LITE model with feature-sparsity regularization
python -m scripts.train --config configs/base_lite_1e-5.yaml --dataset ACSF1

# Fine-tune a pruned model
python -m scripts.train --config configs/finetune_lite.yaml --dataset ACSF1

# Train pruned architecture from scratch (lottery-ticket style)
python -m scripts.train --config configs/scratch_lite.yaml --dataset ACSF1
```

Run on all UCR datasets:

```bash
python -m scripts.train --config configs/base_lite_1e-5.yaml --all-datasets
```

### Config reference

Key fields in a YAML config:

| Field            | Description                                                 |
|------------------|-------------------------------------------------------------|
| `model`          | `lite` or `inception`                                       |
| `n_filters`      | Filters per layer (list for LITE, int for Inception)        |
| `sparse_lambda`  | L₂,₁ regularization coefficient                             |
| `sparse_weights` | Per-layer weighting of the sparsity penalty (e.g. `[4,2,1]`)|
| `epochs`         | Training epochs                                             |
| `seeds`          | List of seeds for repeated runs                             |
| `init_from`      | Path template to load a pruned checkpoint                   |
| `reinit_weights` | If `true`, reset weights before training (scratch mode)     |

See `configs/` for full examples covering every experiment in the paper.

## Analysis

After running experiments, generate paired-comparison plots:

```bash
python -m scripts.analyze results.csv --base-col base_ens_acc --new-col pruned_ens_acc
```

Outputs a 1-vs-1 scatter plot and Wilcoxon signed-rank p-value.

## Citation

If you use this work please make sure you cite this paper:
```bibtex
@article{abdullayev2026adaptive,
  title={Adaptive Structured Pruning of Convolutional Neural Networks for Time Series Classification},
  author={Abdullayev, Javidan and Devanne, Maxime and Meyer, Cyril and Ismail-Fawaz, Ali and Weber, Jonathan and Forestier, Germain},
  journal={arXiv preprint arXiv:2602.12744},
  year={2026}
}
```

## License

MIT — see `LICENSE`.
