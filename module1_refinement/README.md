# Module 1 — Model Refinement

This module adapts a pre-built sign language sequence model to the EthSL
(Ethiopian Sign Language) vocabulary using **transfer learning**.

## What is "refinement"?

Since no large public EthSL landmark dataset exists yet, we start from a model
pre-trained on a **similar task** (general sign language landmark classification)
and *refine* it on our EthSL data. This is much faster than training from scratch
and usually achieves better accuracy with limited data.

## Two-Phase Strategy

```
Phase 1 (Head-only training)
  Frozen encoder layers
  ---> Train only the new EthSL Dense output (60 classes)
  ---> 20 epochs, LR = 1e-3

Phase 2 (Full fine-tuning)
  All layers unfrozen
  ---> End-to-end training with lower learning rate
  ---> 30 epochs, LR = 1e-4
```

## Files

| File | Purpose |
|---|---|
| `config.yaml` | All hyperparameters — edit this, not the Python files |
| `download_pretrained.py` | Download base model weights from public sources |
| `model_architectures.py` | LSTM, BiLSTM, GRU, Transformer definitions |
| `refine_model.py` | Main refinement training script |
| `evaluate.py` | Comprehensive evaluation with confusion matrix |

## Quick Start

```bash
# 1. Download pretrained weights
python download_pretrained.py

# 2. Edit config.yaml if needed (adjust paths, epochs, LR)

# 3. Run refinement
python refine_model.py --config config.yaml

# 4. Evaluate the refined model
python evaluate.py --model ../models/refined/best_model.h5 --config config.yaml
```

## Pretraining on Kaggle ISLR (recommended)

The public pretrained models above are no longer downloadable, so the
starting point is pretrained here on Google's
[Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs)
dataset: 250 ASL signs, ~94k clips, already in MediaPipe Holistic format.
ASL only teaches the lower layers (hand shapes, motion); `refine_model.py`
replaces the 250-sign output layer with the EthSL vocabulary.

One-time setup: create a Kaggle account, accept the competition rules on the
competition's **Data** tab, and save an API token (Kaggle → Settings → API →
Create New Token) as `~/.kaggle/kaggle.json`.

```bash
# From the repo root. --download fetches only the selected clips and deletes
# each raw file after converting it, so it needs almost no disk space.
python scripts/convert_islr.py --download --max_per_sign 100

# Pretrain (architecture must match pretrained.architecture in config.yaml)
python module1_refinement/pretrain_islr.py --arch lstm

# Then refine on EthSL as usual
cd module1_refinement
python refine_model.py --config config.yaml
```

- **Face mode:** convert with the same `--face` as the EthSL data
  (`extract_landmarks.py --face`), or the weights won't fit.
- **Mirroring:** our camera image is flipped before MediaPipe, which swaps
  the hand channels. Once some EthSL clips exist, add
  `--compare_with data/landmarks` to see whether ISLR needs `--mirror`.
- **For the report:** compare EthSL accuracy with this pretraining against
  `--from_scratch`.

## If pretrained weights can't be downloaded

```bash
# Use random initialization (train from scratch)
python refine_model.py --config config.yaml --from_scratch
```

## Output

```
models/refined/
  best_model.h5          <- best model by val_accuracy
  phase1_best.keras      <- best from phase 1 only
  phase2_best.keras      <- best from phase 2 only
  training_summary.json  <- accuracy, loss, epoch counts
  eval_report.json       <- full evaluation metrics
  confusion_matrix.png   <- visualized confusion matrix
```
