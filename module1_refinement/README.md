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

## If pretrained weights can't be downloaded

```bash
# Use random initialization (train from scratch)
python refine_model.py --config config.yaml --from_scratch
```

## Output

```
models/refined/
  best_model.h5          <- best model by val_accuracy
  phase1_best.h5         <- best from phase 1 only
  phase2_best.h5         <- best from phase 2 only
  training_summary.json  <- accuracy, loss, epoch counts
  eval_report.json       <- full evaluation metrics
  confusion_matrix.png   <- visualized confusion matrix
```
