# Module 2 — Fine-Tuning Pipeline

This module provides a **configurable fine-tuning pipeline** that trains and
comparatively evaluates all four sequence architectures on your custom EthSL
landmark dataset.

## Architecture Comparison

The pipeline trains and evaluates all architectures in one run:

| Architecture | Parameters | Strengths |
|---|---|---|
| **LSTM** | ~185K | Baseline, well-understood |
| **BiLSTM** | ~370K | Captures bidirectional temporal patterns |
| **GRU** | ~140K | Faster training, competitive accuracy |
| **Transformer** | ~210K | Long-range dependencies, state-of-the-art |

## Features

- Comparative training: all 4 architectures evaluated in one run
- Signer-dependent AND signer-independent evaluation
- Data augmentation (mirror, noise, jitter, scale, frame dropout)
- Discriminative fine-tuning (frozen layers option)
- Metrics: Accuracy, Precision, Recall, F1 (macro + per-class)
- Latency measurement (ms/sample)
- CSV comparison table output

## Files

| File | Purpose |
|---|---|
| `config.yaml` | All hyperparameters |
| `finetune.py` | Main fine-tuning and comparison script |
| `custom_dataset.py` | Dataset loader for EthSL .npy files |
| `callbacks.py` | Keras training callbacks |
| `evaluate.py` | Standalone evaluation (same as module1) |

## Quick Start

```bash
# Compare all architectures (recommended)
python finetune.py --config config.yaml

# Train a single architecture
python finetune.py --config config.yaml --arch bilstm

# Fine-tune from a refined module1 model
# (Set pretrained.path in config.yaml to point to module1 output)
python finetune.py --config config.yaml
```

## Output

```
models/finetuned/
  best_model.h5              <- Best overall model (copied from best arch)
  comparison_results.csv     <- Ranked comparison table
  all_results.json           <- Full metrics for all architectures
  lstm/
    best.h5                  <- Best LSTM checkpoint
    final_model.h5           <- Final LSTM model
    training_log.csv         <- Epoch-by-epoch metrics
    results.json             <- LSTM evaluation results
  bilstm/ ...
  gru/ ...
  transformer/ ...
```

## Understanding the Results

The comparison table printed at the end ranks all architectures:

```
Architecture   SD Acc%   SD F1%   SI Acc%   SI F1%   ms/sample
--------------------------------------------------------------------
bilstm          94.20%    93.85%    71.30%    70.12%       8.3ms
lstm            91.50%    91.10%    68.90%    67.80%       5.1ms
gru             92.80%    92.40%    69.50%    68.20%       6.7ms
transformer     93.10%    92.80%    72.10%    71.50%      12.4ms
```

- **SD** = Signer-Dependent (easier, same signers in train/test)
- **SI** = Signer-Independent (harder, unseen signer in test)
- Choose the architecture with the best balance of SI accuracy and latency.
