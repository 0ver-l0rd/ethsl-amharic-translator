# 🤟 EthSL → Amharic Translator
### Real-Time Ethiopian Sign Language Recognition & Amharic Translation System

> **Senior Project 2026**
> *Bersabeh Dawit · Emanuel Solomon · Estifanos Behailu · Menase Teshale*
> *Advisor: Dr. Eyob N.*

---

## 📋 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [System Pipeline](#system-pipeline)
- [Quick Start](#quick-start)
- [Module 1 — Model Refinement](#module-1--model-refinement-pretrained)
- [Module 2 — Fine-Tuning](#module-2--fine-tuning-pipeline)
- [Vocabulary](#vocabulary)
- [Evaluation Metrics](#evaluation-metrics)
- [Team Responsibilities](#team-responsibilities)
- [References](#references)

---

## Overview

This system recognizes isolated **Ethiopian Sign Language (EthSL)** signs captured from a single webcam and translates them into **Amharic text** and **synthesized Amharic speech** in real-time.

**Core approach:** MediaPipe Holistic landmark extraction → normalized sequence → sequence classifier → Amharic text + TTS

| Property | Value |
|---|---|
| Vocabulary | 50-100 common day-to-day EthSL signs |
| Input | Single camera, single signer |
| Output | Amharic text + synthesized speech |
| Hardware | CPU-only (no dedicated GPU required) |
| Language | Python 3.10+ |
| Framework | TensorFlow / Keras |

---

## Repository Structure

```
ethsl-amharic-translator/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- setup.py
|
|-- data/
|   |-- raw/                    # Raw recorded sign clips (not committed to git)
|   |-- landmarks/              # Extracted MediaPipe landmark arrays (.npy)
|   |-- processed/              # Normalized, fixed-length sequences
|   |-- vocabulary.json         # Sign label <=> Amharic text mapping
|   `-- splits/                 # train / val / test split indices
|
|-- scripts/
|   |-- collect_data.py         # Record sign clips via webcam
|   |-- split_recording.py      # Split a phone video of repeated signs into clips
|   |-- extract_landmarks.py    # MediaPipe Holistic extraction pipeline
|   |-- preprocess.py           # Normalize + build fixed-length sequences
|   |-- split_dataset.py        # Create signer-dependent/independent splits
|   `-- evaluate_realtime.py    # Live inference latency benchmarking
|
|-- models/
|   |-- pretrained/             # Cloned / downloaded pretrained weights
|   |-- refined/                # Weights after refinement (Module 1)
|   `-- finetuned/              # Weights after fine-tuning (Module 2)
|
|-- src/
|   |-- __init__.py
|   |-- landmark_extractor.py   # MediaPipe Holistic wrapper
|   |-- sequence_builder.py     # Fixed-length sequence constructor
|   |-- augmentation.py         # Spatial & temporal augmentation utilities
|   |-- amharic_tts.py          # Amharic text-to-speech engine wrapper
|   `-- vocabulary.py           # Label <=> Amharic mapping utilities
|
|-- module1_refinement/
|   |-- README.md
|   |-- download_pretrained.py  # Download base model weights
|   |-- refine_model.py         # Transfer-learning refinement training
|   |-- model_architectures.py  # LSTM, BiLSTM, GRU, Transformer definitions
|   |-- config.yaml             # Hyperparameter configuration
|   `-- evaluate.py             # Evaluation on validation set
|
|-- module2_finetune/
|   |-- README.md
|   |-- finetune.py             # Fine-tuning script with frozen/unfrozen layers
|   |-- custom_dataset.py       # Custom EthSL dataset loader
|   |-- config.yaml             # Fine-tuning hyperparameters
|   |-- callbacks.py            # Training callbacks (LR schedule, checkpoints)
|   `-- evaluate.py             # Signer-dependent / independent evaluation
|
|-- app/
|   |-- realtime_app.py         # Main real-time inference application
|   |-- ui.py                   # OpenCV-based display overlay
|   `-- inference_engine.py     # Optimized inference pipeline
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_model_comparison.ipynb
|   |-- 03_refinement_results.ipynb
|   `-- 04_finetuning_results.ipynb
|
`-- tests/
    |-- test_landmark_extractor.py
    |-- test_sequence_builder.py
    `-- test_model_architectures.py
```

---

## System Pipeline

```
+-------------+    +----------------------+    +----------------------+
|  Webcam     |-->>|  MediaPipe Holistic   |-->>|  Sequence Normalizer |
|  Input      |    |  (pose+hands+face)   |    |  (fixed 30-frame     |
|             |    |  543 landmarks x 3   |    |   window, z-norm)    |
+-------------+    +----------------------+    +----------+-----------+
                                                          |
                                                          v
+-------------+    +----------------------+    +----------------------+
|  Amharic    |<<--|  Label -> Amharic    |<<--|  Sequence Classifier |
|  TTS Output |    |  Text Mapping        |    |  (LSTM/BiLSTM/GRU/   |
|  + Display  |    |                      |    |   Transformer)       |
+-------------+    +----------------------+    +----------------------+
```

**Landmark dimensions per frame:**
- Pose: 33 landmarks x 4 values = 132
- Left hand: 21 landmarks x 3 values = 63
- Right hand: 21 landmarks x 3 values = 63
- **Total (pose + hands only):** 258 values/frame x 30 frames = **7,740 features**

---

## Quick Start

### 1. Clone and set up
```bash
git clone https://github.com/YOUR_USERNAME/ethsl-amharic-translator.git
cd ethsl-amharic-translator
pip install -r requirements.txt
```

### 2. Download pre-trained base weights (Module 1)
```bash
cd module1_refinement
python download_pretrained.py
```

### 3. Collect your data
Every person records with their own anonymous signer ID (`s01`, `s02`, ...).
The ID is stored in each clip's filename and is what makes the
signer-independent evaluation possible, so never record without it or reuse
another person's ID.

Each clip is 2.5 s (`SIGN_WINDOW_SECONDS`) saved as a compressed `.mp4`
(~1.7 MB), so landmarks can be re-extracted later with other settings
without re-recording.

```bash
python scripts/collect_data.py --signer s01        # each team member uses their own ID
python scripts/extract_landmarks.py --face subset  # --face none | subset | full (see below)
python scripts/preprocess.py                       # also writes data/processed/signer_ids.npy
python scripts/split_dataset.py                    # --holdout_signer s04 to pick the test signer
```

**Phone recordings.** Instead of `collect_data.py`, a word can be recorded on
a phone as one landscape video of 15 repetitions, with the hands fully down for
about a second between them, named `<signer>_<word>.mp4` (e.g. `s01_01.mp4`).
`split_recording.py` finds the repetitions from the hand landmarks and saves a
2.5 s clip around each one, in the same place `collect_data.py` would. It
writes nothing unless it finds the expected count, so dry-run it first:

```bash
python scripts/split_recording.py s01_01.mp4 --label selam --dry_run --plot
python scripts/split_recording.py s01_01.mp4 --label selam   # then extract_landmarks.py as above
```

**Gonfa et al. dataset.** The isolated-word EthSL dataset of Gonfa et al.,
*Scientific Reports* 15:36181 (2025), is 7 signers x 40 clips per word, already
as MediaPipe landmarks. It was shared with us for non-commercial use on
condition that it is not redistributed, so it is never committed. Convert it
in place of `collect_data.py` + `extract_landmarks.py`; this also rewrites
`data/vocabulary.json` to the words that have data. Its signers are named
`g01`-`g07`, so they never share an ID with our own `s01`, `s02`, ...;
re-running it replaces its earlier output but keeps our recorded clips:

```bash
python scripts/convert_gonfa.py --dataset_dir "<folder with the EthSL_Dataset download parts>"
```

**Signer-independent accuracy.** Train on all signers but one, test on that
one, for every signer in turn:

```bash
cd module2_finetune
python leave_one_signer_out.py --arch lstm --pretrained ../models/pretrained/islr_lstm.h5 --tag lstm_islr
```

**Face landmarks** (`--face`), compared as an experiment in Module 2:

| Mode | Face points | Features / frame |
|---|---|---|
| `none` | none (pose + hands) | 258 |
| `subset` | 92: lips, eyes, eyebrows | 534 |
| `full` | all 468 (face mesh) | 1662 |

Training reads the feature size from the data, and the live app reads it from
the model, so the mode only has to be chosen once, at extraction time.

### 4. Refine the pre-trained model
```bash
cd module1_refinement
python refine_model.py --config config.yaml
```

### 5. Fine-tune on your custom data
```bash
cd module2_finetune
python finetune.py --config config.yaml --pretrained ../models/refined/best_model.h5
```

### 6. Run the real-time app
```bash
python app/realtime_app.py --model models/finetuned/best_model.h5
```

---

## Module 1 — Model Refinement (Pretrained)

Loads a pre-trained sign-language sequence model and refines it on EthSL landmark data using **transfer learning**.

**Strategy:**
1. Load base LSTM/Transformer weights pretrained on general SL landmark data
2. Freeze encoder layers, replace and train new EthSL classification head
3. Gradually unfreeze deeper layers (discriminative fine-tuning)
4. Evaluate on EthSL validation split

See `module1_refinement/README.md` for full details.

---

## Module 2 — Fine-Tuning Pipeline

A configurable fine-tuning pipeline for any sequence model on your custom EthSL dataset.

- Supports LSTM, BiLSTM, GRU, and lightweight Transformer
- Signer-dependent and signer-independent split evaluation
- Data augmentation (mirror, rotation, temporal jitter)
- Metrics: accuracy, precision, recall, F1 (per-class + macro)
- Real-time latency measurement
- Configurable via config.yaml

See `module2_finetune/README.md` for full details.

---

## Vocabulary

The system targets **50-100 common EthSL signs**. `data/vocabulary.json` is
currently written by `scripts/convert_gonfa.py` (13 words from the Gonfa et al.
dataset); the planned everyday words are:

| Category | Examples (Amharic) |
|---|---|
| Greetings | selam, dehna neh, dehna nesh, dehna hun |
| Common questions | simih man new?, min felegh?, wagaw sint new? |
| Basic responses | awo, ay, algebagn, silezih |
| Daily life | wiha, migib, hospital, timhirt bet, bet |
| Numbers (1-10) | and, hulet, sost...asr |
| Emergency | irdeta, himem, police |

The full mapping is in `data/vocabulary.json`.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Top-1 Accuracy | Correct classification rate |
| Precision (macro) | Per-class precision averaged |
| Recall (macro) | Per-class recall averaged |
| F1-score (macro) | Harmonic mean of P & R |
| Latency (ms) | End-to-end inference time per frame |

Evaluations run under both:
- **Signer-dependent split** (train/test from same signers)
- **Signer-independent split** (test signers unseen during training)

---

## Team Responsibilities

| Member | Tasks |
|---|---|
| Bersabeh Dawit | Data collection, MediaPipe landmark extraction, preprocessing |
| Menase Teshale | Data collection, real-time app integration, UI overlay |
| Estifanos Behailu | Model training (LSTM/BiLSTM/GRU/Transformer), Amharic TTS |
| Emanuel Solomon | Model comparison, real-time integration, deployment |

---

## References

1. Gonfa et al., "A deep learning framework for EthSL recognition," Scientific Reports, 2025.
2. Camgoz et al., "Sign Language Transformers," CVPR 2020.
3. Lugaresi et al., "MediaPipe," arXiv:1906.08172, 2019.
4. AAU Digital Dictionary of Ethiopian Sign Language, https://ethsl.aau.edu.et
5. Mengiste, "Word Level Amharic Sign Language Recognition," M.Sc. thesis, AAU, 2022.
