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
|   |-- phrases.json            # Demo phrases: signs in a row -> an Amharic sentence
|   `-- splits/                 # train / val / test split indices
|
|-- scripts/
|   |-- collect_data.py         # Record sign clips via webcam
|   |-- split_recording.py      # Split a phone video of repeated signs into clips
|   |-- split_batch.py          # Split a whole folder of phone videos
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
|   |-- web_app.py              # Browser version: translate, add videos, review, retrain
|   |-- live_session.py         # Per-sign recognition shared by both apps
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

### Try the app (team members)
The trained model is not on GitHub (it is trained on data we may not share),
so get `best_model.h5` from the team's Telegram group. Then, with Python 3.12:
```bash
git clone https://github.com/0ver-l0rd/ethsl-amharic-translator.git
cd ethsl-amharic-translator
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app/web                     # opens the app in the browser
```
The app finds the model itself: it uses `models/finetuned/best_model.h5`, and
if that is missing it copies the newest `best_model.h5` from Downloads (where
Telegram saves it) into place. `python app/realtime_app.py` opens the desktop
window instead. Sign one word at a time; to sign a phrase, pause with
the hands still for about half a second between signs (see `data/phrases.json`).
Speech needs the internet (Google TTS).

### 1. Clone and set up
```bash
git clone https://github.com/0ver-l0rd/ethsl-amharic-translator.git
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
clip around each one, in the same place `collect_data.py` would. Clips are
2.5 s, or longer for a slow sign (the whole sign plus 0.25 s each side, up to
4.5 s); the live app cuts the same window around each sign it sees. Videos
filmed sideways or upside down are turned upright. Reps cut off by the start
or end of the recording, the reach for the phone at the end, and two reps run
together are left out. It writes nothing unless it finds the expected count,
so dry-run it first:

```bash
python scripts/split_recording.py s01_01.mp4 --label selam --dry_run --plot
python scripts/split_recording.py s01_01.mp4 --label selam   # then extract_landmarks.py as above
```

A whole folder of videos named `<signer>_<number>` (numbers from
`data/our_words.tsv`) is split in one go, with one summary table, and its
landmarks extracted. Signers who do more than 15 reps are fine with `--extra`;
bad reps found on review go in the folder's `skip.txt` (e.g. `s02_15  22`).

```bash
python scripts/split_batch.py "~/Desktop/Senior Project/recordings/s02" --extra 15 --dry_run
python scripts/split_batch.py "~/Desktop/Senior Project/recordings/s02" --extra 15
```

**Adding videos from a web page.** `web_app.py` (see step 6) does the same
splitting in a browser, for any number of repetitions, so anyone can add a
word or more examples of one:

- **Add a video:** pick the word (or type a new one in Amharic), upload or
  record the video, and submit. It is split into one clip per repetition.
- **Review** (team): watch the clips, untick bad ones, approve or reject.
  Only approved clips reach `data/landmarks/`; rejected ones are deleted.
- **Retrain** (team): trains a new model into `models/candidate/` and shows its
  scores next to the current one; **Use this model** makes the live app use it.

Contributors are stored as `c01`, `c02`, ...; the table of their names stays
in `data/contributions/`, which is never committed.

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
python app/realtime_app.py --model models/finetuned/best_model.h5   # desktop window
python app/web_app.py            # in the browser: open http://127.0.0.1:7860
python app/web_app.py --public   # Translate and Add a video only (for hosting)
```

Both guess once per sign: a sign starts when the hands come up and ends when
they are lowered, and below 65% confidence the app says "Not sure" instead of
guessing. Signs made one after another can form a phrase from
`data/phrases.json` (እንዴት + አንተ -> "እንዴት ነህ?"); the signs may come in
either order, each within 6 seconds of the one before, and the hands may stay
up if they pause between signs. Only the phrase is said: a word that could
start one is shown at once, but said only if no sign follows within 1.5 s. The browser version also has tabs for adding training videos,
reviewing them, and retraining.

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

The system targets **50-100 common EthSL signs**. `data/vocabulary.json` has
75 words: the 15 from the Gonfa et al. dataset (written first by
`scripts/convert_gonfa.py`, which keeps the words after them), then the team's
own recordings. The team records the 65 words in `data/our_words.tsv`: 60
everyday words and 5 that are also in the Gonfa data:

| Category | Examples (Amharic) |
|---|---|
| Greetings | ሰላም, እንዴት, ደህና, ደህና ሰንብት |
| Questions | ምን, የት, መቼ, ስንት, ስም |
| Responses | አዎ, አይደለም, እሺ, ጥሩ |
| Daily life | ውሃ, ምግብ, መድሃኒት, ገንዘብ, ሆስፒታል, ቤት, ትምህርት ቤት |
| Actions | መርዳት, መፈለግ, መብላት, መጠጣት, መሄድ, መምጣት |
| Time, people, things | ዛሬ, ነገ, ቀን, ልጅ, ተማሪ, ቡና, እንጀራ, መኪና |

A word is added to `data/vocabulary.json` once it has recorded clips.

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
