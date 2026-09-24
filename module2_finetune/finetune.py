"""
module2_finetune/finetune.py
=============================
Fine-tuning pipeline for EthSL sign language recognition.

Features:
  - Comparative evaluation of LSTM, BiLSTM, GRU, and Transformer
  - Configurable frozen layers (discriminative fine-tuning)
  - Signer-dependent AND signer-independent evaluation
  - Per-class F1, precision, recall metrics
  - Saves comparison table to CSV

Usage:
  cd module2_finetune
  python finetune.py --config config.yaml
  python finetune.py --config config.yaml --arch bilstm  # single architecture

Authors: Estifanos Behailu, Emanuel Solomon
"""

import os
import sys
import argparse
import json
import csv
import time
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel("ERROR")
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from module1_refinement.model_architectures import build_model
from src.augmentation import AugmentationPipeline
from src.vocabulary import Vocabulary
from module2_finetune.callbacks import build_callbacks
from module2_finetune.custom_dataset import EthSLDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def measure_latency(model, X_sample, n_warmup=10, n_runs=100) -> float:
    s = X_sample[:1]
    for _ in range(n_warmup):
        model.predict(s, verbose=0)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(s, verbose=0)
    return (time.perf_counter() - t0) / n_runs * 1000


def evaluate_model(model, X, y, labels, split_name: str) -> dict:
    """Run evaluation and return metrics dict."""
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)

    print(f"\n  [{split_name}] Accuracy={acc*100:.2f}% | P={p*100:.2f}% | R={r*100:.2f}% | F1={f1*100:.2f}%")

    return {
        "split":     split_name,
        "accuracy":  float(acc),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "report":    classification_report(y, y_pred, target_names=labels,
                                            zero_division=0, output_dict=True),
    }


# ---------------------------------------------------------------------------
# Single architecture fine-tuning
# ---------------------------------------------------------------------------

def finetune_architecture(arch: str, cfg: dict, datasets: dict, output_dir: Path) -> dict:
    """Fine-tune one architecture and return evaluation results."""
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {arch.upper()}")
    print(f"{'='*60}")

    num_classes  = cfg["data"]["num_classes"]
    seq_len      = cfg["data"]["sequence_length"]
    feat_dim     = cfg["data"]["feature_dim"]
    ft_cfg       = cfg["finetune"]
    tr_cfg       = cfg["training"]
    vocab        = Vocabulary(cfg["data"]["vocabulary"])
    labels       = vocab.get_all_labels()

    # Build or load model
    pretrained_path = Path(cfg["pretrained"]["path"])
    if pretrained_path.exists() and cfg["pretrained"]["architecture"] == arch:
        print(f"  Loading pretrained: {pretrained_path}")
        model = keras.models.load_model(str(pretrained_path))
    else:
        print(f"  Building fresh {arch.upper()} model")
        model = build_model(arch, num_classes=num_classes,
                             sequence_length=seq_len, feature_dim=feat_dim)

    # Freeze layers if requested
    n_freeze = ft_cfg.get("frozen_layers", 0)
    if n_freeze > 0:
        for layer in model.layers[:n_freeze]:
            layer.trainable = False
        print(f"  Frozen first {n_freeze} layers")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ft_cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Augment training data
    aug_cfg = tr_cfg.get("augmentation", {})
    aug = AugmentationPipeline(
        mirror_prob=aug_cfg.get("mirror_prob", 0.5),
        noise_std=aug_cfg.get("noise_std", 0.005),
        jitter_frames=aug_cfg.get("jitter_frames", 3),
        scale_range=(aug_cfg.get("scale_min", 0.85), aug_cfg.get("scale_max", 1.15)),
        dropout_prob=aug_cfg.get("dropout_prob", 0.1),
    ) if tr_cfg.get("use_augmentation", True) else None

    X_train, y_train = datasets["train"]
    if aug:
        X_aug = np.stack([aug(x) for x in X_train])
        X_train = np.concatenate([X_train, X_aug])
        y_train = np.concatenate([y_train, y_train])
        print(f"  Augmented train size: {X_train.shape[0]}")

    X_val, y_val = datasets["val"]

    # Callbacks
    arch_output_dir = output_dir / arch
    arch_output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = build_callbacks(cfg, arch_output_dir)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=ft_cfg["epochs"],
        batch_size=ft_cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # Load best checkpoint
    best_ckpt = arch_output_dir / "best.h5"
    if best_ckpt.exists():
        model = keras.models.load_model(str(best_ckpt))

    # Save final model
    final_path = arch_output_dir / "final_model.h5"
    model.save(str(final_path))

    # Evaluate
    results = {"architecture": arch, "epochs_trained": len(history.history["accuracy"])}

    eval_cfg = cfg.get("evaluation", {})

    if eval_cfg.get("signer_dependent", True) and "test_sd" in datasets:
        X_t, y_t = datasets["test_sd"]
        sd_res = evaluate_model(model, X_t, y_t, labels, "Signer-Dependent")
        results["signer_dependent"] = sd_res

    if eval_cfg.get("signer_independent", True) and "test_si" in datasets:
        X_t, y_t = datasets["test_si"]
        si_res = evaluate_model(model, X_t, y_t, labels, "Signer-Independent")
        results["signer_independent"] = si_res

    if eval_cfg.get("measure_latency", True):
        X_lat = datasets["val"][0]
        lat = measure_latency(model, X_lat,
                               n_warmup=eval_cfg.get("latency_warmup", 10),
                               n_runs=eval_cfg.get("latency_runs", 100))
        results["latency_ms"] = lat
        print(f"  Latency: {lat:.2f} ms per sample")

    # Save per-architecture report
    with open(arch_output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune EthSL models")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--arch",   default=None,
                        help="Single architecture to train. If omitted, compares all from config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tf.random.set_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    output_dir = Path(cfg["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load datasets ────────────────────────────────────────────────────
    print("\n[1/3] Loading datasets...")
    dataset = EthSLDataset(cfg)
    datasets = dataset.load_all()

    # ── Determine architectures ──────────────────────────────────────────
    if args.arch:
        architectures = [args.arch]
    elif cfg["finetune"].get("compare_architectures", True):
        architectures = cfg["finetune"]["architectures_to_compare"]
    else:
        architectures = [cfg["pretrained"]["architecture"]]

    print(f"\n[2/3] Architectures to compare: {architectures}")

    # ── Run fine-tuning ──────────────────────────────────────────────────
    all_results = []
    for arch in architectures:
        res = finetune_architecture(arch, cfg, datasets, output_dir)
        all_results.append(res)

    # ── Comparison table ─────────────────────────────────────────────────
    print(f"\n[3/3] Comparison Summary")
    print(f"{'='*80}")
    print(f"{'Architecture':<14} {'SD Acc%':>8} {'SD F1%':>8} {'SI Acc%':>8} {'SI F1%':>8} {'ms/sample':>10}")
    print(f"{'-'*80}")

    best_f1 = -1
    best_arch = None

    csv_rows = []
    for res in all_results:
        arch = res["architecture"]
        sd_acc = res.get("signer_dependent",   {}).get("accuracy", 0) * 100
        sd_f1  = res.get("signer_dependent",   {}).get("f1",       0) * 100
        si_acc = res.get("signer_independent", {}).get("accuracy", 0) * 100
        si_f1  = res.get("signer_independent", {}).get("f1",       0) * 100
        lat    = res.get("latency_ms", 0)

        print(f"{arch:<14} {sd_acc:>7.2f}% {sd_f1:>7.2f}% {si_acc:>7.2f}% {si_f1:>7.2f}% {lat:>9.1f}ms")
        csv_rows.append([arch, sd_acc, sd_f1, si_acc, si_f1, lat])

        if sd_f1 > best_f1:
            best_f1 = sd_f1
            best_arch = arch

    print(f"{'='*80}")
    print(f"\nBest architecture: {best_arch} (SD F1: {best_f1:.2f}%)")

    # Copy best model to output root
    best_src  = output_dir / best_arch / "final_model.h5"
    best_dest = output_dir / cfg["output"]["best_model"]
    if best_src.exists():
        import shutil
        shutil.copy2(best_src, best_dest)
        print(f"Best model saved to: {best_dest}")

    # Save CSV
    csv_path = output_dir / cfg["output"]["results_csv"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["architecture", "SD_accuracy", "SD_f1", "SI_accuracy", "SI_f1", "latency_ms"])
        writer.writerows(csv_rows)
    print(f"Comparison results saved to: {csv_path}")

    # Save full JSON
    with open(output_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
