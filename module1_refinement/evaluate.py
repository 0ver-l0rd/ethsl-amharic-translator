"""
module1_refinement/evaluate.py
================================
Evaluation script for a refined/trained sign language model.

Computes:
  - Accuracy (top-1)
  - Precision, Recall, F1-score (macro + per-class)
  - Confusion matrix
  - Inference latency (ms per sample)

Outputs a detailed report to the console and saves:
  - models/refined/eval_report.json
  - models/refined/confusion_matrix.png

Usage:
  python evaluate.py --model ../models/refined/best_model.h5 --config config.yaml

Authors: All team members
"""

import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel("ERROR")
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("ERROR: scikit-learn, matplotlib, seaborn required.")
    print("  pip install scikit-learn matplotlib seaborn")
    sys.exit(1)

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vocabulary import Vocabulary


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_test_data(cfg: dict):
    proc_dir   = Path(cfg["data"]["processed_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])

    X = np.load(proc_dir / "X.npy")
    y = np.load(proc_dir / "y.npy")

    # Use signer-independent test split for a more rigorous evaluation
    si_test = splits_dir / "signer_independent_test_idx.npy"
    if si_test.exists():
        print("  Using signer-INDEPENDENT test split")
        test_idx = np.load(si_test)
    else:
        print("  Using signer-DEPENDENT test split")
        test_idx = np.load(splits_dir / "test_idx.npy")

    return X[test_idx], y[test_idx]


def measure_latency(model, X_sample: np.ndarray, n_warmup: int = 10, n_runs: int = 100) -> float:
    """Measure mean inference latency in milliseconds."""
    sample = X_sample[:1]  # single sample

    # Warm up
    for _ in range(n_warmup):
        model.predict(sample, verbose=0)

    # Time
    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(sample, verbose=0)
    elapsed = (time.perf_counter() - start) / n_runs * 1000  # ms

    return elapsed


def plot_confusion_matrix(cm: np.ndarray, labels: list, output_path: Path) -> None:
    """Save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(max(10, len(labels) // 2),
                                     max(8,  len(labels) // 2)))
    sns.heatmap(
        cm,
        annot=len(labels) <= 30,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — EthSL Recognition")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate refined EthSL model")
    parser.add_argument("--model",  required=True, help="Path to .h5 model file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    vocab = Vocabulary(cfg["data"]["vocabulary"])
    labels = vocab.get_all_labels()

    print(f"\n{'='*60}")
    print(f"Evaluating: {args.model}")
    print(f"{'='*60}")

    # Load model
    print("\n[1/4] Loading model...")
    model = keras.models.load_model(args.model)
    model.summary()

    # Load test data
    print("\n[2/4] Loading test data...")
    X_test, y_test = load_test_data(cfg)
    print(f"  Test samples: {len(X_test)}")

    # Predict
    print("\n[3/4] Running predictions...")
    y_prob = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # Metrics
    print("\n[4/4] Computing metrics...")
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    latency_ms = measure_latency(model, X_test)

    print(f"\n{'='*60}")
    print(f"  Top-1 Accuracy :  {acc*100:.2f}%")
    print(f"  Precision (macro): {p*100:.2f}%")
    print(f"  Recall (macro):    {r*100:.2f}%")
    print(f"  F1-score (macro):  {f1*100:.2f}%")
    print(f"  Latency (per sample): {latency_ms:.2f} ms")
    print(f"{'='*60}")

    # Per-class report
    print("\nPer-class classification report:")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    output_dir = Path(args.model).parent
    plot_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")

    # Save JSON report
    report = {
        "model_path":    args.model,
        "accuracy":      float(acc),
        "precision":     float(p),
        "recall":        float(r),
        "f1":            float(f1),
        "latency_ms":    float(latency_ms),
        "test_samples":  int(len(X_test)),
        "per_class":     classification_report(
            y_test, y_pred, target_names=labels, zero_division=0, output_dict=True
        ),
    }
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
