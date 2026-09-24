"""
module1_refinement/refine_model.py
====================================
Transfer-learning refinement of a pretrained sign language model
on the EthSL (Ethiopian Sign Language) landmark dataset.

Pipeline:
  1. Load pretrained base model weights
  2. Replace classification head with EthSL-specific dense layer
  3. Phase 1: Train only the new head (encoder frozen)
  4. Phase 2: Unfreeze all layers, full fine-tuning with low LR
  5. Save best model to models/refined/

Usage:
  cd module1_refinement
  python refine_model.py --config config.yaml
  python refine_model.py --config config.yaml --from_scratch  # random init

Authors: Estifanos Behailu, Emanuel Solomon
"""

import os
import sys
import argparse
import json
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Tensorflow / Keras imports (lazy to give helpful error if not installed)
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.get_logger().setLevel("ERROR")
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_architectures import build_model, MODEL_REGISTRY
from src.augmentation import AugmentationPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(cfg: dict):
    """Load processed landmark sequences from disk."""
    proc_dir = Path(cfg["data"]["processed_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])

    X = np.load(proc_dir / "X.npy")          # (N, T, D)
    y = np.load(proc_dir / "y.npy")          # (N,) integer labels

    # Load signer-dependent split indices
    train_idx = np.load(splits_dir / "train_idx.npy")
    val_idx   = np.load(splits_dir / "val_idx.npy")
    test_idx  = np.load(splits_dir / "test_idx.npy")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def apply_augmentation(X: np.ndarray, y: np.ndarray, aug: AugmentationPipeline) -> tuple:
    """Augment training data."""
    X_aug = np.stack([aug(x) for x in X])
    return np.concatenate([X, X_aug]), np.concatenate([y, y])


def load_base_model(cfg: dict, num_classes: int, from_scratch: bool):
    """Load or create the base model."""
    arch    = cfg["pretrained"]["architecture"]
    base_path = Path(cfg["pretrained"]["path"])

    if from_scratch or not base_path.exists():
        if not from_scratch:
            print(f"  WARNING: Pretrained weights not found at {base_path}")
            print("  Falling back to random initialization (from_scratch=True)")
        print(f"  Building {arch.upper()} model from scratch...")
        model = build_model(
            arch,
            num_classes=num_classes,
            sequence_length=cfg["data"]["sequence_length"],
            feature_dim=cfg["data"]["feature_dim"],
        )
        return model, False

    print(f"  Loading pretrained weights from: {base_path}")
    try:
        base = keras.models.load_model(str(base_path))
        # Rebuild with EthSL head
        model = _replace_head(base, arch, num_classes, cfg)
        return model, True
    except Exception as e:
        print(f"  WARNING: Could not load pretrained model ({e})")
        print("  Falling back to random initialization...")
        model = build_model(arch, num_classes=num_classes,
                            sequence_length=cfg["data"]["sequence_length"],
                            feature_dim=cfg["data"]["feature_dim"])
        return model, False


def _replace_head(base_model, arch: str, num_classes: int, cfg: dict):
    """Replace the output layer of the base model with a new EthSL head."""
    seq_len  = cfg["data"]["sequence_length"]
    feat_dim = cfg["data"]["feature_dim"]

    # Build a fresh model with correct architecture
    new_model = build_model(arch, num_classes=num_classes,
                             sequence_length=seq_len, feature_dim=feat_dim)

    # Transfer weights layer by layer (where shapes match)
    transferred = 0
    for new_layer in new_model.layers:
        try:
            old_layer = base_model.get_layer(new_layer.name)
            if old_layer.get_weights() and new_layer.get_weights():
                old_w = old_layer.get_weights()
                new_w = new_layer.get_weights()
                # Only transfer if shapes match
                if all(o.shape == n.shape for o, n in zip(old_w, new_w)):
                    new_layer.set_weights(old_w)
                    transferred += 1
        except ValueError:
            pass  # layer not in base model — skip

    print(f"  Transferred weights for {transferred} layers")
    return new_model


def freeze_encoder(model, freeze: bool) -> None:
    """Freeze or unfreeze all layers except the output layer."""
    for layer in model.layers:
        if layer.name != "output":
            layer.trainable = not freeze
    # Recompile after changing trainable flags
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=model.optimizer.learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def get_callbacks(cfg: dict, phase: int, output_dir: Path) -> list:
    """Build Keras training callbacks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg["training"]
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"phase{phase}_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=tc["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=tc["reduce_lr_factor"],
            patience=tc["reduce_lr_patience"],
            min_lr=tc["min_lr"],
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(Path(cfg["output"]["logs_dir"]) / f"phase{phase}_{datetime.now():%Y%m%d_%H%M%S}"),
            histogram_freq=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, from_scratch: bool):
    tf.random.set_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    num_classes = cfg["data"]["num_classes"]
    output_dir  = Path(cfg["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(cfg)

    print("\n[2/5] Applying augmentation...")
    if cfg["training"]["use_augmentation"]:
        aug = AugmentationPipeline()
        X_train, y_train = apply_augmentation(X_train, y_train, aug)
        print(f"  Augmented train set: {X_train.shape}")

    print("\n[3/5] Loading / building model...")
    model, has_pretrained = load_base_model(cfg, num_classes, from_scratch)

    # ── Phase 1: Train only new EthSL head ──────────────────────────────
    p1 = cfg["refinement"]["phase1"]
    if has_pretrained and p1["freeze_encoder"]:
        print("\n[4/5] Phase 1 — Training EthSL head only (encoder frozen)...")
        freeze_encoder(model, freeze=True)
        model.optimizer.learning_rate.assign(p1["learning_rate"])
    else:
        print("\n[4/5] Phase 1 — Training from scratch (no frozen layers)...")

    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=p1["epochs"],
        batch_size=p1["batch_size"],
        callbacks=get_callbacks(cfg, phase=1, output_dir=output_dir),
        verbose=1,
    )

    # ── Phase 2: Full fine-tuning ────────────────────────────────────────
    p2 = cfg["refinement"]["phase2"]
    print("\n[5/5] Phase 2 — Full fine-tuning (all layers unfrozen)...")
    freeze_encoder(model, freeze=False)
    keras.backend.set_value(model.optimizer.learning_rate, p2["learning_rate"])

    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=p2["epochs"],
        batch_size=p2["batch_size"],
        callbacks=get_callbacks(cfg, phase=2, output_dir=output_dir),
        verbose=1,
    )

    # ── Final evaluation ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")

    # Save best model
    best_path = output_dir / cfg["output"]["best_model"]
    model.save(str(best_path))
    print(f"\nBest model saved to: {best_path}")

    # Save training summary
    summary = {
        "test_accuracy": float(test_acc),
        "test_loss":     float(test_loss),
        "phase1_epochs": len(history1.history["accuracy"]),
        "phase2_epochs": len(history2.history["accuracy"]),
        "architecture":  cfg["pretrained"]["architecture"],
        "from_scratch":  from_scratch,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine pretrained SL model on EthSL data")
    parser.add_argument("--config",       default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--from_scratch", action="store_true",   help="Skip pretrained weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, from_scratch=args.from_scratch)
