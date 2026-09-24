"""
module2_finetune/callbacks.py
==============================
Training callbacks for Keras fine-tuning.

Authors: Emanuel Solomon
"""

import sys
from pathlib import Path
from datetime import datetime

try:
    from tensorflow import keras
except ImportError:
    print("ERROR: TensorFlow not installed.")
    sys.exit(1)


def build_callbacks(cfg: dict, output_dir: Path) -> list:
    """
    Build a list of Keras training callbacks.

    Parameters
    ----------
    cfg : dict
        Full configuration dict.
    output_dir : Path
        Directory to save checkpoints and logs.

    Returns
    -------
    list of keras.callbacks
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg["training"]
    log_dir = Path(cfg["output"]["logs_dir"]) / f"{output_dir.name}_{datetime.now():%Y%m%d_%H%M%S}"

    return [
        # Save best model by validation accuracy
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),

        # Stop if validation accuracy stops improving
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=tc.get("early_stopping_patience", 15),
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),

        # Reduce learning rate when val_loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=tc.get("reduce_lr_factor", 0.5),
            patience=tc.get("reduce_lr_patience", 7),
            min_lr=tc.get("min_lr", 1e-7),
            verbose=1,
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
        ),

        # CSV log for later analysis
        keras.callbacks.CSVLogger(
            filename=str(output_dir / "training_log.csv"),
            append=False,
        ),
    ]
