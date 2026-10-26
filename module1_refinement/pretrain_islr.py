"""
module1_refinement/pretrain_islr.py
====================================
Pretrain a model on the converted Kaggle ISLR dataset (250 ASL signs), to give
Module 1 a real starting point instead of random weights.

The saved model is the "pretrained" input of refine_model.py, which keeps every
layer except the 250-sign output and attaches a new EthSL output layer. The
architecture and face mode must therefore match what Module 1 trains: the
model is built with build_model(), so layer names line up, and its feature
size comes from the data (convert_islr.py --face).

Validation uses whole held-out participants, so the reported accuracy shows
how well the learned features carry over to signers the model has not seen.

Usage:
  python scripts/convert_islr.py --download --max_per_sign 100
  python module1_refinement/pretrain_islr.py --arch lstm
  python module1_refinement/pretrain_islr.py --arch bilstm --epochs 60

Output: models/pretrained/islr_<arch>.h5 (islr_<arch>_face-<mode>.h5 for face
data) and a .json with its validation accuracy.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel("ERROR")
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from module1_refinement.model_architectures import build_model, MODEL_REGISTRY
from src.landmark_extractor import face_mode_for_dim

DATA_DIR   = ROOT / "data" / "islr"
OUTPUT_DIR = ROOT / "models" / "pretrained"


def participant_split(participants: np.ndarray, n_val: int, seed: int):
    """Hold out every clip of n_val randomly chosen participants for validation."""
    unique = np.unique(participants)
    if len(unique) <= n_val:
        raise ValueError(f"need more than {n_val} participants for validation, found {len(unique)}")
    val_people = np.random.default_rng(seed).choice(unique, n_val, replace=False)
    is_val = np.isin(participants, val_people)
    return np.where(~is_val)[0], np.where(is_val)[0], sorted(val_people.tolist())


def main():
    parser = argparse.ArgumentParser(description="Pretrain on converted ISLR data")
    parser.add_argument("--arch", default="lstm", choices=list(MODEL_REGISTRY),
                        help="Must match pretrained.architecture in config.yaml")
    parser.add_argument("--data_dir",   default=str(DATA_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_participants", type=int, default=3)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy")
    participants = np.load(data_dir / "participant_ids.npy")
    signs = json.loads((data_dir / "signs.json").read_text())

    train_idx, val_idx, val_people = participant_split(participants, args.val_participants, args.seed)
    face_mode = face_mode_for_dim(X.shape[2])

    print("ISLR pretraining")
    print("================")
    print(f"Architecture: {args.arch} | Data: {X.shape} | Classes: {len(signs)} | Face: {face_mode}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} (held-out participants {val_people})")

    model = build_model(args.arch, num_classes=len(signs),
                        sequence_length=X.shape[1], feature_dim=X.shape[2])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=4, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(X[train_idx], y[train_idx],
                        validation_data=(X[val_idx], y[val_idx]),
                        epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=callbacks, verbose=2)

    _, val_acc = model.evaluate(X[val_idx], y[val_idx], verbose=0)
    suffix = "" if face_mode == "none" else f"_face-{face_mode}"
    model_path = output_dir / f"islr_{args.arch}{suffix}.h5"
    model.save(str(model_path))

    with open(model_path.with_suffix(".json"), "w") as f:
        json.dump({
            "source": "Kaggle ISLR (asl-signs), converted with scripts/convert_islr.py",
            "architecture": args.arch,
            "face_mode": face_mode,
            "feature_dim": int(X.shape[2]),
            "num_classes": len(signs),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "val_participants": val_people,
            "epochs_trained": len(history.history["loss"]),
            "val_accuracy": float(val_acc),
        }, f, indent=2)

    print(f"\nValidation accuracy on unseen participants: {val_acc*100:.1f}% "
          f"(chance: {100/len(signs):.1f}%)")
    print(f"Saved: {model_path}")
    print(f"Next: set pretrained.path to ../models/pretrained/{model_path.name} and "
          f"pretrained.architecture to {args.arch} in module1_refinement/config.yaml, then run "
          f"refine_model.py")


if __name__ == "__main__":
    main()
