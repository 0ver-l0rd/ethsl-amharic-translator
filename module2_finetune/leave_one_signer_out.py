"""
module2_finetune/leave_one_signer_out.py
=========================================
The honest accuracy figure: for each signer, train on everyone else and test
on that signer, then average. A third signer's clips pick the epoch to keep,
so the tested signer influences nothing.

Results go to <out_dir>/<tag>_<signer>.json; run with --signers to split the
folds between processes. The summary covers every fold saved under the tag.

Usage:
  cd module2_finetune
  python leave_one_signer_out.py --arch gru --tag gru
  python leave_one_signer_out.py --arch lstm --pretrained ../models/pretrained/islr_lstm.h5 --tag lstm_islr

Authors: EthSL Team
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from module1_refinement.model_architectures import build_model
from src.augmentation import AugmentationPipeline
from src.vocabulary import Vocabulary

ROOT = Path(__file__).parent.parent


class AugmentedBatches(keras.utils.PyDataset):
    """Training batches, freshly augmented every epoch."""

    def __init__(self, X, y, batch_size, aug, repeats):
        super().__init__()
        self.X, self.y, self.batch_size, self.aug, self.repeats = X, y, batch_size, aug, repeats
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch_size))

    def on_epoch_end(self):
        self.order = np.random.permutation(np.tile(np.arange(len(self.y)), self.repeats))

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        return np.stack([self.aug(self.X[j]) for j in idx]), self.y[idx]


def build(arch, num_classes, shape, pretrained, lr):
    model = build_model(arch, num_classes=num_classes, sequence_length=shape[0], feature_dim=shape[1])
    if pretrained:
        base = keras.models.load_model(pretrained)
        for layer in model.layers:
            if layer.name != "output" and layer.get_weights():
                layer.set_weights(base.get_layer(layer.name).get_weights())
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def run_fold(args, X, y, signers, test_signer, num_classes):
    people = sorted(np.unique(signers))
    val_signer = people[(people.index(test_signer) + 1) % len(people)]
    test, val = signers == test_signer, signers == val_signer
    train = ~(test | val)

    tf.keras.utils.set_random_seed(args.seed)
    model = build(args.arch, num_classes, X.shape[1:], args.pretrained, args.lr)
    batches = AugmentedBatches(X[train], y[train], args.batch_size, AugmentationPipeline(), args.repeats)
    model.fit(batches, validation_data=(X[val], y[val]), epochs=args.epochs, verbose=2, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
    ])
    pred = model.predict(X[test], verbose=0).argmax(1)
    return {
        "test_signer": test_signer,
        "val_signer": val_signer,
        "accuracy": float((pred == y[test]).mean()),
        "n_test": int(test.sum()),
        "confusion": confusion_matrix(y[test], pred, labels=range(num_classes)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Leave-one-signer-out evaluation")
    parser.add_argument("--arch", default="gru", choices=["lstm", "bilstm", "gru", "transformer"])
    parser.add_argument("--pretrained", default=None, help="Copy encoder weights from this model (same --arch)")
    parser.add_argument("--tag", required=True, help="Name for this experiment's result files")
    parser.add_argument("--signers", nargs="*", default=None, help="Only run these folds")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3, help="Augmented copies of each clip per epoch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed_dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--vocab", default=str(ROOT / "data" / "vocabulary.json"))
    parser.add_argument("--out_dir", default=str(ROOT / "logs" / "loso"))
    args = parser.parse_args()

    proc = Path(args.processed_dir)
    X = np.load(proc / "X.npy").astype(np.float32)
    y = np.load(proc / "y.npy")
    signers = np.load(proc / "signer_ids.npy").astype(str)
    vocab = Vocabulary(args.vocab)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for signer in args.signers or sorted(np.unique(signers)):
        print(f"\n=== Testing on {signer} ({args.tag}) ===", flush=True)
        result = run_fold(args, X, y, signers, signer, vocab.num_classes)
        (out_dir / f"{args.tag}_{signer}.json").write_text(json.dumps(result))
        print(f"{signer}: {result['accuracy'] * 100:.1f}%", flush=True)

    folds = [json.loads(p.read_text()) for p in sorted(out_dir.glob(f"{args.tag}_s*.json"))]
    acc = np.array([f["accuracy"] for f in folds])
    print(f"\n{args.tag}: {acc.mean() * 100:.1f}% mean over {len(acc)} signers "
          f"(worst {acc.min() * 100:.1f}%, best {acc.max() * 100:.1f}%)")
    total = np.sum([f["confusion"] for f in folds], axis=0)
    for i, row in enumerate(total):
        if row.sum():
            print(f"  {vocab.id_to_label(i):12s} {row[i] / row.sum() * 100:5.1f}%")


if __name__ == "__main__":
    main()
