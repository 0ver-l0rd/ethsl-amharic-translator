"""
scripts/preprocess.py
======================
Build the training dataset from extracted landmark .npy files.

For each sign label, loads all *_kp.npy files, normalizes the sequences,
and assembles X (features) and y (integer labels) arrays saved to data/processed/.

Usage:
  python scripts/preprocess.py
  python scripts/preprocess.py --input_dir data/landmarks --output_dir data/processed

Authors: Bersabeh Dawit
"""

import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sequence_builder import SequenceBuilder
from src.vocabulary import Vocabulary

LANDMARKS_DIR = Path(__file__).parent.parent / "data" / "landmarks"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
VOCAB_PATH    = Path(__file__).parent.parent / "data" / "vocabulary.json"


def main():
    parser = argparse.ArgumentParser(description="Build EthSL training dataset")
    parser.add_argument("--input_dir",  default=str(LANDMARKS_DIR))
    parser.add_argument("--output_dir", default=str(PROCESSED_DIR))
    parser.add_argument("--vocab",      default=str(VOCAB_PATH))
    parser.add_argument("--no_normalize", action="store_true",
                        help="Skip z-score normalization")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary(args.vocab)
    print(f"EthSL Preprocessing")
    print(f"===================")
    print(f"Vocabulary: {vocab.num_classes} signs")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    X_list = []
    y_list = []
    stats  = {}

    for label in tqdm(vocab.get_all_labels(), desc="Building dataset"):
        class_id  = vocab.label_to_id(label)
        sign_dir  = input_dir / label
        if not sign_dir.exists():
            print(f"\n  WARNING: No data for '{label}' — directory not found")
            continue

        kp_files = sorted(sign_dir.glob("*_kp.npy"))
        if not kp_files:
            print(f"\n  WARNING: No _kp.npy files for '{label}'")
            continue

        count = 0
        for kp_path in kp_files:
            seq = np.load(str(kp_path), allow_pickle=False).astype(np.float32)
            if seq.ndim != 2:
                continue

            if not args.no_normalize:
                seq = SequenceBuilder.normalize_sequence(seq)

            X_list.append(seq)
            y_list.append(class_id)
            count += 1

        stats[label] = count

    if not X_list:
        print("\nERROR: No data found. Run collect_data.py and extract_landmarks.py first.")
        return

    X = np.stack(X_list, axis=0)   # (N, T, D)
    y = np.array(y_list, dtype=np.int32)  # (N,)

    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Samples per class (min/max): {min(stats.values())}/{max(stats.values())}")

    # Save
    np.save(str(output_dir / "X.npy"), X)
    np.save(str(output_dir / "y.npy"), y)

    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump({
            "total_samples": len(X_list),
            "num_classes":   vocab.num_classes,
            "sequence_length": X.shape[1],
            "feature_dim":     X.shape[2],
            "samples_per_class": stats,
        }, f, indent=2)

    print(f"\nSaved X.npy, y.npy to {output_dir}")
    print(f"Run split_dataset.py next to create train/val/test splits.")


if __name__ == "__main__":
    main()
