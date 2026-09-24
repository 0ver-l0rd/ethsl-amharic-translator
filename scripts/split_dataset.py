"""
scripts/split_dataset.py
=========================
Create train/val/test splits for the EthSL dataset.

Creates both:
  1. Signer-dependent split  (random stratified split, all signers in all sets)
  2. Signer-independent split (hold out one signer's recordings for test)

Saves index arrays to data/splits/. These indices are loaded by the
training scripts.

Usage:
  python scripts/split_dataset.py
  python scripts/split_dataset.py --val_frac 0.15 --test_frac 0.15

Authors: Bersabeh Dawit, Menase Teshale
"""

import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SPLITS_DIR    = Path(__file__).parent.parent / "data" / "splits"


def stratified_split(y, val_frac, test_frac, seed):
    """Create stratified train/val/test index split."""
    idx = np.arange(len(y))

    # First split: train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(idx, y))

    # Second split: train vs val
    val_frac_adj = val_frac / (1 - test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_adj, random_state=seed)
    train_idx, val_idx = next(sss2.split(train_val_idx, y[train_val_idx]))

    return train_val_idx[train_idx], train_val_idx[val_idx], test_idx


def main():
    parser = argparse.ArgumentParser(description="Create EthSL train/val/test splits")
    parser.add_argument("--val_frac",   type=float, default=0.15)
    parser.add_argument("--test_frac",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--processed_dir", default=str(PROCESSED_DIR))
    parser.add_argument("--splits_dir",    default=str(SPLITS_DIR))
    args = parser.parse_args()

    proc_dir   = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(proc_dir / "X.npy")
    y = np.load(proc_dir / "y.npy")

    print(f"EthSL Dataset Splitting")
    print(f"=======================")
    print(f"Total samples: {len(y)}")

    # ── Signer-dependent split ───────────────────────────────────────────
    train_idx, val_idx, test_idx = stratified_split(y, args.val_frac, args.test_frac, args.seed)

    np.save(str(splits_dir / "train_idx.npy"), train_idx)
    np.save(str(splits_dir / "val_idx.npy"),   val_idx)
    np.save(str(splits_dir / "test_idx.npy"),  test_idx)

    print(f"\nSigner-Dependent (Stratified):")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")

    # ── Signer-independent split ─────────────────────────────────────────
    # Load signer metadata if available
    signer_path = proc_dir / "signer_ids.npy"
    if signer_path.exists():
        signer_ids = np.load(str(signer_path))
        unique_signers = np.unique(signer_ids)
        hold_out = unique_signers[-1]  # Hold out the last signer (volunteer)

        si_test_mask  = signer_ids == hold_out
        si_train_mask = ~si_test_mask

        si_test_idx = np.where(si_test_mask)[0]
        si_all      = np.where(si_train_mask)[0]

        # Val from training signers
        si_train_idx, si_val_idx, _ = stratified_split(
            y[si_all], args.val_frac, 0.0, args.seed
        )
        si_train_idx = si_all[si_train_idx]
        si_val_idx   = si_all[si_val_idx]

        np.save(str(splits_dir / "signer_independent_train_idx.npy"), si_train_idx)
        np.save(str(splits_dir / "signer_independent_val_idx.npy"),   si_val_idx)
        np.save(str(splits_dir / "signer_independent_test_idx.npy"),  si_test_idx)

        print(f"\nSigner-Independent (Held-out signer: {hold_out}):")
        print(f"  Train: {len(si_train_idx)}")
        print(f"  Val:   {len(si_val_idx)}")
        print(f"  Test:  {len(si_test_idx)}")
    else:
        print(f"\nNOTE: signer_ids.npy not found — skipping signer-independent split.")
        print(f"      Add a signer ID to each recording in collect_data.py to enable this.")

    # Save split info
    with open(splits_dir / "split_info.json", "w") as f:
        json.dump({
            "val_frac":   args.val_frac,
            "test_frac":  args.test_frac,
            "seed":       args.seed,
            "total":      int(len(y)),
            "train":      int(len(train_idx)),
            "val":        int(len(val_idx)),
            "test":       int(len(test_idx)),
        }, f, indent=2)

    print(f"\nSplit indices saved to: {splits_dir}")


if __name__ == "__main__":
    main()
