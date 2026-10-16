"""
scripts/split_dataset.py
=========================
Create train/val/test splits for the EthSL dataset.

Creates both:
  1. Signer-dependent split  (random stratified split, all signers in all sets)
  2. Signer-independent split (hold out one signer's recordings for test).
     Requires data/processed/signer_ids.npy (written by preprocess.py) with
     at least two known signers. The held-out signer appears in NONE of the
     signer-independent train/val sets.

Saves index arrays to data/splits/. These indices are loaded by the
training scripts.

Usage:
  python scripts/split_dataset.py
  python scripts/split_dataset.py --val_frac 0.15 --test_frac 0.15
  python scripts/split_dataset.py --holdout_signer s04

Authors: Bersabeh Dawit, Menase Teshale
"""

import sys
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clip_naming import UNKNOWN_SIGNER

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SPLITS_DIR    = Path(__file__).parent.parent / "data" / "splits"

SI_SPLIT_FILES = (
    "signer_independent_train_idx.npy",
    "signer_independent_val_idx.npy",
    "signer_independent_test_idx.npy",
)


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


def signer_independent_split(y, signer_ids, hold_out, val_frac, seed):
    """
    Hold out every sample of one signer as the test set; split the remaining
    signers' samples into stratified train/val.
    """
    test_idx = np.where(signer_ids == hold_out)[0]
    # 'unknown' samples can't be guaranteed to come from a different person
    # than the held-out signer, so they are left out of this split entirely.
    rest_idx = np.where((signer_ids != hold_out) & (signer_ids != UNKNOWN_SIGNER))[0]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_rel, val_rel = next(sss.split(rest_idx, y[rest_idx]))
    return rest_idx[train_rel], rest_idx[val_rel], test_idx


def main():
    parser = argparse.ArgumentParser(description="Create EthSL train/val/test splits")
    parser.add_argument("--val_frac",   type=float, default=0.15)
    parser.add_argument("--test_frac",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--holdout_signer", default=None,
                        help="Signer ID to hold out for the signer-independent test "
                             "(default: the last signer ID in sorted order)")
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
    # Remove stale signer-independent indices from a previous run, so training
    # never evaluates against a split that no longer matches X.npy.
    for name in SI_SPLIT_FILES:
        (splits_dir / name).unlink(missing_ok=True)

    hold_out = None
    signer_path = proc_dir / "signer_ids.npy"
    if not signer_path.exists():
        print(f"\nNOTE: signer_ids.npy not found — skipping signer-independent split.")
        print(f"      Re-run preprocess.py on clips recorded with collect_data.py --signer.")
    else:
        signer_ids = np.load(str(signer_path))
        known = sorted(s for s in np.unique(signer_ids) if s != UNKNOWN_SIGNER)
        print(f"\nSigners found: {known}")

        if len(known) < 2:
            print(f"NOTE: need at least 2 known signers for a signer-independent split "
                  f"(found {len(known)}) — skipping.")
        elif args.holdout_signer is not None and args.holdout_signer not in known:
            print(f"ERROR: --holdout_signer '{args.holdout_signer}' not in {known}")
            sys.exit(1)
        else:
            hold_out = args.holdout_signer or known[-1]
            si_train_idx, si_val_idx, si_test_idx = signer_independent_split(
                y, signer_ids, hold_out, args.val_frac, args.seed
            )

            np.save(str(splits_dir / "signer_independent_train_idx.npy"), si_train_idx)
            np.save(str(splits_dir / "signer_independent_val_idx.npy"),   si_val_idx)
            np.save(str(splits_dir / "signer_independent_test_idx.npy"),  si_test_idx)

            print(f"\nSigner-Independent (Held-out signer: {hold_out}):")
            print(f"  Train: {len(si_train_idx)}")
            print(f"  Val:   {len(si_val_idx)}")
            print(f"  Test:  {len(si_test_idx)}")

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
            "holdout_signer": hold_out,
        }, f, indent=2)

    print(f"\nSplit indices saved to: {splits_dir}")


if __name__ == "__main__":
    main()
