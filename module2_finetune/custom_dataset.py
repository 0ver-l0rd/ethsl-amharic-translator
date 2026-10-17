"""
module2_finetune/custom_dataset.py
====================================
Dataset loader for the EthSL landmark dataset.

Loads processed .npy arrays and creates both:
  - Signer-dependent splits (train/val/test from same signers)
  - Signer-independent splits (held-out signer as test)

Authors: Bersabeh Dawit, Menase Teshale
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class EthSLDataset:
    """
    Loads EthSL landmark sequences and organizes train/val/test splits.

    Parameters
    ----------
    cfg : dict
        Configuration dict loaded from config.yaml.
    """

    def __init__(self, cfg: dict):
        self.proc_dir   = Path(cfg["data"]["processed_dir"])
        self.splits_dir = Path(cfg["data"]["splits_dir"])
        self.cfg        = cfg

    def load_all(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load all available splits.

        Returns dict with keys:
          'train'   : (X_train, y_train)
          'val'     : (X_val,   y_val)
          'test_sd' : (X_test,  y_test)  signer-dependent
          'train_si', 'val_si', 'test_si' : signer-independent (if available).
              The held-out test signer is absent from train_si/val_si, so a
              signer-independent score must come from a model trained on train_si.
        """
        X = np.load(self.proc_dir / "X.npy").astype(np.float32)
        y = np.load(self.proc_dir / "y.npy")

        datasets = {}

        # Signer-dependent split
        train_idx = np.load(self.splits_dir / "train_idx.npy")
        val_idx   = np.load(self.splits_dir / "val_idx.npy")
        test_idx  = np.load(self.splits_dir / "test_idx.npy")

        datasets["train"]   = (X[train_idx], y[train_idx])
        datasets["val"]     = (X[val_idx],   y[val_idx])
        datasets["test_sd"] = (X[test_idx],  y[test_idx])

        print(f"  Signer-Dependent  — "
              f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

        # Signer-independent split (optional)
        si_paths = {
            key: self.splits_dir / f"signer_independent_{part}_idx.npy"
            for key, part in (("train_si", "train"), ("val_si", "val"), ("test_si", "test"))
        }
        if all(p.exists() for p in si_paths.values()):
            sizes = []
            for key, path in si_paths.items():
                idx = np.load(path)
                datasets[key] = (X[idx], y[idx])
                sizes.append(len(idx))
            print(f"  Signer-Independent — "
                  f"Train: {sizes[0]} | Val: {sizes[1]} | Test: {sizes[2]}")
        else:
            print("  Signer-Independent split not found — skipping.")

        return datasets

    @staticmethod
    def from_numpy(
        X: np.ndarray,
        y: np.ndarray,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create splits from raw arrays (useful for quick experiments).

        Parameters
        ----------
        X : (N, T, D)
        y : (N,)
        val_frac, test_frac : fraction of data for val/test
        seed : random seed
        """
        rng = np.random.default_rng(seed)
        N = len(X)
        idx = rng.permutation(N)

        n_test = int(N * test_frac)
        n_val  = int(N * val_frac)
        n_train = N - n_test - n_val

        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]

        return {
            "train":   (X[train_idx], y[train_idx]),
            "val":     (X[val_idx],   y[val_idx]),
            "test_sd": (X[test_idx],  y[test_idx]),
        }
