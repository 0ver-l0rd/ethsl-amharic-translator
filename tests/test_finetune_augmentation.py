"""
tests/test_finetune_augmentation.py
====================================
Tests that fine-tuning uses the augmentation config.yaml asks for, and that
each epoch shows every clip once as recorded plus a new augmented copy.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("tensorflow")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module2_finetune"))

import finetune  # noqa: E402


def test_config_turns_on_signer_variation():
    cfg = yaml.safe_load((ROOT / "module2_finetune" / "config.yaml").read_text(encoding="utf-8"))
    pipe = finetune.augmentation_from_config(cfg["training"])
    assert cfg["training"]["augmentation"]["fresh_each_epoch"] is True
    assert pipe.tilt_degrees == 12
    assert pipe.stretch_x_range == (0.85, 1.15) and pipe.stretch_y_range == (0.9, 1.1)
    assert pipe.hand_scale_range == (0.8, 1.2)
    assert pipe.speed_range == (0.6, 1.6)


def test_old_configs_keep_the_old_augmentation():
    pipe = finetune.augmentation_from_config({"augmentation": {"scale_min": 0.9, "scale_max": 1.1}})
    assert pipe.speed_range == (0.7, 1.4) and pipe.scale_range == (0.9, 1.1)
    assert pipe.tilt_degrees == 0 and pipe.hand_scale_range is None
    assert finetune.augmentation_from_config({"use_augmentation": False}) is None


def test_fresh_copies_show_each_clip_as_recorded_and_augmented_every_epoch():
    X = np.arange(10, dtype=np.float32)[:, None, None] * np.ones((10, 30, 258), np.float32)
    y = np.arange(10)
    batches = finetune.FreshAugmentedCopies(X, y, 4, aug=lambda x: -x - 100)
    for _ in range(2):
        xs, ys = zip(*(batches[i] for i in range(len(batches))))
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        assert sorted(ys) == sorted(np.tile(y, 2))
        recorded = xs[:, 0, 0] >= 0
        assert sorted(xs[recorded, 0, 0]) == list(range(10))                 # each clip once as is
        np.testing.assert_array_equal(-xs[~recorded, 0, 0] - 100, ys[~recorded])  # labels match copies
        batches.on_epoch_end()
