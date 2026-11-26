"""Tests for src/retrain.find_live_model: the apps find the model themselves."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.retrain import find_live_model


def test_live_model_is_used_where_it_is(tmp_path):
    live = tmp_path / "finetuned"
    live.mkdir()
    (live / "best_model.h5").write_bytes(b"live")
    assert find_live_model(live, [tmp_path]) == (live / "best_model.h5", None)


def test_newest_download_is_copied_into_place(tmp_path):
    downloads, live = tmp_path / "Downloads", tmp_path / "models" / "finetuned"
    (downloads / "Telegram Desktop").mkdir(parents=True)
    old, new = downloads / "best_model.h5", downloads / "Telegram Desktop" / "best_model (1).h5"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    os.utime(old, (1_000, 1_000))
    model, copied_from = find_live_model(live, [downloads])
    assert model == live / "best_model.h5" and copied_from == new
    assert model.read_bytes() == b"new"


def test_no_model_anywhere(tmp_path):
    (tmp_path / "previous_model.h5").write_bytes(b"x")     # not a best_model
    assert find_live_model(tmp_path / "finetuned", [tmp_path]) == (None, None)
    assert not (tmp_path / "finetuned").exists()
