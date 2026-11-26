"""Tests for scripts/dictionary_to_landmarks.py (AAU dictionary clips as signer d01)."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dictionary_to_landmarks as d2l
from src.clip_naming import parse_signer


def _video(frames=90, features=1662):
    """A still signer who moves the right hand in the middle of the video."""
    kp = np.zeros((frames, features), np.float32)
    kp[:, :d2l.FEATURES] = 0.5
    t = np.arange(frames)
    moving = (t > 30) & (t < 60)
    kp[moving, 195:258] += 0.1 * np.sin(t[moving] / 2)[:, None]
    return kp


def test_dictionary_clip_shape():
    clip = d2l.dictionary_clip(_video(), fps=30.0)
    assert clip.shape == (30, d2l.FEATURES)
    assert clip.dtype == np.float32


def test_matching_rows_splits_alternatives():
    rows = [{"label": "ሰላም / ደህና"}, {"label": "ሰላምታ"}, {"label": "ደህና"}]
    assert d2l.matching_rows(rows, "ደህና") == [rows[0], rows[2]]
    assert d2l.matching_rows(rows, "ሰላ") == []


def test_main_writes_d01_clips_and_replaces_old(tmp_path, monkeypatch):
    dict_dir, out = tmp_path / "dictionary", tmp_path / "landmarks"
    (dict_dir / "clips").mkdir(parents=True)
    for name in ["1", "2"]:
        np.savez(dict_dir / "clips" / f"{name}.npz", kp=_video(), fps=30.0)
    with open(dict_dir / "index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["filename", "label"])
        w.writeheader()
        w.writerows([{"filename": "1.mov", "label": "ሰላም"}, {"filename": "2.mov", "label": "ሰላም/ደህና"}])
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"signs": {"g": [
        {"id": 0, "label": "selam", "amharic": "ሰላም"},
        {"id": 1, "label": "ehet", "amharic": "እህት"}]}}), encoding="utf-8")
    stale = out / "selam" / "d01_seq_007_kp.npy"
    stale.parent.mkdir(parents=True)
    np.save(stale, np.zeros((30, 258)))
    team = out / "selam" / "s01_seq_000_kp.npy"
    np.save(team, np.zeros((30, 258)))

    monkeypatch.setattr(sys, "argv", ["x", "--dict_dir", str(dict_dir),
                                      "--landmarks_dir", str(out), "--vocab", str(vocab)])
    d2l.main()

    clips = sorted(p.name for p in (out / "selam").glob("d01_*"))
    assert clips == ["d01_seq_000_kp.npy", "d01_seq_001_kp.npy"]
    assert team.exists()
    assert parse_signer(out / "selam" / clips[0]) == "d01"
    assert not (out / "ehet").exists()
