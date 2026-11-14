"""
tests/test_convert_gonfa.py
============================
Tests for scripts/convert_gonfa.py, using a fake download in the dataset's
folder layout.

Run with: pytest tests/
"""

import sys
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import convert_gonfa
from convert_gonfa import to_our_layout, SOURCE_DIM
from src.clip_naming import parse_signer
from src.landmark_extractor import FEATURE_DIMS, FACE_SUBSET_INDICES
from src.vocabulary import Vocabulary

POSE, FACE, HAND = 132, 1404, 63


def source_frames(t=30):
    """Frames whose value is the source column number, so moves are visible."""
    return np.tile(np.arange(SOURCE_DIM, dtype=np.float32), (t, 1))


def test_hands_come_straight_after_pose():
    out = to_our_layout(source_frames(), "none")
    assert out.shape == (30, 258)
    assert out[0, 0] == 0 and out[0, POSE - 1] == POSE - 1
    assert out[0, POSE] == POSE + FACE          # left hand starts after the face
    assert out[0, POSE + HAND] == POSE + FACE + HAND


@pytest.mark.parametrize("mode", ["none", "subset", "full"])
def test_face_modes_match_our_feature_sizes(mode):
    assert to_our_layout(source_frames(), mode).shape == (30, FEATURE_DIMS[mode])


def test_subset_takes_the_right_face_points():
    out = to_our_layout(source_frames(), "subset")
    first = FACE_SUBSET_INDICES[0]
    np.testing.assert_array_equal(out[0, 258:261], POSE + first * 3 + np.arange(3))


def write_clip(root, word, clip, frames):
    d = root / "EthSL_Dataset" / word / str(clip)
    d.mkdir(parents=True, exist_ok=True)
    for i in frames:
        np.save(d / f"{i}.npy", np.full(SOURCE_DIM, clip, dtype=np.float64))


@pytest.fixture
def fake_download(tmp_path):
    a, b = tmp_path / "dl" / "part-1", tmp_path / "dl" / "part-2"
    write_clip(a, "ለምን", 0, range(30))
    write_clip(a, "ለምን", 41, range(30))
    write_clip(a, "ሀዘን", 279, range(20))      # split across the two parts
    write_clip(b, "ሀዘን", 279, range(20, 30))
    write_clip(b, "ሀዘን", 5, range(35))         # stray frames past the last one
    write_clip(b, "ሀዘን", 6, range(28))         # incomplete
    (a / "EthSL_Dataset" / "ይቅርታ" / "0").mkdir(parents=True)  # empty word folder
    return tmp_path / "dl"


def convert(dl, tmp_path, monkeypatch):
    out, vocab = tmp_path / "landmarks", tmp_path / "vocabulary.json"
    monkeypatch.setattr(sys, "argv", ["convert_gonfa.py", "--dataset_dir", str(dl), "--output_dir", str(out),
                                      "--vocab", str(vocab), "--workers", "2"])
    convert_gonfa.main()
    return out, vocab


def test_clips_named_by_signer_and_merged_across_parts(fake_download, tmp_path, monkeypatch):
    out, _ = convert(fake_download, tmp_path, monkeypatch)
    names = {p.relative_to(out).as_posix() for p in out.rglob("*.npy")}
    assert names == {"lemin/g01_seq_000_kp.npy", "lemin/g02_seq_001_kp.npy",
                     "hazen/g07_seq_039_kp.npy", "hazen/g01_seq_005_kp.npy"}
    assert parse_signer(out / "lemin" / "g02_seq_001_kp.npy") == "g02"
    assert np.load(out / "hazen" / "g07_seq_039_kp.npy").shape == (30, 258)


def test_rerun_replaces_old_output_but_keeps_our_recordings(fake_download, tmp_path, monkeypatch):
    lemin = tmp_path / "landmarks" / "lemin"
    lemin.mkdir(parents=True)
    (lemin / "s01_seq_000_kp.npy").touch()        # old converter output (s01-s07 names)
    (lemin / "s03_seq_002.mp4").touch()           # our own recording ...
    (lemin / "s03_seq_002_kp.npy").touch()        # ... and its landmarks
    out, _ = convert(fake_download, tmp_path, monkeypatch)
    names = {p.name for p in (out / "lemin").iterdir()}
    assert names == {"g01_seq_000_kp.npy", "g02_seq_001_kp.npy", "s03_seq_002.mp4", "s03_seq_002_kp.npy"}


def test_vocabulary_lists_only_words_with_data(fake_download, tmp_path, monkeypatch):
    _, path = convert(fake_download, tmp_path, monkeypatch)
    vocab = Vocabulary(str(path))
    assert vocab.num_classes == 2
    assert sorted(vocab.get_all_labels()) == ["hazen", "lemin"]
    assert vocab.label_to_amharic("lemin") == "ለምን"
    assert json.loads(path.read_text(encoding="utf-8"))["metadata"]["total_signs"] == 2


def test_rerun_keeps_our_own_words_after_the_gonfa_ones(fake_download, tmp_path, monkeypatch):
    from src.vocabulary import add_word
    _, path = convert(fake_download, tmp_path, monkeypatch)
    add_word(path, "selam", "ሰላም", "Hello")
    _, path = convert(fake_download, tmp_path, monkeypatch)
    vocab = Vocabulary(str(path))
    assert vocab.num_classes == 3
    assert vocab.label_to_id("selam") == 2           # after the 2 Gonfa words
