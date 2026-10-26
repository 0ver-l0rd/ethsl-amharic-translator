"""
tests/test_convert_islr.py
===========================
Tests for scripts/convert_islr.py and module1_refinement/pretrain_islr.py,
using fake clips in the Kaggle ISLR parquet layout.

Run with: pytest tests/
"""

import sys
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from convert_islr import islr_to_sequence, POSE_MIRROR_PERM, LH_COLS, RH_COLS
from src.landmark_extractor import LandmarkExtractor, FEATURE_DIMS, POSE_DIMS

# ISLR stores each frame as 543 rows in this order
ISLR_TYPES = [("face", 468), ("left_hand", 21), ("pose", 33), ("right_hand", 21)]


def make_landmarks(n_frames, rng, missing=(), offscreen_pose=()):
    """Random landmark coordinates per type; `missing` types are absent (NaN)."""
    lms = {}
    for kind, n in ISLR_TYPES:
        arr = rng.uniform(0.1, 0.9, (n_frames, n, 3)).astype(np.float32)
        if kind in missing:
            arr[:] = np.nan
        if kind == "pose":
            arr[:, list(offscreen_pose), 1] = 1.3   # below the image, like hips/legs
        lms[kind] = arr
    return lms


def to_islr_frame(lms):
    """Long-format table exactly like an ISLR .parquet file."""
    rows = []
    n_frames = len(lms["pose"])
    for f in range(n_frames):
        for kind, n in ISLR_TYPES:
            for i in range(n):
                x, y, z = lms[kind][f, i]
                rows.append((f + 20, f"{f + 20}-{kind}-{i}", kind, i, x, y, z))
    return pd.DataFrame(rows, columns=["frame", "row_id", "type", "landmark_index", "x", "y", "z"])


def our_extractor_frame(lms, f, face_mode):
    """What src.landmark_extractor produces for the same landmarks in frame f."""
    def landmark_list(arr, visibility=None):
        if np.isnan(arr).all():
            return None
        pts = [SimpleNamespace(x=x, y=y, z=z, visibility=None if visibility is None else visibility[i])
               for i, (x, y, z) in enumerate(arr)]
        return SimpleNamespace(landmark=pts)

    pose = lms["pose"][f]
    vis = ((pose[:, 0] >= 0) & (pose[:, 0] <= 1) & (pose[:, 1] >= 0) & (pose[:, 1] <= 1)).astype(np.float32)
    results = SimpleNamespace(
        pose_landmarks=landmark_list(pose, vis),
        left_hand_landmarks=landmark_list(lms["left_hand"][f]),
        right_hand_landmarks=landmark_list(lms["right_hand"][f]),
        face_landmarks=landmark_list(lms["face"][f]),
    )
    ext = LandmarkExtractor.__new__(LandmarkExtractor)
    ext.face_mode = face_mode
    return ext._extract_keypoints(results)


@pytest.mark.parametrize("face_mode", ["none", "subset", "full"])
def test_layout_matches_our_extractor(face_mode):
    # 30 frames so no resampling happens and frames compare one to one
    rng = np.random.default_rng(0)
    lms = make_landmarks(30, rng, missing=("left_hand",), offscreen_pose=(23, 24, 25))
    seq = islr_to_sequence(to_islr_frame(lms), face_mode)

    assert seq.shape == (30, FEATURE_DIMS[face_mode])
    for f in (0, 17, 29):
        np.testing.assert_allclose(seq[f], our_extractor_frame(lms, f, face_mode), atol=1e-6)


def test_visibility_and_missing_hands():
    rng = np.random.default_rng(1)
    lms = make_landmarks(30, rng, missing=("left_hand",), offscreen_pose=(23,))
    seq = islr_to_sequence(to_islr_frame(lms))
    pose = seq[:, :POSE_DIMS].reshape(30, 33, 4)
    assert (pose[:, 0, 3] == 1).all()       # on-screen point: visible
    assert (pose[:, 23, 3] == 0).all()      # off-screen point: not visible
    assert (seq[:, LH_COLS] == 0).all()     # missing hand: zeros
    assert (seq[:, RH_COLS] != 0).any()


def test_resamples_any_length_to_30_frames():
    rng = np.random.default_rng(2)
    for n in (1, 7, 95):
        assert islr_to_sequence(to_islr_frame(make_landmarks(n, rng))).shape == (30, 258)


def test_mirror_flips_x_and_swaps_sides():
    rng = np.random.default_rng(3)
    lms = make_landmarks(30, rng, missing=("left_hand",))
    df = to_islr_frame(lms)
    plain, mirrored = islr_to_sequence(df), islr_to_sequence(df, mirror=True)

    # The detected right hand moves to the left-hand channel, x mirrored
    rh = plain[:, RH_COLS].reshape(30, 21, 3)
    lh_m = mirrored[:, LH_COLS].reshape(30, 21, 3)
    np.testing.assert_allclose(lh_m[..., 0], 1 - rh[..., 0], atol=1e-6)
    np.testing.assert_allclose(lh_m[..., 1:], rh[..., 1:], atol=1e-6)
    assert (mirrored[:, RH_COLS] == 0).all()

    # Left shoulder (11) takes the mirrored right shoulder (12) and vice versa
    p, pm = plain[:, :POSE_DIMS].reshape(30, 33, 4), mirrored[:, :POSE_DIMS].reshape(30, 33, 4)
    np.testing.assert_allclose(pm[:, 11, 0], 1 - p[:, 12, 0], atol=1e-6)
    np.testing.assert_allclose(pm[:, 0, 0], 1 - p[:, 0, 0], atol=1e-6)   # nose stays
    assert sorted(POSE_MIRROR_PERM) == list(range(33))
    assert (POSE_MIRROR_PERM[POSE_MIRROR_PERM] == np.arange(33)).all()


@pytest.fixture
def fake_islr(tmp_path):
    """A tiny ISLR download: 3 signs x 4 participants x 2 clips."""
    rng = np.random.default_rng(4)
    islr = tmp_path / "islr_raw"
    rows, seq_id = [], 1000
    for p in (101, 102, 103, 104):
        for sign in ("book", "dog", "cat"):
            for _ in range(2):
                rel = f"train_landmark_files/{p}/{seq_id}.parquet"
                (islr / rel).parent.mkdir(parents=True, exist_ok=True)
                missing = ("left_hand",) if seq_id % 2 else ()
                to_islr_frame(make_landmarks(int(rng.integers(10, 60)), rng, missing)).to_parquet(islr / rel)
                rows.append((rel, p, seq_id, sign))
                seq_id += 1
    pd.DataFrame(rows, columns=["path", "participant_id", "sequence_id", "sign"]).to_csv(
        islr / "train.csv", index=False)
    return islr


def run(*cmd):
    out = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def test_convert_and_pretrain_end_to_end(fake_islr, tmp_path):
    out_dir, model_dir = tmp_path / "islr", tmp_path / "models"
    run("scripts/convert_islr.py", "--islr_dir", str(fake_islr), "--output_dir", str(out_dir),
        "--max_per_sign", "6")

    X, y = np.load(out_dir / "X.npy"), np.load(out_dir / "y.npy")
    assert X.shape == (18, 30, 258) and y.shape == (18,)
    assert json.loads((out_dir / "signs.json").read_text()) == ["book", "cat", "dog"]
    assert np.bincount(y).tolist() == [6, 6, 6]

    # A second run reuses the cached clips
    assert "Converted: 18 clips" in run("scripts/convert_islr.py", "--islr_dir", str(fake_islr),
                                        "--output_dir", str(out_dir), "--max_per_sign", "6")

    run("module1_refinement/pretrain_islr.py", "--data_dir", str(out_dir), "--output_dir",
        str(model_dir), "--epochs", "1", "--val_participants", "1")
    model_path = model_dir / "islr_lstm.h5"
    assert model_path.exists()
    assert json.loads(model_path.with_suffix(".json").read_text())["num_classes"] == 3

    # Module 1 can load it and keep everything except the output layer
    from tensorflow import keras
    from module1_refinement.refine_model import _replace_head
    cfg = {"data": {"sequence_length": 30, "feature_dim": 258}}
    base = keras.models.load_model(str(model_path))
    ethsl = _replace_head(base, "lstm", 60, cfg)
    assert ethsl.output_shape == (None, 60)
    for name in ("lstm_0", "lstm_1", "dense_hidden"):
        for a, b in zip(ethsl.get_layer(name).get_weights(), base.get_layer(name).get_weights()):
            np.testing.assert_array_equal(a, b)


class FakeKaggle:
    """Serves files from a local folder the way the Kaggle client saves them:
    named after the download URL, and zipped when `zipped` is set."""

    def __init__(self, source: Path, zipped=False):
        self.source, self.zipped, self.calls = source, zipped, []

    def competition_download_file(self, competition, file_name, path, quiet, force):
        import shutil, zipfile
        assert competition == "asl-signs"
        self.calls.append(file_name)
        src = self.source / file_name
        if self.zipped:
            with zipfile.ZipFile(Path(path) / "download.zip", "w") as zf:
                zf.write(src, src.name)
        else:
            shutil.copy(src, Path(path) / f"url-name-{src.name}")


@pytest.mark.parametrize("zipped", [False, True])
def test_fetch_file_handles_url_names_and_zips(fake_islr, tmp_path, monkeypatch, zipped):
    import convert_islr
    monkeypatch.setattr(convert_islr, "_kaggle_api", lambda: FakeKaggle(fake_islr, zipped))
    got = convert_islr.fetch_file("train.csv", tmp_path / "dl")
    assert got == tmp_path / "dl" / "train.csv"
    assert got.read_bytes() == (fake_islr / "train.csv").read_bytes()
    assert [p.name for p in (tmp_path / "dl").iterdir()] == ["train.csv"]   # temp folder cleaned up


def test_download_mode_converts_and_deletes_raw_clips(fake_islr, tmp_path, monkeypatch):
    import convert_islr
    fake = FakeKaggle(fake_islr)
    monkeypatch.setattr(convert_islr, "_kaggle_api", lambda: fake)
    cache, out_dir = tmp_path / "cache", tmp_path / "islr"
    monkeypatch.setattr(sys, "argv", ["convert_islr.py", "--download", "--islr_dir", str(cache),
                                      "--output_dir", str(out_dir), "--max_per_sign", "4"])
    convert_islr.main()

    assert np.load(out_dir / "X.npy").shape == (12, 30, 258)
    assert fake.calls[0] == "train.csv" and len(fake.calls) == 13
    assert not list(cache.rglob("*.parquet")) and not list(out_dir.glob("tmp*"))   # raw clips deleted
