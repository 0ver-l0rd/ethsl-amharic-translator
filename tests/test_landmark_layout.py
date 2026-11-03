"""
tests/test_landmark_layout.py
==============================
Tests for the frame layout, body-relative normalization, mirroring and the
augmentations built on them.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.landmark_layout import (
    layout, body_relative, mirror, coordinate_mask,
    POSE_MIRROR_PERM, FACE_MIRROR_PERM, FALLBACK_SHOULDER_WIDTH,
)
from src.landmark_extractor import FEATURE_DIMS, FACE_SUBSET_INDICES
from src.sequence_builder import SequenceBuilder
from src import augmentation as aug

T = 30


def raw_sequence(feature_dim=258, seed=0, missing_lh=()):
    """Fake raw landmarks: everything detected inside the image, except the
    left hand in the `missing_lh` frames."""
    rng = np.random.default_rng(seed)
    lay = layout(feature_dim)
    seq = np.zeros((T, feature_dim), np.float32)
    pts = rng.uniform(0.2, 0.8, (T, len(lay.xyz), 3)).astype(np.float32)
    pts[:, 11] = [0.6, 0.5, -0.2]    # left shoulder (image right, the camera is mirrored)
    pts[:, 12] = [0.4, 0.5, -0.2]    # right shoulder
    seq[:, lay.xyz] = pts
    seq[:, lay.visibility] = rng.uniform(0.0, 1.0, (T, 33))
    lh = slice(132, 195)
    seq[list(missing_lh), lh] = 0.0
    return seq


@pytest.mark.parametrize("mode", ["none", "subset", "full"])
def test_layout_covers_every_column_once(mode):
    dim = FEATURE_DIMS[mode]
    lay = layout(dim)
    cols = np.concatenate([lay.xyz.ravel(), lay.visibility])
    assert sorted(cols) == list(range(dim))
    assert coordinate_mask(dim).sum() == dim - 33


@pytest.mark.parametrize("mode", ["none", "subset", "full"])
def test_mirror_permutation_is_a_pairing(mode):
    perm = layout(FEATURE_DIMS[mode]).mirror_perm
    assert sorted(perm) == list(range(len(perm)))
    assert (perm[perm] == np.arange(len(perm))).all()


def test_face_mirror_pairs():
    assert (FACE_MIRROR_PERM[FACE_MIRROR_PERM] == np.arange(468)).all()
    # Eye outer corners, eye inner corners, mouth corners; nose tip on the midline
    for a, b in [(33, 263), (133, 362), (61, 291)]:
        assert FACE_MIRROR_PERM[a] == b
    assert FACE_MIRROR_PERM[1] == 1
    assert set(FACE_MIRROR_PERM[FACE_SUBSET_INDICES]) == set(FACE_SUBSET_INDICES)


def test_body_relative_puts_shoulders_at_plus_minus_half():
    out = body_relative(raw_sequence())
    pose = out[:, :132].reshape(T, 33, 4)
    np.testing.assert_allclose(pose[:, 11, :3], [[0.5, 0, 0]] * T, atol=1e-5)
    np.testing.assert_allclose(pose[:, 12, :3], [[-0.5, 0, 0]] * T, atol=1e-5)


def test_body_relative_ignores_position_and_distance():
    """The same sign further from the camera and off to one side."""
    raw = raw_sequence(seed=1)
    lay = layout(258)
    moved = raw.copy()
    pts = moved[:, lay.xyz]
    pts[..., :2] = pts[..., :2] * 0.6 + [0.25, 0.1]
    pts[..., 2] *= 0.6
    moved[:, lay.xyz] = pts
    np.testing.assert_allclose(body_relative(moved), body_relative(raw), atol=1e-4)


def test_body_relative_keeps_missing_and_visibility():
    raw = raw_sequence(missing_lh=[0, 1, 2])
    out = body_relative(raw)
    assert (out[:3, 132:195] == 0).all()
    assert (out[3:, 132:195] != 0).any()
    vis = layout(258).visibility
    np.testing.assert_array_equal(out[:, vis], raw[:, vis])
    assert out.dtype == np.float32


def test_body_relative_without_shoulders():
    raw = raw_sequence()
    raw[:, :132] = 0.0   # no pose at all, hands only
    out = body_relative(raw)
    assert np.isfinite(out).all()
    hands = out[:, 132:258].reshape(T, 42, 3)
    np.testing.assert_allclose(hands[..., :2].reshape(-1, 2).mean(0), [0, 0], atol=1e-4)
    assert np.abs(hands[..., :2]).max() <= 0.6 / FALLBACK_SHOULDER_WIDTH
    assert (body_relative(np.zeros((T, 258))) == 0).all()


def test_normalize_sequence_is_body_relative():
    raw = raw_sequence()
    np.testing.assert_array_equal(SequenceBuilder.normalize_sequence(raw), body_relative(raw))


@pytest.mark.parametrize("mode", ["none", "subset", "full"])
def test_mirror_twice_is_identity(mode):
    seq = body_relative(raw_sequence(FEATURE_DIMS[mode], missing_lh=[4]))
    np.testing.assert_array_equal(mirror(mirror(seq)), seq)


def test_mirror_swaps_hands_and_flips_x():
    seq = body_relative(raw_sequence(missing_lh=range(T)))
    m = mirror(seq)
    rh = seq[:, 195:258].reshape(T, 21, 3)
    lh_m = m[:, 132:195].reshape(T, 21, 3)
    np.testing.assert_allclose(lh_m[..., 0], -rh[..., 0])
    np.testing.assert_allclose(lh_m[..., 1:], rh[..., 1:])
    assert (m[:, 195:258] == 0).all()

    pose, pose_m = seq[:, :132].reshape(T, 33, 4), m[:, :132].reshape(T, 33, 4)
    np.testing.assert_allclose(pose_m[:, 11, 0], -pose[:, 12, 0])
    np.testing.assert_allclose(pose_m[:, 11, 3], pose[:, 12, 3])   # visibility moves with it
    np.testing.assert_allclose(pose_m[:, 0, 0], -pose[:, 0, 0])    # nose stays
    assert sorted(POSE_MIRROR_PERM) == list(range(33))


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def test_noise_skips_visibility_and_missing_landmarks():
    seq = body_relative(raw_sequence(missing_lh=[0, 1]))
    noisy = aug.add_noise(seq, noise_std=0.05)
    vis = layout(258).visibility
    np.testing.assert_array_equal(noisy[:, vis], seq[:, vis])
    assert (noisy[:2, 132:195] == 0).all()
    assert 0.03 < np.std(noisy - seq, where=(seq != 0) & coordinate_mask(258)) < 0.07


def test_temporal_jitter_pads_with_edge_frames():
    seq = np.repeat(np.arange(T, dtype=np.float32)[:, None], 258, axis=1)
    np.random.seed(0)
    for _ in range(20):
        shifted = aug.temporal_jitter(seq, max_shift=3)[:, 0]
        assert (np.diff(shifted) >= 0).all()              # never wraps around
        assert set(np.diff(shifted)) <= {0.0, 1.0}


def test_scale_leaves_visibility_and_missing():
    seq = body_relative(raw_sequence(missing_lh=[0]))
    scaled = aug.scale_landmarks(seq, scale_range=(1.1, 1.1))
    mask = coordinate_mask(258)
    np.testing.assert_allclose(scaled[:, mask], seq[:, mask] * 1.1, rtol=1e-6)
    np.testing.assert_array_equal(scaled[:, ~mask], seq[:, ~mask])
    assert (scaled[0, 132:195] == 0).all()


@pytest.mark.parametrize("mode", ["none", "subset"])
def test_pipeline_keeps_shape_and_missing(mode):
    seq = body_relative(raw_sequence(FEATURE_DIMS[mode], missing_lh=range(T)))
    pipe = aug.AugmentationPipeline(mirror_prob=0.0, dropout_prob=0.0)
    out = pipe(seq)
    assert out.shape == seq.shape
    assert (out[:, 132:195] == 0).all()
