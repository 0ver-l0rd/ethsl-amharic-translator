"""
src/landmark_layout.py
======================
Where each landmark sits in one frame's feature vector, and the two
operations that need to know it: body-relative normalization and
left-right mirroring.

Frame layout (see landmark_extractor.py): pose 33 x (x, y, z, visibility),
left hand 21 x (x, y, z), right hand 21 x (x, y, z), then the face points of
the face mode (none / subset / full) as (x, y, z). Missing landmarks are zeros.

This module does not import MediaPipe (~13 s), so preprocessing and training
stay fast; landmark_extractor.py is loaded only for the face subset indices.

Authors: Estifanos Behailu
"""

from functools import lru_cache
from typing import NamedTuple

import numpy as np

# Same values as landmark_extractor.py (checked in tests/test_landmark_layout.py)
POSE_LANDMARKS, HAND_LANDMARKS, FACE_LANDMARKS = 33, 21, 468
POSE_DIMS = POSE_LANDMARKS * 4
FEATURE_DIM_NO_FACE = POSE_DIMS + HAND_LANDMARKS * 3 * 2   # 258

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

# Typical shoulder width in image coordinates, used only for clips in which
# the shoulders are never detected
FALLBACK_SHOULDER_WIDTH = 0.25

# MediaPipe Pose landmarks that trade places when the image is mirrored:
# eyes (1-3 <-> 4-6), ears, mouth corners, then every left/right body pair.
POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + \
                    [(i, i + 1) for i in range(11, POSE_LANDMARKS, 2)]
POSE_MIRROR_PERM = np.arange(POSE_LANDMARKS)
for _a, _b in POSE_MIRROR_PAIRS:
    POSE_MIRROR_PERM[_a], POSE_MIRROR_PERM[_b] = _b, _a

# Face Mesh left/right pairs, from the x-mirror of MediaPipe's canonical face
# model (mediapipe/modules/face_geometry/data/canonical_face_model.obj): an
# exact match for every point; the 28 points on the midline map to themselves.
# Pairs follow index runs: (first, last, offset) means i <-> i + offset.
_FACE_MIRROR_RUNS = [(20, 93, 230), (95, 150, 229), (153, 163, 227), (165, 167, 226),
                     (169, 174, 225), (176, 194, 224), (201, 247, 220)]
FACE_MIRROR_PAIRS = [(3, 248), (7, 249), (196, 419), (198, 420)] + \
                    [(i, i + off) for lo, hi, off in _FACE_MIRROR_RUNS for i in range(lo, hi + 1)]
FACE_MIRROR_PERM = np.arange(FACE_LANDMARKS)
for _a, _b in FACE_MIRROR_PAIRS:
    FACE_MIRROR_PERM[_a], FACE_MIRROR_PERM[_b] = _b, _a


class Layout(NamedTuple):
    xyz: np.ndarray          # (P, 3) columns of each landmark's x, y, z
    pose_xyz: np.ndarray     # (P,) bool: pose landmark (z is depth from the hips)
    visibility: np.ndarray   # (33,) pose visibility columns
    mirror_perm: np.ndarray  # (P,) landmark that each landmark becomes when mirrored


@lru_cache(maxsize=None)
def layout(feature_dim: int) -> Layout:
    """Column layout for a frame of `feature_dim` features (258, 534 or 1662)."""
    n_face, rest = divmod(feature_dim - FEATURE_DIM_NO_FACE, 3)
    if n_face < 0 or rest:
        raise ValueError(f"Feature dim {feature_dim} is not pose + hands + face points")

    pose = np.arange(POSE_LANDMARKS)[:, None] * 4 + np.arange(3)
    n_other = 2 * HAND_LANDMARKS + n_face
    other = POSE_DIMS + np.arange(n_other)[:, None] * 3 + np.arange(3)
    xyz = np.concatenate([pose, other])

    lh = POSE_LANDMARKS + np.arange(HAND_LANDMARKS)
    rh = lh + HAND_LANDMARKS
    if n_face == FACE_LANDMARKS:
        face_perm = FACE_MIRROR_PERM
    elif n_face:
        from src.landmark_extractor import FACE_SUBSET_INDICES
        if n_face != len(FACE_SUBSET_INDICES):
            raise ValueError(f"Feature dim {feature_dim} matches no face mode")
        position = {idx: k for k, idx in enumerate(FACE_SUBSET_INDICES)}
        face_perm = np.array([position[FACE_MIRROR_PERM[idx]] for idx in FACE_SUBSET_INDICES])
    else:
        face_perm = np.arange(0)
    mirror_perm = np.concatenate([POSE_MIRROR_PERM, rh, lh,
                                  POSE_LANDMARKS + 2 * HAND_LANDMARKS + face_perm])

    pose_xyz = np.arange(len(xyz)) < POSE_LANDMARKS
    visibility = np.arange(POSE_LANDMARKS) * 4 + 3
    return Layout(xyz, pose_xyz, visibility, mirror_perm)


def body_relative(sequence: np.ndarray) -> np.ndarray:
    """
    Express every landmark relative to the signer's body: the origin is the
    mid-shoulder point and the unit is the shoulder width, both averaged over
    the sequence. This removes where the signer stands and how far they are
    from the camera, but keeps hand shape and the hands' position against the
    body, which a per-feature z-score erased.

    Missing landmarks stay zero and visibility is unchanged. Hand and face z
    are already relative to the wrist / head, so they are only scaled.

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D), raw landmarks as extracted

    Returns
    -------
    normalized : np.ndarray of shape (T, D), float32
    """
    lay = layout(sequence.shape[1])
    out = sequence.astype(np.float32, copy=True)
    pts = out[:, lay.xyz]                          # (T, P, 3)
    present = np.abs(pts).sum(-1) > 0              # (T, P)

    shoulders = present[:, LEFT_SHOULDER] & present[:, RIGHT_SHOULDER]
    if shoulders.any():
        ls, rs = pts[shoulders, LEFT_SHOULDER], pts[shoulders, RIGHT_SHOULDER]
        center = ((ls + rs) / 2).mean(0)
        scale = np.linalg.norm((ls - rs)[:, :2], axis=1).mean()
    elif present.any():
        center = np.append(pts[present][:, :2].mean(0), 0.0)
        scale = FALLBACK_SHOULDER_WIDTH
    else:
        return out
    scale = max(scale, 1e-6)

    shift = np.where(lay.pose_xyz[:, None], center, np.append(center[:2], 0.0))  # (P, 3)
    pts = np.where(present[..., None], (pts - shift) / scale, 0.0)
    out[:, lay.xyz] = pts
    return out


def mirror(sequence: np.ndarray) -> np.ndarray:
    """
    Mirror a body-relative sequence left-right: x -> -x about the body's
    midline, and every left landmark trades places with its right partner
    (hands, pose pairs, face pairs). A left-handed version of the same sign.
    """
    lay = layout(sequence.shape[1])
    out = sequence.copy()
    pts = sequence[:, lay.xyz]
    pts[..., 0] *= -1
    out[:, lay.xyz] = pts[:, lay.mirror_perm]
    out[:, lay.visibility] = sequence[:, lay.visibility][:, POSE_MIRROR_PERM]
    return out


def coordinate_mask(feature_dim: int) -> np.ndarray:
    """(D,) bool: True for coordinate columns, False for pose visibility."""
    mask = np.ones(feature_dim, dtype=bool)
    mask[layout(feature_dim).visibility] = False
    return mask
