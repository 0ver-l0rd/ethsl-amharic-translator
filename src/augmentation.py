"""
src/augmentation.py
===================
Data augmentation utilities for sign language landmark sequences.

All augmentations operate on numpy arrays of shape (T, D) where:
  T = sequence length (e.g. 30 frames)
  D = feature dimension (e.g. 258 for pose + hands)

They expect NORMALIZED sequences (SequenceBuilder.normalize_sequence): the
origin is the signer's mid-shoulder point and the unit is the shoulder width,
so x = 0 is the body's midline. Missing landmarks are zeros and stay zeros.

Authors: Estifanos Behailu
"""

import numpy as np
from typing import Optional

from src.landmark_layout import mirror, coordinate_mask


# ---------------------------------------------------------------------------
# Individual augmentation transforms
# ---------------------------------------------------------------------------

def mirror_horizontal(sequence: np.ndarray) -> np.ndarray:
    """
    Mirror the signer left-right: x -> -x about the body's midline, and the
    left and right hands (and left/right pose and face points) swap channels,
    giving the left-handed version of the sign.

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)

    Returns
    -------
    flipped : np.ndarray of shape (T, D)
    """
    return mirror(sequence)


def add_noise(sequence: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
    """
    Add small Gaussian noise to every detected landmark coordinate, like
    MediaPipe's frame-to-frame jitter. Visibility and missing landmarks
    are left alone.

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)
    noise_std : float
        Standard deviation of the Gaussian noise, in shoulder widths.

    Returns
    -------
    noisy : np.ndarray of shape (T, D)
    """
    noise = np.random.normal(0, noise_std, size=sequence.shape).astype(np.float32)
    noise *= (sequence != 0) & coordinate_mask(sequence.shape[1])
    return sequence + noise


def temporal_jitter(
    sequence: np.ndarray,
    max_shift: int = 3,
) -> np.ndarray:
    """
    Shift the sequence in time by a random amount, as if the sign started a
    little earlier or later in the window. Frames shifted in repeat the
    first or last frame (no wrap-around from the other end).

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)
    max_shift : int
        Maximum number of frames to shift.

    Returns
    -------
    shifted : np.ndarray of shape (T, D)
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    idx = np.clip(np.arange(len(sequence)) - shift, 0, len(sequence) - 1)
    return sequence[idx]


def scale_landmarks(
    sequence: np.ndarray,
    scale_range: tuple = (0.85, 1.15),
) -> np.ndarray:
    """
    Randomly scale all landmark coordinates about the mid-shoulder point
    (simulates signers with longer or shorter arms relative to their
    shoulders; distance from the camera is already removed by normalization).
    Visibility is not scaled; missing landmarks stay zero.

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)
    scale_range : tuple (min_scale, max_scale)

    Returns
    -------
    scaled : np.ndarray of shape (T, D)
    """
    scale = np.random.uniform(*scale_range)
    return np.where(coordinate_mask(sequence.shape[1]), sequence * scale, sequence)


def dropout_frames(
    sequence: np.ndarray,
    dropout_prob: float = 0.1,
) -> np.ndarray:
    """
    Randomly zero out entire frames, the same as a frame in which MediaPipe
    detected nothing.

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)
    dropout_prob : float
        Probability of dropping each frame.

    Returns
    -------
    dropped : np.ndarray of shape (T, D)
    """
    dropped = sequence.copy()
    mask = np.random.rand(sequence.shape[0]) < dropout_prob
    dropped[mask] = 0.0
    return dropped


# ---------------------------------------------------------------------------
# Composed augmentation pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """
    Composable augmentation pipeline for training.

    Parameters
    ----------
    mirror_prob : float
        Probability of applying horizontal mirror.
    noise_std : float
        Standard deviation of Gaussian noise, in shoulder widths. Set to 0 to disable.
    jitter_frames : int
        Max frame shift for temporal jitter. Set to 0 to disable.
    scale_range : tuple or None
        (min, max) scale range. Set to None to disable.
    dropout_prob : float
        Frame dropout probability. Set to 0 to disable.
    """

    def __init__(
        self,
        mirror_prob: float = 0.5,
        noise_std: float = 0.02,
        jitter_frames: int = 3,
        scale_range: Optional[tuple] = (0.85, 1.15),
        dropout_prob: float = 0.1,
    ):
        self.mirror_prob   = mirror_prob
        self.noise_std     = noise_std
        self.jitter_frames = jitter_frames
        self.scale_range   = scale_range
        self.dropout_prob  = dropout_prob

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """Apply all enabled augmentations to a sequence."""
        if self.mirror_prob > 0 and np.random.rand() < self.mirror_prob:
            sequence = mirror_horizontal(sequence)
        if self.noise_std > 0:
            sequence = add_noise(sequence, noise_std=self.noise_std)
        if self.jitter_frames > 0:
            sequence = temporal_jitter(sequence, max_shift=self.jitter_frames)
        if self.scale_range is not None:
            sequence = scale_landmarks(sequence, scale_range=self.scale_range)
        if self.dropout_prob > 0:
            sequence = dropout_frames(sequence, dropout_prob=self.dropout_prob)
        return sequence
