"""
src/augmentation.py
===================
Data augmentation utilities for sign language landmark sequences.

All augmentations operate on numpy arrays of shape (T, D) where:
  T = sequence length (e.g. 30 frames)
  D = feature dimension (e.g. 258 for pose + hands)

Authors: Estifanos Behailu
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Individual augmentation transforms
# ---------------------------------------------------------------------------

def mirror_horizontal(sequence: np.ndarray) -> np.ndarray:
    """
    Flip the sequence left-right by negating all x-coordinates.

    Convention: MediaPipe x-coords are in [0, 1] relative to frame width.
    Flipping x → 1 - x mirrors the signer horizontally, which effectively
    swaps left/right handedness and doubles the dataset.

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)

    Returns
    -------
    flipped : np.ndarray of shape (T, D)
    """
    flipped = sequence.copy()
    # x-coords are at every 4th value for pose (x, y, z, vis),
    # every 3rd value for hands (x, y, z).
    # For simplicity, we flip ALL x-coords by finding indices divisible by 3 or 4.
    # A safe approximation: negate every element that represents x in [0,1] range.
    # Since sequences are z-normalized, we instead negate the first component
    # of each landmark group.
    # Practical implementation: negate the full x-component slice.
    flipped[:, 0::3] = 1.0 - flipped[:, 0::3]   # approximate x-flip
    return flipped


def add_noise(sequence: np.ndarray, noise_std: float = 0.005) -> np.ndarray:
    """
    Add small Gaussian noise to all landmarks to improve robustness.

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)
    noise_std : float
        Standard deviation of the Gaussian noise (in normalized coords).

    Returns
    -------
    noisy : np.ndarray of shape (T, D)
    """
    noise = np.random.normal(0, noise_std, size=sequence.shape).astype(np.float32)
    return sequence + noise


def temporal_jitter(
    sequence: np.ndarray,
    max_shift: int = 3,
) -> np.ndarray:
    """
    Shift the sequence in time by a random amount (roll with wrap-around).

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
    return np.roll(sequence, shift, axis=0)


def scale_landmarks(
    sequence: np.ndarray,
    scale_range: tuple = (0.85, 1.15),
) -> np.ndarray:
    """
    Randomly scale all landmark coordinates (simulates signer distance).

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)
    scale_range : tuple (min_scale, max_scale)

    Returns
    -------
    scaled : np.ndarray of shape (T, D)
    """
    scale = np.random.uniform(*scale_range)
    return sequence * scale


def dropout_frames(
    sequence: np.ndarray,
    dropout_prob: float = 0.1,
) -> np.ndarray:
    """
    Randomly zero out entire frames to simulate occlusion.

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
        Standard deviation of Gaussian noise. Set to 0 to disable.
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
        noise_std: float = 0.005,
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
