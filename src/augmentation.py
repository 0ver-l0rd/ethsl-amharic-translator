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

from src.landmark_layout import mirror, coordinate_mask, layout, POSE_DIMS, HAND_LANDMARKS
from src.sequence_builder import _resample


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


def time_stretch(
    sequence: np.ndarray,
    speed_range: tuple = (0.7, 1.4),
) -> np.ndarray:
    """
    Play the sign faster or slower about the middle of the window, as if it
    were signed at a different pace or filmed over a different duration.
    Above speed 1 the ends repeat the first or last frame; below 1 the ends
    of the sequence are cut off.

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)
    speed_range : tuple (min_speed, max_speed)

    Returns
    -------
    stretched : np.ndarray of shape (T, D)
    """
    speed = np.random.uniform(*speed_range)
    times = np.arange(len(sequence), dtype=np.float64)
    middle = times[-1] / 2
    targets = np.clip(middle + (times - middle) * speed, 0, times[-1])
    return _resample(times, sequence, targets)


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


def tilt(sequence: np.ndarray, max_degrees: float = 12) -> np.ndarray:
    """
    Rotate the whole signer in the image plane about the mid-shoulder point by
    a random angle, like a signer leaning or a phone held at a slight angle.
    Only x and y change; missing landmarks stay zero.

    Parameters
    ----------
    sequence : np.ndarray of shape (T, D)
    max_degrees : float
        The angle is drawn from [-max_degrees, max_degrees].
    """
    lay = layout(sequence.shape[1])
    a = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
    out = sequence.copy()
    x, y = sequence[:, lay.xyz[:, 0]], sequence[:, lay.xyz[:, 1]]
    out[:, lay.xyz[:, 0]] = x * np.cos(a) - y * np.sin(a)
    out[:, lay.xyz[:, 1]] = x * np.sin(a) + y * np.cos(a)
    return out


def stretch_xy(
    sequence: np.ndarray,
    x_range: tuple = (0.85, 1.15),
    y_range: tuple = (0.9, 1.1),
) -> np.ndarray:
    """
    Stretch the signing space sideways and up-down by separate random factors,
    like signers who sign wider, narrower, higher or lower than others.
    Missing landmarks stay zero.
    """
    lay = layout(sequence.shape[1])
    out = sequence.copy()
    out[:, lay.xyz[:, 0]] *= np.random.uniform(*x_range)
    out[:, lay.xyz[:, 1]] *= np.random.uniform(*y_range)
    return out


def scale_hands(sequence: np.ndarray, scale_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """
    Make each hand bigger or smaller about its wrist (one random factor per
    hand), like people with bigger or smaller hands. The wrist stays where it
    is; frames in which the hand was not found stay zero.
    """
    out = sequence.copy()
    for first in (POSE_DIMS, POSE_DIMS + HAND_LANDMARKS * 3):
        cols = slice(first, first + HAND_LANDMARKS * 3)
        hand = sequence[:, cols].reshape(len(sequence), HAND_LANDMARKS, 3)
        found = (hand != 0).any(axis=(1, 2))
        wrist = hand[:, :1]
        scaled = wrist + (hand - wrist) * np.random.uniform(*scale_range)
        out[:, cols] = np.where(found[:, None, None], scaled, 0).reshape(len(sequence), -1)
    return out


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
    speed_range : tuple or None
        (min, max) playback speed. Set to None to disable.
    scale_range : tuple or None
        (min, max) scale range. Set to None to disable.
    dropout_prob : float
        Frame dropout probability. Set to 0 to disable.
    tilt_degrees : float
        Max rotation of the signer (tilt). 0 (the default) disables it.
    stretch_x_range, stretch_y_range : tuple or None
        (min, max) sideways / up-down stretch. None (the default) disables it.
    hand_scale_range : tuple or None
        (min, max) hand size about the wrist. None (the default) disables it.

    Tilt, stretch and hand size imitate differences between people; they
    raised the unseen-signer accuracy on the team's 65 words from 69.0% to
    73.3% (leave-one-signer-out, 2026-11-23).
    """

    def __init__(
        self,
        mirror_prob: float = 0.5,
        noise_std: float = 0.02,
        jitter_frames: int = 3,
        speed_range: Optional[tuple] = (0.7, 1.4),
        scale_range: Optional[tuple] = (0.85, 1.15),
        dropout_prob: float = 0.1,
        tilt_degrees: float = 0,
        stretch_x_range: Optional[tuple] = None,
        stretch_y_range: Optional[tuple] = None,
        hand_scale_range: Optional[tuple] = None,
    ):
        self.mirror_prob   = mirror_prob
        self.noise_std     = noise_std
        self.jitter_frames = jitter_frames
        self.speed_range   = speed_range
        self.scale_range   = scale_range
        self.dropout_prob  = dropout_prob
        self.tilt_degrees  = tilt_degrees
        self.stretch_x_range  = stretch_x_range
        self.stretch_y_range  = stretch_y_range
        self.hand_scale_range = hand_scale_range

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """Apply all enabled augmentations to a sequence."""
        if self.tilt_degrees > 0:
            sequence = tilt(sequence, max_degrees=self.tilt_degrees)
        if self.stretch_x_range is not None or self.stretch_y_range is not None:
            sequence = stretch_xy(sequence, x_range=self.stretch_x_range or (1, 1),
                                  y_range=self.stretch_y_range or (1, 1))
        if self.hand_scale_range is not None:
            sequence = scale_hands(sequence, scale_range=self.hand_scale_range)
        if self.mirror_prob > 0 and np.random.rand() < self.mirror_prob:
            sequence = mirror_horizontal(sequence)
        if self.noise_std > 0:
            sequence = add_noise(sequence, noise_std=self.noise_std)
        if self.jitter_frames > 0:
            sequence = temporal_jitter(sequence, max_shift=self.jitter_frames)
        if self.speed_range is not None:
            sequence = time_stretch(sequence, speed_range=self.speed_range)
        if self.scale_range is not None:
            sequence = scale_landmarks(sequence, scale_range=self.scale_range)
        if self.dropout_prob > 0:
            sequence = dropout_frames(sequence, dropout_prob=self.dropout_prob)
        return sequence
