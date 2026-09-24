"""
src/sequence_builder.py
=======================
Builds fixed-length landmark sequences from raw per-frame landmark arrays.

Each sign is represented as a tensor of shape (SEQUENCE_LENGTH, FEATURE_DIM),
e.g. (30, 258) for a 30-frame window with pose+hands only.

Authors: Bersabeh Dawit
"""

import numpy as np
from collections import deque
from typing import List, Optional


SEQUENCE_LENGTH = 30       # frames per sign window (~1 second at 30fps)
FEATURE_DIM = 258          # default: pose(132) + lh(63) + rh(63)


class SequenceBuilder:
    """
    Maintains a rolling buffer of landmark frames and builds
    fixed-length sequences for model inference.

    Parameters
    ----------
    sequence_length : int
        Number of frames per sequence window.
    feature_dim : int
        Number of features per frame.
    """

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH, feature_dim: int = FEATURE_DIM):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self._buffer: deque = deque(maxlen=sequence_length)

    # ------------------------------------------------------------------
    # Real-time buffering
    # ------------------------------------------------------------------

    def add_frame(self, keypoints: np.ndarray) -> None:
        """Append a single frame's keypoints to the rolling buffer."""
        assert keypoints.shape == (self.feature_dim,), (
            f"Expected keypoints of shape ({self.feature_dim},), got {keypoints.shape}"
        )
        self._buffer.append(keypoints)

    def is_ready(self) -> bool:
        """Return True when the buffer holds a full sequence."""
        return len(self._buffer) == self.sequence_length

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Return the current sequence as (sequence_length, feature_dim) array,
        or None if not enough frames yet.
        """
        if not self.is_ready():
            return None
        return np.array(self._buffer, dtype=np.float32)

    def reset(self) -> None:
        """Clear the buffer (call after a sign is recognised)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Offline sequence construction (for training)
    # ------------------------------------------------------------------

    @staticmethod
    def build_from_clip(
        frames: List[np.ndarray],
        target_length: int = SEQUENCE_LENGTH,
    ) -> np.ndarray:
        """
        Convert a variable-length list of landmark frames into a
        fixed-length sequence by linear interpolation.

        Parameters
        ----------
        frames : list of np.ndarray, each (feature_dim,)
        target_length : int

        Returns
        -------
        sequence : np.ndarray of shape (target_length, feature_dim)
        """
        frames_array = np.array(frames, dtype=np.float32)  # (T, D)
        T = len(frames_array)
        if T == target_length:
            return frames_array

        # Resample along time axis
        src_idx = np.linspace(0, T - 1, target_length)
        tgt_idx = np.arange(T)
        resampled = np.stack(
            [np.interp(src_idx, tgt_idx, frames_array[:, d])
             for d in range(frames_array.shape[1])],
            axis=1,
        )
        return resampled.astype(np.float32)

    @staticmethod
    def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
        """
        Z-score normalize a sequence per feature dimension.

        Parameters
        ----------
        sequence : np.ndarray of shape (T, D)

        Returns
        -------
        normalized : np.ndarray of shape (T, D)
        """
        mean = sequence.mean(axis=0, keepdims=True)
        std  = sequence.std(axis=0, keepdims=True) + 1e-8
        return (sequence - mean) / std
