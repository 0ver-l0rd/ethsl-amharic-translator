"""
src/sequence_builder.py
=======================
Builds fixed-length landmark sequences from raw per-frame landmark arrays.

Each sign is represented as a tensor of shape (SEQUENCE_LENGTH, FEATURE_DIM),
e.g. (30, 258) for a 30-frame window with pose+hands only.

A sign window is defined by DURATION (SIGN_WINDOW_SECONDS), not by a frame
count: recording captures exactly that long, and the live app resamples the
most recent SIGN_WINDOW_SECONDS to SEQUENCE_LENGTH evenly spaced frames using
frame timestamps. This keeps training and live sequences on the same time
scale even though the camera runs at ~25 FPS while recording and the live app
(MediaPipe-bound) runs at ~10 FPS.

Authors: Bersabeh Dawit
"""

import time
import numpy as np
from collections import deque
from typing import List, Optional


SEQUENCE_LENGTH     = 30   # frames per sign sequence fed to the model
SIGN_WINDOW_SECONDS = 1.5  # real time covered by one sign sequence
MIN_WINDOW_FRAMES   = 8    # fewer real frames than this per window is too sparse to classify
FEATURE_DIM = 258          # default: pose(132) + lh(63) + rh(63)


def _resample(times: np.ndarray, frames: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Linearly interpolate frames (T, D) sampled at `times` onto `targets`."""
    if len(times) == 1:
        return np.repeat(frames, len(targets), axis=0)
    idx = np.clip(np.searchsorted(times, targets, side="right") - 1, 0, len(times) - 2)
    t0, t1 = times[idx], times[idx + 1]
    w = np.clip((targets - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)[:, None]
    return (frames[idx] * (1.0 - w) + frames[idx + 1] * w).astype(np.float32)


class SequenceBuilder:
    """
    Maintains a rolling, timestamped buffer of landmark frames and builds
    fixed-length sequences covering the last `window_seconds` for inference.

    Parameters
    ----------
    sequence_length : int
        Number of frames per output sequence.
    feature_dim : int
        Number of features per frame.
    window_seconds : float
        Real time covered by one sequence (must match how training clips
        were recorded; see SIGN_WINDOW_SECONDS).
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        feature_dim: int = FEATURE_DIM,
        window_seconds: float = SIGN_WINDOW_SECONDS,
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.window_seconds = window_seconds
        self._buffer: deque = deque()  # (timestamp, keypoints), oldest first

    # ------------------------------------------------------------------
    # Real-time buffering
    # ------------------------------------------------------------------

    def add_frame(self, keypoints: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Append one frame's keypoints. `timestamp` is when the frame was
        captured (seconds, monotonic); defaults to now.
        """
        assert keypoints.shape == (self.feature_dim,), (
            f"Expected keypoints of shape ({self.feature_dim},), got {keypoints.shape}"
        )
        t = time.perf_counter() if timestamp is None else timestamp
        self._buffer.append((t, keypoints))
        # Keep one frame at or before the window start, for interpolation.
        while len(self._buffer) >= 2 and self._buffer[1][0] <= t - self.window_seconds:
            self._buffer.popleft()

    def _span(self) -> float:
        return self._buffer[-1][0] - self._buffer[0][0] if self._buffer else 0.0

    def fill_ratio(self) -> float:
        """How much of the window is covered so far, 0.0 to 1.0 (for the UI)."""
        return min(self._span() / self.window_seconds, 1.0)

    def is_ready(self) -> bool:
        """True once the buffer covers a full window with enough real frames."""
        return self._span() >= self.window_seconds and len(self._buffer) >= MIN_WINDOW_FRAMES

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Return the last `window_seconds` resampled to (sequence_length,
        feature_dim), or None if the window is not full yet.
        """
        if not self.is_ready():
            return None
        times  = np.array([t for t, _ in self._buffer])
        frames = np.array([kp for _, kp in self._buffer], dtype=np.float32)
        end = times[-1]
        targets = np.linspace(end - self.window_seconds, end, self.sequence_length)
        return _resample(times, frames, targets)

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
