"""
src/sign_spotter.py
===================
Finds where each sign starts and ends in the live camera stream, so the app
classifies once per sign instead of guessing on every frame.

A sign is a stretch where a hand is tracked and a wrist is clearly above its
resting height, the same test split_recording.py uses to cut the training
clips. When the hands have been down for MIN_REST, the sign is over; the
classifier gets a SIGN_WINDOW_SECONDS window centred on it, which is how the
training clips are cut too.

No MediaPipe or TensorFlow import, so it can be tested on its own.

Authors: Estifanos Behailu
"""

from collections import deque
from enum import Enum
from typing import NamedTuple, Optional, Tuple

import numpy as np

from src.landmark_layout import LEFT_SHOULDER, POSE_DIMS, RIGHT_SHOULDER
from src.sequence_builder import SEQUENCE_LENGTH, SIGN_WINDOW_SECONDS, resample_window

LEFT_WRIST, RIGHT_WRIST = 15, 16
HAND_DIMS = 63

RAISED_MARGIN = 0.5   # shoulder widths above resting wrist height that count as "hands up"
MIN_REST      = 0.35  # s of hands down that ends a sign (shorter: a hand dipping mid-sign)
MIN_SIGN      = 0.4   # s; shorter bursts are a twitch, not a sign
MAX_SIGN      = 4.0   # s; longer means the hands stayed up, not one sign
NO_BODY_AFTER = 1.0   # s without a body before the status says so

# Resting wrist height, learned from the last REST_HISTORY seconds. Until
# then it is DEFAULT_REST (measured on our phone recordings: -1.9 shoulder
# widths), and it never rises above REST_CEILING, so someone who signs
# without pausing is not taken to be resting.
REST_HISTORY = 20.0
DEFAULT_REST = -1.9
REST_CEILING = -1.0


def activity_signal(keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-frame wrist height and hand tracking from (T, >=258) keypoints in the
    extractor's layout (pose 33x4, left hand, right hand), in image coordinates.

    Returns (height, hand, pose_ok): height is the higher wrist above the
    mid-shoulder point in shoulder widths (image y points down, so it is
    negated); hand is True where either hand was detected; pose_ok is False
    where no body was found (height is then meaningless).
    """
    pose = keypoints[:, :POSE_DIMS].reshape(-1, 33, 4)
    ls, rs = pose[:, LEFT_SHOULDER, :2], pose[:, RIGHT_SHOULDER, :2]
    width = np.linalg.norm(ls - rs, axis=1)
    pose_ok = width > 1e-6
    mid_y = (ls[:, 1] + rs[:, 1]) / 2
    wrist_y = np.minimum(pose[:, LEFT_WRIST, 1], pose[:, RIGHT_WRIST, 1])
    height = np.where(pose_ok, (mid_y - wrist_y) / np.maximum(width, 1e-6), -np.inf)

    hands = keypoints[:, POSE_DIMS:POSE_DIMS + 2 * HAND_DIMS]
    hand = (np.abs(hands[:, :HAND_DIMS]).sum(1) > 0) | (np.abs(hands[:, HAND_DIMS:]).sum(1) > 0)
    return height, hand, pose_ok


class Status(Enum):
    NO_BODY     = "no_body"      # nobody in view
    READY       = "ready"        # hands down, waiting for a sign
    SIGNING     = "signing"      # hands up
    RECOGNISING = "recognising"  # sign over, waiting for the end of its window


class Event(NamedTuple):
    kind: str                            # "sign" or "too_long"
    sequence: Optional[np.ndarray] = None  # (sequence_length, D) for "sign"


class SignSpotter:
    """
    Feed it every frame with add_frame(); it returns an Event when a sign's
    window is complete ("sign", with the sequence to classify) or when the
    hands stay up too long to be one sign ("too_long"), otherwise None.
    """

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH,
                 window_seconds: float = SIGN_WINDOW_SECONDS):
        self.sequence_length = sequence_length
        self.window_seconds = window_seconds
        self._frames: deque = deque()   # (t, keypoints), oldest first
        self._heights: deque = deque()  # (t, wrist height) with a body in view
        self._last_body = -np.inf
        self._now = -np.inf
        self.reset()

    def reset(self) -> None:
        """Forget the sign in progress (the resting height is kept)."""
        self._sign_start: Optional[float] = None
        self._last_active = -np.inf
        self._too_long = False
        self._pending: Optional[Tuple[float, float]] = None  # window still filling

    @property
    def status(self) -> Status:
        if self._sign_start is not None:
            return Status.SIGNING
        if self._pending is not None:
            return Status.RECOGNISING
        if self._now - self._last_body > NO_BODY_AFTER:
            return Status.NO_BODY
        return Status.READY

    def sign_seconds(self) -> float:
        """How long the current sign has lasted (0 when not signing)."""
        return 0.0 if self._sign_start is None else self._now - self._sign_start

    def rest_height(self) -> float:
        if len(self._heights) < 10:
            return DEFAULT_REST
        rest = float(np.percentile([h for _, h in self._heights], 10))
        return min(rest, REST_CEILING)

    def add_frame(self, t: float, keypoints: np.ndarray) -> Optional[Event]:
        self._now = t
        self._frames.append((t, keypoints))
        keep = self.window_seconds + MAX_SIGN + 1.0
        while self._frames and self._frames[0][0] < t - keep:
            self._frames.popleft()

        height, hand, pose_ok = (a[0] for a in activity_signal(keypoints[None]))
        if pose_ok:
            self._last_body = t
            self._heights.append((t, float(height)))
            while self._heights[0][0] < t - REST_HISTORY:
                self._heights.popleft()
        active = bool(pose_ok and hand and height > self.rest_height() + RAISED_MARGIN)

        event = None
        if self._pending is not None and t >= self._pending[1]:
            event = Event("sign", self._window(*self._pending))
            self._pending = None

        if active:
            if self._sign_start is None:
                self._sign_start = t
            self._last_active = t
            if not self._too_long and t - self._sign_start > MAX_SIGN:
                self._too_long = True
                event = event or Event("too_long")
        elif self._sign_start is not None and t - self._last_active >= MIN_REST:
            start, end = self._sign_start, self._last_active
            if not self._too_long and end - start >= MIN_SIGN:
                lo = (start + end) / 2 - self.window_seconds / 2
                self._pending = (lo, lo + self.window_seconds)
            self._sign_start, self._too_long = None, False
        return event

    def _window(self, start: float, end: float) -> np.ndarray:
        times = np.array([f[0] for f in self._frames])
        frames = np.array([f[1] for f in self._frames], dtype=np.float32)
        return resample_window(times, frames, start, end, self.sequence_length)
