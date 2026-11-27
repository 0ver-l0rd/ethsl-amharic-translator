"""
src/sign_spotter.py
===================
Finds where each sign starts and ends in the live camera stream, so the app
classifies once per sign instead of guessing on every frame.

A sign is a stretch where a hand is tracked and a wrist is clearly above its
resting height, close to the test split_recording.py uses to cut the training
clips. The sign is over when the hands have been down for MIN_REST, or when
they keep still in the air for PAUSE_SECONDS after moving for at least
MIN_PAUSED_SIGN, so signs can follow each other without lowering the hands
(as in a phrase, or the way Menase signs). The classifier gets a window
centred on the sign, SIGN_WINDOW_SECONDS long or longer for a slow sign
(sequence_builder.sign_window), which is how the training clips are cut too.

No MediaPipe or TensorFlow import, so it can be tested on its own.
"""

from collections import deque
from enum import Enum
from typing import NamedTuple, Optional, Tuple

import numpy as np

from src.landmark_layout import LEFT_SHOULDER, POSE_DIMS, RIGHT_SHOULDER, body_relative
from src.sequence_builder import MAX_CLIP_SECONDS, SEQUENCE_LENGTH, SIGN_WINDOW_SECONDS, resample_window, sign_window

LEFT_WRIST, RIGHT_WRIST = 15, 16
HAND_DIMS = 63

RAISED_MARGIN = 0.5   # shoulder widths above resting wrist height that count as "hands up"
MIN_REST      = 0.35  # s of hands down that ends a sign (shorter: a hand dipping mid-sign)
MIN_SIGN      = 0.4   # s; shorter bursts are a twitch, not a sign
MAX_SIGN      = 4.0   # s; longer means the hands stayed up, not one sign
MOTION_SMOOTH = 0.25  # s; hand speed is averaged over this long

# Signs one after another with the hands kept up are split where the hands
# pause. Measured on our clips in shoulder widths per second: hands held still
# in the air move 0.2-0.35, signing 4-8; Menase's signs between pauses last
# 0.6-1.3 s. A shorter burst (just raising the hands) runs on into the next.
PAUSE_SPEED     = 0.7   # slower than this is keeping still
PAUSE_SECONDS   = 0.25  # s of keeping still that ends a sign; with the speed averaging,
                        # the hands must be still for about half a second
MIN_PAUSED_SIGN = 0.6   # s; a sign ended by a pause must have moved this long
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


def hand_speed(times: np.ndarray, keypoints: np.ndarray, smooth: float = MOTION_SMOOTH) -> np.ndarray:
    """
    How fast the hands move in each frame (shoulder widths per second, smoothed
    over `smooth` seconds): the fastest of the two wrists and the two hands'
    points, measured relative to the body. Points not found count as still.
    """
    rel = body_relative(keypoints)
    parts = [rel[:, [w * 4, w * 4 + 1]] for w in (LEFT_WRIST, RIGHT_WRIST)]
    for first in (POSE_DIMS, POSE_DIMS + HAND_DIMS):
        parts.append(rel[:, first:first + HAND_DIMS].reshape(len(rel), -1, 3)[..., :2].reshape(len(rel), -1))
    dt = np.diff(times)
    speed = np.zeros(len(rel))
    for p in parts:
        found = (p != 0).any(axis=1)
        both = found[1:] & found[:-1]
        step = np.linalg.norm((p[1:] - p[:-1]).reshape(len(dt), -1, 2), axis=2).mean(axis=1)
        moved = np.zeros(len(rel))
        moved[1:][both] = step[both] / np.maximum(dt[both], 1e-6)
        speed = np.maximum(speed, moved)
    k = max(1, int(round(smooth / np.median(dt)))) if len(dt) else 1
    return np.convolve(speed, np.ones(k) / k, mode="same")


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
        self._speeds: deque = deque()   # (t, hand speed) over the last MOTION_SMOOTH
        self._heights: deque = deque()  # (t, wrist height) with a body in view
        self._last_body = -np.inf
        self._now = -np.inf
        self.reset()

    def reset(self) -> None:
        """Forget the sign in progress (the resting height is kept)."""
        self._sign_start: Optional[float] = None
        self._last_active = -np.inf
        self._too_long = False
        self._hands_up = False          # up, though maybe keeping still between signs
        self._after_pause = False       # this sign started after a pause, hands up
        self._still_since: Optional[float] = None
        self._pending: deque = deque()  # (start, end) of windows still filling

    @property
    def status(self) -> Status:
        if self._sign_start is not None:
            return Status.SIGNING
        if self._pending:
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
        keep = max(self.window_seconds, MAX_CLIP_SECONDS) + MAX_SIGN + 1.0
        while self._frames and self._frames[0][0] < t - keep:
            self._frames.popleft()

        height, hand, pose_ok = (a[0] for a in activity_signal(keypoints[None]))
        if pose_ok:
            self._last_body = t
            self._heights.append((t, float(height)))
            while self._heights[0][0] < t - REST_HISTORY:
                self._heights.popleft()
        active = bool(pose_ok and hand and height > self.rest_height() + RAISED_MARGIN)

        moving = self._speed(t, keypoints) > PAUSE_SPEED

        event = None
        if self._pending and t >= self._pending[0][1]:
            event = Event("sign", self._window(*self._pending.popleft()))

        if active:
            if self._sign_start is None and (not self._hands_up or moving):
                # hands just came up, or moving again after a pause
                self._sign_start, self._after_pause = t, self._hands_up
                self._still_since = None
            self._hands_up = True
            self._last_active = t
            if self._sign_start is not None:
                if moving:
                    self._still_since = None
                elif self._still_since is None:
                    self._still_since = t
                if self._still_since is not None and t - self._still_since >= PAUSE_SECONDS \
                        and self._still_since - self._sign_start >= MIN_PAUSED_SIGN:
                    if not self._too_long:
                        self._queue(self._sign_start, self._still_since)
                    self._sign_start, self._too_long = None, False   # wait for the next move
                elif not self._too_long and t - self._sign_start > MAX_SIGN:
                    self._too_long = True
                    event = event or Event("too_long")
        elif self._hands_up and t - self._last_active >= MIN_REST:
            if self._sign_start is not None:
                start, end = self._sign_start, self._last_active
                shortest = MIN_PAUSED_SIGN if self._after_pause else MIN_SIGN   # else just lowering them
                if not self._too_long and end - start >= shortest:
                    self._queue(start, end)
            self._sign_start, self._too_long, self._hands_up = None, False, False
        return event

    def _queue(self, start: float, end: float) -> None:
        length = sign_window(start, end, self.window_seconds)
        lo = (start + end) / 2 - length / 2
        self._pending.append((lo, lo + length))

    def _speed(self, t: float, keypoints: np.ndarray) -> float:
        """Hand speed now, averaged over the last MOTION_SMOOTH seconds."""
        if len(self._frames) >= 2:
            (t0, k0), (t1, k1) = self._frames[-2], self._frames[-1]
            step = hand_speed(np.array([t0, t1]), np.stack([k0, k1]), smooth=0)[1]
            self._speeds.append((t, float(step)))
        while self._speeds and self._speeds[0][0] <= t - MOTION_SMOOTH:
            self._speeds.popleft()
        return float(np.mean([s for _, s in self._speeds])) if self._speeds else 0.0

    def _window(self, start: float, end: float) -> np.ndarray:
        times = np.array([f[0] for f in self._frames])
        frames = np.array([f[1] for f in self._frames], dtype=np.float32)
        return resample_window(times, frames, start, end, self.sequence_length)
