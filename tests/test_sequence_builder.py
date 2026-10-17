"""
tests/test_sequence_builder.py
===============================
Unit tests for SequenceBuilder: offline clip resampling and the time-based
real-time window.

Run with: pytest tests/
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sequence_builder import (
    SequenceBuilder, SEQUENCE_LENGTH, SIGN_WINDOW_SECONDS, MIN_WINDOW_FRAMES,
)

FEATURE_DIM = 6


def motion(t: float) -> np.ndarray:
    """A smooth fake 'sign': every feature moves linearly with time."""
    return np.full(FEATURE_DIM, t, dtype=np.float32)


def feed(builder: SequenceBuilder, fps: float, seconds: float, t0: float = 100.0) -> None:
    for i in range(int(seconds * fps) + 1):
        t = t0 + i / fps
        builder.add_frame(motion(t), timestamp=t)


class TestBuildFromClip:

    def test_output_shape(self):
        frames = [np.random.rand(FEATURE_DIM) for _ in range(47)]
        out = SequenceBuilder.build_from_clip(frames, target_length=SEQUENCE_LENGTH)
        assert out.shape == (SEQUENCE_LENGTH, FEATURE_DIM)

    def test_keeps_endpoints(self):
        frames = [motion(i) for i in range(12)]
        out = SequenceBuilder.build_from_clip(frames, target_length=SEQUENCE_LENGTH)
        np.testing.assert_allclose(out[0], frames[0])
        np.testing.assert_allclose(out[-1], frames[-1])


class TestTimeWindow:

    def test_not_ready_until_window_covered(self):
        b = SequenceBuilder(feature_dim=FEATURE_DIM)
        feed(b, fps=30, seconds=SIGN_WINDOW_SECONDS * 0.5)
        assert not b.is_ready()
        assert b.get_sequence() is None
        assert 0.4 < b.fill_ratio() < 0.6

    def test_ready_after_window(self):
        b = SequenceBuilder(feature_dim=FEATURE_DIM)
        feed(b, fps=30, seconds=SIGN_WINDOW_SECONDS + 0.2)
        assert b.is_ready()
        assert b.fill_ratio() == 1.0
        assert b.get_sequence().shape == (SEQUENCE_LENGTH, FEATURE_DIM)

    def test_same_sequence_at_any_frame_rate(self):
        # The point of the time-based window: a sign looks the same to the
        # model whether the app runs at 10 FPS or the camera at 30 FPS.
        slow, fast = SequenceBuilder(feature_dim=FEATURE_DIM), SequenceBuilder(feature_dim=FEATURE_DIM)
        feed(slow, fps=10, seconds=3.0)
        feed(fast, fps=30, seconds=3.0)
        np.testing.assert_allclose(slow.get_sequence(), fast.get_sequence(), atol=1e-4)

    def test_covers_exactly_the_last_window(self):
        b = SequenceBuilder(feature_dim=FEATURE_DIM)
        feed(b, fps=12, seconds=4.0, t0=0.0)
        seq = b.get_sequence()
        end = 4.0
        np.testing.assert_allclose(seq[-1], motion(end), atol=1e-4)
        np.testing.assert_allclose(seq[0], motion(end - SIGN_WINDOW_SECONDS), atol=1e-4)

    def test_too_few_frames_is_not_ready(self):
        b = SequenceBuilder(feature_dim=FEATURE_DIM)
        fps = (MIN_WINDOW_FRAMES - 2) / SIGN_WINDOW_SECONDS   # far too slow
        feed(b, fps=fps, seconds=SIGN_WINDOW_SECONDS * 2)
        assert not b.is_ready()

    def test_buffer_stays_bounded(self):
        b = SequenceBuilder(feature_dim=FEATURE_DIM)
        feed(b, fps=30, seconds=60.0)
        assert len(b._buffer) <= int(SIGN_WINDOW_SECONDS * 30) + 2

    def test_reset(self):
        b = SequenceBuilder(feature_dim=FEATURE_DIM)
        feed(b, fps=30, seconds=SIGN_WINDOW_SECONDS + 0.2)
        b.reset()
        assert not b.is_ready()
        assert b.fill_ratio() == 0.0
