"""
tests/test_video_io.py
=======================
Round-trip tests for saving and loading recorded clips.

Run with: pytest tests/
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_io import write_clip, read_clip


def smooth_frames(n=20, h=480, w=640):
    """Frames with a moving gradient: realistic enough for lossy video codecs."""
    y, x = np.mgrid[0:h, 0:w]
    return [np.dstack([(x + 8 * i) % 256, (y + 4 * i) % 256, np.full((h, w), 128)]).astype(np.uint8)
            for i in range(n)]


def test_mp4_roundtrip(tmp_path):
    frames = smooth_frames()
    path = tmp_path / "s01_seq_000.mp4"
    codec = write_clip(path, frames, fps=30)
    assert codec in ("avc1", "mp4v")

    back = read_clip(path)
    assert len(back) == len(frames)
    assert back[0].shape == frames[0].shape
    # Lossy, but close: mean absolute pixel error well under 10 / 255
    assert np.mean([np.abs(a.astype(int) - b.astype(int)).mean() for a, b in zip(frames, back)]) < 10
    # And much smaller than raw frames
    assert path.stat().st_size < sum(f.nbytes for f in frames) / 10


def test_legacy_npy_clip(tmp_path):
    frames = smooth_frames(5)
    path = tmp_path / "s01_seq_000.npy"
    np.save(path, np.array(frames))
    back = read_clip(path)
    assert len(back) == 5
    np.testing.assert_array_equal(back[2], frames[2])
