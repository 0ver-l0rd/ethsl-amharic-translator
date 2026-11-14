"""
tests/test_split_recording.py
=============================
Tests for scripts/split_recording.py, using synthetic landmark tracks and a
small synthetic video.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from split_recording import (
    parse_recording_name, find_repetitions, clip_window, existing_clips, write_clips,
)
from src.landmark_layout import LEFT_SHOULDER, RIGHT_SHOULDER, POSE_DIMS
from src.sign_spotter import HAND_DIMS, LEFT_WRIST, RIGHT_WRIST

FPS = 30


def track(segments, duration=20.0, fps=FPS):
    """
    Keypoints for a signer resting with hands down, raising them (and having
    them tracked) during each (start, end) segment. Returns (times, keypoints).
    """
    times = np.arange(0, duration, 1 / fps)
    kp = np.zeros((len(times), POSE_DIMS + 2 * HAND_DIMS), dtype=np.float32)
    pose = kp[:, :POSE_DIMS].reshape(-1, 33, 4)      # view into kp
    pose[:, LEFT_SHOULDER, :2] = (0.6, 0.4)
    pose[:, RIGHT_SHOULDER, :2] = (0.4, 0.4)          # shoulder width 0.2
    pose[:, [LEFT_WRIST, RIGHT_WRIST], 1] = 0.8       # 2 shoulder widths below
    for start, end in segments:
        up = (times >= start) & (times <= end)
        pose[up, RIGHT_WRIST, 1] = 0.35
        kp[up, POSE_DIMS + HAND_DIMS:] = 0.5          # right hand tracked
    return times, kp


def test_recording_names():
    assert parse_recording_name(Path("s01_01.mp4")) == ("s01", "01")
    assert parse_recording_name(Path("/x/s02_E1.mov")) == ("s02", "E1")
    with pytest.raises(ValueError):
        parse_recording_name(Path("IMG_1234 copy.mov"))


def test_finds_each_repetition():
    segs = [(1.0 + 2.5 * k, 2.6 + 2.5 * k) for k in range(5)]
    found = find_repetitions(*track(segs))
    assert len(found) == 5
    for (s, e), (fs, fe) in zip(segs, found):
        assert abs(fs - s) < 0.05 and abs(fe - e) < 0.05


def test_brief_dropout_inside_a_sign_is_not_a_rest():
    # Tracking lost for 0.1 s mid-sign must not split the sign in two
    found = find_repetitions(*track([(1.0, 1.9), (2.0, 2.8), (6.0, 7.5)]))
    assert len(found) == 2
    assert found[0][0] == pytest.approx(1.0, abs=0.05) and found[0][1] == pytest.approx(2.8, abs=0.05)


def test_twitch_is_ignored():
    found = find_repetitions(*track([(1.0, 2.5), (5.0, 5.1), (8.0, 9.5)]))
    assert len(found) == 2


def test_no_body_finds_nothing():
    times = np.arange(0, 5, 1 / FPS)
    assert find_repetitions(times, np.zeros((len(times), POSE_DIMS + 2 * HAND_DIMS))) == []


def test_clip_window_is_centred_and_kept_inside_video():
    assert clip_window(4.0, 5.0, 2.5, 20.0) == pytest.approx((3.25, 5.75))
    assert clip_window(0.1, 1.0, 2.5, 20.0) == pytest.approx((0.0, 2.5))
    assert clip_window(19.0, 19.9, 2.5, 20.0) == pytest.approx((17.5, 20.0))
    assert clip_window(0.5, 1.0, 2.5, 1.5) == pytest.approx((0.0, 1.5))


def test_imported_landmarks_are_told_apart_from_recordings(tmp_path):
    sign = tmp_path / "misgana"
    sign.mkdir()
    (sign / "s01_seq_000_kp.npy").touch()          # imported dataset: no video
    (sign / "s02_seq_000.mp4").touch()             # our recording ...
    (sign / "s02_seq_000_kp.npy").touch()          # ... and its landmarks
    assert existing_clips(sign, "s01") == ([], [sign / "s01_seq_000_kp.npy"])
    assert existing_clips(sign, "s02") == ([sign / "s02_seq_000.mp4"], [])
    assert existing_clips(sign, "s03") == ([], [])
    assert existing_clips(tmp_path / "missing", "s01") == ([], [])


def test_write_clips_cuts_windows_at_output_rate(tmp_path):
    video = tmp_path / "s01_01.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 60, (64, 48))
    for i in range(240):                            # 4 s at 60 fps
        writer.write(np.full((48, 64, 3), i % 256, np.uint8))
    writer.release()

    paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    write_clips(video, [(0.5, 1.5), (2.0, 3.5)], paths, out_fps=30, max_height=720)
    for path, seconds in zip(paths, (1.0, 1.5)):
        cap = cv2.VideoCapture(str(path))
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30, abs=0.5)
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(seconds * 30, abs=1)
        cap.release()
