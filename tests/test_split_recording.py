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
    parse_recording_name, find_repetitions, clip_window, existing_clips, write_clips, is_stop_reach,
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


def test_upright_check_spots_a_sideways_body():
    from split_recording import is_upright, NOSE
    _, kp = track([], duration=0.1)
    pose = kp[:, :POSE_DIMS].reshape(-1, 33, 4)
    pose[:, NOSE, :2] = (0.5, 0.2)                    # head above the shoulders
    assert is_upright(kp).all()
    side = kp.copy()
    p = side[:, :POSE_DIMS].reshape(-1, 33, 4)
    p[:, :, :2] = p[:, :, 1::-1]                      # swap x and y: lying on its side
    assert not is_upright(side).any()


def test_reaching_for_the_phone_at_the_end_is_not_a_sign():
    assert is_stop_reach(start=33.2, end=34.3, hand_tracked=0.03, video_duration=34.32)
    assert is_stop_reach(start=47.79, end=48.19, hand_tracked=0.46, video_duration=48.9)
    assert not is_stop_reach(start=33.2, end=34.3, hand_tracked=0.9, video_duration=34.32)  # a real last sign
    assert not is_stop_reach(start=29.0, end=30.0, hand_tracked=0.03, video_duration=34.32)  # not at the end
    assert not is_stop_reach(start=45.55, end=49.15, hand_tracked=0.39, video_duration=49.2)  # sign + reach


def test_hands_resting_in_view_still_split_by_height():
    segs = [(1.0 + 2.5 * k, 2.6 + 2.5 * k) for k in range(5)]
    times, kp = track(segs)
    kp[:, POSE_DIMS + HAND_DIMS:] = 0.5               # hand tracked all the time (resting in the lap)
    found = find_repetitions(times, kp)
    assert len(found) == 5
    for (s, e), (fs, fe) in zip(segs, found):
        assert abs(fs - s) < 0.05 and abs(fe - e) < 0.05


def test_hand_seen_at_rest_height_is_not_a_sign():
    segs = [(1.0 + 2.5 * k, 2.6 + 2.5 * k) for k in range(6)]
    times, kp = track(segs)
    late = times > 8.0
    kp[late, POSE_DIMS + HAND_DIMS:] = 0.5            # from 8 s the resting hand stays in view
    assert len(find_repetitions(times, kp)) == 6


def test_slow_sign_gets_a_longer_clip():
    from src.sequence_builder import sign_window
    assert sign_window(0.0, 1.1) == pytest.approx(2.5)        # fast sign: the usual window
    assert sign_window(0.0, 3.3) == pytest.approx(3.8)        # slow sign: all of it plus 0.25 s each side
    assert sign_window(0.0, 9.0) == pytest.approx(4.5)        # capped
    assert clip_window(10.0, 13.3, 2.5, 60.0) == pytest.approx((9.75, 13.55))


def test_rep_cut_off_by_the_recording_is_noticed():
    from split_recording import cut_by_recording
    assert cut_by_recording(0.0, 1.2, 40.0)                   # already signing at the start
    assert cut_by_recording(38.5, 40.0, 40.0)                 # still signing at the end
    assert not cut_by_recording(0.2, 1.4, 40.0)
