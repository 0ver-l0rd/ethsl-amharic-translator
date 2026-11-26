"""Tests for marking reps by hand: scripts/mark_reps.py and split_recording's --marks."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from mark_reps import Marks
from split_recording import read_marks, write_marks, reps_from_marks, marks_path
from test_split_recording import hands_up_track


def test_marks_file_round_trip(tmp_path):
    path = marks_path(tmp_path / "s04_08.mp4")
    assert path.name == "s04_08.marks.txt"
    write_marks(path, [(3.5, False), (1.25, True)], end=6.0, video_name="s04_08.mp4")
    assert read_marks(path) == ([(1.25, True), (3.5, False)], 6.0)
    path.write_text("# comment\n2.0\n\n4.0 skip  # bad one\n", encoding="utf-8")
    assert read_marks(path) == ([(2.0, False), (4.0, True)], None)


def test_marks_add_undo_skip_and_end():
    m = Marks()
    m.add(1.0)
    m.add(3.0)
    m.add(3.1)                          # pressed twice: moves the mark, doesn't add one
    assert m.as_list() == [(1.0, False), (3.1, False)]
    assert m.toggle_skip(2.0) == 0      # the rep the playhead is in
    assert m.as_list()[0] == (1.0, True)
    assert m.toggle_skip(0.5) is None   # before the first rep
    m.set_end(5.0)
    assert m.undo() and m.end is None
    assert m.undo() and m.as_list()[0] == (1.0, False)
    assert m.undo() and m.as_list() == [(1.0, False), (3.0, False)]
    m.undo(); m.undo()
    assert m.as_list() == [] and not m.undo()


def test_marked_reps_fit_the_movement_not_the_pause():
    segs = [(2.0 + 2.4 * k, 3.0 + 2.4 * k) for k in range(15)]
    times, kp = hands_up_track(segs)
    # marks pressed a little early or late, as a person would
    starts = [s + d for (s, _), d in zip(segs, np.tile([-0.15, 0.1, 0.0], 5))]
    found = reps_from_marks(times, kp, starts, end=segs[-1][1] + 0.6)
    assert len(found) == 15
    for (s, e), (fs, fe) in zip(segs, found):
        assert abs(fe - e) < 0.2                    # ends at the pause, not at the next mark
        assert fs >= s - 0.2 and fs <= s + 0.15     # starts with the movement after the mark


def test_last_rep_without_an_end_mark_gets_a_typical_length():
    segs = [(2.0 + 2.4 * k, 3.0 + 2.4 * k) for k in range(4)]
    times, kp = hands_up_track(segs, up=(1.0, 12.0), duration=14.0)
    found = reps_from_marks(times, kp, [s for s, _ in segs])
    assert len(found) == 4 and abs(found[-1][1] - segs[-1][1]) < 0.2


def test_a_mark_with_no_movement_keeps_its_whole_stretch():
    times, kp = hands_up_track([], up=(1.0, 9.0), duration=10.0)
    assert reps_from_marks(times, kp, [2.0, 4.5], end=7.0) == [(2.0, 4.5), (4.5, 7.0)]


def test_split_recording_cuts_the_marked_reps_and_leaves_out_bad_ones(tmp_path, monkeypatch, capsys):
    import split_recording
    segs = [(2.0 + 2.4 * k, 3.0 + 2.4 * k) for k in range(15)]
    times, kp = hands_up_track(segs)
    video = tmp_path / "s04_08.mp4"
    video.write_bytes(b"")
    write_marks(marks_path(video), [(s, k == 3) for k, (s, _) in enumerate(segs)], end=39.0)
    monkeypatch.setattr(split_recording, "analyse", lambda *a, **k: (times, kp, 30.0, 40.0))
    written = []
    monkeypatch.setattr(split_recording, "write_clips", lambda v, windows, paths, **k: written.extend(paths))
    code = split_recording.main([str(video), "--label", "test", "--output_dir", str(tmp_path / "out"),
                                 "--rotate", "0", "--marks"])
    assert code == 0 and len(written) == 14            # 15 marked, 1 bad; no --expected needed
    assert "marked by hand" in capsys.readouterr().out


def test_split_recording_refuses_without_a_marks_file(tmp_path):
    import split_recording
    video = tmp_path / "s04_09.mp4"
    video.write_bytes(b"")
    assert split_recording.main([str(video), "--label", "test", "--output_dir", str(tmp_path),
                                 "--rotate", "0", "--marks"]) == 2


def test_marks_pressed_at_the_end_of_each_sign_still_find_the_signs():
    # a signer who holds the end of the sign: the marks fall just after each sign,
    # so each sign lies at the end of the stretch before the next mark
    segs = [(2.0 + 2.4 * k, 3.0 + 2.4 * k) for k in range(15)]
    times, kp = hands_up_track(segs)
    marks = [e + 0.1 for _, e in segs]
    found = reps_from_marks(times, kp, marks)
    assert len(found) == 15
    for (s, e), (fs, fe) in zip(segs[1:], found[:-1]):     # mark k's rep is the sign after it
        assert abs(fs - s) < 0.25 and abs(fe - e) < 0.25
