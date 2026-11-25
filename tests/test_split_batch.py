"""Tests for scripts/split_batch.py: name parsing, the word table and which videos are split."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from split_batch import load_words, parse_name


def test_names():
    assert parse_name(Path("s03_07.MP4")) == ("s03", "07", "")
    assert parse_name(Path("s03_7.mov")) == ("s03", "07", "")
    assert parse_name(Path("s03_29 a.MP4")) == ("s03", "29", "a")
    assert parse_name(Path("S02_e1.mp4")) == ("s02", "E1", "")
    assert parse_name(Path("IMG_8921.MP4")) is None or parse_name(Path("IMG_8921.MP4"))[1] not in load_words()


def test_word_table_is_complete_and_unique():
    words = load_words()
    assert len(words) == 65
    labels = [label for label, _ in words.values()]
    assert len(set(labels)) == 65
    assert words["01"] == ("selam", "ሰላም") and words["E1"] == ("misgana", "ምስጋና")


def test_one_version_brings_in_the_other_versions_of_the_word(tmp_path):
    from split_batch import select_videos
    for name in ["s01_29.mp4", "s01_29 b.mp4", "s01_30.mp4", "s01_55 b.mp4", "s01_55 a.mp4", "s02_29.mp4"]:
        (tmp_path / name).touch()
    names = lambda only: [p.name for p in select_videos(tmp_path, only)]
    assert names(["s01_29 b"]) == ["s01_29.mp4", "s01_29 b.mp4"]
    assert names(["s01_55 b"]) == ["s01_55 a.mp4", "s01_55 b.mp4"]
    assert names(["s01_30"]) == ["s01_30.mp4"]
    assert len(names(None)) == 6


def test_by_motion_is_passed_to_the_splitter(tmp_path, monkeypatch):
    import split_batch
    import split_recording
    (tmp_path / "s04_30.mp4").write_bytes(b"")
    calls = []
    monkeypatch.setattr(split_recording, "main", lambda argv: calls.append(argv) or 0)
    split_batch.main([str(tmp_path), "--by_motion", "--dry_run"])
    split_batch.main([str(tmp_path), "--dry_run"])
    assert "--by_motion" in calls[0] and "--by_motion" not in calls[1]


def test_marks_are_passed_and_unmarked_videos_skipped(tmp_path, monkeypatch):
    import split_batch
    import split_recording
    (tmp_path / "s04_30.mp4").write_bytes(b"")
    (tmp_path / "s04_31.mp4").write_bytes(b"")
    split_recording.write_marks(tmp_path / "s04_30.marks.txt", [(1.0, False), (3.0, False)])
    calls = []
    monkeypatch.setattr(split_recording, "main",
                        lambda argv: calls.append(argv) or print("  Found 2 repetitions") or 0)
    assert split_batch.main([str(tmp_path), "--marks", "--no_extract"]) == 1   # s04_31 isn't marked
    assert len(calls) == 1 and calls[0][0].endswith("s04_30.mp4")
    assert calls[0][calls[0].index("--marks") + 1].endswith("s04_30.marks.txt")
