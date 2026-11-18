"""Tests for scripts/split_batch.py name parsing and the word table."""

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
