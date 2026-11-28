"""
tests/test_speech_cache.py
==========================
Speech that keeps working offline (src/amharic_tts.py): each text is made
once and kept, prepare() makes the words and phrases in advance, and offline
an unmade text is joined from made pieces. Also the apps' word lists.

Run with: pytest tests/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.amharic_tts import AmharicTTS, speech_texts  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402


class FakeGTTS:
    """Stands in for gTTS: writes the text as bytes, or fails like a lost connection."""
    online = True
    made = []

    def __init__(self, text, lang, slow):
        self.text = text

    def save(self, path):
        if not FakeGTTS.online:
            raise ConnectionError("no internet")
        FakeGTTS.made.append(self.text)
        Path(path).write_bytes(f"<{self.text}>".encode())


def tts(tmp_path, online=True):
    FakeGTTS.online, FakeGTTS.made = online, []
    t = AmharicTTS(cache_dir=str(tmp_path))
    t._gtts, t._gtts_available = FakeGTTS, True
    return t


def test_speech_is_made_once_and_kept(tmp_path):
    t = tts(tmp_path)
    first = t.to_file("ሰላም")
    assert Path(first).read_bytes() == "<ሰላም>".encode()
    FakeGTTS.online = False                       # later, offline: still there
    assert t.to_file("ሰላም") == first and FakeGTTS.made == ["ሰላም"]


def test_prepare_makes_everything_and_counts(tmp_path):
    t = tts(tmp_path)
    assert t.prepare(["ሰላም", "እንዴት ነህ?", "ሰላም", ""]) == 2
    assert all(t.is_cached(x) for x in ["ሰላም", "እንዴት ነህ?"])


def test_prepare_stops_when_offline(tmp_path):
    t = tts(tmp_path, online=False)
    assert t.prepare(["ሰላም", "አንተ"]) == 0 and not t.is_cached("ሰላም")


def test_offline_text_is_joined_from_made_pieces(tmp_path):
    t = tts(tmp_path)
    t.prepare(["ሰላም", "እንዴት ነህ?"])
    FakeGTTS.online = False
    path = t.to_file("ሰላም እንዴት ነህ?")             # never made as a whole
    assert Path(path).read_bytes() == "<ሰላም><እንዴት ነህ?>".encode()
    assert t.to_file("ሰላም ውሃ") is None             # ውሃ was never made


def test_speech_texts_cover_words_and_phrases():
    vocab = Vocabulary(str(ROOT / "data" / "vocabulary.json"))
    texts = speech_texts(vocab)
    assert texts[:vocab.num_classes] == [vocab.id_to_amharic(i) for i in range(vocab.num_classes)]
    assert "እንዴት ነህ?" in texts and len(texts) == len(set(texts))


def test_words_by_category_cover_the_vocabulary():
    vocab = Vocabulary(str(ROOT / "data" / "vocabulary.json"))
    groups = vocab.by_category()
    assert sum(len(g) for g in groups.values()) == vocab.num_classes


def test_desktop_word_list_fits_without_overlap():
    from app.ui import SignLanguageUI
    ui = SignLanguageUI(Vocabulary(str(ROOT / "data" / "vocabulary.json")))
    size, line, rows, cols, col_w = ui._word_grid(640 - 32, 480 - 125 - 6 - 86 - 34)
    assert rows * cols >= len(ui.words) and rows * line <= 480 - 125 - 6 - 86 - 34
