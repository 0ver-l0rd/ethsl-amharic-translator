"""Tests for src/phrases.py (the demo phrase table) and its use in app/live_session.py."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.live_session import SAY_WAIT, LiveSession
from src.phrases import Phrase, PhraseBuilder, load_phrases, phrases_for
from src.vocabulary import Vocabulary
from tests.test_split_recording import track

HOW = Phrase(("እንዴት", "አንተ"), "እንዴት ነህ?", "How are you?")
FINE = Phrase(("እኔ", "ደህና"), "ደህና ነኝ")
FINE_THANKS = Phrase(("እኔ", "ደህና", "ምስጋና"), "ደህና ነኝ፣ አመሰግናለሁ")


def say(builder, words, step=2.0):
    return [builder.add(w, i * step) for i, w in enumerate(words)]


def test_shipped_table_uses_only_vocabulary_words():
    vocab = Vocabulary(str(ROOT / "data" / "vocabulary.json"))
    phrases = phrases_for(vocab)
    assert len(phrases) == len(json.loads((ROOT / "data" / "phrases.json").read_text(encoding="utf-8"))["phrases"])
    assert any(p.amharic == "እንዴት ነህ?" for p in phrases)


def test_unknown_sign_or_single_sign_is_refused(tmp_path):
    path = tmp_path / "phrases.json"
    path.write_text(json.dumps({"phrases": [{"signs": ["እኔ", "ሻይ"], "amharic": "ሻይ"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="ሻይ"):
        load_phrases(path, ["እኔ"])
    path.write_text(json.dumps({"phrases": [{"signs": ["እኔ"], "amharic": "እኔ"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="two signs"):
        load_phrases(path)


def test_phrase_in_either_order():
    for words in (["እንዴት", "አንተ"], ["አንተ", "እንዴት"]):
        b = PhraseBuilder([HOW])
        assert say(b, words) == [None, HOW]
        assert b.sentence() == ["እንዴት ነህ?"] and b.joined == words


def test_words_too_far_apart_stay_words():
    b = PhraseBuilder([HOW], gap_seconds=6.0)
    assert say(b, ["እንዴት", "አንተ"], step=10.0) == [None, None]
    assert b.sentence() == ["እንዴት", "አንተ"]


def test_longer_phrase_takes_over_a_shorter_one():
    b = PhraseBuilder([FINE, FINE_THANKS])
    assert say(b, ["እኔ", "ደህና", "ምስጋና"]) == [None, FINE, FINE_THANKS]
    assert b.sentence() == ["ደህና ነኝ፣ አመሰግናለሁ"]


def test_a_made_phrase_is_not_split():
    b = PhraseBuilder([HOW, Phrase(("አንተ", "ደህና"), "ደህና ነህ")])
    assert say(b, ["እንዴት", "አንተ", "ደህና"]) == [None, HOW, None]
    assert b.sentence() == ["እንዴት ነህ?", "ደህና"]


def test_other_words_and_repeats_around_phrases():
    b = PhraseBuilder([HOW])
    say(b, ["ሰላም", "እንዴት", "አንተ", "እንዴት", "አንተ"])
    assert b.sentence() == ["ሰላም", "እንዴት ነህ?", "እንዴት ነህ?"]
    b.clear()
    assert b.sentence() == [] and b.joined == []


class WordsEngine:
    """Fake model: says the given words in turn."""
    def __init__(self, words):
        self.words = list(words)

    def best_guess(self, sequence):
        word = self.words.pop(0)
        return word, word, 0.9


def test_live_session_says_the_phrase():
    session = LiveSession(WordsEngine(["ሰላም", "እንዴት", "አንተ"]), phrases=[HOW])
    times, kp = track([(1.0, 2.2), (3.5, 4.7), (6.0, 7.2)], duration=9.0, fps=15)
    said = [w for t, k in zip(times, kp) if (w := session.update(float(t), k))]
    assert said == ["ሰላም", "እንዴት ነህ?"]        # እንዴት alone is never said
    assert session.amharic == "አንተ" and session.phrase == HOW
    assert session.phrase_signs == ["እንዴት", "አንተ"]
    assert session.history == ["ሰላም", "እንዴት ነህ?"]
    session.clear_history()
    assert session.history == [] and session.phrase is None


def spoken(words, segments, duration, phrases):
    """(time, text) of everything the session says for a fake model saying `words`."""
    session = LiveSession(WordsEngine(words), phrases=phrases)
    times, kp = track(segments, duration=duration, fps=15)
    return [(float(t), w) for t, k in zip(times, kp) if (w := session.update(float(t), k))]


def test_a_word_that_could_start_a_phrase_is_said_when_nothing_follows():
    said = spoken(["እንዴት"], [(1.0, 2.2)], 8.0, [HOW])
    assert [w for _, w in said] == ["እንዴት"]
    recognised = 2.2 + 0.35                     # about when the hands are down and the word is found
    assert recognised + SAY_WAIT - 0.5 < said[0][0] < recognised + SAY_WAIT + 1.5


def test_a_phrase_that_could_grow_waits_for_the_longer_one():
    said = spoken(["እኔ", "ደህና", "ምስጋና"], [(1.0, 2.2), (3.5, 4.7), (6.0, 7.2)], 11.0, [FINE, FINE_THANKS])
    assert [w for _, w in said] == ["ደህና ነኝ፣ አመሰግናለሁ"]
    said = spoken(["እኔ", "ደህና"], [(1.0, 2.2), (3.5, 4.7)], 9.0, [FINE, FINE_THANKS])
    assert [w for _, w in said] == ["ደህና ነኝ"]


def test_open_from_names_the_words_that_may_still_join():
    b = PhraseBuilder([HOW, FINE_THANKS])
    b.add("ሰላም", 0.0)
    assert b.open_from() is None                # in no phrase
    b.add("እኔ", 1.0)
    assert b.open_from() == 2
    b.add("ደህና", 2.0)
    assert b.open_from() == 2                   # እኔ ደህና: two thirds of a phrase
    assert b.pieces(0, 2) == ["ሰላም"]
    b.add("ምስጋና", 3.0)
    assert b.open_from() is None                # complete, can't grow
    assert b.pieces(1) == ["ደህና ነኝ፣ አመሰግናለሁ"]
