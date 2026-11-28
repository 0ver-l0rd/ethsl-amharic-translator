"""
tests/test_live_session.py
==========================
Tests for app/live_session.py, the per-sign logic shared by the desktop and
web apps, with a fake model.

Run with: pytest tests/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.live_session import LiveSession, unsure_text
from src.sign_spotter import MAX_SIGN
from tests.test_split_recording import track

FPS = 15


class FakeEngine:
    def __init__(self, confidence):
        self.confidence, self.calls = confidence, 0

    def best_guess(self, sequence):
        self.calls += 1
        return "selam", "ሰላም", self.confidence


def run(confidence, segments, duration=10.0):
    engine = FakeEngine(confidence)
    session = LiveSession(engine)
    times, kp = track(segments, duration=duration, fps=FPS)
    words, messages = [], set()
    for t, k in zip(times, kp):
        w = session.update(float(t), k)
        if w:
            words.append(w)
        if session.message:
            messages.add(session.message)
    return session, engine, words, messages


def test_confident_sign_is_recognised_once():
    s, engine, words, _ = run(0.9, [(2.0, 3.4)])
    assert words == ["ሰላም"] and engine.calls == 1
    assert s.amharic == "ሰላም" and s.history == ["ሰላም"]


def test_unsure_sign_says_so_instead_of_guessing():
    s, _, words, messages = run(0.4, [(2.0, 3.4)])
    assert words == [] and s.amharic == "" and s.history == []
    assert messages == {"Not sure - maybe ሰላም?"}
    assert s.message == ""                      # the notice goes away again


class TopThreeEngine(FakeEngine):
    def predict_top_k(self, sequence, k):
        self.calls += 1
        return [("endet", "እንዴት", self.confidence), ("yet", "የት", 0.3), ("min", "ምን", 0.1)][:k]


def test_unsure_sign_names_the_three_best_guesses():
    session = LiveSession(TopThreeEngine(0.4))
    times, kp = track([(2.0, 3.4)], duration=10.0, fps=FPS)
    messages = set()
    for t, k in zip(times, kp):
        session.update(float(t), k)
        messages.add(session.message)
    assert "Not sure - maybe እንዴት, የት or ምን?" in messages
    assert session.history == [] and session.guesses == ["እንዴት", "የት", "ምን"]


def test_a_confident_top_guess_is_used_as_before():
    session = LiveSession(TopThreeEngine(0.9))
    times, kp = track([(2.0, 3.4)], duration=10.0, fps=FPS)
    for t, k in zip(times, kp):
        session.update(float(t), k)
    assert session.history == ["እንዴት"] and session.guesses == []


def test_unsure_text():
    assert unsure_text(["ሰላም"]) == "Not sure - maybe ሰላም?"
    assert unsure_text(["ሰላም", "አንተ"]) == "Not sure - maybe ሰላም or አንተ?"
    assert unsure_text(["ሰላም", "ሰላም", "አንተ"]) == "Not sure - maybe ሰላም or አንተ?"


def test_same_word_twice_counts_twice():
    _, _, words, _ = run(0.9, [(2.0, 3.4), (5.5, 6.8)])
    assert words == ["ሰላም", "ሰላም"]


def test_hands_up_too_long_gives_a_hint():
    _, engine, words, messages = run(0.9, [(1.0, 1.0 + MAX_SIGN + 2)], duration=12.0)
    assert words == [] and engine.calls == 0
    assert "Pause or lower your hands between signs" in messages


def test_reset_and_clear():
    s, _, _, _ = run(0.9, [(2.0, 3.4)])
    s.reset()
    assert s.amharic == "" and s.history == ["ሰላም"]
    s.clear_history()
    assert s.history == []
