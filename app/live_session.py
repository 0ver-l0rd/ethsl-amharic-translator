"""
app/live_session.py
===================
What the live translator does with each camera frame, shared by the desktop
app (app/realtime_app.py) and the web page (app/web_app.py):

  - the sign spotter (src/sign_spotter.py) finds where each sign starts and
    ends; nothing is guessed while the hands are down
  - after each sign the model classifies it once; below the confidence
    threshold it says "Not sure" instead of guessing (the engine's threshold
    by default: 0.65 for one model, 0.43 for an ensemble)
  - words signed one after another can make a phrase (src/phrases.py):
    እንዴት + አንተ -> "እንዴት ነህ?". Only the phrase is said: a word that could
    start one is shown at once but said only if no sign follows within
    SAY_WAIT seconds
  - it keeps the last recognised word, the history and short notices
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.phrases import Phrase, PhraseBuilder
from src.sign_spotter import SignSpotter, Status

CONFIDENCE_THRESHOLD = 0.65   # minimum confidence to accept a prediction, if the engine has none
MESSAGE_SECONDS      = 2.5    # how long a notice like "Not sure" stays up
HISTORY_LENGTH       = 20
SAY_WAIT             = 1.5    # s with no new sign before a word that could start a phrase is said


class LiveSession:
    """
    One person's live translation. `engine` needs best_guess(sequence) ->
    (label, amharic, confidence), like app.inference_engine.InferenceEngine.
    `phrases` (from src.phrases.load_phrases) turns sign sequences into
    sentences; without it every word stands alone.
    """

    def __init__(self, engine, threshold: Optional[float] = None, sequence_length: int = 30,
                 phrases: Optional[List[Phrase]] = None):
        self.engine = engine
        self.threshold = threshold if threshold is not None else getattr(engine, "threshold", CONFIDENCE_THRESHOLD)
        self.spotter = SignSpotter(sequence_length=sequence_length)
        self.builder = PhraseBuilder(phrases or [])
        self.label = ""
        self.amharic = ""
        self.confidence = 0.0
        self.phrase: Optional[Phrase] = None   # the phrase the last word completed
        self.message = ""
        self._message_until = 0.0
        self._said = 0                # number of the last word said (PhraseBuilder.last_seq)
        self._last_busy = 0.0         # last time a sign was recognised or under way

    @property
    def status(self) -> Status:
        return self.spotter.status

    def update(self, t: float, keypoints: np.ndarray) -> Optional[str]:
        """
        Feed one frame's keypoints, captured at time t (seconds). Returns what
        to say now (words and phrases, space-separated), or None.
        """
        event = self.spotter.add_frame(t, keypoints)
        if self.status in (Status.SIGNING, Status.RECOGNISING):
            self._last_busy = t
        if event and event.kind == "sign":
            label, amharic, confidence = self.engine.best_guess(event.sequence)
            if confidence >= self.threshold:
                self.label, self.amharic, self.confidence = label, amharic, confidence
                self.phrase = self.builder.add(amharic, t)
                self.message = ""
                self._last_busy = t
            else:
                self._notice(f"Not sure - try again (maybe {amharic}?)", t)
        elif event and event.kind == "too_long":
            self._notice("Pause or lower your hands between signs", t)
        if self.message and t > self._message_until:
            self.message = ""
        # Signing lately: hold back the words that may still become a phrase
        waiting = t - self._last_busy < SAY_WAIT
        return self._say(self.builder.open_from() if waiting else None)

    def _say(self, before: Optional[int]) -> Optional[str]:
        """Say the unsaid words numbered below `before` (None: all of them)."""
        pieces = self.builder.pieces(self._said, before)
        if not pieces:
            return None
        self._said = self.builder.last_seq if before is None else before - 1
        return " ".join(pieces)

    @property
    def history(self) -> List[str]:
        """The words and phrases said so far, oldest first."""
        return self.builder.sentence()[-HISTORY_LENGTH:]

    @property
    def phrase_signs(self) -> List[str]:
        """The signs of the current phrase, in the order they were signed."""
        return self.builder.joined if self.phrase else []

    def _notice(self, text: str, t: float) -> None:
        self.message, self._message_until = text, t + MESSAGE_SECONDS

    def reset(self) -> None:
        """Drop the sign in progress and the last word (the history stays)."""
        self.spotter.reset()
        self.label, self.amharic, self.confidence, self.message = "", "", 0.0, ""
        self.phrase = None
        self._said = self.builder.last_seq   # an unsaid word is dropped too

    def clear_history(self) -> None:
        self.builder.clear()
        self.phrase = None
        self._said = self.builder.last_seq
