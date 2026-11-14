"""
app/live_session.py
===================
What the live translator does with each camera frame, shared by the desktop
app (app/realtime_app.py) and the web page (app/web_app.py):

  - the sign spotter (src/sign_spotter.py) finds where each sign starts and
    ends; nothing is guessed while the hands are down
  - after each sign the model classifies it once; below the confidence
    threshold it says "Not sure" instead of guessing
  - it keeps the last recognised word, the history and short notices

Authors: Estifanos Behailu
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sign_spotter import SignSpotter, Status

CONFIDENCE_THRESHOLD = 0.65   # minimum confidence to accept a prediction
MESSAGE_SECONDS      = 2.5    # how long a notice like "Not sure" stays up
HISTORY_LENGTH       = 20


class LiveSession:
    """
    One person's live translation. `engine` needs best_guess(sequence) ->
    (label, amharic, confidence), like app.inference_engine.InferenceEngine.
    """

    def __init__(self, engine, threshold: float = CONFIDENCE_THRESHOLD, sequence_length: int = 30):
        self.engine = engine
        self.threshold = threshold
        self.spotter = SignSpotter(sequence_length=sequence_length)
        self.label = ""
        self.amharic = ""
        self.confidence = 0.0
        self.history: List[str] = []
        self.message = ""
        self._message_until = 0.0

    @property
    def status(self) -> Status:
        return self.spotter.status

    def update(self, t: float, keypoints: np.ndarray) -> Optional[str]:
        """
        Feed one frame's keypoints, captured at time t (seconds). Returns the
        Amharic word when a sign has just been recognised, otherwise None.
        """
        event = self.spotter.add_frame(t, keypoints)
        recognised = None
        if event and event.kind == "sign":
            label, amharic, confidence = self.engine.best_guess(event.sequence)
            if confidence >= self.threshold:
                self.label, self.amharic, self.confidence = label, amharic, confidence
                self.history = (self.history + [amharic])[-HISTORY_LENGTH:]
                self.message = ""
                recognised = amharic
            else:
                self._notice(f"Not sure - try again (maybe {amharic}?)", t)
        elif event and event.kind == "too_long":
            self._notice("Lower your hands between signs", t)
        if self.message and t > self._message_until:
            self.message = ""
        return recognised

    def _notice(self, text: str, t: float) -> None:
        self.message, self._message_until = text, t + MESSAGE_SECONDS

    def reset(self) -> None:
        """Drop the sign in progress and the last word (the history stays)."""
        self.spotter.reset()
        self.label, self.amharic, self.confidence, self.message = "", "", 0.0, ""

    def clear_history(self) -> None:
        self.history = []
