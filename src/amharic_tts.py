"""
src/amharic_tts.py
==================
Amharic Text-to-Speech (TTS) engine wrapper.

Uses Google Text-to-Speech (gTTS) with language code 'am' (Amharic).
Audio is saved to a temp file and played back via playsound.

Authors: Estifanos Behailu
"""

import os
import tempfile
import threading
import time
from typing import Optional


class AmharicTTS:
    """
    Synthesizes Amharic text to speech.

    Parameters
    ----------
    lang : str
        BCP-47 language code. 'am' = Amharic (Google TTS).
    slow : bool
        If True, generate slower speech for clarity.
    use_offline_fallback : bool
        If True and gTTS fails, print text to console as fallback.
    """

    def __init__(
        self,
        lang: str = "am",
        slow: bool = False,
        use_offline_fallback: bool = True,
    ):
        self.lang = lang
        self.slow = slow
        self.use_offline_fallback = use_offline_fallback
        self._last_spoken: Optional[str] = None
        self._min_interval_sec = 1.5  # avoid rapid repeated speech

        # Check if gTTS is available
        try:
            from gtts import gTTS
            self._gtts = gTTS
            self._gtts_available = True
        except ImportError:
            print("[AmharicTTS] gTTS not installed. Run: pip install gTTS")
            self._gtts_available = False

        # Check if playsound is available
        try:
            from playsound import playsound
            self._playsound = playsound
            self._playsound_available = True
        except ImportError:
            print("[AmharicTTS] playsound not installed. Run: pip install playsound")
            self._playsound_available = False

    def speak(self, text: str, async_play: bool = True) -> None:
        """
        Synthesize and play Amharic speech for the given text.

        Parameters
        ----------
        text : str
            Amharic text to synthesize (UTF-8 Ethiopic script).
        async_play : bool
            If True, play audio in a background thread.
        """
        if async_play:
            t = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
            t.start()
        else:
            self._speak_sync(text)

    def _speak_sync(self, text: str) -> None:
        """Internal synchronous TTS playback."""
        now = time.time()
        if self._last_spoken == text:
            return  # suppress immediate repeats

        if self._gtts_available and self._playsound_available:
            try:
                tts = self._gtts(text=text, lang=self.lang, slow=self.slow)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                    tmp_path = fp.name
                tts.save(tmp_path)
                self._playsound(tmp_path)
                os.unlink(tmp_path)
                self._last_spoken = text
                return
            except Exception as e:
                print(f"[AmharicTTS] gTTS error: {e}")

        # Fallback: print to console
        if self.use_offline_fallback:
            print(f"[TTS Fallback] {text}")
            self._last_spoken = text

    def to_file(self, text: str, folder: str) -> Optional[str]:
        """
        Save the speech for `text` as an MP3 in `folder` and return its path,
        for playback in a browser. Each text is synthesized once and reused.
        Returns None if gTTS is unavailable or fails (it needs the internet).
        """
        if not self._gtts_available:
            return None
        import hashlib
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, hashlib.sha1(f"{self.lang}:{text}".encode()).hexdigest()[:16] + ".mp3")
        if not os.path.exists(path):
            try:
                self._gtts(text=text, lang=self.lang, slow=self.slow).save(path + ".part")
                os.replace(path + ".part", path)
            except Exception as e:
                print(f"[AmharicTTS] gTTS error: {e}")
                return None
        return path

    def reset(self) -> None:
        """Reset last-spoken cache (forces next speak() to always play)."""
        self._last_spoken = None
