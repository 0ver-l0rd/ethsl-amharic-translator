"""
src/amharic_tts.py
==================
Amharic Text-to-Speech (TTS) engine wrapper.

Uses Google Text-to-Speech (gTTS) with language code 'am' (Amharic), which
needs the internet. Every text is synthesized once and kept as an MP3 in
data/speech/, so speech still works offline for anything said or prepared
before: run `python scripts/prepare_speech.py` (or just start an app while
online; both apps prepare the 75 words and the phrases in the background).
Offline, a text that was never prepared is put together from prepared words
and phrases where it can be ("ሰላም እንዴት ነህ?" from "ሰላም" + "እንዴት ነህ?").
"""

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

SPEECH_DIR = Path(__file__).resolve().parent.parent / "data" / "speech"


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
        cache_dir: Optional[str] = None,
    ):
        self.lang = lang
        self.cache_dir = str(cache_dir or SPEECH_DIR)
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
        if self._last_spoken == text:
            return  # suppress immediate repeats

        path = self.to_file(text) if self._playsound_available else None
        if path:
            try:
                self._playsound(path)
                self._last_spoken = text
                return
            except Exception as e:
                print(f"[AmharicTTS] playback error: {e}")

        # Fallback: print to console
        if self.use_offline_fallback:
            print(f"[TTS Fallback] {text}")
            self._last_spoken = text

    def cached_path(self, text: str, folder: Optional[str] = None) -> str:
        """Where the speech for `text` is (or would be) kept."""
        name = hashlib.sha1(f"{self.lang}:{text}".encode()).hexdigest()[:16] + ".mp3"
        return os.path.join(folder or self.cache_dir, name)

    def is_cached(self, text: str, folder: Optional[str] = None) -> bool:
        return os.path.exists(self.cached_path(text, folder))

    def to_file(self, text: str, folder: Optional[str] = None) -> Optional[str]:
        """
        The MP3 with the speech for `text` (in `folder`, by default data/speech/),
        made with gTTS the first time and reused after that. Offline, a text
        that was never made is joined from made words and phrases if possible.
        Returns None if there is no speech for it.
        """
        folder = folder or self.cache_dir
        path = self.cached_path(text, folder)
        if os.path.exists(path):
            return path
        os.makedirs(folder, exist_ok=True)
        if self._gtts_available:
            try:
                self._gtts(text=text, lang=self.lang, slow=self.slow).save(path + ".part")
                os.replace(path + ".part", path)
                return path
            except Exception as e:
                print(f"[AmharicTTS] gTTS error (offline?): {e}")
        parts = self._from_cache(text, folder)
        if parts:
            with open(path + ".part", "wb") as out:
                for p in parts:
                    with open(p, "rb") as f:
                        out.write(f.read())
            os.replace(path + ".part", path)
            return path
        return None

    def _from_cache(self, text: str, folder: str) -> Optional[List[str]]:
        """Paths of made pieces that together say `text`, longest pieces first; None if a word is missing."""
        words, parts, i = text.split(), [], 0
        while i < len(words):
            for j in range(len(words), i, -1):
                piece = self.cached_path(" ".join(words[i:j]), folder)
                if os.path.exists(piece):
                    parts.append(piece)
                    i = j
                    break
            else:
                return None
        return parts if len(parts) > 1 else None

    def prepare(self, texts: Iterable[str]) -> int:
        """
        Make the speech for every text that has none yet (needs the internet).
        Stops at the first failure, as that usually means no connection.
        Returns how many of the texts now have speech.
        """
        texts = list(dict.fromkeys(t for t in texts if t))
        ready = 0
        for text in texts:
            if self.is_cached(text):
                ready += 1
                continue
            if not self._gtts_available:
                break
            try:
                path = self.cached_path(text)
                os.makedirs(self.cache_dir, exist_ok=True)
                self._gtts(text=text, lang=self.lang, slow=self.slow).save(path + ".part")
                os.replace(path + ".part", path)
                ready += 1
            except Exception:
                break
        return ready

    def prepare_in_background(self, texts: Iterable[str]) -> threading.Thread:
        """prepare() on a background thread, so an app can start at once."""
        th = threading.Thread(target=self.prepare, args=(list(texts),), daemon=True)
        th.start()
        return th

    def reset(self) -> None:
        """Reset last-spoken cache (forces next speak() to always play)."""
        self._last_spoken = None


def speech_texts(vocab=None, phrases=None) -> List[str]:
    """Everything the apps can say: the Amharic of every word, then of every phrase."""
    from src.phrases import phrases_for
    from src.vocabulary import Vocabulary
    root = Path(__file__).resolve().parent.parent
    vocab = vocab or Vocabulary(str(root / "data" / "vocabulary.json"))
    if phrases is None:
        phrases = phrases_for(vocab)
    words = [vocab.id_to_amharic(i) for i in range(vocab.num_classes)]
    return list(dict.fromkeys(words + [p.amharic for p in phrases]))

