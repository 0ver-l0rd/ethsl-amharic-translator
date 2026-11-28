"""
scripts/prepare_speech.py
=========================
Make the Amharic speech for all 75 words and every phrase now, while there is
internet, so both apps can speak them later without a connection (they are
kept in data/speech/). The apps also do this in the background when they
start online; this script just does it at once and says how many are ready.

Usage:
  python scripts/prepare_speech.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.amharic_tts import SPEECH_DIR, AmharicTTS, speech_texts


def main():
    texts = speech_texts()
    ready = AmharicTTS().prepare(texts)
    print(f"Speech ready for {ready} of {len(texts)} words and phrases, in {SPEECH_DIR}")
    if ready < len(texts):
        print("Not all of them: is there an internet connection? Run this again when online.")
        sys.exit(1)


if __name__ == "__main__":
    main()
