"""
app/realtime_app.py
====================
Real-time EthSL sign language recognition application.

Pipeline (per frame):
  1. Capture frame from webcam
  2. Extract MediaPipe Holistic landmarks
  3. The sign spotter (src/sign_spotter.py) watches the hands: a sign starts
     when they come up and ends when they have been down for a moment
  4. After each sign: resample a SIGN_WINDOW_SECONDS window centred on it to
     30 frames (cut the same way as the training clips) and classify it once
  5. Display the sign (Amharic text) with its confidence, or "Not sure" if
     the confidence is below the threshold
  6. Synthesize Amharic speech for recognised signs

Nothing is guessed while the hands are down.

Usage:
  python app/realtime_app.py --model models/finetuned/best_model.h5
  python app/realtime_app.py --model models/refined/best_model.h5 --threshold 0.7

Keyboard controls:
  q     : Quit
  r     : Reset buffer (clear current sign window)
  s     : Toggle speech output on/off
  c     : Clear the sentence history

Authors: Emanuel Solomon, Menase Teshale
"""

import cv2
import numpy as np
import argparse
import sys
import time
import threading
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel("ERROR")
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

from src.landmark_extractor import LandmarkExtractor, face_mode_for_dim
from src.vocabulary import Vocabulary
from src.amharic_tts import AmharicTTS
from app.ui import SignLanguageUI
from app.inference_engine import InferenceEngine
from app.live_session import CONFIDENCE_THRESHOLD, LiveSession


SEQUENCE_LENGTH     = 30
VOCAB_PATH          = Path(__file__).parent.parent / "data" / "vocabulary.json"


class LatestFrameReader:
    """
    Reads the camera on a background thread and keeps only the newest frame.

    cap.read() blocks until the webcam delivers the next frame (~40 ms here).
    Reading in the background overlaps that wait with MediaPipe processing,
    and the main loop always gets the most recent frame instead of a stale one.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap   = cap
        self._cond  = threading.Condition()
        self._frame = None
        self._time  = 0.0        # perf_counter() when _frame was captured
        self._seq   = 0          # increments with every new frame
        self._ok    = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._ok:
            ok, frame = self._cap.read()
            t_captured = time.perf_counter()
            with self._cond:
                if not ok:
                    self._ok = False
                else:
                    self._frame, self._time, self._seq = frame, t_captured, self._seq + 1
                self._cond.notify_all()

    def read(self, last_seq: int, timeout: float = 2.0):
        """
        Wait for a frame newer than last_seq.
        Returns (ok, frame, capture_time, seq).
        """
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self._ok, timeout)
            if self._seq == last_seq:
                return False, None, 0.0, last_seq
            return True, self._frame, self._time, self._seq

    def stop(self):
        self._ok = False
        self._thread.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser(description="Real-time EthSL translator")
    parser.add_argument("--model",     required=True,   help="Path to .h5 model")
    parser.add_argument("--camera",    type=int, default=0, help="Camera device index")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Confidence threshold for accepting a prediction")
    parser.add_argument("--no_speech", action="store_true", help="Disable TTS output")
    parser.add_argument("--lite", action="store_true",
                        help="Use MediaPipe's lite pose model (faster, slightly less accurate). "
                             "Best if training data was also extracted in lite mode.")
    args = parser.parse_args()

    print("EthSL Real-Time Translator")
    print("=" * 40)
    print(f"Model:     {args.model}")
    print(f"Camera:    {args.camera}")
    print(f"Threshold: {args.threshold}")
    print(f"Speech:    {'OFF' if args.no_speech else 'ON'}")
    print(f"MediaPipe: {'lite' if args.lite else 'full'}")

    # Initialize components
    vocab    = Vocabulary(str(VOCAB_PATH))
    engine   = InferenceEngine(args.model, vocab, args.threshold)

    # The model's input size says which face landmarks it was trained with
    feature_dim = engine.model.input_shape[-1]
    try:
        face_mode = face_mode_for_dim(feature_dim)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Face mode: {face_mode} ({feature_dim} features, from the model)")
    print()

    session  = LiveSession(engine, args.threshold, SEQUENCE_LENGTH)
    tts      = AmharicTTS() if not args.no_speech else None
    ui       = SignLanguageUI(vocab)
    speech_enabled = not args.no_speech

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    reader = LatestFrameReader(cap)
    frame_seq = 0

    print("Press 'q' to quit | 'r' to reset | 's' to toggle speech")
    print()

    fps_counter = deque(maxlen=30)

    with LandmarkExtractor(face_mode=face_mode,
                           model_complexity=0 if args.lite else 1) as extractor:
        while True:
            t0 = time.perf_counter()
            ret, frame, t_captured, frame_seq = reader.read(frame_seq)
            if not ret:
                print("ERROR: Failed to read frame")
                break

            frame = cv2.flip(frame, 1)  # Mirror for natural interaction

            # Extract landmarks
            keypoints, results = extractor.process_frame(frame)
            recognised = session.update(t_captured, keypoints)

            # Draw landmarks
            frame = extractor.draw_landmarks(frame, results)

            if recognised:
                print(f"{recognised}  ({session.label}, {session.confidence:.0%})")
                if speech_enabled and tts:
                    tts.reset()          # each sign is new, so say repeats too
                    tts.speak(recognised)

            # FPS tracking
            fps_counter.append(time.perf_counter() - t0)
            fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

            # Draw UI overlay
            frame = ui.draw(
                frame=frame,
                status=session.status,
                message=session.message,
                current_label=session.label,
                current_amharic=session.amharic,
                confidence=session.confidence,
                sentence_history=session.history[-5:],  # last 5 signs
                fps=fps,
                speech_enabled=speech_enabled,
            )

            cv2.imshow("EthSL Amharic Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                session.reset()
                print("Reset.")
            elif key == ord("s"):
                speech_enabled = not speech_enabled
                print(f"Speech: {'ON' if speech_enabled else 'OFF'}")
            elif key == ord("c"):
                session.clear_history()
                print("Sentence history cleared.")

    reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended.")
    if session.history:
        print("Signs recognized this session:")
        print(" | ".join(session.history))


if __name__ == "__main__":
    main()
