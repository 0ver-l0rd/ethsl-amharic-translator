"""
app/realtime_app.py
====================
Real-time EthSL sign language recognition application.

Pipeline (per frame):
  1. Capture frame from webcam
  2. Extract MediaPipe Holistic landmarks
  3. Add frame to rolling 30-frame buffer
  4. When buffer is full: run model inference
  5. Display top-1 prediction (Amharic text) with confidence
  6. Synthesize Amharic speech for confirmed predictions

Usage:
  python app/realtime_app.py --model models/finetuned/best_model.h5
  python app/realtime_app.py --model models/refined/best_model.h5 --threshold 0.7

Keyboard controls:
  q     : Quit
  r     : Reset buffer (clear current sign window)
  s     : Toggle speech output on/off

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

from src.landmark_extractor import LandmarkExtractor
from src.sequence_builder import SequenceBuilder
from src.vocabulary import Vocabulary
from src.amharic_tts import AmharicTTS
from app.ui import SignLanguageUI
from app.inference_engine import InferenceEngine


SEQUENCE_LENGTH     = 30
CONFIDENCE_THRESHOLD = 0.65   # minimum confidence to accept a prediction
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
        self._seq   = 0          # increments with every new frame
        self._ok    = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._ok:
            ok, frame = self._cap.read()
            with self._cond:
                if not ok:
                    self._ok = False
                else:
                    self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self, last_seq: int, timeout: float = 2.0):
        """Wait for a frame newer than last_seq. Returns (ok, frame, seq)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self._ok, timeout)
            if self._seq == last_seq:
                return False, None, last_seq
            return True, self._frame, self._seq

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
    parser.add_argument("--include_face", action="store_true", help="Include face landmarks")
    parser.add_argument("--infer_every", type=int, default=2,
                        help="Run the classifier every N frames (1 = every frame). "
                             "Higher values raise FPS; a sign lasts ~30 frames.")
    parser.add_argument("--lite", action="store_true",
                        help="Use MediaPipe's lite pose model (faster, slightly less accurate). "
                             "Best if training data was also extracted in lite mode.")
    args = parser.parse_args()
    if args.infer_every < 1:
        parser.error("--infer_every must be >= 1")

    print("EthSL Real-Time Translator")
    print("=" * 40)
    print(f"Model:     {args.model}")
    print(f"Camera:    {args.camera}")
    print(f"Threshold: {args.threshold}")
    print(f"Speech:    {'OFF' if args.no_speech else 'ON'}")
    print(f"Classify:  every {args.infer_every} frame(s)")
    print(f"MediaPipe: {'lite' if args.lite else 'full'}")
    print()

    # Initialize components
    vocab    = Vocabulary(str(VOCAB_PATH))
    engine   = InferenceEngine(args.model, vocab, args.threshold)
    seq_builder = SequenceBuilder(sequence_length=SEQUENCE_LENGTH,
                                   feature_dim=1662 if args.include_face else 258)
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

    # State
    current_prediction  = None
    current_amharic     = ""
    current_confidence  = 0.0
    sentence_history    = []
    fps_counter         = deque(maxlen=30)
    last_stable_pred    = None
    stable_count        = 0
    STABLE_THRESHOLD    = 3   # consecutive same predictions before accepting
    frame_idx           = 0

    with LandmarkExtractor(include_face=args.include_face,
                           model_complexity=0 if args.lite else 1) as extractor:
        while True:
            t0 = time.perf_counter()
            ret, frame, frame_seq = reader.read(frame_seq)
            if not ret:
                print("ERROR: Failed to read frame")
                break

            frame = cv2.flip(frame, 1)  # Mirror for natural interaction

            # Extract landmarks
            keypoints, results = extractor.process_frame(frame)
            seq_builder.add_frame(keypoints)

            # Draw landmarks
            frame = extractor.draw_landmarks(frame, results)

            # Run inference when buffer is ready (every infer_every frames)
            frame_idx += 1
            if seq_builder.is_ready() and frame_idx % args.infer_every == 0:
                sequence = seq_builder.get_sequence()
                pred_label, pred_amharic, confidence = engine.predict(sequence)

                if pred_label is not None:
                    # Stability check: require N consistent predictions
                    if pred_label == last_stable_pred:
                        stable_count += 1
                    else:
                        stable_count = 1
                        last_stable_pred = pred_label

                    if stable_count >= STABLE_THRESHOLD and confidence >= args.threshold:
                        if current_prediction != pred_label:
                            current_prediction = pred_label
                            current_amharic    = pred_amharic
                            current_confidence = confidence
                            sentence_history.append(pred_amharic)

                            # Speak the Amharic text
                            if speech_enabled and tts:
                                tts.speak(pred_amharic)

                            # Reset buffer after confirmed prediction
                            seq_builder.reset()
                            stable_count = 0

            # FPS tracking
            fps_counter.append(time.perf_counter() - t0)
            fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

            # Draw UI overlay
            frame = ui.draw(
                frame=frame,
                current_label=current_prediction or "",
                current_amharic=current_amharic,
                confidence=current_confidence,
                sentence_history=sentence_history[-5:],  # last 5 signs
                buffer_fill=len(seq_builder._buffer) / SEQUENCE_LENGTH,
                fps=fps,
                speech_enabled=speech_enabled,
            )

            cv2.imshow("EthSL Amharic Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                seq_builder.reset()
                current_prediction = None
                current_amharic    = ""
                current_confidence = 0.0
                if tts:
                    tts.reset()
                print("Buffer reset.")
            elif key == ord("s"):
                speech_enabled = not speech_enabled
                print(f"Speech: {'ON' if speech_enabled else 'OFF'}")
            elif key == ord("c"):
                sentence_history.clear()
                print("Sentence history cleared.")

    reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended.")
    if sentence_history:
        print("Signs recognized this session:")
        print(" | ".join(sentence_history))


if __name__ == "__main__":
    main()
