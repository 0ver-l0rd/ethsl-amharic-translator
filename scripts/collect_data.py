"""
scripts/collect_data.py
========================
Record sign clips via webcam and save raw video frames to disk.

For each sign in the vocabulary, the script:
  1. Displays a countdown prompt showing which sign to perform
  2. Records N_SEQUENCES clips of SEQUENCE_LENGTH frames each
  3. Saves each clip as a numpy array of BGR frames

This script is designed for the EthSL data collection protocol
described in the project proposal (§5.2).

Usage:
  python scripts/collect_data.py
  python scripts/collect_data.py --signs selam,awo,ay  # specific signs only
  python scripts/collect_data.py --sequences 20        # 20 clips per sign

Authors: Bersabeh Dawit, Menase Teshale
"""

import cv2
import numpy as np
import os
import json
import argparse
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30         # frames per clip
N_SEQUENCES     = 30         # number of clips per sign (30 × 4 signers = 120/sign)
FRAME_DELAY_MS  = 33         # ~30 fps
DATA_DIR        = Path(__file__).parent.parent / "data" / "landmarks"
VOCAB_PATH      = Path(__file__).parent.parent / "data" / "vocabulary.json"

# ── Unicode Amharic fonts require special handling in OpenCV ───────────────
# We display the sign label (ASCII) since cv2 text doesn't support Ethiopic.
# Amharic text is shown separately in the terminal.


def load_vocabulary(vocab_path: Path) -> dict:
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    signs = {}
    for category, items in data["signs"].items():
        for item in items:
            signs[item["label"]] = {
                "amharic":     item["amharic"],
                "description": item["description"],
                "id":          item["id"],
            }
    return signs


def draw_text(frame, text: str, pos, color=(255, 255, 255), scale=1.0, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def show_countdown(frame, seconds: int, label: str, amharic: str, description: str):
    """Display countdown overlay on frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    draw_text(frame, f"Sign: {label} ({description})", (10, 35), color=(0, 255, 0), scale=0.9)
    draw_text(frame, f"Amharic: {amharic}", (10, 70), color=(255, 255, 0), scale=0.8)
    draw_text(frame, f"Starting in: {seconds}s  (Press 'q' to skip)", (10, 105), color=(200, 200, 200), scale=0.7)


def record_sign(
    cap: cv2.VideoCapture,
    sign_label: str,
    sign_info: dict,
    output_dir: Path,
    n_sequences: int,
    sequence_length: int,
) -> int:
    """Record n_sequences clips for a given sign. Returns clips recorded."""
    sign_dir = output_dir / sign_label
    sign_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Sign: {sign_label}")
    print(f"  Amharic: {sign_info['amharic']}")
    print(f"  Description: {sign_info['description']}")
    print(f"  Perform this sign {n_sequences} times.\n")

    recorded = 0

    for seq_idx in range(n_sequences):
        # --- Countdown ---
        for cd in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            show_countdown(frame, cd, sign_label, sign_info["amharic"], sign_info["description"])
            cv2.imshow("EthSL Data Collection", frame)
            if cv2.waitKey(1000) & 0xFF == ord("q"):
                return recorded

        # --- Recording ---
        frames = []
        for f_idx in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Status overlay
            progress = int((f_idx + 1) / sequence_length * frame.shape[1])
            cv2.rectangle(frame, (0, 0), (progress, 8), (0, 255, 0), -1)
            draw_text(frame, f"RECORDING [{seq_idx+1}/{n_sequences}]", (10, 40),
                      color=(0, 0, 255), scale=1.0, thickness=2)
            draw_text(frame, f"Frame {f_idx+1}/{sequence_length}", (10, 75), color=(255, 255, 255))

            cv2.imshow("EthSL Data Collection", frame)
            cv2.waitKey(FRAME_DELAY_MS)

            frames.append(frame.copy())

        # Save frames
        save_path = sign_dir / f"seq_{seq_idx:03d}.npy"
        np.save(str(save_path), np.array(frames))
        recorded += 1
        print(f"    Saved clip {seq_idx+1}/{n_sequences} -> {save_path}")

        # Brief pause between clips
        time.sleep(0.5)

    return recorded


def main():
    parser = argparse.ArgumentParser(description="Record EthSL sign clips via webcam")
    parser.add_argument("--signs",      default=None,
                        help="Comma-separated sign labels to record (default: all)")
    parser.add_argument("--sequences",  type=int, default=N_SEQUENCES,
                        help=f"Number of clips per sign (default: {N_SEQUENCES})")
    parser.add_argument("--output_dir", default=str(DATA_DIR),
                        help="Output directory for landmark arrays")
    parser.add_argument("--camera",     type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    vocab    = load_vocabulary(VOCAB_PATH)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter signs if specified
    if args.signs:
        selected = [s.strip() for s in args.signs.split(",")]
        vocab = {k: v for k, v in vocab.items() if k in selected}
        if not vocab:
            print(f"ERROR: No matching signs found for: {selected}")
            return

    print(f"EthSL Data Collection")
    print(f"=====================")
    print(f"Signs to record: {len(vocab)}")
    print(f"Clips per sign:  {args.sequences}")
    print(f"Output:          {out_dir}")
    print(f"\nPress 'q' to skip a sign, Ctrl+C to quit.\n")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        total_recorded = 0
        for label, info in vocab.items():
            n = record_sign(cap, label, info, out_dir, args.sequences, SEQUENCE_LENGTH)
            total_recorded += n

        print(f"\nData collection complete.")
        print(f"Total clips recorded: {total_recorded}")
        print(f"Run extract_landmarks.py next to process the raw frames.")

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
