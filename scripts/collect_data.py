"""
scripts/collect_data.py
========================
Record sign clips via webcam and save them to disk as compressed .mp4 video.

For each sign in the vocabulary, the script:
  1. Displays a countdown prompt showing which sign to perform
  2. Records N_SEQUENCES clips, each exactly SIGN_WINDOW_SECONDS long
     (the same window the live app uses, whatever the camera's frame rate)
  3. Saves each clip as an .mp4 video (~1% of the size of raw frames), named
     <signer>_seq_XXX.mp4 so every clip records who performed it. Keeping
     the video means landmarks can be re-extracted later (e.g. with another
     face mode) without re-recording.

Every recording session MUST pass --signer. The ID is what makes
signer-independent evaluation possible, and it keeps each person's clips
from overwriting another's. Use anonymous codes (s01, s02, ...) and keep
the code -> person mapping outside the repo.

This script is designed for the EthSL data collection protocol
described in the project proposal (§5.2).

Usage:
  python scripts/collect_data.py --signer s01
  python scripts/collect_data.py --signer s02 --signs selam,awo,ay  # specific signs only
  python scripts/collect_data.py --signer s01 --sequences 20        # 20 clips per sign

Authors: Bersabeh Dawit, Menase Teshale
"""

import cv2
import numpy as np
import os
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clip_naming import validate_signer_id, clip_filename, next_clip_index
from src.sequence_builder import SIGN_WINDOW_SECONDS, MIN_WINDOW_FRAMES
from src.video_io import write_clip

# ── Configuration ──────────────────────────────────────────────────────────
N_SEQUENCES     = 30         # number of clips per sign (30 × 4 signers = 120/sign)
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
    window_seconds: float,
    signer: str,
) -> int:
    """Record n_sequences clips for a given sign. Returns clips recorded."""
    sign_dir = output_dir / sign_label
    sign_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_clip_index(sign_dir, signer)

    print(f"\n  Sign: {sign_label}  (signer {signer}, existing clips: {start_idx})")
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

        # --- Recording: every camera frame for exactly window_seconds ---
        for _ in range(5):          # drop frames buffered during the countdown
            cap.grab()
        frames = []
        t_start = time.perf_counter()
        while True:
            ret, frame = cap.read()
            elapsed = time.perf_counter() - t_start
            if not ret or elapsed > window_seconds:
                break
            frame = cv2.flip(frame, 1)
            frames.append(frame.copy())   # save the clean frame, before any overlay

            # Status overlay (display only)
            progress = int(min(elapsed / window_seconds, 1.0) * frame.shape[1])
            cv2.rectangle(frame, (0, 0), (progress, 8), (0, 255, 0), -1)
            draw_text(frame, f"RECORDING [{seq_idx+1}/{n_sequences}]  signer {signer}", (10, 40),
                      color=(0, 0, 255), scale=1.0, thickness=2)
            draw_text(frame, f"{elapsed:.1f}s / {window_seconds:.1f}s", (10, 75), color=(255, 255, 255))

            cv2.imshow("EthSL Data Collection", frame)
            cv2.waitKey(1)

        if len(frames) < MIN_WINDOW_FRAMES:
            print(f"    WARNING: only {len(frames)} frames captured — clip discarded, "
                  f"check the camera.")
            continue

        # Save frames
        save_path = sign_dir / clip_filename(signer, start_idx + recorded)
        write_clip(save_path, frames, fps=len(frames) / window_seconds)
        recorded += 1
        print(f"    Saved clip {seq_idx+1}/{n_sequences} -> {save_path}  "
              f"({len(frames)} frames, {len(frames) / window_seconds:.0f} FPS, "
              f"{save_path.stat().st_size / 1e6:.1f} MB)")

        # Brief pause between clips
        time.sleep(0.5)

    return recorded


def main():
    parser = argparse.ArgumentParser(description="Record EthSL sign clips via webcam")
    parser.add_argument("--signer",     required=True,
                        help="Anonymous ID of the person signing, e.g. s01 (letters/digits/'-' only)")
    parser.add_argument("--signs",      default=None,
                        help="Comma-separated sign labels to record (default: all)")
    parser.add_argument("--sequences",  type=int, default=N_SEQUENCES,
                        help=f"Number of clips per sign (default: {N_SEQUENCES})")
    parser.add_argument("--output_dir", default=str(DATA_DIR),
                        help="Output directory for landmark arrays")
    parser.add_argument("--camera",     type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    try:
        signer = validate_signer_id(args.signer)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

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
    print(f"Signer:          {signer}")
    print(f"Signs to record: {len(vocab)}")
    print(f"Clips per sign:  {args.sequences}")
    print(f"Clip length:     {SIGN_WINDOW_SECONDS:.1f} s (SIGN_WINDOW_SECONDS)")
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
            n = record_sign(cap, label, info, out_dir, args.sequences, SIGN_WINDOW_SECONDS, signer)
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
