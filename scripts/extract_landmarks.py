"""
scripts/extract_landmarks.py
==============================
MediaPipe Holistic landmark extraction pipeline.

Processes raw video frames saved by collect_data.py and extracts
MediaPipe pose + hand landmarks, saving them as normalized .npy arrays.

Input:  data/landmarks/<sign_label>/seq_XXX.npy  (raw BGR frames)
Output: data/landmarks/<sign_label>/seq_XXX_kp.npy  (landmark arrays)

Usage:
  python scripts/extract_landmarks.py
  python scripts/extract_landmarks.py --input_dir data/landmarks --output_dir data/landmarks

Authors: Bersabeh Dawit
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.landmark_extractor import LandmarkExtractor
from src.sequence_builder import SequenceBuilder

SEQUENCE_LENGTH = 30
INPUT_DIR       = Path(__file__).parent.parent / "data" / "landmarks"
OUTPUT_DIR      = Path(__file__).parent.parent / "data" / "landmarks"


def process_clip(frames: np.ndarray, extractor: LandmarkExtractor) -> np.ndarray:
    """
    Extract MediaPipe landmarks from a clip of BGR frames.

    Parameters
    ----------
    frames : np.ndarray of shape (T, H, W, 3)
    extractor : LandmarkExtractor

    Returns
    -------
    keypoints : np.ndarray of shape (T, feature_dim)
    """
    keypoints = []
    for frame in frames:
        kp, _ = extractor.process_frame(frame)
        keypoints.append(kp)
    return np.array(keypoints, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from sign clips")
    parser.add_argument("--input_dir",    default=str(INPUT_DIR))
    parser.add_argument("--output_dir",   default=str(OUTPUT_DIR))
    parser.add_argument("--include_face", action="store_true",
                        help="Include face landmarks (increases feature dim to 1662)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all raw clip files
    clip_files = sorted(input_dir.rglob("seq_*.npy"))
    kp_files   = [f for f in clip_files if not f.name.endswith("_kp.npy")]

    if not kp_files:
        print(f"ERROR: No clip files found in {input_dir}")
        print("       Run collect_data.py first to record sign clips.")
        return

    print(f"EthSL Landmark Extraction")
    print(f"=========================")
    print(f"Found {len(kp_files)} clips")
    print(f"Include face: {args.include_face}")
    print(f"Feature dim: {1662 if args.include_face else 258} per frame")
    print()

    with LandmarkExtractor(include_face=args.include_face) as extractor:
        processed = 0
        skipped   = 0

        for clip_path in tqdm(kp_files, desc="Extracting landmarks"):
            # Output path: same location, _kp suffix
            kp_path = clip_path.with_name(clip_path.stem + "_kp.npy")

            if kp_path.exists():
                skipped += 1
                continue

            # Load raw frames
            try:
                frames = np.load(str(clip_path), allow_pickle=False)
            except Exception as e:
                print(f"\n  WARNING: Could not load {clip_path}: {e}")
                continue

            # Extract landmarks
            keypoints = process_clip(frames, extractor)  # (T, D)

            # Resample to fixed length
            keypoints = SequenceBuilder.build_from_clip(
                list(keypoints), target_length=SEQUENCE_LENGTH
            )

            # Save
            np.save(str(kp_path), keypoints)
            processed += 1

    print(f"\nDone.")
    print(f"  Processed: {processed} clips")
    print(f"  Skipped (already exist): {skipped} clips")
    print(f"\nRun preprocess.py next to build the training dataset.")


if __name__ == "__main__":
    main()
