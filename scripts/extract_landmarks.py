"""
scripts/extract_landmarks.py
==============================
MediaPipe Holistic landmark extraction pipeline.

Processes the clips saved by collect_data.py and extracts MediaPipe pose +
hand (+ optional face) landmarks, resampled to SEQUENCE_LENGTH frames.

Input:  data/landmarks/<sign_label>/<signer>_seq_XXX.mp4   (or legacy .npy frames)
Output: data/landmarks/<sign_label>/<signer>_seq_XXX_kp.npy  (landmark arrays)

Existing _kp.npy files are reused only if they match the requested face
mode, so re-running with a different --face re-extracts everything.

Usage:
  python scripts/extract_landmarks.py                 # pose + hands (258 features)
  python scripts/extract_landmarks.py --face subset   # + lips/eyes/eyebrows (534)
  python scripts/extract_landmarks.py --face full     # + full face mesh (1662)

Authors: Bersabeh Dawit
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.landmark_extractor import LandmarkExtractor, FACE_MODES
from src.sequence_builder import SequenceBuilder, SEQUENCE_LENGTH
from src.video_io import read_clip

INPUT_DIR       = Path(__file__).parent.parent / "data" / "landmarks"
OUTPUT_DIR      = Path(__file__).parent.parent / "data" / "landmarks"


def process_clip(frames: np.ndarray, extractor: LandmarkExtractor) -> np.ndarray:
    """
    Extract MediaPipe landmarks from a clip of BGR frames.

    Parameters
    ----------
    frames : sequence of T BGR frames, each (H, W, 3)
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
    parser.add_argument("--face", choices=FACE_MODES, default="none",
                        help="Face landmarks to include: none (258 features), "
                             "subset = lips/eyes/eyebrows (534), full (1662)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all raw clips: <signer>_seq_XXX.mp4, plus legacy .npy frame arrays
    clip_files = sorted(
        f for pattern in ("*seq_*.mp4", "*seq_*.npy")
        for f in input_dir.rglob(pattern)
        if not f.name.endswith("_kp.npy")
    )

    if not clip_files:
        print(f"ERROR: No clip files found in {input_dir}")
        print("       Run collect_data.py first to record sign clips.")
        return

    with LandmarkExtractor(face_mode=args.face) as extractor:
        print(f"EthSL Landmark Extraction")
        print(f"=========================")
        print(f"Found {len(clip_files)} clips")
        print(f"Face mode:   {args.face}")
        print(f"Feature dim: {extractor.feature_dim} per frame")
        print()

        processed = 0
        skipped   = 0

        for clip_path in tqdm(clip_files, desc="Extracting landmarks"):
            # Output path: same location, _kp suffix
            kp_path = clip_path.with_name(clip_path.stem + "_kp.npy")

            # Reuse only if it was extracted with the same face mode
            if kp_path.exists() and np.load(str(kp_path), mmap_mode="r").shape[1] == extractor.feature_dim:
                skipped += 1
                continue

            # Load raw frames
            try:
                frames = read_clip(clip_path)
            except Exception as e:
                print(f"\n  WARNING: Could not load {clip_path}: {e}")
                continue
            if not frames:
                print(f"\n  WARNING: No frames in {clip_path} — skipped")
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
