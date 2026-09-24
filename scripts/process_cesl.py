"""
scripts/process_cesl.py
========================
Helper script to process the downloaded CESL (Continuous Ethiopian Sign Language)
dataset videos into MediaPipe landmark sequences for our pipeline.

Instructions:
1. Request access to CESL on Zenodo: https://zenodo.org/records/10800699
2. Download the videos and extract them into data/raw_cesl/
3. Run this script to extract landmarks and save them to data/landmarks/

Usage:
  python scripts/process_cesl.py --input_dir data/raw_cesl --output_dir data/landmarks

Authors: EthSL Team
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.landmark_extractor import LandmarkExtractor
    from src.sequence_builder import SequenceBuilder
except ImportError:
    print("ERROR: Make sure you run this from the project root directory.")
    sys.exit(1)

SEQUENCE_LENGTH = 30


def extract_landmarks_from_video(video_path: Path, extractor: LandmarkExtractor) -> np.ndarray:
    """Reads a video file, extracts landmarks frame-by-frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"\n  [ERROR] Cannot open video: {video_path}")
        return None

    keypoints_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # MediaPipe extraction
        kp, _ = extractor.process_frame(frame)
        keypoints_list.append(kp)

    cap.release()
    
    if not keypoints_list:
        return None
        
    return np.array(keypoints_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Process CESL dataset videos into landmarks")
    parser.add_argument("--input_dir",  default="data/raw_cesl", help="Directory containing CESL .mp4 videos")
    parser.add_argument("--output_dir", default="data/landmarks", help="Directory to save extracted .npy arrays")
    parser.add_argument("--include_face", action="store_true", help="Include face landmarks")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"\nERROR: Input directory '{input_dir}' not found.")
        print("Please download the CESL dataset from Zenodo and extract it there.")
        return

    # Find all videos (assuming they might be organized in subfolders by word/sentence)
    video_files = list(input_dir.rglob("*.mp4")) + list(input_dir.rglob("*.avi"))
    
    if not video_files:
        print(f"\nERROR: No video files found in '{input_dir}'.")
        return

    print(f"CESL Dataset Processing")
    print(f"=======================")
    print(f"Found {len(video_files)} videos to process.")
    print(f"Output directory: {output_dir}")
    print()

    with LandmarkExtractor(include_face=args.include_face) as extractor:
        processed_count = 0
        
        for video_path in tqdm(video_files, desc="Extracting landmarks"):
            # Assume folder name is the sign label (e.g., raw_cesl/water/video1.mp4)
            # If CESL puts all videos in one folder with names like "water_01.mp4", 
            # we extract the label from the filename.
            
            # This is a safe fallback assuming the parent folder is the label
            label = video_path.parent.name
            if label == input_dir.name:
                # If they are all dumped in the root directory, guess label from filename (e.g., 'selam_01.mp4' -> 'selam')
                label = video_path.stem.split('_')[0] 

            sign_out_dir = output_dir / label
            sign_out_dir.mkdir(parents=True, exist_ok=True)
            
            out_file = sign_out_dir / f"{video_path.stem}_kp.npy"
            if out_file.exists():
                continue

            # Extract full variable-length sequence
            keypoints = extract_landmarks_from_video(video_path, extractor)
            if keypoints is None:
                continue

            # Interpolate/Resample to fixed length of 30 frames for our models
            keypoints = SequenceBuilder.build_from_clip(list(keypoints), target_length=SEQUENCE_LENGTH)

            # Save the sequence
            np.save(str(out_file), keypoints)
            processed_count += 1

    print(f"\nDone! Successfully processed {processed_count} videos.")
    print("You can now run 'python scripts/preprocess.py' to generate the training splits.")


if __name__ == "__main__":
    main()
