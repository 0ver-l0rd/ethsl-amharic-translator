"""
scripts/convert_dictionary.py
==============================
Extract landmarks from the AAU EthSL dictionary videos (2,946 scraped words,
one clip per word, one signer).

Every frame is kept (no resampling) with the full face mesh (1662 features),
so any face mode and any window length can be derived later without
re-running MediaPipe:
    none   = frame[:, :258]
    subset = frame[:, :258] + face[FACE_SUBSET_INDICES]
    full   = frame

Input:  <dict_dir>/DataSet/<id>.mov|.mp4  and  <dict_dir>/words-metadata/labels.csv
Output: data/dictionary/clips/<id>.npz   kp (T, 1662) float16, fps
        data/dictionary/index.csv        filename, label, frames, fps, hand_frames

Clips already converted are skipped, so the script can be stopped and re-run.

Usage:
  python scripts/convert_dictionary.py --dict_dir "C:/.../ETHSL-2946-WORDS/ETHSL-2946-WORDS"
  python scripts/convert_dictionary.py --dict_dir ... --limit 10      # quick test

Authors: EthSL Team
"""

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.landmark_extractor import LandmarkExtractor, FEATURE_DIM_NO_FACE, POSE_DIMS

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "dictionary"
PROCESS_WIDTH = 640     # MediaPipe downsizes internally; 1080p input only costs time

_extractor = None


def _init_worker():
    global _extractor
    _extractor = LandmarkExtractor(face_mode="full")


def convert_clip(job):
    """Extract every frame of one video. Returns (filename, error or None)."""
    video, out_path = job
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (PROCESS_WIDTH, round(h * PROCESS_WIDTH / w)))
        frames.append(_extractor.process_frame(frame)[0])
    cap.release()
    if not frames:
        return video.name, "no frames"
    np.savez_compressed(out_path, kp=np.array(frames, dtype=np.float16), fps=np.float32(fps))
    return video.name, None


def read_labels(dict_dir: Path):
    with open(dict_dir / "words-metadata" / "labels.csv", encoding="utf-8-sig") as f:
        return {row["filename"]: row["label"].strip() for row in csv.DictReader(f)}


def write_index(output_dir: Path, labels: dict):
    rows = []
    for name, label in labels.items():
        path = output_dir / "clips" / (Path(name).stem + ".npz")
        if not path.exists():
            continue
        with np.load(path) as d:
            kp = d["kp"]
            hands = (kp[:, POSE_DIMS:FEATURE_DIM_NO_FACE] != 0).any(axis=1)
            rows.append((name, label, len(kp), round(float(d["fps"]), 2), int(hands.sum())))
    with open(output_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "frames", "fps", "hand_frames"])
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract landmarks from the AAU EthSL dictionary videos")
    parser.add_argument("--dict_dir", required=True,
                        help="Folder containing DataSet/ and words-metadata/")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=6, help="Parallel MediaPipe processes")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N clips")
    args = parser.parse_args()

    dict_dir, output_dir = Path(args.dict_dir), Path(args.output_dir)
    (output_dir / "clips").mkdir(parents=True, exist_ok=True)
    labels = read_labels(dict_dir)

    videos = [dict_dir / "DataSet" / name for name in sorted(labels)]
    missing = [v.name for v in videos if not v.exists()]
    videos = [v for v in videos if v.exists()][:args.limit]
    jobs = [(v, output_dir / "clips" / (v.stem + ".npz")) for v in videos]
    todo = [j for j in jobs if not j[1].exists()]

    print("EthSL Dictionary Conversion")
    print("===========================")
    print(f"Labelled clips: {len(labels)}  (video missing: {len(missing)})")
    print(f"To convert:     {len(todo)}  (already done: {len(jobs) - len(todo)})")
    print(f"Workers:        {args.workers}\n")

    failed = []
    if todo:
        with Pool(args.workers, initializer=_init_worker) as pool:
            for name, err in tqdm(pool.imap_unordered(convert_clip, todo), total=len(todo),
                                  desc="Extracting", smoothing=0.05):
                if err:
                    failed.append((name, err))

    rows = write_index(output_dir, labels)
    print(f"\nDone. Index: {len(rows)} clips, {len({r[1] for r in rows})} distinct words "
          f"-> {output_dir / 'index.csv'}")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")


if __name__ == "__main__":
    main()
