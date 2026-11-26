"""
scripts/dictionary_to_landmarks.py
==================================
Add the AAU dictionary's clips of our words to the training data, as one more
signer (d01), so preprocess.py and retrain.py pick them up like any other clip.

Each dictionary clip (data/dictionary/clips/<id>.npz, made by
convert_dictionary.py) is cut like a team clip: the sign is found by hand
movement, a 2.5 s window is centred on it and resampled to 30 frames. The
clips are saved un-normalized, as preprocess.py normalizes every clip.

Leave-one-signer-out tests with these clips added to training raised accuracy
on an unseen team signer from 72.4% to 73.9% (12 runs, 2026-11-25). The
dictionary's signer is never used as a test signer.

Input:  data/dictionary/index.csv and data/dictionary/clips/
Output: data/landmarks/<label>/d01_seq_NNN_kp.npy (old d01 clips are replaced)

The dictionary is not ours to share: data/dictionary and data/landmarks are
gitignored, and so are the models trained on them.

Usage:
  python scripts/dictionary_to_landmarks.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from split_recording import clip_window, find_repetitions
from src.sequence_builder import SIGN_WINDOW_SECONDS, resample_window

SIGNER = "d01"
FEATURES = 258   # pose + hands, no face, like the team's clips


def dictionary_clip(kp: np.ndarray, fps: float) -> np.ndarray:
    """One dictionary video's keypoints (T, >=258) -> a (30, 258) sign clip."""
    kp = kp[:, :FEATURES].astype(np.float32)
    times = np.arange(len(kp)) / fps
    duration = len(kp) / fps
    spans = find_repetitions(times, kp)
    start, end = max(spans, key=lambda s: s[1] - s[0]) if spans else (0.0, duration)
    lo, hi = clip_window(start, end, SIGN_WINDOW_SECONDS, duration)
    return resample_window(times, kp, lo, hi)


def matching_rows(rows, amharic: str):
    """Dictionary entries whose label (alternatives split by '/') is this word."""
    return [r for r in rows if amharic in [p.strip() for p in r["label"].split("/")]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--dict_dir", default=str(ROOT / "data" / "dictionary"))
    parser.add_argument("--landmarks_dir", default=str(ROOT / "data" / "landmarks"))
    parser.add_argument("--vocab", default=str(ROOT / "data" / "vocabulary.json"))
    args = parser.parse_args()

    dict_dir, out_dir = Path(args.dict_dir), Path(args.landmarks_dir)
    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    signs = [s for group in vocab["signs"].values() for s in group]
    with open(dict_dir / "index.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total, missing = 0, []
    for sign in signs:
        found = matching_rows(rows, sign["amharic"])
        if not found:
            missing.append(sign["amharic"])
            continue
        sign_dir = out_dir / sign["label"]
        sign_dir.mkdir(parents=True, exist_ok=True)
        for old in sign_dir.glob(f"{SIGNER}_seq_*_kp.npy"):
            old.unlink()
        for i, row in enumerate(found):
            data = np.load(dict_dir / "clips" / f"{Path(row['filename']).stem}.npz")
            clip = dictionary_clip(data["kp"], float(data["fps"]))
            np.save(sign_dir / f"{SIGNER}_seq_{i:03d}_kp.npy", clip)
            total += 1

    print(f"Saved {total} dictionary clips of {len(signs) - len(missing)} words as signer {SIGNER}.")
    if missing:
        print(f"Not in the dictionary ({len(missing)}): {' '.join(missing)}")


if __name__ == "__main__":
    main()
