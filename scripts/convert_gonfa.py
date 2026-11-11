"""
scripts/convert_gonfa.py
=========================
Convert the isolated-word EthSL dataset of Gonfa et al., Sci. Rep. 15:36181
(2025) into the layout used by preprocess.py, and write data/vocabulary.json
for the words that have data.

The dataset is 20 words x 7 signers x 40 clips x 30 frames of MediaPipe
Holistic landmarks. It was shared for non-commercial use only, on condition
that it is not redistributed and the paper is cited: do not commit it.

Input:  <dataset_dir>/**/EthSL_Dataset/<word>/<clip 0-279>/<frame 0-29>.npy
        Google Drive splits the download into several folders and one clip's
        frames can land in different ones, so all of them are merged.
Output: data/landmarks/<label>/<signer>_seq_XXX_kp.npy
        data/vocabulary.json

Each word's clips are in signer order, 40 per signer, so clip // 40 is the
signer. Frames are stored as pose, face, left hand, right hand; our layout is
pose, left hand, right hand, face. Clips with missing frames are skipped.

Usage:
  python scripts/convert_gonfa.py --dataset_dir "C:/.../Senior Project"
  python scripts/convert_gonfa.py --dataset_dir ... --face subset

Authors: EthSL Team
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.landmark_layout import FACE_LANDMARKS, FEATURE_DIM_NO_FACE, POSE_DIMS
from src.landmark_extractor import FACE_MODES, FACE_SUBSET_INDICES
from src.sequence_builder import SEQUENCE_LENGTH

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "landmarks"
VOCAB_PATH = Path(__file__).parent.parent / "data" / "vocabulary.json"

CLIPS_PER_SIGNER = 40
SOURCE_DIM = FEATURE_DIM_NO_FACE + FACE_LANDMARKS * 3

# Amharic word: (category, label, English)
WORDS = {
    "ሀዘን":    ("feelings",    "hazen",      "Sadness"),
    "መናደድ":   ("feelings",    "menaded",    "To be angry"),
    "መገረም":   ("feelings",    "megerem",    "To be surprised"),
    "መጨነቅ":   ("feelings",    "mechenek",   "To worry"),
    "ህመም":    ("feelings",    "himem",      "Pain / illness"),
    "ለምን":    ("questions",   "lemin",      "Why"),
    "ልምምድ":   ("actions",     "limimid",    "Practice"),
    "መጀመር":   ("actions",     "mejemer",    "To start"),
    "መጨረስ":   ("actions",     "mecheres",   "To finish"),
    "ማሸነፍ":   ("actions",     "mashenef",   "To win"),
    "ተነስ":    ("actions",     "tenes",      "Stand up"),
    "ምስጋና":   ("social",      "misgana",    "Thanks"),
    "ይቅርታ":   ("social",      "yikirta",    "Sorry"),
    "እህት":    ("family",      "ehit",       "Sister"),
    "ወንድም":   ("family",      "wendim",     "Brother"),
    "መራራ":    ("descriptive", "merara",     "Bitter"),
    "መጥፎ":    ("descriptive", "metfo",      "Bad"),
    "ቀላል":    ("descriptive", "kelal",      "Easy"),
    "አስቸጋሪ":  ("descriptive", "aschegari",  "Difficult"),
    "ጎበዝ":    ("descriptive", "gobez",      "Clever / skilled"),
}


def to_our_layout(frames: np.ndarray, face_mode: str) -> np.ndarray:
    """Reorder (T, 1662) source frames into our layout for `face_mode`."""
    face_end = POSE_DIMS + FACE_LANDMARKS * 3
    body = np.concatenate([frames[:, :POSE_DIMS], frames[:, face_end:]], axis=1)
    if face_mode == "none":
        return body
    face = frames[:, POSE_DIMS:face_end]
    if face_mode == "subset":
        face = face.reshape(len(frames), FACE_LANDMARKS, 3)[:, FACE_SUBSET_INDICES].reshape(len(frames), -1)
    return np.concatenate([body, face], axis=1)


def find_clips(dataset_dir: Path) -> dict:
    """{word: {clip number: {frame number: path}}}, merged over every download part."""
    clips = defaultdict(lambda: defaultdict(dict))
    for root in dataset_dir.glob("**/EthSL_Dataset"):
        for path in root.glob("*/*/*.npy"):
            if re.fullmatch(r"\d+", path.stem) and path.parent.name.isdigit():
                clips[path.parent.parent.name][int(path.parent.name)][int(path.stem)] = path
    return clips


def convert_clip(job) -> bool:
    """Write one clip; returns False if the source frames are unusable."""
    frame_paths, out_path, face_mode = job
    frames = np.stack([np.load(p, allow_pickle=False) for p in frame_paths]).astype(np.float32)
    if frames.shape != (SEQUENCE_LENGTH, SOURCE_DIM):
        return False
    np.save(out_path, to_our_layout(frames, face_mode))
    return True


def write_vocabulary(path: Path, words: list) -> None:
    """Write vocabulary.json for `words`, grouped by category with contiguous IDs."""
    signs = defaultdict(list)
    for word in sorted(words, key=lambda w: list(WORDS).index(w)):
        category, label, english = WORDS[word]
        signs[category].append({"label": label, "amharic": word, "description": english})
    next_id = 0
    for group in signs.values():
        for sign in group:
            sign["id"] = next_id
            next_id += 1
    meta = {
        "version": "2.0",
        "total_signs": next_id,
        "language": "Ethiopian Sign Language (EthSL)",
        "output_language": "Amharic",
        "reference": "Gonfa et al., Scientific Reports 15:36181 (2025)",
        "categories": list(signs),
    }
    path.write_text(json.dumps({"metadata": meta, "signs": signs}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert the Gonfa et al. EthSL dataset")
    parser.add_argument("--dataset_dir", required=True,
                        help="Folder containing the EthSL_Dataset download part(s)")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--vocab", default=str(VOCAB_PATH))
    parser.add_argument("--face", choices=FACE_MODES, default="none")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("Scanning dataset (about 100,000 small files)...")
    clips = find_clips(Path(args.dataset_dir))

    unknown = sorted(set(clips) - set(WORDS))
    if unknown:
        sys.exit(f"ERROR: words missing from the WORDS table: {unknown}")

    jobs, per_word, incomplete = [], {}, 0
    for word, by_clip in clips.items():
        label = WORDS[word][1]
        (output_dir / label).mkdir(parents=True, exist_ok=True)
        complete = 0
        for clip, frames in sorted(by_clip.items()):
            # A few clips have stray extra frames past the last one; they are ignored
            if not all(i in frames for i in range(SEQUENCE_LENGTH)):
                incomplete += 1
                continue
            complete += 1
            signer = f"s{clip // CLIPS_PER_SIGNER + 1:02d}"
            out = output_dir / label / f"{signer}_seq_{clip % CLIPS_PER_SIGNER:03d}_kp.npy"
            jobs.append(([frames[i] for i in range(SEQUENCE_LENGTH)], out, args.face))
        per_word[word] = complete

    print(f"Words with data: {len(per_word)} of {len(WORDS)}")
    print(f"Complete clips:  {len(jobs)}  (incomplete, skipped: {incomplete})")
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(tqdm(pool.map(convert_clip, jobs), total=len(jobs), desc="Converting"))
    if not all(results):
        print(f"  WARNING: {results.count(False)} clips had unexpected shapes and were skipped")

    write_vocabulary(Path(args.vocab), list(per_word))
    print(f"\nVocabulary: {len(per_word)} words -> {args.vocab}")
    for word in WORDS:
        print(f"  {per_word.get(word, 0):4d}  {word}  ({WORDS[word][1]})")
    print("\nRun preprocess.py next to build the training dataset.")


if __name__ == "__main__":
    main()
