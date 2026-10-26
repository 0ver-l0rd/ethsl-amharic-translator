"""
scripts/convert_islr.py
========================
Convert Google's Isolated Sign Language Recognition (ISLR) Kaggle dataset
(competition "asl-signs": 250 ASL signs, ~94k clips, MediaPipe Holistic
landmarks) into this project's sequence format, for pretraining Module 1.

ISLR is ASL, not EthSL. It is used only to pretrain the model's lower layers
(hand shapes, motion over time); Module 1 then replaces the 250-sign output
layer with the EthSL vocabulary and trains on our own recordings.

Each clip becomes a (30, D) sequence laid out exactly like extract_landmarks.py
output: pose (33 x [x, y, z, visibility]), left hand, right hand, then the face
points selected by --face. ISLR has no pose visibility, so it is set to 1 for a
detected point inside the image and 0 otherwise (MediaPipe gives off-screen
points a low visibility). Missing landmarks become zeros, as in our extractor.
Clips have varying lengths and are resampled to 30 frames over the whole sign.

Two ways to get the data:
  1. Already downloaded and unzipped into --islr_dir (needs ~tens of GB):
       python scripts/convert_islr.py --max_per_sign 100
  2. --download: fetch train.csv and only the selected clips through the Kaggle
     API, converting and deleting each one, so almost no disk space is needed:
       python scripts/convert_islr.py --download --max_per_sign 100
     Needs `pip install kaggle`, an API token in ~/.kaggle/kaggle.json, and the
     competition rules accepted on kaggle.com/competitions/asl-signs.

Converted clips are cached in <output_dir>/clips_<mode>/, so an interrupted run
resumes where it stopped. Output (in --output_dir, default data/islr):
  X.npy (N, 30, D) normalized like preprocess.py, y.npy, participant_ids.npy,
  signs.json (class index -> sign name), dataset_stats.json

Mirroring: our recorder and live app flip the camera image before MediaPipe,
which swaps the left/right hand channels. --mirror applies the same flip to
ISLR (x -> 1 - x, left/right swapped). Use --compare_with data/landmarks to see
which hand channel dominates in each dataset and pick the matching setting.

Usage:
  python scripts/convert_islr.py --download --max_per_sign 100
  python scripts/convert_islr.py --face subset --mirror
  python scripts/convert_islr.py --download --max_per_sign 100 --compare_with data/landmarks
"""

import sys
import json
import shutil
import zipfile
import argparse
import tempfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.landmark_extractor import (
    POSE_LANDMARKS, HAND_LANDMARKS, FACE_LANDMARKS, FACE_SUBSET_INDICES,
    POSE_DIMS, HAND_DIMS, FACE_MODES,
)
from src.sequence_builder import SequenceBuilder, SEQUENCE_LENGTH

COMPETITION = "asl-signs"
ISLR_DIR    = ROOT / "data" / "islr_raw"
OUTPUT_DIR  = ROOT / "data" / "islr"

# MediaPipe Pose landmarks that trade places when the image is mirrored:
# eyes (1-3 <-> 4-6), ears, mouth corners, then every left/right body pair.
POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + \
                    [(i, i + 1) for i in range(11, POSE_LANDMARKS, 2)]
POSE_MIRROR_PERM = np.arange(POSE_LANDMARKS)
for a, b in POSE_MIRROR_PAIRS:
    POSE_MIRROR_PERM[a], POSE_MIRROR_PERM[b] = b, a

LH_COLS = slice(POSE_DIMS, POSE_DIMS + HAND_DIMS)
RH_COLS = slice(POSE_DIMS + HAND_DIMS, POSE_DIMS + 2 * HAND_DIMS)


# ---------------------------------------------------------------------------
# Conversion of one clip
# ---------------------------------------------------------------------------

def _landmark_block(df: pd.DataFrame, frame_ids: np.ndarray, kind: str, n: int) -> np.ndarray:
    """(T, n, 3) array of one landmark type; NaN where missing."""
    out = np.full((len(frame_ids), n, 3), np.nan, np.float32)
    part = df[df["type"] == kind]
    if len(part):
        fi = np.searchsorted(frame_ids, part["frame"].to_numpy())
        out[fi, part["landmark_index"].to_numpy()] = part[["x", "y", "z"]].to_numpy(np.float32)
    return out


def islr_to_sequence(df: pd.DataFrame, face_mode: str = "none",
                     mirror: bool = False) -> np.ndarray:
    """
    Convert one ISLR clip (long-format landmark table) to a (SEQUENCE_LENGTH, D)
    array in extract_landmarks.py's layout, before normalization.
    """
    frame_ids = np.sort(df["frame"].unique())
    pose = _landmark_block(df, frame_ids, "pose", POSE_LANDMARKS)
    lh   = _landmark_block(df, frame_ids, "left_hand", HAND_LANDMARKS)
    rh   = _landmark_block(df, frame_ids, "right_hand", HAND_LANDMARKS)
    face = _landmark_block(df, frame_ids, "face", FACE_LANDMARKS) if face_mode != "none" else None

    if mirror:
        for block in (pose, lh, rh):
            block[..., 0] = 1.0 - block[..., 0]
        pose = pose[:, POSE_MIRROR_PERM]
        lh, rh = rh, lh

    x, y = pose[..., 0], pose[..., 1]
    visible = ~np.isnan(x) & (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    pose = np.concatenate([pose, visible[..., None].astype(np.float32)], axis=-1)

    T = len(frame_ids)
    parts = [pose.reshape(T, -1), lh.reshape(T, -1), rh.reshape(T, -1)]
    if face_mode == "subset":
        parts.append(face[:, FACE_SUBSET_INDICES].reshape(T, -1))
    elif face_mode == "full":
        parts.append(face.reshape(T, -1))

    frames = np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0)
    return SequenceBuilder.build_from_clip(list(frames), SEQUENCE_LENGTH)


def hand_presence(X: np.ndarray) -> tuple:
    """Fraction of frames in which the left / right hand channel is detected."""
    lh = np.abs(X[..., LH_COLS]).sum(-1) > 0
    rh = np.abs(X[..., RH_COLS]).sum(-1) > 0
    return float(lh.mean()), float(rh.mean())


# ---------------------------------------------------------------------------
# Kaggle download helpers
# ---------------------------------------------------------------------------

_local = threading.local()


def _kaggle_api():
    """One authenticated Kaggle API client per thread."""
    if not hasattr(_local, "api"):
        try:
            # Importing the kaggle package authenticates immediately
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            sys.exit("ERROR: the kaggle package is not installed. Run: pip install kaggle")
        except OSError as e:
            sys.exit(f"ERROR: Kaggle API token not found ({e}).\n"
                     f"Create one at kaggle.com -> Settings -> API -> Create New Token and "
                     f"save it as {Path.home() / '.kaggle' / 'kaggle.json'}")
        api = KaggleApi()
        api.authenticate()
        _local.api = api
    return _local.api


def fetch_file(name: str, dest_dir: Path) -> Path:
    """
    Download one competition file (e.g. 'train.csv') to dest_dir/<its base name>.

    The Kaggle client names the saved file after the download URL and may
    deliver it zipped, so it is downloaded into an empty folder and whatever
    single file arrives there is unzipped if needed and renamed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest_dir))
    try:
        _kaggle_api().competition_download_file(COMPETITION, name, path=str(tmp),
                                                quiet=True, force=True)
        files = [p for p in tmp.iterdir() if p.is_file()]
        if len(files) != 1:
            raise FileNotFoundError(f"Kaggle download of {name} produced {len(files)} files")
        got = files[0]
        if zipfile.is_zipfile(got):
            with zipfile.ZipFile(got) as zf:
                members = [m for m in zf.namelist() if not m.endswith("/")]
                if len(members) != 1:
                    raise FileNotFoundError(f"{got.name} holds {len(members)} files, expected 1")
                got = Path(zf.extract(members[0], tmp))
        target = dest_dir / Path(name).name
        got.replace(target)
        return target
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_clips(train: pd.DataFrame, max_per_sign, seed: int) -> pd.DataFrame:
    """Up to max_per_sign clips per sign, chosen at random but reproducibly."""
    if max_per_sign:
        train = train.sample(frac=1, random_state=seed).groupby("sign").head(max_per_sign)
    return train.sort_values("sequence_id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Convert the Kaggle ISLR dataset for pretraining")
    parser.add_argument("--islr_dir",   default=str(ISLR_DIR),
                        help="Folder with train.csv and train_landmark_files/ (download cache)")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--face", choices=FACE_MODES, default="none",
                        help="Face points to include; must match the EthSL data's mode")
    parser.add_argument("--mirror", action="store_true",
                        help="Mirror clips like our flipped camera image (pose and hands only)")
    parser.add_argument("--max_per_sign", type=int, default=None,
                        help="Clips per sign (default: all, ~380). 100 gives 25k clips.")
    parser.add_argument("--download", action="store_true",
                        help="Fetch missing files through the Kaggle API, deleting raw clips after conversion")
    parser.add_argument("--keep_raw", action="store_true",
                        help="With --download, keep the raw .parquet files")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_with", default=None,
                        help="Our landmark folder (e.g. data/landmarks) to compare hand channels with")
    args = parser.parse_args()

    if args.mirror and args.face != "none":
        parser.error("--mirror supports --face none only (mirroring face points needs a "
                     "face-mesh symmetry map)")

    islr_dir   = Path(args.islr_dir)
    output_dir = Path(args.output_dir)
    clip_dir   = output_dir / f"clips_{args.face}{'_mirror' if args.mirror else ''}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        _kaggle_api()   # fail early with a clear message if the token is missing

    # ── Clip list and sign names ────────────────────────────────────────
    train_csv = islr_dir / "train.csv"
    if not train_csv.exists():
        if not args.download:
            sys.exit(f"ERROR: {train_csv} not found. Download the dataset into {islr_dir} "
                     f"or add --download.")
        print("Downloading train.csv ...")
        fetch_file("train.csv", islr_dir)

    train = pd.read_csv(train_csv)
    signs = sorted(train["sign"].unique())
    sign_to_id = {s: i for i, s in enumerate(signs)}
    clips = select_clips(train, args.max_per_sign, args.seed)

    print("ISLR conversion")
    print("===============")
    print(f"Signs: {len(signs)} | Clips selected: {len(clips)} of {len(train)} | "
          f"Face: {args.face} | Mirror: {args.mirror}")

    # ── Convert each clip (cached) ──────────────────────────────────────
    def convert(row) -> bool:
        out_path = clip_dir / f"{row.sequence_id}.npy"
        if out_path.exists():
            return True
        raw_path = islr_dir / row.path
        downloaded = False
        try:
            if not raw_path.exists():
                if not args.download:
                    return False
                tmp = Path(tempfile.mkdtemp(dir=clip_dir.parent))
                try:
                    fetched = fetch_file(row.path, tmp)
                    if args.keep_raw:
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(fetched), raw_path)
                        shutil.rmtree(tmp, ignore_errors=True)
                    else:
                        raw_path = fetched
                        downloaded = True
                except Exception:
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise
            df = pd.read_parquet(raw_path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
            seq = islr_to_sequence(df, args.face, args.mirror)
            np.save(out_path, seq)
            return True
        except Exception as e:
            tqdm.write(f"  skipped {row.path}: {e}")
            return False
        finally:
            if downloaded:
                shutil.rmtree(raw_path.parent, ignore_errors=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        ok = list(tqdm(pool.map(convert, clips.itertuples(index=False)),
                       total=len(clips), desc="Converting"))

    clips = clips[np.array(ok, dtype=bool)]
    if clips.empty:
        sys.exit("ERROR: no clips converted.")
    print(f"Converted: {len(clips)} clips ({len(ok) - len(clips)} missing or failed)")

    # ── Assemble the dataset ────────────────────────────────────────────
    raw = np.stack([np.load(clip_dir / f"{sid}.npy") for sid in clips["sequence_id"]])
    X = np.stack([SequenceBuilder.normalize_sequence(s) for s in raw]).astype(np.float32)
    y = clips["sign"].map(sign_to_id).to_numpy(np.int32)
    participants = clips["participant_id"].to_numpy().astype(str)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "participant_ids.npy", participants)
    with open(output_dir / "signs.json", "w") as f:
        json.dump(signs, f, indent=2)

    lh, rh = hand_presence(raw)
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump({
            "source": f"kaggle competition {COMPETITION}",
            "total_samples": int(len(X)),
            "num_classes": len(signs),
            "sequence_length": int(X.shape[1]),
            "feature_dim": int(X.shape[2]),
            "face_mode": args.face,
            "mirrored": args.mirror,
            "participants": int(len(np.unique(participants))),
            "left_hand_presence": lh,
            "right_hand_presence": rh,
        }, f, indent=2)

    print(f"\nDataset: X={X.shape}, y={y.shape}, participants={len(np.unique(participants))}")
    print(f"Hand channel detected in frames: left {lh:.0%}, right {rh:.0%}")

    if args.compare_with:
        own = [np.load(p) for p in Path(args.compare_with).glob("*/*_kp.npy")]
        own = [a for a in own if a.ndim == 2]
        if own:
            olh, orh = hand_presence(np.concatenate(own)[None])
            print(f"Our recordings ({len(own)} clips):  left {olh:.0%}, right {orh:.0%}")
            if (lh > rh) != (olh > orh):
                print("  The dominant hand is in different channels. Re-run with "
                      f"{'no ' if args.mirror else ''}--mirror so the datasets match.")
            else:
                print("  The dominant hand is in the same channel in both datasets.")
        else:
            print(f"No *_kp.npy files found under {args.compare_with} to compare with.")

    print(f"\nSaved to {output_dir}. Next: python module1_refinement/pretrain_islr.py")


if __name__ == "__main__":
    main()
