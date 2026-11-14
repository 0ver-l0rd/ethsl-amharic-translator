"""
scripts/split_recording.py
==========================
Split a phone video of one word signed several times into one clip per
repetition, ready for extract_landmarks.py.

The signer rests with their hands fully down for about a second between
repetitions. MediaPipe tracks the wrists on every frame; a repetition is a
stretch where a hand is tracked or a wrist is clearly above its resting
height. Each repetition is saved as a SIGN_WINDOW_SECONDS clip centred on
it, the same length collect_data.py records, so training clips and the live
app's window cover the same amount of time.

Input:  a video named <signer>_<word>.mp4 / .mov, e.g. s01_01.mp4, s02_E1.mov
Output: data/landmarks/<label>/<signer>_seq_XXX.mp4  (one per repetition)

Nothing is written unless exactly --expected repetitions are found (use
--force to write anyway). Check the printed table, and the --plot image if
the count is off.

Usage:
  python scripts/split_recording.py ~/Downloads/s01_01.mp4 --label selam --dry_run --plot
  python scripts/split_recording.py ~/Downloads/s01_01.mp4 --label selam
  python scripts/split_recording.py s02_E1.mov --label misgana --expected 15 --replace

Then run extract_landmarks.py and preprocess.py as usual.

Authors: EthSL Team
"""

import re
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clip_naming import clip_filename, next_clip_index, parse_signer, validate_signer_id
from src.sequence_builder import SIGN_WINDOW_SECONDS
from src.sign_spotter import activity_signal   # shared with the live app
from src.video_io import open_clip_writer

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "landmarks"
VOCAB_PATH = Path(__file__).parent.parent / "data" / "vocabulary.json"

ANALYSIS_FPS  = 30     # frames per second MediaPipe looks at to find the signs
OUTPUT_FPS    = 30     # frame rate of the saved clips
MAX_HEIGHT    = 720    # saved clips are scaled down to this height (MediaPipe needs no more)
RAISED_MARGIN = 0.5    # shoulder widths above resting wrist height that count as "hands up"
MIN_REST      = 0.3    # s; shorter gaps are a hand dipping mid-sign, not a rest
MIN_SIGN      = 0.4    # s; shorter bursts are a twitch, not a sign

_NAME_RE = re.compile(r"^(?P<signer>[A-Za-z0-9-]+)_(?P<word>[A-Za-z0-9-]+)$")


@dataclass
class Repetition:
    start: float          # s, first frame with hands up
    end: float            # s, last frame with hands up
    clip_start: float     # s, saved window
    clip_end: float
    hand_tracked: float   # fraction of the repetition's frames with a hand detected

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_recording_name(path: Path) -> Tuple[str, str]:
    """(signer, word) from a recording named <signer>_<word>, e.g. s01_01 -> ('s01', '01')."""
    m = _NAME_RE.match(Path(path).stem)
    if not m:
        raise ValueError(
            f"Can't read signer and word from '{Path(path).name}'. "
            f"Name recordings <signer>_<word>, e.g. s01_01.mp4 or s02_E1.mov, or pass --signer."
        )
    return m.group("signer"), m.group("word")


def find_repetitions(
    times: np.ndarray,
    keypoints: np.ndarray,
    min_rest: float = MIN_REST,
    min_sign: float = MIN_SIGN,
    raised_margin: float = RAISED_MARGIN,
) -> List[Tuple[float, float]]:
    """
    (start, end) times of each repetition: stretches where a hand is tracked
    or a wrist is more than raised_margin shoulder widths above its resting
    height, ignoring rests shorter than min_rest and bursts shorter than
    min_sign.
    """
    height, hand, pose_ok = activity_signal(keypoints)
    if not pose_ok.any():
        return []
    rest = np.percentile(height[pose_ok], 10)     # signers rest more than 10% of the time
    active = pose_ok & (hand | (height > rest + raised_margin))

    # Runs of active frames, as index ranges [a, b]
    edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))

    merged: List[List[int]] = []
    for a, b in runs:
        if merged and times[a] - times[merged[-1][1]] < min_rest:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(float(times[a]), float(times[b])) for a, b in merged if times[b] - times[a] >= min_sign]


def clip_window(start: float, end: float, length: float, video_duration: float) -> Tuple[float, float]:
    """A window of the given length centred on [start, end], kept inside the video."""
    length = min(length, video_duration)
    lo = min(max((start + end) / 2 - length / 2, 0.0), video_duration - length)
    return lo, lo + length


def analyse(video: Path, fps: float = ANALYSIS_FPS) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run MediaPipe on the video at about `fps`. Returns (times, keypoints, source_fps, duration)."""
    from src.landmark_extractor import LandmarkExtractor   # slow import; keep --help fast

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    times, kps = [], []
    i = 0
    with LandmarkExtractor(face_mode="none") as extractor:
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    kp, _ = extractor.process_frame(frame)
                    times.append(i / src_fps)
                    kps.append(kp)
            i += 1
    cap.release()
    if not kps:
        raise RuntimeError(f"No frames could be read from {video}")
    return np.array(times), np.array(kps, dtype=np.float32), src_fps, i / src_fps


def write_clips(video: Path, windows: List[Tuple[float, float]], paths: List[Path],
                out_fps: float = OUTPUT_FPS, max_height: int = MAX_HEIGHT) -> None:
    """Copy each [start, end) window of the video to its path, in one pass."""
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / out_fps))
    writers = [None] * len(windows)
    i = 0
    while cap.grab():
        t = i / src_fps
        i += 1
        if (i - 1) % step:
            continue
        open_now = [k for k, (a, b) in enumerate(windows) if a <= t < b]
        if not open_now:
            if t >= windows[-1][1]:
                break
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        h, w = frame.shape[:2]
        if h > max_height:
            frame = cv2.resize(frame, (round(w * max_height / h), max_height), interpolation=cv2.INTER_AREA)
        for k in open_now:
            if writers[k] is None:
                writers[k], _ = open_clip_writer(paths[k], src_fps / step, (frame.shape[1], frame.shape[0]))
            writers[k].write(frame)
    cap.release()
    for wr in writers:
        if wr is not None:
            wr.release()


def existing_clips(sign_dir: Path, signer: str) -> Tuple[List[Path], List[Path]]:
    """(recorded videos, landmark-only files) this signer already has for the sign."""
    if not sign_dir.is_dir():
        return [], []
    files = [p for p in sign_dir.iterdir() if parse_signer(p) == signer]
    videos = [p for p in files if p.suffix in (".mp4", ".npy") and not p.stem.endswith("_kp")]
    video_stems = {p.stem for p in videos}
    imported = [p for p in files if p.stem.endswith("_kp") and p.stem[:-3] not in video_stems]
    return videos, imported


def save_plot(path: Path, times, keypoints, reps: List[Repetition]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, hand, pose_ok = activity_signal(keypoints)
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(times, np.where(pose_ok, height, np.nan), lw=1, label="higher wrist (shoulder widths)")
    ax.fill_between(times, ax.get_ylim()[0], ax.get_ylim()[1], where=hand, alpha=0.12,
                    color="tab:green", label="hand tracked")
    for n, r in enumerate(reps, 1):
        ax.axvspan(r.clip_start, r.clip_end, color="tab:orange", alpha=0.15)
        ax.text((r.start + r.end) / 2, ax.get_ylim()[1], str(n), ha="center", va="top")
    ax.set_xlabel("seconds")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Split a multi-repetition sign video into clips")
    parser.add_argument("video", help="Recording named <signer>_<word>, e.g. s01_01.mp4")
    parser.add_argument("--label", required=True, help="Sign label (folder name), e.g. selam")
    parser.add_argument("--signer", default=None, help="Signer ID (default: from the file name)")
    parser.add_argument("--expected", type=int, default=15, help="Repetitions in the video (default 15)")
    parser.add_argument("--window", type=float, default=SIGN_WINDOW_SECONDS,
                        help=f"Seconds per saved clip (default SIGN_WINDOW_SECONDS = {SIGN_WINDOW_SECONDS})")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--dry_run", action="store_true", help="Only report; write nothing")
    parser.add_argument("--force", action="store_true", help="Write even if the count isn't --expected")
    parser.add_argument("--replace", action="store_true",
                        help="Delete this signer's existing clips for the label first")
    parser.add_argument("--append", action="store_true",
                        help="Add to this signer's existing clips for the label")
    parser.add_argument("--plot", nargs="?", const="", default=None,
                        help="Save a PNG of the signal (default: next to the video)")
    args = parser.parse_args(argv)

    video = Path(args.video).expanduser()
    signer = args.signer or parse_recording_name(video)[0]
    validate_signer_id(signer)
    sign_dir = Path(args.output_dir) / args.label

    # Refuse before the slow part if the clips would collide with others
    videos, imported = existing_clips(sign_dir, signer)
    if imported:
        print(f"ERROR: {sign_dir} already has {len(imported)} landmark files for signer '{signer}' "
              f"that weren't recorded by us (e.g. {imported[0].name}, likely an imported dataset).\n"
              f"       Two different people would share one signer ID. Use a different --signer.")
        return 2
    if videos and not (args.replace or args.append or args.dry_run):
        print(f"ERROR: signer '{signer}' already has {len(videos)} clips in {sign_dir}. "
              f"Pass --replace to redo them or --append to add more.")
        return 2

    if VOCAB_PATH.exists() and f'"{args.label}"' not in VOCAB_PATH.read_text(encoding="utf-8"):
        print(f"Note: '{args.label}' isn't in {VOCAB_PATH.name} yet; add it before training.")

    print(f"Analysing {video.name} ...")
    times, keypoints, src_fps, duration = analyse(video)
    _, hand, pose_ok = activity_signal(keypoints)
    print(f"  {duration:.1f} s at {src_fps:.0f} fps; body found in {pose_ok.mean():.0%} of frames")

    reps = []
    for start, end in find_repetitions(times, keypoints):
        lo, hi = clip_window(start, end, args.window, duration)
        inside = (times >= start) & (times <= end)
        reps.append(Repetition(start, end, lo, hi, float(hand[inside].mean()) if inside.any() else 0.0))

    print(f"\n  {'#':>3}  {'start':>6}  {'end':>6}  {'sign':>5}  {'clip':>13}  hands")
    for n, r in enumerate(reps, 1):
        flags = []
        if r.duration > args.window:
            flags.append("LONGER THAN CLIP")
        if r.hand_tracked < 0.6:
            flags.append("weak hand tracking")
        print(f"  {n:>3}  {r.start:6.2f}  {r.end:6.2f}  {r.duration:5.2f}  "
              f"{r.clip_start:5.2f}-{r.clip_end:5.2f} s  {r.hand_tracked:4.0%}  {' '.join(flags)}")
    for a, b in zip(reps, reps[1:]):
        if a.clip_end > b.start or b.clip_start < a.end:
            print(f"  Warning: clip around {a.start:.1f} s overlaps the next sign; rest longer between them.")

    if reps:
        d = np.array([r.duration for r in reps])
        print(f"\n  Sign length: median {np.median(d):.2f} s, range {d.min():.2f}-{d.max():.2f} s "
              f"(clip window {args.window:.2f} s)")

    if args.plot is not None:
        plot_path = Path(args.plot) if args.plot else video.with_name(video.stem + "_split.png")
        save_plot(plot_path, times, keypoints, reps)
        print(f"  Plot: {plot_path}")

    count_ok = len(reps) == args.expected
    print(f"\n  Found {len(reps)} repetitions (expected {args.expected})"
          + ("" if count_ok else " -- check the table/plot; re-record or pass --expected / --force"))
    if args.dry_run or not reps or not (count_ok or args.force):
        print("  Nothing written.")
        return 0 if (count_ok or args.dry_run) else 1

    sign_dir.mkdir(parents=True, exist_ok=True)
    if args.replace:
        for p in videos:
            p.unlink()
            p.with_name(p.stem + "_kp.npy").unlink(missing_ok=True)
    first = next_clip_index(sign_dir, signer)
    paths = [sign_dir / clip_filename(signer, first + n) for n in range(len(reps))]
    write_clips(video, [(r.clip_start, r.clip_end) for r in reps], paths)
    print(f"  Wrote {len(paths)} clips: {paths[0].name} .. {paths[-1].name} in {sign_dir}")
    print("  Next: python scripts/extract_landmarks.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
