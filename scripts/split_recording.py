"""
scripts/split_recording.py
==========================
Split a phone video of one word signed several times into one clip per
repetition, ready for extract_landmarks.py.

The signer rests with their hands fully down for about a second between
repetitions. MediaPipe tracks the wrists on every frame; a repetition is a
stretch where a hand is tracked or a wrist is clearly above its resting
height. Each repetition is saved as a clip centred on it, SIGN_WINDOW_SECONDS
long or longer for a slow sign (sequence_builder.sign_window), the same way
the live app cuts its window around each sign, so both cover the same amount
of time.

Some signers keep their hands up in front of the chest and only pause
between repetitions. For them, --by_motion finds the repetitions as bursts
of hand movement between still moments instead; check its plot or table.

When neither works, mark the start of each repetition by hand while watching
the video (scripts/mark_reps.py) and split with --marks; the reps are lined
up by their movement, so the marks only need to fall at the same moment of
each rep.

Input:  a video named <signer>_<word>.mp4 / .mov, e.g. s01_01.mp4, s02_E1.mov
Output: data/landmarks/<label>/<signer>_seq_XXX.mp4  (one per repetition)

Nothing is written unless exactly --expected repetitions are found (or up
to --extra more, for signers who do a spare one or two; use
--force to write anyway). Check the printed table, and the --plot image if
the count is off. With --marks the marked reps are cut, whatever their number.

Usage:
  python scripts/split_recording.py ~/Downloads/s01_01.mp4 --label selam --dry_run --plot
  python scripts/split_recording.py ~/Downloads/s01_01.mp4 --label selam
  python scripts/split_recording.py s02_E1.mov --label misgana --expected 15 --replace

Then run extract_landmarks.py and preprocess.py as usual.
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
from src.landmark_layout import LEFT_SHOULDER, RIGHT_SHOULDER, POSE_DIMS, body_relative
from src.sequence_builder import SIGN_WINDOW_SECONDS, sign_window
from src.sign_spotter import activity_signal, hand_speed, HAND_DIMS, LEFT_WRIST, RIGHT_WRIST   # shared with the live app
from src.video_io import open_clip_writer

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "landmarks"
VOCAB_PATH = Path(__file__).parent.parent / "data" / "vocabulary.json"

ANALYSIS_FPS  = 30     # frames per second MediaPipe looks at to find the signs
OUTPUT_FPS    = 30     # frame rate of the saved clips
MAX_HEIGHT    = 720    # saved clips are scaled down to this height (MediaPipe needs no more)
RAISED_MARGIN = 0.5    # shoulder widths above resting wrist height that count as "hands up"
MIN_REST      = 0.3    # s; shorter gaps are a hand dipping mid-sign, not a rest
MIN_SIGN      = 0.4    # s; shorter bursts are a twitch, not a sign
HAND_REST_MARGIN = 0.3   # shoulder widths; a tracked hand lower than rest + this is resting
MERGED_FACTOR = 2.0       # a rep this many times the median sign length is two reps run together
HANDS_ALWAYS_SEEN = 0.85  # hands tracked in more frames than this: they rest in view, so
                          # "hand tracked" can't mark a sign and only wrist height is used
# A short last "repetition" just before the end of the video, with the hands
# poorly tracked, is the signer reaching for the phone to stop recording. (A
# long one is a real sign that ran into the reach, so it is kept.)
STOP_REACH_END   = 1.0  # s from the end of the video
STOP_REACH_HANDS = 0.5  # below this share of frames with a hand detected
STOP_REACH_LEN   = 2.5  # s; longer stretches hold a real sign
EDGE_TOLERANCE   = 0.05 # s; a rep this close to the start or end of the video was cut off
# --by_motion: signers who keep their hands up and only pause between reps
MOTION_EDGE      = 0.1  # s; a burst this close to the hands coming up / going down ...
MOTION_SHORT     = 0.6  # ... and shorter than this x the typical burst is raising / lowering them
MOTION_MIN_PAUSE = 0.35 # s; shorter still moments are inside one sign
MOTION_LEVELS    = tuple(np.round(np.arange(0.10, 0.56, 0.05), 2))  # tried moving-speed thresholds

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
    raised = height > rest + raised_margin
    # A tracked hand counts as signing too (for signs made low), but not when
    # it is at resting height: some signers rest with their hands in view.
    off_rest = height > rest + HAND_REST_MARGIN
    if hand[pose_ok].mean() > HANDS_ALWAYS_SEEN:
        active = pose_ok & raised                 # hands always in view (e.g. in the lap): height only
    else:
        active = pose_ok & ((hand & off_rest) | raised)

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


def motion_bursts(times: np.ndarray, keypoints: np.ndarray, level: float,
                  speed: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
    """
    (start, end) of each burst of hand movement between still pauses, while the
    hands are up. A frame is moving when its speed is more than `level` of the
    way from the still speed (15th percentile) to the signing speed (90th).
    A short burst right where the hands come up or go down is only the raising
    or lowering of the hands, and is left out.
    """
    speed = hand_speed(times, keypoints) if speed is None else speed
    _, hand, _ = activity_signal(keypoints)
    seen = np.flatnonzero(hand)
    if len(seen) == 0:
        return []
    up = (times >= times[seen[0]]) & (times <= times[seen[-1]])
    still, signing = np.percentile(speed[up], 15), np.percentile(speed[up], 90)
    moving = up & (speed > still + level * (signing - still))

    edges = np.diff(np.concatenate([[0], moving.astype(int), [0]]))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))
    merged: List[List[int]] = []
    for a, b in runs:
        if merged and times[a] - times[merged[-1][1]] < MOTION_MIN_PAUSE:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    bursts = [(float(times[a]), float(times[b])) for a, b in merged if times[b] - times[a] >= MIN_SIGN]
    if len(bursts) >= 3:
        typical = float(np.median([e - s for s, e in bursts]))
        first, last = times[seen[0]], times[seen[-1]]
        if bursts[-1][1] >= last - MOTION_EDGE and bursts[-1][1] - bursts[-1][0] < MOTION_SHORT * typical:
            bursts.pop()                                   # lowering the hands
        if bursts[0][0] <= first + MOTION_EDGE and bursts[0][1] - bursts[0][0] < MOTION_SHORT * typical:
            bursts.pop(0)                                  # raising the hands
    return bursts


def find_repetitions_by_motion(times: np.ndarray, keypoints: np.ndarray, expected: int,
                               extra: int = 0) -> Tuple[List[Tuple[float, float]], float]:
    """
    Repetitions of a signer who keeps their hands up and only pauses between
    them, as (spans, level). Every level in MOTION_LEVELS is tried; of those
    giving expected..expected+extra usable bursts (not two reps run together,
    which main() leaves out), the count found at the most levels wins (the
    steadiest answer) and its middle level is used. If no level gives a count
    in range, MOTION_LEVELS' middle level is used and the count is off.
    """
    def usable(bursts):
        if len(bursts) < 5:
            return len(bursts)
        typical = float(np.median([e - s for s, e in bursts]))
        return sum(e - s <= MERGED_FACTOR * typical for s, e in bursts)

    speed = hand_speed(times, keypoints)
    found = {level: motion_bursts(times, keypoints, level, speed) for level in MOTION_LEVELS}
    count = {level: usable(b) for level, b in found.items()}
    fits = [lv for lv in MOTION_LEVELS if expected <= count[lv] <= expected + extra]
    if not fits:
        level = MOTION_LEVELS[len(MOTION_LEVELS) // 2]
        return found[level], level
    counts = [count[lv] for lv in fits]
    best = max(set(counts), key=lambda c: (counts.count(c), -abs(c - expected)))
    levels = [lv for lv in fits if count[lv] == best]
    level = levels[len(levels) // 2]
    return found[level], level


def marks_path(video: Path) -> Path:
    """Where mark_reps.py saves the marks for a video: next to it, <stem>.marks.txt."""
    return Path(video).with_name(Path(video).stem + ".marks.txt")


def read_marks(path: Path) -> Tuple[List[Tuple[float, bool]], Optional[float]]:
    """([(start, skip), ...] sorted by time, end of the last rep or None) from a marks file.

    One line per repetition: its start in seconds, followed by "skip" if it is
    a bad rep to leave out. An "end <seconds>" line marks where the last rep
    finishes. Lines starting with # are comments.
    """
    marks, end = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("#")[0].split()
        if not parts:
            continue
        if parts[0].lower() == "end":
            end = float(parts[1])
        else:
            marks.append((float(parts[0]), len(parts) > 1 and parts[1].lower() == "skip"))
    return sorted(marks), end


def write_marks(path: Path, marks: List[Tuple[float, bool]], end: Optional[float] = None,
                video_name: str = "") -> None:
    """Save marks in the format read_marks reads."""
    lines = [f"# Repetition starts in {video_name} (seconds), from mark_reps.py.",
             "# 'skip' = a bad rep to leave out; 'end' = where the last rep finishes."]
    lines += [f"{t:.2f}" + (" skip" if skip else "") for t, skip in sorted(marks)]
    if end is not None:
        lines.append(f"end {end:.2f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


MARK_SHIFT = 0.6    # s; each marked rep may move this much to line up with the others
MARK_LEVEL = 0.3    # moving-speed level, from still to fast, in the word's average movement
MARK_MERGE = 0.4    # s; shorter still moments inside the average movement belong to the sign
MARK_PART  = 0.15   # a burst with this share of the strongest one's movement is part of the sign


def reps_from_marks(times: np.ndarray, keypoints: np.ndarray, starts: List[float],
                    end: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    (start, end) of each repetition marked by hand, one per mark.

    The marks only need to fall at the same moment of every rep, not at its
    exact start: a signer who holds the end of a sign until the next rep makes
    "where a rep starts" hard to see. The hand speed after each mark is lined
    up with the others (each rep may move up to MARK_SHIFT), and their median
    is the word's typical movement over one mark-to-mark gap. From the first
    to the last clear burst of it (the still hold left out; it may run past
    the next mark) is the sign, and each rep gets that stretch after its own
    mark. `end` is
    accepted for older marks files and not needed. A rep with much less
    movement than the others (weak_marked_reps) is not a sign.
    """
    starts = np.sort(np.asarray(starts, dtype=float))
    if len(starts) == 0:
        return []
    gap = float(np.median(np.diff(starts))) if len(starts) > 1 else SIGN_WINDOW_SECONDS
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / ANALYSIS_FPS
    speed = hand_speed(times, keypoints)
    k = int(round(MARK_SHIFT / dt))
    n = max(int(round(gap / dt)), 1)
    tau = np.arange(-k, n + k) * dt                       # around each mark, room to shift
    profiles = np.array([np.interp(m + tau, times, speed) for m in starts])
    core = slice(k, k + n)                                # one gap after the mark

    shifts = np.zeros(len(starts), dtype=int)             # in samples
    for _ in range(3):
        template = np.median([p[k + s:k + s + n] for p, s in zip(profiles, shifts)], axis=0)
        t0 = template - template.mean()
        for i, p in enumerate(profiles):
            scores = [float(((p[k + d:k + d + n] - p[k + d:k + d + n].mean()) * t0).sum())
                      for d in range(-k, k + 1)]
            shifts[i] = int(np.argmax(scores)) - k
    template = np.median([p[k + s:k + s + n] for p, s in zip(profiles, shifts)], axis=0)

    lo, hi = np.percentile(template, 10), np.percentile(template, 95)
    if hi - lo < 1e-6:                                    # no movement to go by: mark to mark
        nxt = list(starts[1:]) + [starts[-1] + gap]
        return [(float(a), float(b)) for a, b in zip(starts, nxt)]
    # The movement repeats every gap and may wrap past the next mark: look for it
    # starting from the stillest moment, so it is never cut in two.
    w = max(int(round(0.2 / dt)), 1)
    rot = int(np.argmin(np.convolve(np.concatenate([template, template[:w]]), np.ones(w) / w, "valid")[:n]))
    turned = np.roll(template, -rot)
    runs: List[List[int]] = []
    for i in np.flatnonzero(turned > lo + MARK_LEVEL * (hi - lo)):
        if runs and (i - runs[-1][1]) * dt < MARK_MERGE:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    # every clear burst is part of the sign (e.g. to the chin and back out again)
    energy = [float((turned[a:b + 1] - lo).sum()) for a, b in runs]
    clear = [r for r, e in zip(runs, energy) if e >= MARK_PART * max(energy)] if runs else [[0, n - 1]]
    best = [clear[0][0], clear[-1][1]]
    first = (best[0] + rot) % n
    if first >= 0.75 * n:                                 # just before the mark: that rep's sign
        first -= n
    on, off = first * dt, (first + best[1] - best[0]) * dt
    return [(float(m + s * dt + on), float(m + s * dt + off)) for m, s in zip(starts, shifts)]


MARK_WEAK = 0.4     # a marked rep moving less than this x the typical rep holds no sign


def weak_marked_reps(times: np.ndarray, keypoints: np.ndarray, spans: List[Tuple[float, float]]) -> List[bool]:
    """True for each span with much less hand movement than the typical one (e.g. a
    mark past the last sign)."""
    speed = hand_speed(times, keypoints)
    still = np.percentile(speed, 10)
    moved = np.array([float(np.mean(speed[(times >= a) & (times <= b)] - still))
                      if ((times >= a) & (times <= b)).any() else 0.0 for a, b in spans])
    typical = np.median(moved) if len(moved) else 0.0
    return [bool(typical > 0 and m < MARK_WEAK * typical) for m in moved]

def cut_by_recording(start: float, end: float, video_duration: float) -> bool:
    """True for a rep already going when the recording started or still going when it
    stopped: part of the sign is missing."""
    return start <= EDGE_TOLERANCE or video_duration - end <= EDGE_TOLERANCE


def is_stop_reach(start: float, end: float, hand_tracked: float, video_duration: float) -> bool:
    """True for the signer reaching for the phone at the end of the video (not a sign)."""
    return (video_duration - end <= STOP_REACH_END and hand_tracked < STOP_REACH_HANDS
            and end - start < STOP_REACH_LEN)


def clip_window(start: float, end: float, length: float, video_duration: float) -> Tuple[float, float]:
    """(start, end) of the clip around a sign: at least `length` seconds, longer for a
    slow sign (sign_window), centred on the sign and kept inside the video."""
    length = min(sign_window(start, end, length), video_duration)
    lo = min(max((start + end) / 2 - length / 2, 0.0), video_duration - length)
    return lo, lo + length


# Degrees to turn a frame clockwise to make it upright. Phones sometimes save a
# landscape recording sideways with no rotation tag (e.g. after Telegram).
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
NOSE = 0


def rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    return frame if not degrees else cv2.rotate(frame, ROTATIONS[degrees])


def is_upright(keypoints: np.ndarray) -> np.ndarray:
    """Per frame: a body was found with the nose above level shoulders."""
    pose = keypoints[:, :33 * 4].reshape(-1, 33, 4)
    ls, rs, nose = pose[:, LEFT_SHOULDER, :2], pose[:, RIGHT_SHOULDER, :2], pose[:, NOSE, :2]
    width = np.linalg.norm(ls - rs, axis=1)
    mid = (ls + rs) / 2
    return (width > 1e-6) & (np.abs(ls[:, 1] - rs[:, 1]) < 0.5 * width) & (mid[:, 1] - nose[:, 1] > 0.3 * width)


def detect_rotation(video: Path, samples: int = 10) -> int:
    """The rotation (0/90/180/270) that shows the signer upright in most sampled frames."""
    from src.landmark_extractor import LandmarkExtractor

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * n / samples))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    scores = {}
    for degrees in ROTATIONS:
        with LandmarkExtractor(face_mode="none", static_image_mode=True) as extractor:
            kps = np.array([extractor.process_frame(rotate(f, degrees))[0] for f in frames])
        scores[degrees] = int(is_upright(kps).sum()) if len(kps) else 0
    best = max(scores.values())
    return 0 if scores[0] == best else max(scores, key=scores.get)


def analyse(video: Path, fps: float = ANALYSIS_FPS, rotation: int = 0) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run MediaPipe on the video at about `fps`, turning frames by `rotation` degrees
    clockwise first. Returns (times, keypoints, source_fps, duration)."""
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
                    kp, _ = extractor.process_frame(rotate(frame, rotation))
                    times.append(i / src_fps)
                    kps.append(kp)
            i += 1
    cap.release()
    if not kps:
        raise RuntimeError(f"No frames could be read from {video}")
    return np.array(times), np.array(kps, dtype=np.float32), src_fps, i / src_fps


def write_clips(video: Path, windows: List[Tuple[float, float]], paths: List[Path],
                out_fps: float = OUTPUT_FPS, max_height: int = MAX_HEIGHT, rotation: int = 0) -> None:
    """Copy each [start, end) window of the video to its path, in one pass, turned
    `rotation` degrees clockwise."""
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
        frame = rotate(frame, rotation)
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
    parser.add_argument("--rotate", default="auto", choices=["auto", "0", "90", "180", "270"],
                        help="Degrees to turn the video clockwise to make it upright "
                             "(default: detect it)")
    parser.add_argument("--dry_run", action="store_true", help="Only report; write nothing")
    parser.add_argument("--extra", type=int, default=0,
                        help="Also accept up to this many repetitions more than --expected")
    parser.add_argument("--skip", type=int, nargs="*", default=[],
                        help="Rep numbers to leave out (e.g. wrong ones the signer noted)")
    parser.add_argument("--force", action="store_true", help="Write even if the count isn't --expected")
    parser.add_argument("--by_motion", action="store_true",
                        help="Find reps by hand movement and still pauses, for signers who keep "
                             "their hands up between reps instead of lowering them")
    parser.add_argument("--marks", nargs="?", const="", default=None,
                        help="Cut the reps marked by hand with mark_reps.py (default file: "
                             "<video>.marks.txt next to the video) instead of finding them")
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

    marks_file = None
    if args.marks is not None:
        marks_file = Path(args.marks).expanduser() if args.marks else marks_path(video)
        if not marks_file.exists():
            print(f"ERROR: no marks file {marks_file}. Mark the reps first: python scripts/mark_reps.py {video}")
            return 2

    if VOCAB_PATH.exists() and f'"{args.label}"' not in VOCAB_PATH.read_text(encoding="utf-8"):
        print(f"Note: '{args.label}' isn't in {VOCAB_PATH.name} yet; add it before training.")

    rotation = detect_rotation(video) if args.rotate == "auto" else int(args.rotate)
    if rotation:
        print(f"The video is sideways or upside down; turning it {rotation} degrees clockwise.")
    print(f"Analysing {video.name} ...")
    times, keypoints, src_fps, duration = analyse(video, rotation=rotation)
    _, hand, pose_ok = activity_signal(keypoints)
    print(f"  {duration:.1f} s at {src_fps:.0f} fps; body found in {pose_ok.mean():.0%} of frames")

    reps = []
    keep = None
    if marks_file is not None:
        marks, marks_end = read_marks(marks_file)
        spans = reps_from_marks(times, keypoints, [t for t, _ in marks], marks_end)
        keep = [not skip for _, skip in marks]
        print(f"  Using the {len(marks)} reps marked by hand in {marks_file.name}"
              + (f"; leaving out the {keep.count(False)} marked skip" if not all(keep) else ""))
        for n, ((a, b), weak) in enumerate(zip(spans, weak_marked_reps(times, keypoints, spans)), 1):
            if weak and keep[n - 1]:
                print(f"  Left out marked rep {n} ({a:.1f}-{b:.1f} s): hardly any movement, no sign there.")
                keep[n - 1] = False
    elif args.by_motion:
        spans, level = find_repetitions_by_motion(times, keypoints, args.expected, args.extra)
        print(f"  Split by movement (hands kept up between reps); moving-speed level {level:.2f}")
    else:
        spans = find_repetitions(times, keypoints)
    for start, end in spans:
        lo, hi = clip_window(start, end, args.window, duration)
        inside = (times >= start) & (times <= end)
        reps.append(Repetition(start, end, lo, hi, float(hand[inside].mean()) if inside.any() else 0.0))
    if keep is not None:                  # marked by hand: the marks decide, no automatic checks
        reps = [r for r, k in zip(reps, keep) if k]
    if keep is None and reps and is_stop_reach(reps[-1].start, reps[-1].end, reps[-1].hand_tracked, duration):
        print(f"  Left out the movement at {reps[-1].start:.1f} s: reaching for the phone at the end, not a sign.")
        reps.pop()
    for r in [r for r in reps[:1] + reps[-1:] if keep is None and cut_by_recording(r.start, r.end, duration)]:
        print(f"  Left out {r.start:.1f}-{r.end:.1f} s: the recording "
              f"{'started' if r.start <= EDGE_TOLERANCE else 'stopped'} in the middle of it.")
        reps.remove(r)
    if keep is None and len(reps) >= 5:
        typical = float(np.median([r.duration for r in reps]))
        merged = [r for r in reps if r.duration > MERGED_FACTOR * typical]
        for r in merged:
            print(f"  Left out {r.start:.1f}-{r.end:.1f} s ({r.duration:.1f} s, typical {typical:.1f} s): "
                  f"probably two reps without a rest between them.")
        reps = [r for r in reps if r not in merged]
    if args.skip:
        print(f"  Left out rep(s) {', '.join(map(str, args.skip))} as asked (--skip; numbers as in a run without it).")
        reps = [r for n, r in enumerate(reps, 1) if n not in set(args.skip)]

    print(f"\n  {'#':>3}  {'start':>6}  {'end':>6}  {'sign':>5}  {'clip':>13}  hands")
    for n, r in enumerate(reps, 1):
        flags = []
        if r.duration > r.clip_end - r.clip_start:
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
              f"(clips {args.window:.2f} s, longer for slow signs)")

    if args.plot is not None:
        plot_path = Path(args.plot) if args.plot else video.with_name(video.stem + "_split.png")
        save_plot(plot_path, times, keypoints, reps)
        print(f"  Plot: {plot_path}")

    count_ok = args.expected <= len(reps) <= args.expected + args.extra
    wanted = f"{args.expected}" + (f"-{args.expected + args.extra}" if args.extra else "")
    if keep is not None:                  # the marks are the count
        count_ok, wanted = len(reps) > 0, "as marked"
    print(f"\n  Found {len(reps)} repetitions (expected {wanted})"
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
    write_clips(video, [(r.clip_start, r.clip_end) for r in reps], paths, rotation=rotation)
    print(f"  Wrote {len(paths)} clips: {paths[0].name} .. {paths[-1].name} in {sign_dir}")
    print("  Next: python scripts/extract_landmarks.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
