"""
scripts/mark_reps.py
====================
Watch a recording and mark where each repetition starts, for videos the
splitter can't cut on its own (e.g. a signer who keeps their hands up and
barely pauses between reps). Press space as each rep begins; the marks are
saved next to the video as <video>.marks.txt after every change, and
split_batch.py --marks (or split_recording.py --marks) cuts the clips from
them. The reps are lined up by their movement, so press at the same moment of
every rep (best: as the hand starts to move); exactness doesn't matter.

Keys (click the video window first):
  space      a rep starts here (while playing, your reaction time is taken off)
  backspace  undo the last change
  x          the rep under the playhead is a bad one: leave it out (x again: keep it)
  p          pause / play
  h          half speed on / off
  a / d      back / forward 2 s (or the left / right arrow keys)
  , / .      one frame back / forward (pauses)
  n          next video
  q / esc    stop (the marks are already saved)

Marking while paused, after stepping frame by frame, puts the mark exactly on
the frame shown.

Usage:
  python scripts/mark_reps.py "~/Desktop/Senior Project/recordings/s04" --only s04_08 s04_10
  python scripts/mark_reps.py "~/Desktop/Senior Project/recordings/s04/s04_08.mp4"
  (a folder without --only opens every video that has no marks yet)

Then:
  python scripts/split_batch.py "~/Desktop/Senior Project/recordings/s04" --marks --only s04_08 s04_10
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from split_recording import marks_path, read_marks, write_marks, rotate, detect_rotation  # noqa: E402
from split_batch import load_words, parse_name, select_videos  # noqa: E402

WINDOW = "mark reps"
REACTION = 0.2      # s between seeing a rep start and pressing space (at normal speed)
SAME_MARK = 0.3     # s; a new mark this close to an old one moves it instead of adding one
MAX_W, MAX_H = 1100, 650
BAR_H = 70
HELP = "space=rep starts  backspace=undo  x=bad rep  p=pause  h=half speed  a/d=2s  ,/.=frame  n=next  q=quit"
LEFT_KEYS, RIGHT_KEYS = {63234, 2424832, 65361}, {63235, 2555904, 65363}


class Marks:
    """The marks of one video: rep starts (each with a skip flag) and an optional end,
    with undo."""

    def __init__(self, starts=None, end: Optional[float] = None):
        self.starts: List[List] = [[t, skip] for t, skip in (starts or [])]
        self.end = end
        self._undo = []

    def _save_state(self):
        self._undo.append(([list(m) for m in self.starts], self.end))

    def add(self, t: float) -> None:
        self._save_state()
        near = [m for m in self.starts if abs(m[0] - t) < SAME_MARK]
        if near:
            near[0][0] = t
        else:
            self.starts.append([t, False])
        self.starts.sort()

    def current(self, t: float) -> Optional[int]:
        """Index of the rep the time falls in (the last mark at or before it)."""
        before = [i for i, m in enumerate(self.starts) if m[0] <= t + 1e-6]
        return before[-1] if before else None

    def toggle_skip(self, t: float) -> Optional[int]:
        i = self.current(t)
        if i is not None:
            self._save_state()
            self.starts[i][1] = not self.starts[i][1]
        return i

    def set_end(self, t: float) -> None:
        self._save_state()
        self.end = t

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.starts, self.end = self._undo.pop()
        return True

    def as_list(self):
        return [(float(t), bool(skip)) for t, skip in self.starts]


class Player:
    """Frame-accurate playback of one video, turned upright and scaled to fit the screen."""

    def __init__(self, path: Path, rotation: int):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.rotation = rotation
        self.pos, self.frame = -1, None
        self.next_frame()

    @property
    def duration(self) -> float:
        return self.count / self.fps

    @property
    def time(self) -> float:
        return max(self.pos, 0) / self.fps

    def _show(self, frame):
        frame = rotate(frame, self.rotation)
        h, w = frame.shape[:2]
        s = min(MAX_W / w, MAX_H / h, 1.0)
        self.frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)

    def next_frame(self, skip: int = 0) -> bool:
        for _ in range(skip):
            if not self.cap.grab():
                return False
            self.pos += 1
        ok, frame = self.cap.read()
        if not ok:
            self.count = max(self.pos + 1, 1)       # the real end (the header count can be off)
            return False
        self.pos += 1
        self._show(frame)
        return True

    def seek(self, index: int) -> None:
        index = int(np.clip(index, 0, self.count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if ok:
            self.pos = index
            self._show(frame)

    def release(self):
        self.cap.release()


def draw(player: Player, marks: Marks, name: str, playing: bool, speed: float, flash: str) -> np.ndarray:
    img = player.frame
    h, w = img.shape[:2]
    canvas = np.zeros((h + BAR_H, w, 3), np.uint8)
    canvas[:h] = img
    t, dur = player.time, max(player.duration, 1e-6)
    skips = sum(s for _, s in marks.starts)
    cur = marks.current(t)
    status = (f"{name}   {t:5.2f} / {dur:.1f} s   {'playing' if playing else 'PAUSED'}  {speed:g}x   "
              f"reps marked: {len(marks.starts)}" + (f" ({skips} bad)" if skips else "")
              + (f"   now in rep {cur + 1}" if cur is not None else "")
              + ("   end set" if marks.end is not None else ""))
    cv2.rectangle(canvas, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(canvas, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if flash:
        cv2.putText(canvas, flash, (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3,
                    cv2.LINE_AA)
    # timeline
    x0, x1, y = 10, w - 10, h + 22
    cv2.line(canvas, (x0, y), (x1, y), (90, 90, 90), 6)
    to_x = lambda s: int(x0 + (x1 - x0) * min(max(s / dur, 0), 1))
    for n, (s, bad) in enumerate(marks.starts, 1):
        cv2.line(canvas, (to_x(s), y - 12), (to_x(s), y + 12), (0, 0, 255) if bad else (0, 200, 0), 2)
        cv2.putText(canvas, str(n), (to_x(s) + 2, y - 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (200, 200, 200), 1)
    if marks.end is not None:
        cv2.line(canvas, (to_x(marks.end), y - 12), (to_x(marks.end), y + 12), (255, 140, 0), 3)
    cv2.line(canvas, (to_x(t), y - 16), (to_x(t), y + 16), (255, 255, 255), 2)
    cv2.putText(canvas, HELP, (8, h + BAR_H - 10), cv2.FONT_HERSHEY_PLAIN, 0.95, (200, 200, 200), 1,
                cv2.LINE_AA)
    return canvas


def mark_video(video: Path, rotation: int, title: str) -> str:
    """Show one video for marking. Returns 'next' or 'quit'."""
    path = marks_path(video)
    marks = Marks(*read_marks(path)) if path.exists() else Marks()
    if path.exists():
        print(f"  Loaded {len(marks.starts)} earlier marks from {path.name}")
    player = Player(video, rotation)
    try:
        cv2.setWindowTitle(WINDOW, title)
    except cv2.error:
        pass

    playing, speed, flash, flash_until = True, 1.0, "", 0.0
    clock = time.perf_counter()

    def changed(message):
        nonlocal flash, flash_until
        write_marks(path, marks.as_list(), marks.end, video.name)
        flash, flash_until = message, time.perf_counter() + 0.5

    result = "next"
    try:
        while True:
            now = time.perf_counter()
            if playing:
                frame_time = 1 / (player.fps * speed)
                behind = int((now - clock) / frame_time)        # frames to drop if decoding is slow
                if behind >= 1:
                    if not player.next_frame(skip=max(behind - 1, 0)):
                        playing = False
                        flash, flash_until = "end of video: n = next", now + 2.0
                    clock += behind * frame_time
                    if now - clock > 0.5:
                        clock = now
            if flash and now > flash_until:
                flash = ""
            cv2.imshow(WINDOW, draw(player, marks, video.stem, playing, speed, flash))
            wait = max(1, int((clock + 1 / (player.fps * speed) - time.perf_counter()) * 1000)) if playing else 30
            key = cv2.waitKeyEx(wait)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                result = "quit"
                break
            if key == -1:
                continue
            ch = key & 0xFF if key < 256 else None
            t = player.time - (REACTION * speed if playing else 0.0)
            if ch == 32:                                     # space
                marks.add(max(t, 0.0))
                changed(f"rep {marks.current(max(t, 0.0)) + 1}")
            elif ch in (8, 127):                             # backspace / delete
                if marks.undo():
                    changed("undone")
            elif ch == ord("x"):
                i = marks.toggle_skip(player.time)
                if i is not None:
                    changed(f"rep {i + 1} {'BAD' if marks.starts[i][1] else 'kept'}")
            elif ch == ord("p"):
                playing = not playing
                clock = time.perf_counter()
            elif ch == ord("h"):
                speed = 0.5 if speed == 1.0 else 1.0
                clock = time.perf_counter()
            elif ch == ord("a") or key in LEFT_KEYS:
                player.seek(player.pos - int(2 * player.fps))
                clock = time.perf_counter()
            elif ch == ord("d") or key in RIGHT_KEYS:
                player.seek(player.pos + int(2 * player.fps))
                clock = time.perf_counter()
            elif ch == ord(","):
                playing = False
                player.seek(player.pos - 1)
            elif ch == ord("."):
                playing = False
                player.next_frame()
            elif ch == ord("n"):
                break
            elif ch in (ord("q"), 27):
                result = "quit"
                break
    finally:
        player.release()
    skips = sum(s for _, s in marks.starts)
    print(f"  {video.stem}: {len(marks.starts)} reps marked" + (f", {skips} bad" if skips else "")
          + (", end set" if marks.end is not None else "") + f" -> {path}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mark where each repetition starts in a recording")
    parser.add_argument("path", help="A video, or a folder of <signer>_<number> videos")
    parser.add_argument("--only", nargs="*", help="Only these videos (file names without extension)")
    parser.add_argument("--rotate", default="auto", choices=["auto", "0", "90", "180", "270"],
                        help="Degrees to turn the video clockwise to make it upright (default: detect it)")
    args = parser.parse_args(argv)

    path = Path(args.path).expanduser()
    if path.is_dir():
        videos = select_videos(path, args.only)
        if not args.only:
            videos = [v for v in videos if not marks_path(v).exists()]
    else:
        videos = [path]
    if not videos:
        print("Nothing to mark (every video already has a .marks.txt; use --only to reopen one).")
        return 0

    words = load_words()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    marked = []
    for n, video in enumerate(videos, 1):
        parsed = parse_name(video)
        label, amharic = words.get(parsed[1], ("?", "?")) if parsed else ("?", "?")
        rotation = int(args.rotate) if args.rotate != "auto" else detect_rotation(video)
        print(f"[{n}/{len(videos)}] {video.name}: {amharic} ({label})"
              + (f", turned {rotation} degrees" if rotation else ""), flush=True)
        result = mark_video(video, rotation, f"{video.stem}  {amharic} ({label})  [{n}/{len(videos)}]")
        if marks_path(video).exists():
            marked.append(video.stem)
        if result == "quit":
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if marked:
        folder = videos[0].parent
        print(f"\nNext, cut the clips:\n  python scripts/split_batch.py \"{folder}\" --marks --only {' '.join(marked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
