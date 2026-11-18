"""
scripts/split_batch.py
======================
Split a whole folder of phone recordings with split_recording.py and print one
summary table.

Videos are named <signer>_<number>, e.g. s03_07.mp4; the number is looked up
in data/our_words.tsv for the word's label. A second version of a word
(another way to sign it) can be added as a letter, e.g. "s03_29 a.mp4" and
"s03_29 b.mp4"; both go to the same word.

Each video is turned upright if needed and must contain --expected
repetitions; videos that don't are left out and listed at the end with a plot
of what the splitter saw (in <folder>/split_report/). Re-running replaces the
signer's earlier clips of a word. Landmarks are extracted at the end.

Bad reps found on review go in <folder>/skip.txt, one video per line, e.g.
  s02_13  5        (rep 5 of s02_13 is left out on every re-run)

Usage:
  python scripts/split_batch.py "~/Desktop/Senior Project/recordings/s03" --dry_run
  python scripts/split_batch.py "~/Desktop/Senior Project/recordings/s03"

Authors: Estifanos Behailu
"""

import argparse
import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

WORDS_PATH = ROOT / "data" / "our_words.tsv"
VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
_NAME_RE = re.compile(r"^(?P<signer>[A-Za-z0-9-]+)_(?P<num>[A-Za-z]?\d+)(?:[ _-]?(?P<variant>[a-z]))?$",
                      re.IGNORECASE)


def load_words(path: Path = WORDS_PATH) -> Dict[str, Tuple[str, str]]:
    """{number: (label, amharic)} from our_words.tsv, numbers like '07' or 'E1'."""
    words = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            num, label, amharic, *_ = line.split("\t")
            words[num.upper()] = (label, amharic)
    return words


def parse_name(path: Path) -> Optional[Tuple[str, str, str]]:
    """(signer, number, variant) from e.g. 's03_29 a.MP4' -> ('s03', '29', 'a'); None if unreadable."""
    m = _NAME_RE.match(path.stem.strip())
    if not m:
        return None
    num = m.group("num").upper()
    if num.isdigit():
        num = f"{int(num):02d}"
    return m.group("signer").lower(), num, (m.group("variant") or "").lower()


def main(argv=None) -> int:
    import split_recording

    parser = argparse.ArgumentParser(description="Split a folder of recordings")
    parser.add_argument("folder", help="Folder with <signer>_<number>.mp4 videos")
    parser.add_argument("--expected", type=int, default=15)
    parser.add_argument("--extra", type=int, default=0,
                        help="Also accept up to this many repetitions more than --expected")
    parser.add_argument("--only", nargs="*", help="Only these videos (file names without extension)")
    parser.add_argument("--no_extract", action="store_true", help="Don't extract landmarks at the end")
    parser.add_argument("--dry_run", action="store_true", help="Only report; write nothing")
    args = parser.parse_args(argv)

    folder = Path(args.folder).expanduser()
    words = load_words()
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS
                    and (not args.only or p.stem in args.only))
    report = folder / "split_report"
    report.mkdir(exist_ok=True)

    skips = {}
    if (folder / "skip.txt").exists():
        for line in (folder / "skip.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split("#")[0].split()
            if len(parts) > 1:
                skips[" ".join(p for p in parts if not p.isdigit())] = [p for p in parts if p.isdigit()]

    rows, done = [], set()
    for n, video in enumerate(videos, 1):
        parsed = parse_name(video)
        if not parsed or parsed[1] not in words:
            rows.append((video.name, "?", "", "", "", "SKIPPED: name not <signer>_<number> from our_words.tsv"))
            continue
        signer, num, variant = parsed
        label, amharic = words[num]
        mode = "--append" if (signer, label) in done else "--replace"
        cmd = [str(video), "--label", label, "--signer", signer, "--expected", str(args.expected), "--extra", str(args.extra), mode,
               "--plot", str(report / f"{video.stem}_split.png")] + (["--dry_run"] if args.dry_run else []) \
            + (["--skip", *skips[video.stem]] if video.stem in skips else [])
        print(f"[{n}/{len(videos)}] {video.name} -> {amharic} ({label})", flush=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = split_recording.main(cmd)
        text = out.getvalue()
        (report / f"{video.stem}.log").write_text(text, encoding="utf-8")
        found = re.search(r"Found (\d+) repetitions", text)
        found = int(found.group(1)) if found else 0
        turned = re.search(r"turning it (\d+) degrees", text)
        median = re.search(r"median ([\d.]+) s", text)
        ok = args.expected <= found <= args.expected + args.extra and code == 0
        if ok and not args.dry_run:
            done.add((signer, label))
        status = ("ok" if args.dry_run else "written") if ok else \
            (f"CHECK: found {found}, see split_report/{video.stem}_split.png" if found else
             text.strip().splitlines()[-1] if text.strip() else f"failed (exit {code})")
        rows.append((video.name, f"{amharic} ({label})", str(found), turned.group(1) if turned else "0",
                     median.group(1) if median else "", status))

    print(f"\n{'video':<16} {'word':<26} {'reps':>4} {'turned':>6} {'sign s':>6}  status")
    for r in rows:
        print(f"{r[0]:<16} {r[1]:<26} {r[2]:>4} {r[3]:>6} {r[4]:>6}  {r[5]}")
    good = sum(1 for r in rows if r[5] in ("ok", "written"))
    print(f"\n{good} of {len(rows)} videos {'would split' if args.dry_run else 'split'} cleanly.")

    if not args.dry_run and not args.no_extract and good:
        print("\nExtracting landmarks for the new clips ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "extract_landmarks.py")], check=True)
    return 0 if good == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
