"""
src/contributions.py
====================
Videos people upload to help train the model (app/web_app.py).

  1. split_upload(): an uploaded video of one word signed any number of
     times is split into one clip per repetition, the same way
     scripts/split_recording.py splits the team's recordings. The clips wait
     in data/contributions/uploads/<id>/ with status "pending"; the original
     video is not kept.
  2. A team member watches the clips, unticks bad ones, and calls approve()
     or reject(). Approving copies the kept clips into data/landmarks/<label>/
     under the contributor's signer ID, extracts their landmarks, and adds the
     word to vocabulary.json if it is new. Rejecting deletes the clips.
  3. Retraining (app/web_app.py) then uses them like any other clips.

Contributors get anonymous signer IDs (c01, c02, ...; the team keeps
s01-s04), so the signer-independent evaluation can tell people apart. The
name -> ID table stays in data/contributions/signers.json, which is never
committed.

Authors: Estifanos Behailu
"""

import json
import re
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.clip_naming import clip_filename, next_clip_index
from src.sequence_builder import SIGN_WINDOW_SECONDS
from src.vocabulary import add_word

CONTRIB_DIR = ROOT / "data" / "contributions"
LANDMARKS_DIR = ROOT / "data" / "landmarks"
VOCAB_PATH = ROOT / "data" / "vocabulary.json"

TEAM = {"estifanos": "s01", "emanuel": "s02", "bersabeh": "s03", "menase": "s04"}
CONTRIBUTOR_PREFIX = "c"
WEAK_TRACKING = 0.6     # below this share of frames with a hand, a rep is flagged


@dataclass
class Rep:
    start: float          # s, hands up
    end: float            # s, hands down
    clip_start: float     # s, saved window
    clip_end: float
    hand_tracked: float   # share of the rep's frames with a hand detected
    clip: str             # file name in the upload folder

    @property
    def flags(self) -> List[str]:
        out = []
        if self.end - self.start > self.clip_end - self.clip_start:
            out.append("longer than the clip")
        if self.hand_tracked < WEAK_TRACKING:
            out.append("hands hard to see")
        return out


@dataclass
class Upload:
    id: str
    label: str
    amharic: str
    description: str
    new_word: bool
    signer: str
    created: float
    status: str = "pending"            # pending | approved | rejected
    reps: List[Rep] = field(default_factory=list)
    kept_clips: List[str] = field(default_factory=list)  # in data/landmarks after approval

    @property
    def folder(self) -> Path:
        return CONTRIB_DIR / "uploads" / self.id

    def clip_paths(self, root: Path = None) -> List[Path]:
        folder = (root / "uploads" / self.id) if root else self.folder
        return [folder / r.clip for r in self.reps]


def normalise_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def signer_id_for(name: str, root: Path = CONTRIB_DIR, team: bool = True) -> str:
    """
    The contributor's signer ID, assigning the next free cNN for a new name.
    With `team`, the team's first names map to s01-s04 (off on a public page,
    where anyone could type them).
    """
    key = normalise_name(name)
    if not key:
        raise ValueError("Please enter your name.")
    if team and key.split(" ")[0] in TEAM:
        return TEAM[key.split(" ")[0]]
    path = root / "signers.json"
    table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if key not in table:
        used = [int(v[1:]) for v in table.values() if v.startswith(CONTRIBUTOR_PREFIX)]
        table[key] = f"{CONTRIBUTOR_PREFIX}{max(used, default=0) + 1:02d}"
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return table[key]


def _save(upload: Upload, root: Path) -> None:
    folder = root / "uploads" / upload.id
    folder.mkdir(parents=True, exist_ok=True)
    data = asdict(upload)
    (folder / "meta.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")


def load(upload_id: str, root: Path = CONTRIB_DIR) -> Upload:
    data = json.loads((root / "uploads" / upload_id / "meta.json").read_text(encoding="utf-8"))
    data["reps"] = [Rep(**r) for r in data["reps"]]
    return Upload(**data)


def list_uploads(status: Optional[str] = None, root: Path = CONTRIB_DIR) -> List[Upload]:
    """Uploads, oldest first, optionally only those with `status`."""
    folder = root / "uploads"
    if not folder.is_dir():
        return []
    uploads = [load(p.name, root) for p in folder.iterdir() if (p / "meta.json").exists()]
    return sorted((u for u in uploads if status is None or u.status == status), key=lambda u: u.created)


def split_upload(video: Path, label: str, amharic: str, description: str, new_word: bool,
                 contributor: str, root: Path = CONTRIB_DIR, team: bool = True,
                 analyse: Optional[Callable] = None) -> Upload:
    """
    Find the repetitions in an uploaded video and save one clip per repetition
    for review. `analyse` (video -> times, keypoints, fps, duration) defaults to
    the splitter's MediaPipe pass; tests pass a fake.
    """
    import split_recording as sr

    signer = signer_id_for(contributor, root, team)
    analyse = analyse or sr.analyse
    times, keypoints, _, duration = analyse(Path(video))
    _, hand, _ = sr.activity_signal(keypoints)

    upload = Upload(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], label=label,
                    amharic=amharic.strip(), description=description.strip(), new_word=new_word,
                    signer=signer, created=time.time())
    for n, (start, end) in enumerate(sr.find_repetitions(times, keypoints), 1):
        lo, hi = sr.clip_window(start, end, SIGN_WINDOW_SECONDS, duration)
        inside = (times >= start) & (times <= end)
        upload.reps.append(Rep(start, end, lo, hi, float(hand[inside].mean()) if inside.any() else 0.0,
                               f"rep_{n:02d}.mp4"))
    _save(upload, root)
    if upload.reps:
        sr.write_clips(Path(video), [(r.clip_start, r.clip_end) for r in upload.reps],
                       upload.clip_paths(root))
    return upload


def approve(upload_id: str, keep: List[int], root: Path = CONTRIB_DIR,
            landmarks_dir: Path = LANDMARKS_DIR, vocab_path: Path = VOCAB_PATH,
            extract: Optional[Callable[[List[Path]], None]] = None) -> Upload:
    """
    Add the kept repetitions (1-based numbers) to the training data. `extract`
    writes landmarks for the given clip paths; it defaults to MediaPipe.
    """
    upload = load(upload_id, root)
    if upload.status != "pending":
        raise ValueError(f"Upload {upload_id} is already {upload.status}.")
    chosen = [upload.reps[k - 1] for k in sorted(set(keep)) if 1 <= k <= len(upload.reps)]
    if not chosen:
        raise ValueError("Keep at least one repetition, or reject the upload.")

    add_word(vocab_path, upload.label, upload.amharic, upload.description)
    sign_dir = landmarks_dir / upload.label
    sign_dir.mkdir(parents=True, exist_ok=True)
    first = next_clip_index(sign_dir, upload.signer)
    dests = []
    for n, rep in enumerate(chosen):
        dest = sign_dir / clip_filename(upload.signer, first + n)
        shutil.copy2(root / "uploads" / upload.id / rep.clip, dest)
        dests.append(dest)
    (extract or _extract_landmarks)(dests)

    upload.status, upload.kept_clips = "approved", [d.name for d in dests]
    _save(upload, root)
    return upload


def reject(upload_id: str, root: Path = CONTRIB_DIR) -> Upload:
    """Mark the upload rejected and delete its clips (people's videos aren't kept for nothing)."""
    upload = load(upload_id, root)
    for p in upload.clip_paths(root):
        p.unlink(missing_ok=True)
    upload.status = "rejected"
    _save(upload, root)
    return upload


def _extract_landmarks(clips: List[Path]) -> None:
    from extract_landmarks import extract_clip
    from src.landmark_extractor import LandmarkExtractor

    with LandmarkExtractor(face_mode="none") as extractor:
        for clip in clips:
            extract_clip(clip, extractor)
