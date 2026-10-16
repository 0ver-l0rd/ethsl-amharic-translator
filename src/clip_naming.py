"""
src/clip_naming.py
==================
Single source of truth for how recorded clips are named, so that the
signer who performed each clip survives every pipeline stage:

  collect_data.py      ->  data/landmarks/<label>/<signer>_seq_007.npy
  extract_landmarks.py ->  data/landmarks/<label>/<signer>_seq_007_kp.npy
  preprocess.py        ->  data/processed/signer_ids.npy  (one ID per sample)

Signer IDs are short codes like "s01" rather than names, so the dataset
stays anonymous. They must not contain underscores (used as the separator).

Authors: EthSL Team
"""

import re
from pathlib import Path
from typing import Optional

UNKNOWN_SIGNER = "unknown"

_SIGNER_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")
_CLIP_RE      = re.compile(r"^(?P<signer>[A-Za-z0-9-]+)_seq_(?P<idx>\d+)(?:_kp)?$")


def validate_signer_id(signer: str) -> str:
    """Return the signer ID unchanged, or raise ValueError if it is unusable."""
    if not _SIGNER_ID_RE.match(signer) or signer == UNKNOWN_SIGNER:
        raise ValueError(
            f"Invalid signer ID '{signer}'. Use letters, digits or '-' only "
            f"(no spaces or underscores), e.g. s01, s02."
        )
    return signer


def clip_filename(signer: str, idx: int) -> str:
    """Raw clip filename for a signer's idx-th recording of a sign."""
    return f"{signer}_seq_{idx:03d}.npy"


def parse_signer(path: Path) -> Optional[str]:
    """Signer ID encoded in a clip/keypoint filename, or None if absent."""
    m = _CLIP_RE.match(Path(path).stem)
    return m.group("signer") if m else None


def next_clip_index(sign_dir: Path, signer: str) -> int:
    """First unused clip index for this signer, so re-runs never overwrite."""
    used = [
        int(m.group("idx"))
        for p in sign_dir.glob(f"{signer}_seq_*.npy")
        if (m := _CLIP_RE.match(p.stem)) and m.group("signer") == signer
    ]
    return max(used) + 1 if used else 0
