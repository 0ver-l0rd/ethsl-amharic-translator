"""
src/video_io.py
===============
Saving and loading recorded sign clips.

Clips are stored as compressed .mp4 video (~1% of the size of raw frame
arrays) so landmarks can be re-extracted later with different settings
(e.g. a different face mode) without re-recording. Legacy .npy clips
(raw BGR frame arrays) can still be read.

Authors: EthSL Team
"""

import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

# (backend, codec) pairs tried in order. H.264 via Windows' built-in Media
# Foundation encoder gives the best quality per MB; MPEG-4 Part 2 via FFmpeg
# ships with every opencv-python build. The backend is named explicitly because
# letting OpenCV try H.264 through FFmpeg first prints "[ERROR] Could not open
# codec libopenh264" on every clip (harmless, but alarming while recording).
_WRITERS = ([(cv2.CAP_MSMF, "avc1")] if sys.platform == "win32" else []) + [(cv2.CAP_FFMPEG, "mp4v")]


def write_clip(path: Path, frames: List[np.ndarray], fps: float) -> str:
    """
    Write BGR frames to an .mp4 file at the given frame rate.
    Returns the codec used. Raises RuntimeError if no codec is available.
    """
    h, w = frames[0].shape[:2]
    for backend, codec in _WRITERS:
        writer = cv2.VideoWriter(str(path), backend, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            for frame in frames:
                writer.write(frame)
            writer.release()
            return codec
        writer.release()
    raise RuntimeError(f"No usable video writer for {path} (tried {[c for _, c in _WRITERS]})")


def read_clip(path: Path) -> List[np.ndarray]:
    """Read a recorded clip (.mp4 video or legacy .npy frame array) as BGR frames."""
    path = Path(path)
    if path.suffix == ".npy":
        return list(np.load(str(path), allow_pickle=False))

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames
