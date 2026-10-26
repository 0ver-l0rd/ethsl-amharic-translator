"""Tests for app/ui.py Amharic (Ethiopic) text rendering."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import app.ui as ui

needs_font = pytest.mark.skipif(ui.ETHIOPIC_FONT is None,
                                reason="no Ethiopic font installed")


def blank():
    return np.zeros((120, 400, 3), np.uint8)


@needs_font
def test_ethiopic_text_is_drawn():
    frame = blank()
    ui.put_unicode_text(frame, "ስምህ ማን ነው?", (10, 60), size_px=30,
                        color=(255, 140, 60))
    assert frame.any()
    # Text sits around the baseline, not elsewhere in the frame
    rows = np.where(frame.any(axis=(1, 2)))[0]
    assert rows.min() > 20 and rows.max() < 80


@needs_font
def test_glyphs_differ_from_missing_glyph_boxes():
    # Different Ethiopic words must render differently; a font without
    # Ethiopic glyphs would draw identical "tofu" boxes for both.
    a, b = blank(), blank()
    ui.put_unicode_text(a, "ሰላም", (10, 60))
    ui.put_unicode_text(b, "ውሃ", (10, 60))
    assert not np.array_equal(a, b)


@needs_font
@pytest.mark.parametrize("pos", [(-50, 60), (380, 60), (10, 5), (10, 200), (-500, -500)])
def test_text_partly_or_fully_off_frame_does_not_crash(pos):
    frame = blank()
    ui.put_unicode_text(frame, "አመሰግናለሁ", pos, size_px=30)


def test_falls_back_to_cv2_without_font(monkeypatch):
    monkeypatch.setattr(ui, "ETHIOPIC_FONT", None)
    frame = blank()
    ui.put_unicode_text(frame, "Amharic: ?", (10, 60))
    assert frame.any()
