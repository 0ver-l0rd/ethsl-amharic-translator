"""
tests/test_landmark_extractor.py
==================================
Unit tests for LandmarkExtractor.

Run with: pytest tests/
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

try:
    from src.landmark_extractor import LandmarkExtractor, FEATURE_DIM_NO_FACE, FEATURE_DIM_WITH_FACE
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@pytest.mark.skipif(not MEDIAPIPE_AVAILABLE, reason="MediaPipe not installed")
class TestLandmarkExtractor:

    def test_output_shape_no_face(self):
        """Landmark array should have (FEATURE_DIM_NO_FACE,) = (258,) shape."""
        extractor = LandmarkExtractor(include_face=False)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp, _ = extractor.process_frame(dummy_frame)
        assert kp.shape == (FEATURE_DIM_NO_FACE,), f"Expected ({FEATURE_DIM_NO_FACE},), got {kp.shape}"
        extractor.close()

    def test_output_shape_with_face(self):
        """With face landmarks enabled, output should have 1662 values."""
        extractor = LandmarkExtractor(include_face=True)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp, _ = extractor.process_frame(dummy_frame)
        assert kp.shape == (FEATURE_DIM_WITH_FACE,)
        extractor.close()

    def test_zero_frame_gives_zero_keypoints(self):
        """A completely black frame should give zero-padded landmarks."""
        extractor = LandmarkExtractor(include_face=False)
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        kp, _ = extractor.process_frame(black)
        # With no detected landmarks, all zeros expected
        assert kp.shape == (FEATURE_DIM_NO_FACE,)
        extractor.close()

    def test_context_manager(self):
        """LandmarkExtractor should work as a context manager."""
        with LandmarkExtractor() as ext:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            kp, _ = ext.process_frame(frame)
            assert kp is not None
