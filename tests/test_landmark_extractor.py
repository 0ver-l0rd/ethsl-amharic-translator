"""
tests/test_landmark_extractor.py
==================================
Unit tests for LandmarkExtractor.

Run with: pytest tests/
"""

import sys
import numpy as np
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

try:
    from src.landmark_extractor import (
        LandmarkExtractor, FEATURE_DIM_NO_FACE, FEATURE_DIM_FACE_SUBSET, FEATURE_DIM_WITH_FACE,
        FEATURE_DIMS, FACE_SUBSET_INDICES, face_mode_for_dim,
    )
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


def fake_results(n_face=468):
    """MediaPipe-like results where every coordinate encodes its landmark index."""
    def lms(n, offset):
        return SimpleNamespace(landmark=[
            SimpleNamespace(x=offset + i, y=offset + i, z=offset + i, visibility=1.0)
            for i in range(n)
        ])
    return SimpleNamespace(
        pose_landmarks=lms(33, 0), left_hand_landmarks=lms(21, 0),
        right_hand_landmarks=lms(21, 0), face_landmarks=lms(n_face, 1000),
    )


@pytest.mark.skipif(not MEDIAPIPE_AVAILABLE, reason="MediaPipe not installed")
class TestLandmarkExtractor:

    @pytest.mark.parametrize("mode", ["none", "subset", "full"])
    def test_output_shape(self, mode):
        """Each face mode produces its documented feature size, even with nothing detected."""
        with LandmarkExtractor(face_mode=mode) as extractor:
            kp, _ = extractor.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        assert kp.shape == (FEATURE_DIMS[mode],)
        assert extractor.feature_dim == FEATURE_DIMS[mode]

    def test_feature_dims(self):
        assert (FEATURE_DIM_NO_FACE, FEATURE_DIM_FACE_SUBSET, FEATURE_DIM_WITH_FACE) == (258, 534, 1662)
        assert len(FACE_SUBSET_INDICES) == 92

    def test_subset_keeps_the_right_face_points(self):
        with LandmarkExtractor(face_mode="subset") as extractor:
            kp = extractor._extract_keypoints(fake_results())
        face = kp[FEATURE_DIM_NO_FACE:].reshape(-1, 3)
        np.testing.assert_array_equal(face[:, 0], 1000 + FACE_SUBSET_INDICES)

    def test_face_mode_for_dim(self):
        for mode, dim in FEATURE_DIMS.items():
            assert face_mode_for_dim(dim) == mode
        with pytest.raises(ValueError):
            face_mode_for_dim(300)

    def test_invalid_face_mode(self):
        with pytest.raises(ValueError):
            LandmarkExtractor(face_mode="eyes")

    def test_zero_frame_gives_zero_keypoints(self):
        """A completely black frame should give zero-padded landmarks."""
        with LandmarkExtractor() as extractor:
            kp, _ = extractor.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        assert kp.shape == (FEATURE_DIM_NO_FACE,)
        assert not kp.any()
