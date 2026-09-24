"""
src/landmark_extractor.py
=========================
MediaPipe Holistic wrapper for extracting pose, hand, and face landmarks
from webcam frames or video files.

Produces a flat numpy array of shape (N_LANDMARKS,) per frame,
where N_LANDMARKS = 258 (pose=132, left_hand=63, right_hand=63)
or 1662 when face landmarks are included.

Authors: Menase Teshale, Bersabeh Dawit
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSE_LANDMARKS = 33        # MediaPipe Pose
HAND_LANDMARKS = 21        # MediaPipe Hands (per hand)
FACE_LANDMARKS = 468       # MediaPipe Face Mesh

POSE_DIMS   = POSE_LANDMARKS * 4    # x, y, z, visibility = 132
HAND_DIMS   = HAND_LANDMARKS * 3    # x, y, z             = 63
FACE_DIMS   = FACE_LANDMARKS * 3    # x, y, z             = 1404

# Default: pose + both hands only (no face) = 258 values/frame
FEATURE_DIM_NO_FACE = POSE_DIMS + HAND_DIMS * 2          # 258
FEATURE_DIM_WITH_FACE = POSE_DIMS + HAND_DIMS * 2 + FACE_DIMS  # 1662


class LandmarkExtractor:
    """
    Wraps MediaPipe Holistic to extract skeleton landmarks from frames.

    Parameters
    ----------
    include_face : bool
        If True, include 468 face landmarks (increases feature dim to 1662).
    min_detection_confidence : float
        Minimum detection confidence threshold for MediaPipe.
    min_tracking_confidence : float
        Minimum tracking confidence threshold for MediaPipe.
    """

    def __init__(
        self,
        include_face: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.include_face = include_face
        self.feature_dim = FEATURE_DIM_WITH_FACE if include_face else FEATURE_DIM_NO_FACE

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing  = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, object]:
        """
        Extract landmarks from a single BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            Raw frame from cv2.VideoCapture.

        Returns
        -------
        keypoints : np.ndarray of shape (feature_dim,)
            Flattened landmark array. Zero-padded if landmarks not detected.
        results : mediapipe Results object
            Raw MediaPipe results for visualization.
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        keypoints = self._extract_keypoints(results)
        return keypoints, results

    def draw_landmarks(self, bgr_frame: np.ndarray, results) -> np.ndarray:
        """Draw MediaPipe landmarks on the frame for visualization."""
        frame = bgr_frame.copy()

        # Pose
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        # Left hand
        self.mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
        )
        # Right hand
        self.mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
        )
        # Face (optional)
        if self.include_face:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
        return frame

    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_keypoints(self, results) -> np.ndarray:
        """Flatten MediaPipe results into a 1-D numpy array."""
        # Pose: 33 landmarks x 4 (x, y, z, visibility)
        pose = (
            np.array([[lm.x, lm.y, lm.z, lm.visibility]
                       for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks
            else np.zeros(POSE_DIMS)
        )

        # Left hand: 21 landmarks x 3 (x, y, z)
        lh = (
            np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks
            else np.zeros(HAND_DIMS)
        )

        # Right hand: 21 landmarks x 3 (x, y, z)
        rh = (
            np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks
            else np.zeros(HAND_DIMS)
        )

        if self.include_face:
            face = (
                np.array([[lm.x, lm.y, lm.z]
                           for lm in results.face_landmarks.landmark]).flatten()
                if results.face_landmarks
                else np.zeros(FACE_DIMS)
            )
            return np.concatenate([pose, lh, rh, face])

        return np.concatenate([pose, lh, rh])
