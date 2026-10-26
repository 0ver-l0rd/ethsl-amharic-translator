"""
src/landmark_extractor.py
=========================
MediaPipe Holistic wrapper for extracting pose, hand, and face landmarks
from webcam frames or video files.

Produces a flat numpy array of shape (feature_dim,) per frame. The face
mode decides how much of MediaPipe's 468-point face mesh is kept:

  face_mode   face points kept                     feature_dim
  ---------   ----------------------------------   -----------
  "none"      none (pose + both hands only)            258
  "subset"    92: lips, eyes, eyebrows                 534
  "full"      all 468                                 1662

The subset keeps the facial regions that carry meaning in sign language
(eyebrows mark questions, the mouth and eyes carry expression) without
letting jaw/cheek contours dominate the input. Holistic computes the face
mesh on every frame regardless, so the mode does not change speed.

Authors: Menase Teshale, Bersabeh Dawit
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import face_mesh_connections as _fmc
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSE_LANDMARKS = 33        # MediaPipe Pose
HAND_LANDMARKS = 21        # MediaPipe Hands (per hand)
FACE_LANDMARKS = 468       # MediaPipe Face Mesh

# Face-mesh regions kept in "subset" mode (MediaPipe's own region definitions)
_FACE_SUBSET_CONNECTIONS = frozenset().union(
    _fmc.FACEMESH_LIPS,
    _fmc.FACEMESH_LEFT_EYE, _fmc.FACEMESH_RIGHT_EYE,
    _fmc.FACEMESH_LEFT_EYEBROW, _fmc.FACEMESH_RIGHT_EYEBROW,
)
FACE_SUBSET_INDICES = np.array(sorted({i for edge in _FACE_SUBSET_CONNECTIONS for i in edge}))

POSE_DIMS   = POSE_LANDMARKS * 4    # x, y, z, visibility = 132
HAND_DIMS   = HAND_LANDMARKS * 3    # x, y, z             = 63
FACE_DIMS   = FACE_LANDMARKS * 3    # x, y, z             = 1404
FACE_SUBSET_DIMS = len(FACE_SUBSET_INDICES) * 3          # 92 x 3 = 276

FACE_MODES = ("none", "subset", "full")

# Default: pose + both hands only (no face) = 258 values/frame
FEATURE_DIM_NO_FACE   = POSE_DIMS + HAND_DIMS * 2                     # 258
FEATURE_DIM_FACE_SUBSET = FEATURE_DIM_NO_FACE + FACE_SUBSET_DIMS      # 534
FEATURE_DIM_WITH_FACE = FEATURE_DIM_NO_FACE + FACE_DIMS               # 1662

FEATURE_DIMS = {
    "none":   FEATURE_DIM_NO_FACE,
    "subset": FEATURE_DIM_FACE_SUBSET,
    "full":   FEATURE_DIM_WITH_FACE,
}


def face_mode_for_dim(feature_dim: int) -> str:
    """Face mode that produces `feature_dim` features (e.g. from a model's input shape)."""
    for mode, dim in FEATURE_DIMS.items():
        if dim == feature_dim:
            return mode
    raise ValueError(
        f"Feature dim {feature_dim} matches no face mode; expected one of {FEATURE_DIMS}"
    )


class LandmarkExtractor:
    """
    Wraps MediaPipe Holistic to extract skeleton landmarks from frames.

    Parameters
    ----------
    face_mode : str
        "none" (default), "subset" or "full"; see the module docstring.
        Training data and the live app must use the same mode.
    min_detection_confidence : float
        Minimum detection confidence threshold for MediaPipe.
    min_tracking_confidence : float
        Minimum tracking confidence threshold for MediaPipe.
    model_complexity : int
        MediaPipe pose model size: 0 (lite, fastest), 1 (default), 2 (heavy).
        Landmarks differ slightly between sizes, so use the same value for
        training data extraction and live inference.
    """

    def __init__(
        self,
        face_mode: str = "none",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        if face_mode not in FACE_MODES:
            raise ValueError(f"face_mode must be one of {FACE_MODES}, got {face_mode!r}")
        self.face_mode = face_mode
        self.feature_dim = FEATURE_DIMS[face_mode]

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing  = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            model_complexity=model_complexity,
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
        # Face: draw exactly the points the model receives
        if self.face_mode != "none":
            connections = (_FACE_SUBSET_CONNECTIONS if self.face_mode == "subset"
                           else self.mp_holistic.FACEMESH_CONTOURS)
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(80, 220, 220), thickness=1, circle_radius=1),
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

        if self.face_mode == "none":
            return np.concatenate([pose, lh, rh])

        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
            if self.face_mode == "subset":
                face = face[FACE_SUBSET_INDICES]
            face = face.flatten()
        else:
            face = np.zeros(FACE_SUBSET_DIMS if self.face_mode == "subset" else FACE_DIMS)
        return np.concatenate([pose, lh, rh, face])
