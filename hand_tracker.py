"""Thin wrapper around MediaPipe's HandLandmarker task."""

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarksConnections  # noqa: F401 (re-exported)

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# Landmark indices, per the MediaPipe Hands topology.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


class HandTracker:
    def __init__(self, num_hands: int = 1, min_detection_confidence: float = 0.5):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Download it with:\n"
                f'curl -L -o "{MODEL_PATH}" '
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_rgb, timestamp_ms: int):
        """Returns the HandLandmarkerResult for one BGR->RGB video frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()
