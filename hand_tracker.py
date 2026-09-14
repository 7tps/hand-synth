"""Thin wrapper around MediaPipe's GestureRecognizer task.

GestureRecognizer runs the same underlying hand-landmark detector as
HandLandmarker (so .hand_landmarks / .handedness work the same way) and adds
.gestures on top - a canned gesture classification per hand (e.g. "Victory",
"Thumb_Up", "Open_Palm"), so one pass gets us both instead of running two
separate models."""

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarksConnections  # noqa: F401 (re-exported)

MODEL_PATH = Path(__file__).parent / "models" / "gesture_recognizer.task"

# Landmark indices, per the MediaPipe Hands topology.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


class HandTracker:
    def __init__(
        self,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Download it with:\n"
                f'curl -L -o "{MODEL_PATH}" '
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                "gesture_recognizer/float16/1/gesture_recognizer.task"
            )

        # Loosening these to tolerate frame-edge/pinch flicker (a past fix)
        # backfired: it also let MediaPipe keep "tracking" a hand for a while
        # after it actually left frame, using stale/extrapolated positions
        # instead of just dropping it (main.py's handedness-confidence gate
        # is the real fix for that staleness; keep these at their defaults).
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def process(self, frame_rgb, timestamp_ms: int):
        """Returns the GestureRecognizerResult for one BGR->RGB video frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._recognizer.recognize_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        self._recognizer.close()
