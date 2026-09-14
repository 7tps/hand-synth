"""Hand-tracked synthesizer.

Point your index finger at the camera:
  - Vertical position of the index fingertip sets pitch (higher = higher pitch).
  - Distance between thumb tip and index fingertip sets volume (spread = louder).

Press 'q' to quit.
"""

import time

import cv2
import numpy as np

from audio_engine import SineSynth
from hand_tracker import HandTracker, HandLandmarksConnections, INDEX_TIP, THUMB_TIP

MIN_FREQ_HZ = 130.81  # C3
OCTAVE_RANGE = 3.0  # pitch spans this many octaves top-to-bottom of frame

MIN_PINCH = 0.03  # normalized thumb-index distance -> silence
MAX_PINCH = 0.35  # normalized thumb-index distance -> full volume

CAM_INDEX = 0


def freq_from_y(y_norm: float) -> float:
    """y_norm in [0, 1], 0 = top of frame. Higher hand -> higher pitch."""
    y_norm = float(np.clip(y_norm, 0.0, 1.0))
    octaves_up = (1.0 - y_norm) * OCTAVE_RANGE
    return MIN_FREQ_HZ * (2.0 ** octaves_up)


def amp_from_pinch(pinch_dist: float) -> float:
    t = (pinch_dist - MIN_PINCH) / (MAX_PINCH - MIN_PINCH)
    return float(np.clip(t, 0.0, 1.0))


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    cv2.circle(frame, pts[INDEX_TIP], 9, (0, 0, 255), 2)
    cv2.circle(frame, pts[THUMB_TIP], 9, (255, 0, 0), 2)
    cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], (255, 0, 255), 2)


def main():
    tracker = HandTracker(num_hands=1)
    synth = SineSynth()
    synth.start()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera")
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = tracker.process(frame_rgb, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                index_tip = landmarks[INDEX_TIP]
                thumb_tip = landmarks[THUMB_TIP]

                freq = freq_from_y(index_tip.y)
                pinch_dist = float(np.hypot(
                    index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y
                ))
                amp = amp_from_pinch(pinch_dist)

                synth.set_frequency(freq)
                synth.set_amplitude(amp)

                draw_hand(frame, landmarks)
                cv2.putText(
                    frame, f"{freq:6.1f} Hz  vol {amp:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                )
            else:
                synth.set_amplitude(0.0)
                cv2.putText(
                    frame, "no hand detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )

            cv2.imshow("Hand Synth", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        synth.stop()
        tracker.close()


if __name__ == "__main__":
    main()
