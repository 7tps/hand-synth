"""Hand-tracked synthesizer.

  - Right hand: vertical position of the index fingertip sets pitch
    (higher on screen = higher pitch).
  - Left hand: pinch (thumb tip touching index fingertip) grabs a volume
    fader. While pinched, the vertical position of the pinch point sets
    volume (higher = louder). Releasing the pinch holds the volume at
    whatever it was last set to. If the left hand isn't visible at all,
    volume defaults to DEFAULT_AMP as long as the right hand is present.
  - Right hand not visible -> silent.

Press 'q' to quit.
"""

import time

import cv2
import numpy as np

from audio_engine import SineSynth
from hand_tracker import HandTracker, HandLandmarksConnections, INDEX_TIP, THUMB_TIP

MIN_FREQ_HZ = 130.81  # C3
OCTAVE_RANGE = 3.0  # pitch spans this many octaves top-to-bottom of frame

PINCH_THRESHOLD = 0.1  # normalized thumb-index distance below which fingers count as touching
DEFAULT_AMP = 0.6  # volume used when the left (volume) hand has never been pinched yet

CAM_INDEX = 0


def freq_from_y(y_norm: float) -> float:
    """y_norm in [0, 1], 0 = top of frame. Higher hand -> higher pitch."""
    y_norm = float(np.clip(y_norm, 0.0, 1.0))
    octaves_up = (1.0 - y_norm) * OCTAVE_RANGE
    return MIN_FREQ_HZ * (2.0 ** octaves_up)


def amp_from_y(y_norm: float) -> float:
    """y_norm in [0, 1], 0 = top of frame. Higher hand -> louder."""
    return float(np.clip(1.0 - y_norm, 0.0, 1.0))


def draw_hand(frame, landmarks, label, pinched=False):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    if label == "Right":  # pitch hand
        cv2.circle(frame, pts[INDEX_TIP], 9, (0, 0, 255), 2)
    elif label == "Left":  # volume hand
        pinch_color = (0, 255, 0) if pinched else (255, 0, 0)
        cv2.circle(frame, pts[INDEX_TIP], 9, pinch_color, 2)
        cv2.circle(frame, pts[THUMB_TIP], 9, pinch_color, 2)
        cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], pinch_color, 2)

    wrist_x, wrist_y = pts[0]
    cv2.putText(
        frame, label, (wrist_x - 20, wrist_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )


def main():
    tracker = HandTracker(num_hands=2)
    synth = SineSynth()
    synth.start()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    start_time = time.time()
    current_amp = DEFAULT_AMP
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

            hands = {}
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hands[handedness[0].category_name] = landmarks

            right = hands.get("Right")
            left = hands.get("Left")

            pinched = False
            if left is not None:
                thumb_tip, index_tip = left[THUMB_TIP], left[INDEX_TIP]
                pinch_dist = float(np.hypot(
                    index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y
                ))
                pinched = pinch_dist < PINCH_THRESHOLD
                if pinched:
                    pinch_y = (index_tip.y + thumb_tip.y) / 2.0
                    current_amp = amp_from_y(pinch_y)
                draw_hand(frame, left, "Left", pinched)

            if right is not None:
                draw_hand(frame, right, "Right")
                freq = freq_from_y(right[INDEX_TIP].y)
                synth.set_frequency(freq)
                synth.set_amplitude(current_amp)

                cv2.putText(
                    frame, f"{freq:6.1f} Hz  vol {current_amp:.2f}"
                    + ("  PINCH" if pinched else ""),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                )
            else:
                synth.set_amplitude(0.0)
                cv2.putText(
                    frame, "show right hand for pitch",
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
