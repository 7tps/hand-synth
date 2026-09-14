"""Hand-tracked synthesizer.

  - Right hand: pinch (thumb tip touching index fingertip) grabs a pitch
    fader. While pinched, the vertical position of the pinch point sets
    pitch (higher = higher pitch, spans 3 octaves from C3). Releasing the
    pinch holds the pitch at whatever it was last set to.
  - Left hand: pinch (thumb tip touching index fingertip) grabs a volume
    fader. While pinched, the vertical position of the pinch point sets
    volume (higher = louder). Releasing the pinch holds the volume at
    whatever it was last set to. If the left hand isn't visible at all,
    volume defaults to DEFAULT_AMP as long as the right hand is present.
  - Right hand not visible -> silent.
  - Peace sign (Victory gesture) with the right hand freezes pitch and
    volume exactly where they are - pinches and hand visibility stop
    affecting the sound at all. Peace sign with the left hand unfreezes.

Press 'q' to quit.
"""

import time

import cv2
import numpy as np

from audio_engine import SineSynth
from hand_tracker import HandTracker, HandLandmarksConnections, INDEX_TIP, THUMB_TIP

# MediaPipe's built-in canned gesture for a peace sign.
PEACE_GESTURE = "Victory"
GESTURE_CONFIDENCE_THRESHOLD = 0.5

MIN_FREQ_HZ = 130.81  # C3
OCTAVE_RANGE = 3.0  # pitch spans this many octaves top-to-bottom of frame

PINCH_THRESHOLD = 0.1  # normalized fingertip distance below which fingers count as touching
DEFAULT_AMP = 0.6  # volume used when the left (volume) hand has never been pinched yet

# Hand tracking flickers frame-to-frame - a hand near the frame edge, or
# mid-pinch (fingers occluding each other), can drop out for a frame or two
# even though it never actually left. Tolerate a brief gap before treating
# the right hand as truly gone, instead of cutting audio on every flicker.
RIGHT_HAND_GRACE_MS = 400

# As a hand exits the frame, MediaPipe's landmark regressor can't see the
# real fingertip position anymore and extrapolates - visibly "sticking" a
# fingertip near the frame border (e.g. the thumb pinning to y=1) instead of
# just losing the hand. Ignore fader-position updates once a controlling
# fingertip is this close to any edge, so that glitch can't drag the volume
# or pitch to an extreme right as the hand leaves.
EDGE_MARGIN = 0.1

# MediaPipe doesn't expose a separate "is this really a hand" quality score
# in this API, but its Left/Right handedness confidence reliably drops during
# degraded/ghost tracking (e.g. still reporting a hand for a few frames after
# it's actually left frame) even when the reported position isn't obviously
# near an edge. Below this, treat the frame as if the hand weren't seen at
# all - both for trusting its position and for "is it still there" timing.
HANDEDNESS_CONFIDENCE_THRESHOLD = 0.85

CAM_INDEX = 0
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def freq_from_y(y_norm: float) -> float:
    """y_norm in [0, 1], 0 = top of frame. Higher hand -> higher pitch."""
    y_norm = float(np.clip(y_norm, 0.0, 1.0))
    octaves_up = (1.0 - y_norm) * OCTAVE_RANGE
    return MIN_FREQ_HZ * (2.0 ** octaves_up)


def amp_from_y(y_norm: float) -> float:
    """y_norm in [0, 1], 0 = top of frame. Higher hand -> louder."""
    return float(np.clip(1.0 - y_norm, 0.0, 1.0))


def in_bounds(*landmarks, margin: float = EDGE_MARGIN) -> bool:
    """False if any landmark is close enough to the frame border that its
    position may be a MediaPipe extrapolation artifact rather than real."""
    return all(
        margin <= lm.x <= 1.0 - margin and margin <= lm.y <= 1.0 - margin
        for lm in landmarks
    )


def is_trustworthy(handedness_score: float, *landmarks) -> bool:
    """False if this frame's hand data looks like degraded/ghost tracking
    rather than a real, clearly-visible hand."""
    return (
        handedness_score >= HANDEDNESS_CONFIDENCE_THRESHOLD
        and in_bounds(*landmarks)
    )


def draw_hand(frame, landmarks, label, pinched=False):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    base_color = (0, 0, 255) if label == "Right" else (255, 0, 0)
    pinch_color = (0, 255, 0) if pinched else base_color
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    start_time = time.time()
    current_amp = DEFAULT_AMP
    current_freq = freq_from_y(0.5)
    last_right_seen_ms = None
    frozen = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera")
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            if frame.shape[1] != DISPLAY_WIDTH or frame.shape[0] != DISPLAY_HEIGHT:
                frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = tracker.process(frame_rgb, timestamp_ms)

            hands = {}
            scores = {}
            gestures = {}
            for landmarks, handedness, gesture in zip(
                result.hand_landmarks, result.handedness, result.gestures
            ):
                name = handedness[0].category_name
                hands[name] = landmarks
                scores[name] = handedness[0].score
                gestures[name] = (gesture[0].category_name, gesture[0].score) if gesture else (None, 0.0)

            right = hands.get("Right")
            left = hands.get("Left")
            right_gesture, right_gesture_score = gestures.get("Right", (None, 0.0))
            left_gesture, left_gesture_score = gestures.get("Left", (None, 0.0))

            # Check transitions against the state at the start of the frame,
            # so a peace sign on both hands at once can't freeze and
            # immediately unfreeze (or vice versa) within the same frame.
            frozen_before = frozen
            if (
                not frozen_before
                and right_gesture == PEACE_GESTURE
                and right_gesture_score >= GESTURE_CONFIDENCE_THRESHOLD
            ):
                frozen = True
            if (
                frozen_before
                and left_gesture == PEACE_GESTURE
                and left_gesture_score >= GESTURE_CONFIDENCE_THRESHOLD
            ):
                frozen = False
                # Give the right hand a fresh grace window instead of muting
                # immediately just because it wasn't seen during the freeze.
                last_right_seen_ms = timestamp_ms

            if frozen:
                if left is not None:
                    draw_hand(frame, left, "Left")
                if right is not None:
                    draw_hand(frame, right, "Right")

                synth.set_frequency(current_freq)
                synth.set_amplitude(current_amp)
                cv2.putText(
                    frame,
                    f"FROZEN  {current_freq:6.1f} Hz  vol {current_amp:.2f}"
                    "  (peace sign w/ left hand to resume)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2,
                )
                cv2.imshow("Hand Synth", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            vol_pinched = False
            if left is not None:
                thumb_tip, index_tip = left[THUMB_TIP], left[INDEX_TIP]
                pinch_dist = float(np.hypot(
                    index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y
                ))
                vol_pinched = pinch_dist < PINCH_THRESHOLD
                if vol_pinched and is_trustworthy(scores["Left"], thumb_tip, index_tip):
                    pinch_y = (index_tip.y + thumb_tip.y) / 2.0
                    current_amp = amp_from_y(pinch_y)
                draw_hand(frame, left, "Left", vol_pinched)

            pitch_pinched = False
            if right is not None:
                thumb_tip, index_tip = right[THUMB_TIP], right[INDEX_TIP]
                pinch_dist = float(np.hypot(
                    index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y
                ))
                pitch_pinched = pinch_dist < PINCH_THRESHOLD
                trustworthy = is_trustworthy(scores["Right"], thumb_tip, index_tip)
                if trustworthy:
                    last_right_seen_ms = timestamp_ms
                if pitch_pinched and trustworthy:
                    pinch_y = (index_tip.y + thumb_tip.y) / 2.0
                    current_freq = freq_from_y(pinch_y)
                draw_hand(frame, right, "Right", pitch_pinched)

            right_active = (
                last_right_seen_ms is not None
                and timestamp_ms - last_right_seen_ms <= RIGHT_HAND_GRACE_MS
            )

            if right_active:
                synth.set_frequency(current_freq)
                synth.set_amplitude(current_amp)

                hud = f"{current_freq:6.1f} Hz  vol {current_amp:.2f}"
                if pitch_pinched:
                    hud += "  PITCH"
                if vol_pinched:
                    hud += "  VOL"
                cv2.putText(
                    frame, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
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
