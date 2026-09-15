"""Hand-tracked synthesizer.

  - Left hand, thumb+index pinch: grabs a pitch fader (shown as a slider on
    the left edge of the frame). While pinched, the vertical position of
    the pinch point sets the root pitch (higher = higher pitch, spans 3
    octaves from C3). Releasing the pinch holds the pitch at whatever it
    was last set to. Left hand not visible -> silent, since pitch is what
    makes a note meaningful at all.
  - Right hand, thumb+index pinch: grabs a volume fader. While pinched, the
    vertical position of the pinch point sets volume (higher = louder).
    Releasing the pinch holds the volume at whatever it was last set to. If
    the right hand isn't visible at all, volume defaults to DEFAULT_AMP as
    long as the left (pitch) hand is present.
  - Right hand, thumb+middle pinch: grabs a chord pad instead of the volume
    fader. While pinched, a 3x3 grid appears (shown on screen only while
    pinching); moving your hand around the frame picks a cell. "Base" plays
    just the root pitch with no chord on top of it (the plain single-note
    sound); the other 8 cells are chord presets built above the current
    root pitch. Releasing holds whichever was picked until you pinch
    thumb+middle again. Volume and chord share the right hand's two
    pinches, so only one of them can be changed at a time; pitch is on its
    own hand, so it can always be changed alongside either.
  - Peace sign (Victory gesture) with the right hand freezes pitch/chord and
    volume exactly where they are - pinches and hand visibility stop
    affecting the sound at all. Peace sign with the left hand unfreezes.

Press 'q' to quit.
"""

import time

import cv2
import numpy as np

from audio_engine import SineSynth
from hand_tracker import (
    HandTracker,
    HandLandmarksConnections,
    INDEX_TIP,
    MIDDLE_TIP,
    THUMB_TIP,
)

# MediaPipe's built-in canned gesture for a peace sign.
PEACE_GESTURE = "Victory"
GESTURE_CONFIDENCE_THRESHOLD = 0.5

MIN_FREQ_HZ = 130.81  # C3
OCTAVE_RANGE = 3.0  # pitch spans this many octaves top-to-bottom of frame

PINCH_THRESHOLD = 0.1  # normalized fingertip distance below which fingers count as touching
DEFAULT_AMP = 0.6  # volume used when the right (volume) hand has never been pinched yet

# Hand tracking flickers frame-to-frame - a hand near the frame edge, or
# mid-pinch (fingers occluding each other), can drop out for a frame or two
# even though it never actually left. Tolerate a brief gap before treating
# the pitch (left) hand as truly gone, instead of cutting audio on every
# flicker.
PITCH_HAND_GRACE_MS = 400

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

# Chord presets (equal temperament), as semitone offsets from whatever the
# current root pitch is. "Base" is just the root pitch by itself.
CHORDS = [
    ("Base", [0]),
    ("Augmented", [0, 4, 8]),
    ("Dominant 7th", [0, 4, 7, 10]),
    ("Major 7th", [0, 4, 7, 11]),
    ("Diminished 7th", [0, 3, 6, 9]),
    ("6th", [0, 4, 7, 9]),
    ("sus2", [0, 2, 7]),
    ("sus4", [0, 5, 7]),
    ("9th", [0, 4, 10, 14]),
]
CHORD_GRID_COLS = 3
CHORD_GRID_ROWS = 3  # CHORD_GRID_COLS * CHORD_GRID_ROWS must equal len(CHORDS)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_MATCH_CENTS = 20  # within this many cents of a note, show just that note; otherwise show both neighbors

PITCH_SLIDER_X0 = 30
PITCH_SLIDER_WIDTH = 50
PITCH_SLIDER_MARGIN_Y = 60


def freq_from_y(y_norm: float) -> float:
    """y_norm in [0, 1], 0 = top of frame. Higher hand -> higher pitch."""
    y_norm = float(np.clip(y_norm, 0.0, 1.0))
    octaves_up = (1.0 - y_norm) * OCTAVE_RANGE
    return MIN_FREQ_HZ * (2.0 ** octaves_up)


def y_from_freq(freq_hz: float) -> float:
    """Inverse of freq_from_y - normalized y in [0, 1] for a given frequency."""
    octaves_up = np.log2(freq_hz / MIN_FREQ_HZ)
    y_norm = 1.0 - octaves_up / OCTAVE_RANGE
    return float(np.clip(y_norm, 0.0, 1.0))


def amp_from_y(y_norm: float) -> float:
    """y_norm in [0, 1], 0 = top of frame. Higher hand -> louder."""
    return float(np.clip(1.0 - y_norm, 0.0, 1.0))


def chord_freqs_for(chord_idx: int, root_hz: float):
    """Frequencies of the given chord preset, built above root_hz."""
    _, semitones = CHORDS[chord_idx]
    return [root_hz * (2.0 ** (s / 12.0)) for s in semitones]


def note_name_from_midi(midi_num: int) -> str:
    return f"{NOTE_NAMES[midi_num % 12]}{midi_num // 12 - 1}"


def note_label_from_freq(freq_hz: float) -> str:
    """The note name at freq_hz, or both neighboring notes if it's not
    within NOTE_MATCH_CENTS of any single note."""
    midi = 69.0 + 12.0 * np.log2(freq_hz / 440.0)
    nearest = round(midi)
    cents_off = (midi - nearest) * 100.0
    if abs(cents_off) <= NOTE_MATCH_CENTS:
        return note_name_from_midi(int(nearest))
    lower = int(np.floor(midi))
    return f"{note_name_from_midi(lower)} / {note_name_from_midi(lower + 1)}"


def describe_sound(chord_idx: int, freq_hz: float) -> str:
    chord_name = CHORDS[chord_idx][0]
    note = note_label_from_freq(freq_hz)
    if chord_name == "Base":
        return f"{freq_hz:6.1f} Hz ({note})"
    return f"{chord_name}  root {freq_hz:6.1f} Hz ({note})"


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


def chord_index_from_xy(x_norm: float, y_norm: float) -> int:
    """Maps a point in the frame to one of the CHORDS grid cells."""
    col = int(np.clip(x_norm, 0.0, 0.999) * CHORD_GRID_COLS)
    row = int(np.clip(y_norm, 0.0, 0.999) * CHORD_GRID_ROWS)
    col = int(np.clip(col, 0, CHORD_GRID_COLS - 1))
    row = int(np.clip(row, 0, CHORD_GRID_ROWS - 1))
    return row * CHORD_GRID_COLS + col


def draw_chord_grid(frame, current_idx: int):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    cell_w, cell_h = w / CHORD_GRID_COLS, h / CHORD_GRID_ROWS
    for i, (name, _) in enumerate(CHORDS):
        row, col = divmod(i, CHORD_GRID_COLS)
        x0, y0 = int(col * cell_w), int(row * cell_h)
        x1, y1 = int((col + 1) * cell_w), int((row + 1) * cell_h)
        active = i == current_idx

        border_color = (0, 255, 0) if active else (120, 120, 120)
        cv2.rectangle(frame, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4), border_color, 3 if active else 1)

        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx, ty = x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 + th) // 2
        if active:
            cv2.rectangle(frame, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (0, 255, 0), -1)
        text_color = (0, 0, 0) if active else (220, 220, 220)
        cv2.putText(frame, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


def draw_pitch_slider(frame, current_freq: float):
    h, w = frame.shape[:2]
    x0, x1 = PITCH_SLIDER_X0, PITCH_SLIDER_X0 + PITCH_SLIDER_WIDTH
    y0, y1 = PITCH_SLIDER_MARGIN_Y, h - PITCH_SLIDER_MARGIN_Y

    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 2)
    for i in range(int(OCTAVE_RANGE) + 1):
        ty = int(y0 + (1.0 - i / OCTAVE_RANGE) * (y1 - y0))
        cv2.line(frame, (x0 - 10, ty), (x1 + 10, ty), (140, 140, 140), 1)

    handle_y = int(y0 + y_from_freq(current_freq) * (y1 - y0))
    cv2.rectangle(frame, (x0 - 8, handle_y - 5), (x1 + 8, handle_y + 5), (0, 0, 255), -1)
    cv2.putText(
        frame, note_label_from_freq(current_freq), (x1 + 16, handle_y + 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
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
    current_chord_idx = 0  # "Base"
    last_pitch_hand_seen_ms = None
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
                # Give the pitch hand a fresh grace window instead of muting
                # immediately just because it wasn't seen during the freeze.
                last_pitch_hand_seen_ms = timestamp_ms

            if frozen:
                if left is not None:
                    draw_hand(frame, left, "Left")
                if right is not None:
                    draw_hand(frame, right, "Right")
                draw_pitch_slider(frame, current_freq)

                synth.set_chord(chord_freqs_for(current_chord_idx, current_freq))
                synth.set_amplitude(current_amp)
                cv2.putText(
                    frame,
                    f"FROZEN  {describe_sound(current_chord_idx, current_freq)}"
                    f"  vol {current_amp:.2f}  (peace sign w/ left hand to resume)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2,
                )
                cv2.imshow("Hand Synth", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            pitch_pinched = False
            if left is not None:
                thumb_tip, index_tip = left[THUMB_TIP], left[INDEX_TIP]
                presence_trustworthy = is_trustworthy(scores["Left"], thumb_tip, index_tip)
                if presence_trustworthy:
                    last_pitch_hand_seen_ms = timestamp_ms

                pinch_dist = float(np.hypot(
                    index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y
                ))
                pitch_pinched = pinch_dist < PINCH_THRESHOLD
                if pitch_pinched and presence_trustworthy:
                    pinch_y = (index_tip.y + thumb_tip.y) / 2.0
                    current_freq = freq_from_y(pinch_y)
                draw_hand(frame, left, "Left", pitch_pinched)

            vol_pinched = False
            chord_pinched = False
            if right is not None:
                thumb_tip = right[THUMB_TIP]
                index_tip = right[INDEX_TIP]
                middle_tip = right[MIDDLE_TIP]

                vol_dist = float(np.hypot(
                    index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y
                ))
                chord_dist = float(np.hypot(
                    middle_tip.x - thumb_tip.x, middle_tip.y - thumb_tip.y
                ))
                # Index and middle fingertips sit close together, so closing
                # either one against the thumb often brings the other within
                # range too. Only trust whichever is actually closer, so
                # volume and chord (they share this hand) can't both change
                # from a single pinch.
                vol_pinched = vol_dist < PINCH_THRESHOLD and vol_dist <= chord_dist
                chord_pinched = chord_dist < PINCH_THRESHOLD and chord_dist < vol_dist

                if vol_pinched and is_trustworthy(scores["Right"], thumb_tip, index_tip):
                    pinch_y = (index_tip.y + thumb_tip.y) / 2.0
                    current_amp = amp_from_y(pinch_y)

                if chord_pinched and is_trustworthy(scores["Right"], thumb_tip, middle_tip):
                    select_x = (middle_tip.x + thumb_tip.x) / 2.0
                    select_y = (middle_tip.y + thumb_tip.y) / 2.0
                    current_chord_idx = chord_index_from_xy(select_x, select_y)

                if chord_pinched:
                    draw_chord_grid(frame, current_chord_idx)
                draw_hand(frame, right, "Right", vol_pinched or chord_pinched)

            pitch_hand_active = (
                last_pitch_hand_seen_ms is not None
                and timestamp_ms - last_pitch_hand_seen_ms <= PITCH_HAND_GRACE_MS
            )

            if not chord_pinched:
                draw_pitch_slider(frame, current_freq)

            if pitch_hand_active:
                synth.set_chord(chord_freqs_for(current_chord_idx, current_freq))
                synth.set_amplitude(current_amp)

                hud = f"{describe_sound(current_chord_idx, current_freq)}  vol {current_amp:.2f}"
                if pitch_pinched:
                    hud += "  PITCH"
                if chord_pinched:
                    hud += "  CHORD"
                if vol_pinched:
                    hud += "  VOL"
                cv2.putText(
                    frame, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                )
            else:
                synth.set_amplitude(0.0)
                cv2.putText(
                    frame, "show left hand for pitch",
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
