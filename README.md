# hand-synth

A webcam-controlled synthesizer. MediaPipe tracks both hands; OpenCV shows a
1280x720 camera feed with an overlay; a polyphonic sine synth plays through
your speakers.

- **Pitch** (left hand, thumb+index pinch): grab a pitch fader - shown as a slider on the left edge of the frame, with the note name (or the two nearest notes, if you're between them) displayed next to the handle - then move your hand up/down to set the root pitch (higher = higher pitch, spans 3 octaves from C3). Releasing the pinch holds the pitch where you left it. This root pitch is always in effect, whether you're playing it plain or under a chord. Left hand not in frame -> silent.
- **Volume** (right hand, thumb+index pinch): grab a volume fader, then move your hand up/down to set the level (higher = louder). Releasing the pinch holds the volume where you left it. If the right hand isn't visible at all, volume defaults to a fixed level as long as the left (pitch) hand is.
- **Chords** (right hand, thumb+middle pinch): grab a chord pad instead of the volume fader. While pinched, a 3x3 grid appears filling whichever half of the frame your right hand is currently on (it follows the hand if you cross the midline mid-pinch); moving your hand within that half picks a cell. "Base" plays the root pitch alone (no chord); the other 8 cells are chord presets (equal temperament) built on top of whatever the current root pitch is - changing pitch transposes the chord rather than interrupting it. Releasing holds whichever was picked until you pinch thumb+middle again. Presets: Augmented, Dominant 7th, Major 7th, Diminished 7th, 6th, sus2, sus4, 9th.
- Volume and chords share the right hand's two pinches, so only one of them can be changed at a time. Pitch is on its own hand (the left), so it can always be changed alongside either one.
- **Freeze**: peace sign (✌️) with the right hand freezes pitch/chord and volume exactly where they are - pinches and hand visibility stop affecting the sound entirely. Peace sign with the left hand unfreezes and resumes normal control.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The gesture recognizer model (`models/gesture_recognizer.task`) is already
downloaded in this repo - it includes hand landmark tracking plus MediaPipe's
canned gesture classifier (Victory, Thumb_Up, Open_Palm, etc.) in one model.
If it's ever missing, re-fetch it with:

```bash
curl -L -o models/gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

## Run

```bash
.venv\Scripts\python main.py
```

Press `q` in the video window to quit.

## Files

- `main.py` - camera loop, pitch/chord/volume/freeze mapping, on-screen overlay.
- `hand_tracker.py` - wraps MediaPipe's `GestureRecognizer` task.
- `audio_engine.py` - real-time polyphonic sine synth (`sounddevice` callback,
  up to 4 voices, with smoothed frequency/gain/amplitude to avoid clicks and
  zipper noise between notes, chords, and volume changes).