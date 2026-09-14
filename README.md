# hand-synth

A webcam-controlled synthesizer. MediaPipe tracks both hands; OpenCV shows a
1280x720 camera feed with an overlay; a sine oscillator plays through your
speakers.

- **Pitch** (right hand): pinch your thumb and index finger together to grab a pitch fader, then move your hand up/down to set it (higher = higher pitch, spans 3 octaves from C3). Releasing the pinch holds the pitch where you left it.
- **Volume** (left hand): pinch your thumb and index finger together to grab a volume fader, then move your hand up/down to set the level (higher = louder). Releasing the pinch holds the volume where you left it. If the left hand isn't visible at all, volume defaults to a fixed level as long as the right hand is.
- Right hand not in frame -> silent.
- **Freeze**: peace sign (✌️) with the right hand freezes pitch and volume exactly where they are - pinches and hand visibility stop affecting the sound entirely. Peace sign with the left hand unfreezes and resumes normal control.

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

- `main.py` - camera loop, pitch/volume/freeze mapping, on-screen overlay.
- `hand_tracker.py` - wraps MediaPipe's `GestureRecognizer` task.
- `audio_engine.py` - real-time sine oscillator (`sounddevice` callback with
  smoothed frequency/amplitude to avoid clicks and zipper noise).