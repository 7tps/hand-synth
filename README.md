# hand-synth

A webcam-controlled synthesizer. MediaPipe tracks both hands; OpenCV shows the
camera feed with an overlay; a sine oscillator plays through your speakers.

- **Pitch** (right hand): vertical position of your index fingertip (higher on screen = higher pitch, spans 3 octaves from C3).
- **Volume** (left hand): distance between your thumb tip and index fingertip (spread apart = louder). If the left hand isn't visible, volume defaults to a fixed level as long as the right hand is.
- Right hand not in frame -> silent.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The hand landmark model (`models/hand_landmarker.task`) is already downloaded
in this repo. If it's ever missing, re-fetch it with:

```bash
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Run

```bash
.venv\Scripts\python main.py
```

Press `q` in the video window to quit.

## Files

- `main.py` - camera loop, pitch/volume mapping, on-screen overlay.
- `hand_tracker.py` - wraps MediaPipe's `HandLandmarker` task.
- `audio_engine.py` - real-time sine oscillator (`sounddevice` callback with
  smoothed frequency/amplitude to avoid clicks and zipper noise).