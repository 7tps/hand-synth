"""Real-time polyphonic sine-wave synthesizer. Runs on a background audio
thread via sounddevice; the main (video) thread just calls
set_chord()/set_amplitude().

Internally there are always NUM_VOICES oscillators. set_chord() spreads up to
NUM_VOICES frequencies across them, each at an equal share of the overall
amplitude - a single-frequency list plays as one plain tone. Both frequency
and per-voice gain are smoothed so switching between chords ramps rather
than clicks."""

import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
BLOCK_SIZE = 256
NUM_VOICES = 4

# How quickly the audio catches up to new target values, in seconds.
# Small values avoid clicks/zipper noise when the hand moves fast.
FREQ_SMOOTHING_SEC = 0.05
GAIN_SMOOTHING_SEC = 0.05
AMP_SMOOTHING_SEC = 0.03


class SineSynth:
    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size

        self._lock = threading.Lock()
        self._target_freqs = np.full(NUM_VOICES, 220.0)
        self._target_gains = np.zeros(NUM_VOICES)
        self._target_gains[0] = 1.0
        self._target_amp = 0.0

        self._current_freqs = self._target_freqs.copy()
        self._current_gains = self._target_gains.copy()
        self._current_amp = 0.0
        self._phases = np.zeros(NUM_VOICES)

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def set_chord(self, freqs_hz) -> None:
        """Play all given frequencies at once, evenly mixed. A single
        frequency plays as one plain tone."""
        with self._lock:
            n = min(len(freqs_hz), NUM_VOICES)
            for i in range(n):
                self._target_freqs[i] = float(freqs_hz[i])
            self._target_gains[:] = 0.0
            self._target_gains[:n] = 1.0 / n

    def set_amplitude(self, amp: float) -> None:
        with self._lock:
            self._target_amp = float(np.clip(amp, 0.0, 1.0))

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(status)

        with self._lock:
            target_freqs = self._target_freqs.copy()
            target_gains = self._target_gains.copy()
            target_amp = self._target_amp

        freq_alpha = _smoothing_alpha(FREQ_SMOOTHING_SEC, self.sample_rate)
        gain_alpha = _smoothing_alpha(GAIN_SMOOTHING_SEC, self.sample_rate)
        amp_alpha = _smoothing_alpha(AMP_SMOOTHING_SEC, self.sample_rate)

        freqs_buf = np.empty((frames, NUM_VOICES))
        gains_buf = np.empty((frames, NUM_VOICES))
        amps_buf = np.empty(frames)

        freq = self._current_freqs.copy()
        gain = self._current_gains.copy()
        amp = self._current_amp
        for i in range(frames):
            freq += (target_freqs - freq) * freq_alpha
            gain += (target_gains - gain) * gain_alpha
            amp += (target_amp - amp) * amp_alpha
            freqs_buf[i] = freq
            gains_buf[i] = gain
            amps_buf[i] = amp
        self._current_freqs = freq
        self._current_gains = gain
        self._current_amp = amp

        phase_inc = 2.0 * np.pi * freqs_buf / self.sample_rate
        phases = self._phases[np.newaxis, :] + np.cumsum(phase_inc, axis=0)
        self._phases = phases[-1] % (2.0 * np.pi)

        voices = np.sin(phases) * gains_buf
        mixed = voices.sum(axis=1) * amps_buf
        outdata[:, 0] = mixed.astype(np.float32)


def _smoothing_alpha(smoothing_sec: float, sample_rate: int) -> float:
    """One-pole smoothing coefficient for a given time constant, per-sample."""
    if smoothing_sec <= 0:
        return 1.0
    return 1.0 - np.exp(-1.0 / (smoothing_sec * sample_rate))
