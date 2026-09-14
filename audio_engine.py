"""Real-time sine-wave synthesizer driven by a continuously updated target
frequency and amplitude. Runs on a background audio thread via sounddevice;
the main (video) thread just calls set_frequency()/set_amplitude()."""

import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
BLOCK_SIZE = 256

# How quickly the audio catches up to new target values, in seconds.
# Small values avoid clicks/zipper noise when the hand moves fast.
FREQ_SMOOTHING_SEC = 0.05
AMP_SMOOTHING_SEC = 0.03


class SineSynth:
    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size

        self._lock = threading.Lock()
        self._target_freq = 220.0
        self._target_amp = 0.0

        self._current_freq = 220.0
        self._current_amp = 0.0
        self._phase = 0.0

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def set_frequency(self, freq_hz: float) -> None:
        with self._lock:
            self._target_freq = float(freq_hz)

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
            target_freq = self._target_freq
            target_amp = self._target_amp

        freq_alpha = _smoothing_alpha(FREQ_SMOOTHING_SEC, self.sample_rate)
        amp_alpha = _smoothing_alpha(AMP_SMOOTHING_SEC, self.sample_rate)

        freqs = np.empty(frames, dtype=np.float64)
        amps = np.empty(frames, dtype=np.float64)
        freq = self._current_freq
        amp = self._current_amp
        for i in range(frames):
            freq += (target_freq - freq) * freq_alpha
            amp += (target_amp - amp) * amp_alpha
            freqs[i] = freq
            amps[i] = amp
        self._current_freq = freq
        self._current_amp = amp

        phase_inc = 2.0 * np.pi * freqs / self.sample_rate
        phases = self._phase + np.cumsum(phase_inc)
        self._phase = phases[-1] % (2.0 * np.pi)

        samples = np.sin(phases) * amps
        outdata[:, 0] = samples.astype(np.float32)


def _smoothing_alpha(smoothing_sec: float, sample_rate: int) -> float:
    """One-pole smoothing coefficient for a given time constant, per-sample."""
    if smoothing_sec <= 0:
        return 1.0
    return 1.0 - np.exp(-1.0 / (smoothing_sec * sample_rate))
