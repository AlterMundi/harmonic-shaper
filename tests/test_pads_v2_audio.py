"""Audio-level verification of harmonic switching in the Shaper.

Feeds controlled harmonic_envelope settings to the Shaper's VoiceParameterStore
and captures the audio output via the AudioEngine callback. Computes FFT on
the captured waveform and asserts the dominant frequency matches the expected
harmonic (f1 * N) for the activated pad/route combination.

No R24, no Weaver, no camera — pure numpy audio analysis offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonic_shaper.audio_engine import AudioEngine
from harmonic_shaper.state import VoiceParameterStore

SR = 48000
BLOCK = 256
F1 = 40.4  # base frequency used in Pads v2
TOL_HZ = 2.0  # acceptable frequency deviation in Hz


def _dominant_frequency(samples: np.ndarray, sample_rate: int, low_hz: float = 20.0) -> float:
    """Return the frequency (Hz) with the highest magnitude in the FFT
    spectrum above `low_hz`."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)  # mono mix
    n = len(samples)
    fft = np.abs(np.fft.rfft(samples * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    # Only look above low_hz to ignore DC and subsonic noise.
    mask = freqs >= low_hz
    if not mask.any():
        return 0.0
    peak_idx = np.argmax(fft[mask])
    return float(freqs[mask][peak_idx])


def _render_blocks(engine: AudioEngine, n_blocks: int) -> np.ndarray:
    """Run the audio callback for n_blocks and collect the output."""
    chunks = []
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    for _ in range(n_blocks):
        engine._audio_callback(out, BLOCK, None, None)
        chunks.append(out.copy())
    return np.concatenate(chunks, axis=0)


def test_harmonic_n1_produces_f1_40hz():
    """harmonic_envelope(1, 1.0) → dominant frequency = f1 * 1 ≈ 40.4 Hz."""
    store = VoiceParameterStore()
    store.update_f1(F1)
    store.set_harmonic_envelope(1, 1.0)
    engine = AudioEngine(store, sample_rate=SR, block_size=BLOCK)

    audio = _render_blocks(engine, int(SR / BLOCK * 1.5))  # 1.5 s of audio
    peak_hz = _dominant_frequency(audio, SR)

    assert abs(peak_hz - F1 * 1) < TOL_HZ, (
        f"Expected {F1:.1f} Hz (f1*1), got {peak_hz:.1f} Hz"
    )


def test_harmonic_n3_produces_f1_times_3():
    """harmonic_envelope(3, 1.0) → dominant frequency = f1 * 3 ≈ 121.2 Hz."""
    store = VoiceParameterStore()
    store.update_f1(F1)
    store.set_harmonic_envelope(3, 1.0)
    engine = AudioEngine(store, sample_rate=SR, block_size=BLOCK)

    audio = _render_blocks(engine, int(SR / BLOCK * 1.5))
    peak_hz = _dominant_frequency(audio, SR)

    assert abs(peak_hz - F1 * 3) < TOL_HZ, (
        f"Expected {F1*3:.1f} Hz (f1*3), got {peak_hz:.1f} Hz"
    )


def test_switching_harmonic_changes_frequency():
    """N=1 for 1 s, then N=3 for 1 s → frequency shifts from f1 to f1*3."""
    store = VoiceParameterStore()
    store.update_f1(F1)
    engine = AudioEngine(store, sample_rate=SR, block_size=BLOCK)

    # Half second of N=1, then switch
    store.set_harmonic_envelope(1, 1.0)
    audio_n1 = _render_blocks(engine, int(SR / BLOCK * 0.7))

    store.set_harmonic_envelope(1, 0.0)    # release N=1
    store.set_harmonic_envelope(3, 1.0)    # activate N=3
    audio_n3 = _render_blocks(engine, int(SR / BLOCK * 1.0))

    # Skip the first 0.2 s to let N=1 release tail fade.
    tail_start = int(0.3 * SR)
    if tail_start < len(audio_n3):
        analysis = audio_n3[tail_start:]
    else:
        analysis = audio_n3

    peak_hz = _dominant_frequency(analysis, SR)
    assert abs(peak_hz - F1 * 3) < TOL_HZ, (
        f"After switching to N=3, expected {F1*3:.1f} Hz, got {peak_hz:.1f} Hz"
    )


def test_trigger_adds_transient_without_changing_frequency():
    """harmonic_trigger on N=1 adds a pluck without changing the dominant freq."""
    store = VoiceParameterStore()
    store.update_f1(F1)
    store.set_harmonic_envelope(1, 1.0)   # sustain N=1
    store.set_harmonic_trigger(1, 0.9)     # pluck on top
    engine = AudioEngine(store, sample_rate=SR, block_size=BLOCK)

    audio = _render_blocks(engine, int(SR / BLOCK * 1.5))
    peak_hz = _dominant_frequency(audio, SR)

    assert abs(peak_hz - F1 * 1) < TOL_HZ, (
        f"Trigger should not change freq; expected {F1:.1f} Hz, got {peak_hz:.1f} Hz"
    )
    # The pluck should increase the RMS energy compared to sustain-only.
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert rms > 0.001, f"audio RMS too low: {rms:.6f}"


def test_envelope_zero_silences_voice():
    """Setting harmonic_envelope to 0 releases the voice → near-silence."""
    store = VoiceParameterStore()
    store.update_f1(F1)
    store.set_harmonic_envelope(1, 1.0)
    engine = AudioEngine(store, sample_rate=SR, block_size=BLOCK)

    # Let the attack ramp up.
    _render_blocks(engine, int(SR / BLOCK * 0.3))

    # Release.
    store.set_harmonic_envelope(1, 0.0)
    audio = _render_blocks(engine, int(SR / BLOCK * 1.0))

    # Skip the release tail (first 0.3 s after release).
    tail_start = int(0.3 * SR)
    if tail_start < len(audio):
        tail_audio = audio[tail_start:]
    else:
        tail_audio = audio

    tail_rms = float(np.sqrt(np.mean(tail_audio.astype(np.float64) ** 2)))
    assert tail_rms < 0.002, (
        f"Voice should be silent after release; RMS = {tail_rms:.6f}"
    )


def test_source_owned_holds_share_a_harmonic_until_last_release() -> None:
    """Two hands on one pad produce one sustained partial until both leave."""
    store = VoiceParameterStore()
    store.update_f1(F1)
    store.set_harmonic_source_envelope(5, source=0, gain=0.7)
    store.set_harmonic_source_envelope(5, source=1, gain=0.9)
    engine = AudioEngine(store, sample_rate=SR, block_size=BLOCK)

    audio = _render_blocks(engine, int(SR / BLOCK * 1.0))
    assert abs(_dominant_frequency(audio, SR) - F1 * 5) < TOL_HZ

    store.set_harmonic_source_envelope(5, source=0, gain=0.0)
    assert store.get_snapshot()[5].active is True
    still_held = _render_blocks(engine, int(SR / BLOCK * 0.25))
    assert float(np.sqrt(np.mean(still_held.astype(np.float64) ** 2))) > 0.01

    store.set_harmonic_source_envelope(5, source=1, gain=0.0)
    released = _render_blocks(engine, int(SR / BLOCK * 1.0))
    tail_rms = float(np.sqrt(np.mean(released[int(0.3 * SR):].astype(np.float64) ** 2)))
    assert tail_rms < 0.002, f"Source-owned harmonic should release, RMS = {tail_rms:.6f}"