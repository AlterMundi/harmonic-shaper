"""Source-owned pluck envelope on top of the sustain envelope.

Behavior contract:
- set_harmonic_trigger(n, amp) commits a pluck target for harmonic n.
- Audio callback chases the target with pluck_attack_s (rising) and
  pluck_release_s (falling). pluck_env decays back to 0 after the target
  is left at 0.
- Phase is never reset by a trigger.
- amp=0 only zeros the target; the sustain envelope (harmonic_envelope)
  is independent and unaffected.
- Repeated triggers add (with clamp) — the env never resets on a
  re-trigger; the existing tail decays naturally.
"""

from __future__ import annotations

import math

import numpy as np

from harmonic_shaper.state import VoiceParameterStore


SR = 48000
BLOCK = 256


def _silence_engine(store: VoiceParameterStore):
    """Build a headless AudioEngine that drains one block silently. We only
    exercise the state-level envelope ramps here — the audio output path
    is verified end-to-end on the live R24 rig, not in unit tests."""
    from harmonic_shaper.audio_engine import AudioEngine

    return AudioEngine(store, sample_rate=SR, block_size=BLOCK)


def test_set_trigger_commits_target_and_clamps_to_unit_interval():
    s = VoiceParameterStore()
    s.set_harmonic_trigger(1, 1.5)
    assert s._pluck_target[1] == 1.0
    s.set_harmonic_trigger(2, -0.5)
    assert s._pluck_target[2] == 0.0
    s.set_harmonic_trigger(3, 0.42)
    assert s._pluck_target[3] == 0.42


def test_trigger_does_not_touch_sustain_envelope():
    s = VoiceParameterStore()
    s.set_harmonic_envelope(1, 0.7)
    pre_env_target = s._voices[1].gain
    s.set_harmonic_trigger(1, 0.9)
    # Sustain envelope gain and voice_id are unchanged.
    assert s._voices[1].gain == pre_env_target == 0.7
    assert s._pluck_target[1] == 0.9
    s.set_harmonic_trigger(1, 0.0)
    # Setting trigger to 0 zeros the pluck target only; the sustain
    # envelope stays at 0.7 (not released).
    assert s._pluck_target[1] == 0.0
    assert s._voices[1].gain == 0.7
    assert s._voices[1].active is True


def test_trigger_to_zero_does_not_release_sustain():
    s = VoiceParameterStore()
    s.set_harmonic_envelope(1, 0.6)
    assert s._voices[1].active is True
    s.set_harmonic_trigger(1, 0.0)
    # Sustain is unaffected by trigger=0.
    assert s._voices[1].active is True
    assert s._voices[1].gain == 0.6


def test_pluck_envelope_attack_rises():
    s = VoiceParameterStore()
    s.set_pluck_attack(0.010)
    s.set_pluck_release(0.250)
    s.set_harmonic_trigger(1, 1.0)
    eng = _silence_engine(s)
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    eng._audio_callback(out, BLOCK, None, None)
    # After 5.33 ms of audio (one block), pluck_env should be roughly halfway.
    pluck_env, _ = s.get_pluck_state(1)
    assert pluck_env > 0.4
    assert pluck_env <= 1.0


def test_pluck_envelope_release_falls_after_target_zero():
    s = VoiceParameterStore()
    s.set_pluck_attack(0.005)
    s.set_pluck_release(0.050)
    s.set_harmonic_trigger(1, 1.0)
    eng = _silence_engine(s)
    # Drive the attack up over a few blocks (~15 ms at 5.33 ms/block).
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    for _ in range(3):
        eng._audio_callback(out, BLOCK, None, None)
    pluck_env, pluck_target = s.get_pluck_state(1)
    assert pluck_env > 0.9
    # Now zero the target and confirm the env decays.
    s.set_harmonic_trigger(1, 0.0)
    for _ in range(3):
        eng._audio_callback(out, BLOCK, None, None)
    pluck_env_after, pluck_target_after = s.get_pluck_state(1)
    assert pluck_target_after == 0.0
    assert pluck_env_after < pluck_env


def test_repeated_triggers_add_with_clamp():
    s = VoiceParameterStore()
    s.set_pluck_attack(0.005)
    s.set_pluck_release(0.250)
    eng = _silence_engine(s)
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    # Two triggers close in time: the env chases the new target without
    # resetting; if the second is higher it adds (up to the target).
    s.set_harmonic_trigger(1, 0.5)
    eng._audio_callback(out, BLOCK, None, None)
    pluck_env_first, _ = s.get_pluck_state(1)
    s.set_harmonic_trigger(1, 0.9)
    eng._audio_callback(out, BLOCK, None, None)
    pluck_env_second, _ = s.get_pluck_state(1)
    # The env either rose (added), or stayed at the first ramp position
    # if it had already reached 0.5 before the second trigger landed.
    assert pluck_env_second >= pluck_env_first * 0.99
    assert pluck_env_second <= 0.9


def test_trigger_on_inactive_voice_claims_it_without_releasing_sustain():
    s = VoiceParameterStore()
    # No sustain envelope; voice is inactive.
    assert s._voices.get(1) is None or s._voices[1].active is False
    s.set_harmonic_trigger(1, 0.5)
    # Pluck alone activates the voice so the audio callback can ramp the
    # pluck envelope on it. The voice is owned by the envelope id.
    assert s._voices[1].active is True
    assert s._voices[1].voice_id == s._envelope_voice_id(1)
    assert s._pluck_target[1] == 0.5


def test_phase_unchanged_across_trigger():
    s = VoiceParameterStore()
    s.set_harmonic_envelope(1, 0.6)
    phase_before = s._voices[1].phase
    s.set_harmonic_trigger(1, 0.9)
    assert s._voices[1].phase == phase_before
    s.set_harmonic_trigger(1, 0.0)
    assert s._voices[1].phase == phase_before


def test_pluck_attack_and_release_configurable():
    s = VoiceParameterStore()
    s.set_pluck_attack(0.5)
    s.set_pluck_release(0.7)
    assert s.get_pluck_attack() == 0.5
    assert s.get_pluck_release() == 0.7
    # Negative values clamp to 0 (no envelope ramp, instant jump).
    s.set_pluck_attack(-0.1)
    s.set_pluck_release(-1.0)
    assert s.get_pluck_attack() == 0.0
    assert s.get_pluck_release() == 0.0


def test_regression_envelope_tests_still_green():
    """Sanity: set_harmonic_envelope semantics untouched by the pluck addition."""
    s = VoiceParameterStore()
    s.set_harmonic_envelope(1, 0.8)
    assert s._voices[1].active is True
    assert s._voices[1].gain == 0.8
    s.set_harmonic_envelope(1, 0.0)
    assert s._voices[1].active is False
    # Pluck state untouched.
    assert s._pluck_target.get(1, 0.0) == 0.0
    assert s._pluck_env.get(1, 0.0) == 0.0