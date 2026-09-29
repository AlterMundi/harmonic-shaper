"""Atomic, leased control frames for the local movement laboratory.

Owns voice IDs 7101..7132 only, never panics unrelated voices. Validation is
completed before acquiring the store lock; the callback expires a lost owner.
"""
import math
import re
import time

from .state import VoiceParams


class LaboratoryConflict(ValueError):
    pass


class LaboratoryInput:
    def __init__(self, store):
        self.store = store
        self.owner = None
        self.sequence = -1
        self.expires_at = 0.
        self.owned = set()

    @staticmethod
    def validate(body):
        if not isinstance(body, dict) or set(body) != {"schema_version", "owner", "sequence", "lease_ms", "voices"}:
            raise ValueError("expected version, owner, sequence, lease_ms and voices")
        if type(body["schema_version"]) is not int or body["schema_version"] != 1:
            raise ValueError("unsupported laboratory frame version")
        if not isinstance(body["owner"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", body["owner"]):
            raise ValueError("invalid owner")
        if type(body["sequence"]) is not int or body["sequence"] < 0:
            raise ValueError("sequence must be non-negative")
        if type(body["lease_ms"]) is not int or not 100 <= body["lease_ms"] <= 2000:
            raise ValueError("lease_ms must be 100..2000")
        if not isinstance(body["voices"], list) or len(body["voices"]) > 32:
            raise ValueError("at most 32 voices")
        limits = {"frequency_hz": (1, 20000), "gain": (0, 1), "phase_deg": (-720, 720),
                  "pan": (-1, 1), "shape": (0, 1), "release_s": (0, 2)}
        seen = set()
        prepared = []
        for voice in body["voices"]:
            if not isinstance(voice, dict) or set(voice) != {"id", *limits}:
                raise ValueError("invalid voice fields")
            n = voice["id"]
            if type(n) is not int or not 1 <= n <= 32 or n in seen:
                raise ValueError("voice IDs must be distinct and in 1..32")
            seen.add(n)
            for key, (low, high) in limits.items():
                value = voice[key]
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"{key}: expected {low}..{high}")
            prepared.append(VoiceParams(harmonic_n=n, voice_id=7100+n, freq=float(voice["frequency_hz"]),
                                        gain=float(voice["gain"]), phase=math.radians(voice["phase_deg"]),
                                        pan=float(voice["pan"]), shape=float(voice["shape"]), attack_s=0.,
                                        release_s=float(voice["release_s"]), active=voice["gain"]>0,
                                        envelope_profile="laboratory"))
        return prepared

    def _release(self):
        for n in self.owned:
            voice = self.store._voices.get(n)
            if voice is not None and voice.voice_id == 7100+n:
                voice.active = False
                if n in self.store._active_history:
                    self.store._active_history.remove(n)
        self.owned.clear()
        self.store._recompute_poly_gains()

    def expire(self, now=None):
        now = time.monotonic() if now is None else now
        with self.store._lock:
            if self.owner is not None and now >= self.expires_at:
                self._release()
                self.owner, self.sequence = None, -1

    def release_seconds(self, harmonic_n, voice_id):
        """Include a release-time edit arriving with the zero-gain frame."""
        with self.store._lock:
            voice = self.store._voices.get(harmonic_n)
            if voice_id == 7100+harmonic_n and voice is not None and voice.voice_id == voice_id:
                return voice.release_s
            return None

    def apply(self, body, now=None):
        prepared = self.validate(body)
        now = time.monotonic() if now is None else now
        with self.store._lock:
            self.expire(now)
            if self.owner is not None and self.owner != body["owner"]:
                raise LaboratoryConflict("another laboratory owner holds the lease")
            if self.owner == body["owner"] and body["sequence"] <= self.sequence:
                raise LaboratoryConflict("stale control frame")
            for voice in prepared:
                existing = self.store._voices.get(voice.harmonic_n)
                if existing is not None and existing.active and existing.voice_id != voice.voice_id:
                    raise LaboratoryConflict(f"harmonic {voice.harmonic_n} belongs to another controller")
            requested = {v.harmonic_n for v in prepared}
            for n in self.owned-requested:
                voice = self.store._voices.get(n)
                if voice is not None and voice.voice_id == 7100+n:
                    voice.active = False
                    if n in self.store._active_history:
                        self.store._active_history.remove(n)
            for voice in prepared:
                n = voice.harmonic_n
                existing = self.store._voices.get(n)
                # Retain the last nonzero gain while the engine renders release.
                if not voice.active and existing is not None and existing.voice_id == voice.voice_id:
                    existing.active = False
                    existing.release_s = voice.release_s
                else:
                    self.store._voices[n] = voice
                    self.store._voice_base_gain[n] = voice.gain
                if voice.active and n not in self.store._active_history:
                    self.store._active_history.append(n)
                elif not voice.active and n in self.store._active_history:
                    self.store._active_history.remove(n)
            self.store._recompute_poly_gains()
            self.owned = requested
            self.owner, self.sequence = body["owner"], body["sequence"]
            self.expires_at = now+body["lease_ms"]/1000
        self.store._notify()
        return {"applied_sequence": self.sequence, "owner": self.owner}
