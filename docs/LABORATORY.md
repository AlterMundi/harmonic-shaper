# Local laboratory control and effective voice telemetry

Companion to AlterMundi/harmonic-weaver#12 and #30. The existing OSC/MIDI APIs
remain available. The laboratory should launch a dedicated Shaper instance.

`POST /api/laboratory/frame` accepts one validated control generation:

```json
{"schema_version":1,"owner":"session-id","sequence":0,"lease_ms":500,"voices":[{"id":1,"frequency_hz":40.4,"gain":0.2,"phase_deg":0,"pan":0,"shape":0,"release_s":0.15}]}
```

There are at most 32 distinct harmonic IDs (1..32), mapped to owned voice IDs
7101..7132. The complete frame is validated before changing state and installed
under the same store lock used by snapshots. Sequence must increase. Conflicting
owners or active voices belonging to another controller return 409. Invalid
values return 422 without a partial application. An empty frame releases only
owned voices. If the controller disappears, the audio callback expires its lease
and renders the configured release; it does not panic unrelated voices.

`GET /api/audio/voices` reports the latest completed audio-block snapshot:

- sample index/rate and block size;
- callback monotonic time and PortAudio output-buffer DAC time when available;
- actual oscillator frequency, integrated phase (radians), effective gain,
  envelope, pan, shape and whether each voice is releasing;
- `stage: oscillators_pre_shape_limiter` and actual engine running state.

Gain includes polyphonic normalization, envelope/pluck, master and sidechain.
A phase is anchored at the first sample of the reported block. This is not a
PCM tap: shaping and the output limiter can add harmonics not represented by
these oscillator phasors. The figure consumer must label that distinction.
No serialization/network I/O occurs inside the callback; the API copies its
latest snapshot. No audio engine yields 503, not simulated phases.

Hardware-free verification reconstructs the unshaped PCM from telemetry for
1, 6 and 32 voices, including phase continuity and gains, then compares after
the same limiter. Further tests cover atomic validation, stale/foreign control,
lease expiry, release tails and HTTP behavior. These do not establish physical
output latency or human listening acceptance.
# Rendered control identity

Voice telemetry additionally includes `control_owner`, `control_sequence`,
`control_applied_monotonic_s` and `control_sampled_monotonic_s`. Parameter snapshot
and identity are read under the same store lock. The sampled timestamp is the
first audio callback consuming that control frame and remains stable for its
subsequent blocks. An HTTP acknowledgement alone does not establish that audio
has consumed a revision. These timestamps measure host software, not physical
speaker latency.
