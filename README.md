# harmonic-shaper

The canonical additive harmonic synthesizer of the Harmonic Beacon ecosystem:
32 voices + waveshaper + per-voice LFO + sidechain.

This package is the reconciled standalone extraction of the evolved Shaper in
`digital-beacon`. The older `NaturalHarmony/harmonic_shaper` implementation is
historical; its Minilab3 controller was recovered here because it had been lost
from the fork.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

PortAudio must be available on the host for real-time output through
`sounddevice`.

## Run

```bash
python -m harmonic_shaper
# or, after installation:
harmonic-shaper
```

The default process starts the audio engine, native OSC listener, MIDI
controllers when present (Launchpad, Minilab3, and a **native keyboard note
source**), and the FastAPI state service.

Standalone mode defaults to sequential natural-harmonic banks: adjacent keys
select `n=1..32` at exact `f1*n` frequencies. The lower configured bank is
momentary; the upper bank is toggle/sustain. No 12-TET pitch adaptation is
applied. See [Native MIDI Harmonic Banks](docs/NATIVE_MIDI_HARMONIC_BANKS.md)
for configuration and controller calibration. No NaturalHarmony beacon is
required. Useful flags:

```bash
python -m harmonic_shaper --slave            # opt in to NH /beacon/* broadcasts
python -m harmonic_shaper --f1 40.40 --anchor 24
python -m harmonic_shaper --native-midi-momentary-start 24 --native-midi-toggle-start 72
python -m harmonic_shaper --native-midi-mode legacy_hybrid
python -m harmonic_shaper --no-native-midi   # keyboards off; pads/CC still on
python -m harmonic_shaper --no-audio         # headless control/API process
python -m harmonic_shaper --no-midi
python -m harmonic_shaper --no-api
python -m harmonic_shaper --help
```

Default bindings:

- UDP `:9002`: current v1 wire protocol under `/digital/*`.
- UDP `:9001`: optional `/beacon/*` slave input, only with `--slave`.
- HTTP `127.0.0.1:8080`: `GET /api/state`, `POST /api/shaper/*`, and
  WebSocket `/ws`.

`/shaper/*` is the planned native namespace. It is intentionally not mapped on
the wire yet because renaming `/digital/*` requires a contract version bump.

## Pure reference renderer

The extracted offline reference is also installable:

```bash
harmonic-shaper-synth-pure input.wav --out output.wav
```

It retains the fork's voice-analysis and NumPy rendering path for the clipping
work tracked after this extraction.

## Test

```bash
pytest -q
```

See [the extraction report](docs/T2.2_EXTRACTION_REPORT.md) for the module map,
fork reconciliation, dependency audit, and clipping notes.

## License

MIT — see [LICENSE](LICENSE).

The movement laboratory can render PCM without opening an audio device using
`AudioEngine.render_block(now=<logical_seconds>)`. It runs the same kernel as
the production callback, with the configured block size and a caller-owned
monotonic clock. Submit `LaboratoryInput` controls with that same logical time;
leases and envelopes retain their production behavior. Each offline run owns a
fresh store and engine. Rendering on an engine that is running a stream is
rejected. Audio remains stereo float32 after shaping/master/soft limiting;
`voice_frame()` describes oscillators before shaping/limiting, not the PCM.

## Optional laboratory PCM capture (development)

The capture tap is **after** the soft limiter and copies the exact float32
samples assigned to PortAudio. It is separate from the legacy pre-limiter
`attach_recorder` hook. Nothing is recorded until an explicit start.

- `POST /api/audio/capture/start`: `{"max_seconds":120,"queue_blocks":128}`.
  Requires a running audio engine; rejects overlapping starts and invalid limits.
  Optional `owner` is an opaque 1..80-character alphanumeric/underscore/hyphen
  nonce. Repeating that owner with the same settings returns its existing capture
  (also after completion), allowing recovery of a lost start acknowledgement.
  Reusing it with different settings is rejected; it is not authentication.
- `GET /api/audio/capture`: progress/error, sample bounds, queue size and output
  directory. `POST /api/audio/capture/stop`: optionally `{"id":"<capture-id>"}`;
  a stale identifier cannot stop a newer capture.
- Default private output: `~/.local/share/harmonic-shaper/laboratory-captures/<id>/`.
  `audio.wav` is stereo float after shape/master/limiter; `blocks.jsonl` preserves
  sample index, callback monotonic/DAC timestamps, pre-shape voices and crop size.
  `manifest.json` is written atomically after the files close.

A bounded single-producer/single-consumer deque holds copied blocks. The callback
never waits for disk or queue capacity; the writer runs separately. Queue overflow,
clock discontinuity, sample-rate changes, callback status or stream failure mark
capture failed. Overflow/disk failure do not stop synthesis. Duration ends exactly
at the requested sample count; early stop closes the current contiguous interval.
The RIFF size limit is validated before starting. Start/stop ownership is serialized
outside the callback. This is Python software, not a hard realtime deadline guarantee.

Captured PCM is Shaper's digital output, excluding downstream mixer/device gain or
other programs. This first foundation does not capture video or laboratory
configuration changes; the Weaver UI/session collector and audiovisual alignment
are subsequent LAB-09 work. No camera capture or retrospective buffer is enabled.
Abrupt process termination can leave an interrupted file: this version does not
claim crash recovery. The running laboratory is not upgraded by checking out this
branch. A change in AudioEngine also changes the strict offline engine hash; use
the original pinned checkout for an old frozen PCM request, or create a new run.

Verification: `pytest tests/test_capture.py tests/test_offline_render.py
 tests/test_laboratory.py tests/test_audio_smoke.py tests/test_pads_v2_audio.py -q`:
32 tests passed. Capture tests cover exact post-limiter samples versus the separate
pre-limiter tap, sample limits/crop, early stop, bounded overflow, disk/clock/stream
errors, stale stop, concurrent starts and validation/API. These are hardware-free
checks; real device latency, audiovisual synchronization and listening remain open.
