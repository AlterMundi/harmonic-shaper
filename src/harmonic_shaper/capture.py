"""Explicit, bounded post-limiter PCM capture; writer never runs in callback."""
import json
import fcntl
import hashlib
import platform
import re
from pathlib import Path
from collections import deque
import time
import threading
from uuid import uuid4

import numpy as np
import soundfile as sf


class PCMCapture:
    def __init__(self, root, sample_rate, *, max_seconds=120., queue_blocks=128, owner=None):
        if type(max_seconds) not in (int,float) or not np.isfinite(max_seconds) or not .1 <= max_seconds <= 3600:
            raise ValueError('max_seconds must be .1..3600')
        if type(queue_blocks) is not int or not 4 <= queue_blocks <= 1024:
            raise ValueError('queue_blocks must be 4..1024')
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if round(max_seconds*sample_rate)*8 > 2**32-1024:
            raise ValueError("Duration exceeds RIFF WAVE size limit; lower max_seconds")
        if owner is not None and (not isinstance(owner,str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}",owner)):
            raise ValueError("Invalid capture owner")
        package=Path(__file__).resolve().parent
        files={name:hashlib.sha256((package/name).read_bytes()).hexdigest()
               for name in ('capture.py','capture_recovery.py','audio_engine.py','state.py','laboratory.py','audio_levels.py','config.py')}
        self.code={'files':files,'code_sha256':hashlib.sha256(
            json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'python':platform.python_version(),'numpy':np.__version__,'soundfile':sf.__version__}
        self.owner = owner
        self.id = uuid4().hex
        self.folder = Path(root)/self.id
        self.folder.mkdir(mode=0o700,parents=True,exist_ok=False)
        self.sample_rate = sample_rate
        self.maximum = round(max_seconds*sample_rate)
        self.queue = deque(maxlen=queue_blocks)
        self._offer_lock = threading.Lock()
        self.accepting = True
        self.stopping = False
        self.ready = threading.Event()
        self.written = self.accepted = 0
        self.first_sample = self.last_sample = None
        self.error = None
        self.status = 'starting'
        self.dropped_blocks = 0
        self.thread = threading.Thread(target=self._write,daemon=True,name='shaper-pcm-writer')
        self._writer_lock=(self.folder/'writer.lock').open('w+b')
        fcntl.flock(self._writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            (self.folder/'capture.json').write_text(json.dumps(self.snapshot(),indent=2,allow_nan=False))
            self.thread.start()
        except Exception:
            self._writer_lock.close()
            raise
        if not self.ready.wait(5):
            self.error = 'Writer initialization timed out'
            self.accepting = False
            self.stopping = True

    def offer(self, pcm, frame):
        if not self._offer_lock.acquire(blocking=False):
            return  # stop owns the closing boundary; callback never waits
        try:
            self._offer(pcm, frame)
        finally:
            self._offer_lock.release()

    def _offer(self, pcm, frame):
        if not self.accepting:
            return
        count = min(len(pcm), self.maximum-self.accepted)
        if count <= 0:
            self.accepting = False
            self.stopping = True
            return
        entry = dict(frame, audio_stage='post_shape_master_soft_limiter',
                     capture_file_sample_start=self.accepted, capture_frames=count)
        # Single producer (serialized offer), single consumer. CPython deque
        # append/popleft are atomic; no queue mutex or writer wait in callback.
        if len(self.queue) >= self.queue.maxlen:
            self.dropped_blocks += 1
            self.error = 'Capture queue overflow; missing audio block'
            self.accepting = False
            self.stopping = True
            return
        self.queue.append((pcm[:count].copy(),entry))
        self.accepted += count
        if self.accepted >= self.maximum:
            self.accepting = False
            self.stopping = True

    def _write(self):
        try:
            with sf.SoundFile(self.folder/'audio.wav',mode='w',samplerate=self.sample_rate,
                              channels=2,format='WAV',subtype='FLOAT') as audio, (self.folder/'blocks.jsonl').open('w') as blocks:
                self.status = 'recording'
                last_flush=-float('inf')
                self.ready.set()
                while not self.stopping or self.queue:
                    try: pcm,frame = self.queue.popleft()
                    except IndexError:
                        time.sleep(.005)
                        continue
                    if frame['sample_rate'] != self.sample_rate:
                        raise ValueError('Capture sample rate changed')
                    if self.last_sample is not None and frame['sample_index'] != self.last_sample:
                        raise ValueError('Discontinuous audio sample clock')
                    if self.first_sample is None: self.first_sample = frame['sample_index']
                    audio.write(pcm)
                    blocks.write(json.dumps(frame,sort_keys=True,allow_nan=False)+'\n')
                    self.written += len(pcm)
                    self.last_sample = frame['sample_index']+len(pcm)
                    if time.monotonic()-last_flush>=.25:
                        audio.flush();blocks.flush()
                        last_flush=time.monotonic()
                self.status = 'finalizing'
        except Exception as exc:
            self.error = str(exc)
            self.status = 'failed'
        finally:
            self.accepting = False
            self.ready.set()
            try:
                temporary = self.folder/'manifest.tmp'
                terminal = 'failed' if self.error else 'complete'
                temporary.write_text(json.dumps(dict(self.snapshot(),status=terminal,writer_alive=False),indent=2,allow_nan=False))
                temporary.replace(self.folder/'manifest.json')
                self.status = terminal
            except OSError as exc:
                self.error = f'{self.error or ""}; manifest: {exc}'
                self.status = 'failed'
            finally:
                self._writer_lock.close()

    def abort(self, message):
        self.error = message
        self.accepting = False
        self.stopping = True

    def stop(self):
        with self._offer_lock:
            self.accepting = False
            self.stopping = True
        self.thread.join(timeout=5)
        return self.snapshot()

    def snapshot(self):
        return dict(schema_version=1,id=self.id,owner=self.owner,code=self.code,status=self.status,error=self.error,
                    directory=str(self.folder),sample_rate=self.sample_rate,channels=2,
                    subtype='FLOAT',stage='post_shape_master_soft_limiter',
                    accepted_samples=self.accepted,written_samples=self.written,
                    first_sample_index=self.first_sample,last_sample_exclusive=self.last_sample,
                    dropped_blocks=self.dropped_blocks,max_samples=self.maximum,
                    queue_blocks=self.queue.maxlen,queued_blocks=len(self.queue),
                    writer_alive=self.thread.is_alive(),
                    limits=['Shaper digital output, not downstream device/mixer sound',
                            'No video or laboratory configuration event log in this audio-only artifact'])
