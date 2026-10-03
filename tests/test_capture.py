import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import threading

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from harmonic_shaper.audio_engine import AudioEngine
from harmonic_shaper.capture import PCMCapture
from harmonic_shaper.state import VoiceParameterStore
from harmonic_shaper.api import create_app


def engine():
    store=VoiceParameterStore()
    store.voice_on(1,7101,200.,.8)
    return AudioEngine(store,sample_rate=48000,block_size=256)


def test_capture_exactly_matches_emitted_post_limiter_samples_and_clock(tmp_path):
    e=engine();e.render_block(now=0)
    assert not list(tmp_path.iterdir()) # no implicit recording
    before_limiter=[];e.attach_recorder(before_limiter)
    e.start_capture(tmp_path,max_seconds=.1)
    emitted=[]
    for i in range(30): emitted.append(e.render_block(now=(i+1)*256/48000))
    state=e.stop_capture()
    assert state['status']=='complete' and not state['writer_alive']
    import hashlib
    assert state['code']['files']['audio_engine.py']==hashlib.sha256(
        (Path(__file__).parents[1]/'src/harmonic_shaper/audio_engine.py').read_bytes()).hexdigest()
    assert state['written_samples']==4800 and state['first_sample_index']==256
    samples,sr=sf.read(Path(state['directory'])/'audio.wav',dtype='float32',always_2d=True)
    np.testing.assert_array_equal(samples,np.concatenate(emitted)[:4800])
    assert not np.array_equal(samples,np.concatenate(before_limiter)[:4800])
    frames=[json.loads(line) for line in (Path(state['directory'])/'blocks.jsonl').read_text().splitlines()]
    assert frames[0]['sample_index']==256 and frames[-1]['capture_frames']==192
    assert all(f['audio_stage']=='post_shape_master_soft_limiter' for f in frames)
    assert frames[0]['stage']=='oscillators_pre_shape_limiter'
    assert state['last_sample_exclusive']==5056
    manifest=json.loads((Path(state['directory'])/'manifest.json').read_text())
    assert manifest['status']=='complete' and not manifest['writer_alive']


def test_overflow_is_bounded_explicit_and_never_blocks_audio(tmp_path):
    ready=threading.Event();release=threading.Event()
    original=sf.SoundFile.write
    def blocked(self, data):
        ready.set();assert release.wait(3);return original(self,data)
    e=engine()
    with patch.object(sf.SoundFile,'write',blocked):
        e.start_capture(tmp_path,queue_blocks=4)
        first=e.render_block(now=0);assert ready.wait(1)
        for i in range(1,20):
            pcm=e.render_block(now=i*256/48000)
            assert np.max(np.abs(pcm))>0
        assert e.capture_state()['dropped_blocks']==1
        assert e.capture_state()['queued_blocks']<=4
        release.set();state=e.stop_capture()
    assert state['status']=='failed' and 'overflow' in state['error']
    assert state['written_samples']==5*256


def test_disk_failure_and_early_stop_leave_audio_alive(tmp_path):
    e=engine();e.start_capture(tmp_path)
    with patch.object(sf.SoundFile,'write',side_effect=OSError('disk full')):
        e.render_block(now=0)
        e._capture.thread.join(2)
    state=e.stop_capture()
    assert state['status']=='failed' and 'disk full' in state['error']
    assert np.max(np.abs(e.render_block(now=.1)))>0
    e.start_capture(tmp_path);e.render_block(now=.2);state=e.stop_capture()
    assert state['status']=='complete' and state['written_samples']==256


def test_capture_api_is_explicit_validated_and_requires_running_audio(tmp_path):
    e=engine()
    with TestClient(create_app(e._store,e,capture_root=tmp_path)) as client:
        assert client.get('/api/audio/capture').json()['status']=='idle'
        assert client.post('/api/audio/capture/start',json={}).status_code==503
        e._running=True;e._stream=SimpleNamespace(active=True)
        assert client.post('/api/audio/capture/start',json={'path':'elsewhere'}).status_code==422
        started=client.post('/api/audio/capture/start',json={'max_seconds':1,'queue_blocks':4})
        assert started.status_code==200 and started.json()['status']=='recording'
        assert client.post('/api/audio/capture/start',json={}).status_code==422
        assert client.post('/api/audio/capture/stop',json={'id':'stale'}).status_code==409
        assert e.capture_state()['writer_alive']
        pcm=np.empty((256,2),dtype='float32');e._audio_callback(pcm,256,None,None)
        stopped=client.post('/api/audio/capture/stop').json()
        assert stopped['written_samples']==256 and stopped['status']=='complete'


def test_clock_gap_and_stream_failure_are_explicit(tmp_path):
    e=engine();e.start_capture(tmp_path)
    e.render_block(now=0);e._sample_index+=256;e.render_block(now=.02)
    e._capture.thread.join(2)
    assert 'Discontinuous' in e.stop_capture()['error']
    e.start_capture(tmp_path);e.render_block(now=.03)
    e._running=True;e._on_stream_finished()
    e._capture.thread.join(2)
    assert e.stop_capture()['status']=='failed'
    assert np.max(np.abs(e.render_block(now=.04)))>0


def test_callback_status_fails_capture_without_interrupting_output(tmp_path):
    e=engine();e.start_capture(tmp_path)
    pcm=np.empty((256,2),dtype='float32')
    e._audio_callback(pcm,256,None,'output underflow')
    e._capture.thread.join(2)
    assert np.max(np.abs(pcm))>0
    assert 'underflow' in e.stop_capture()['error']


def test_concurrent_starts_have_one_owner_and_no_orphan_writer(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    e=engine()
    def start():
        try: return e.start_capture(tmp_path)['id']
        except ValueError: return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:start(),range(2)))
    assert sum(v is not None for v in results)==1
    assert len(list(tmp_path.iterdir()))==1
    e.stop_capture()


def test_capture_limits_are_validated_before_creating_files(tmp_path):
    import pytest
    for settings in ({'max_seconds':True},{'queue_blocks':True},{'max_seconds':float('nan')}):
        with pytest.raises(ValueError): PCMCapture(tmp_path,48000,**settings)
    with pytest.raises(ValueError,match='RIFF'): PCMCapture(tmp_path,192000,max_seconds=3600)
    assert not list(tmp_path.iterdir())


def test_owner_retry_is_idempotent_even_after_capture_completes(tmp_path):
    e=engine()
    a=e.start_capture(tmp_path,owner='weaver_test',max_seconds=.1)
    b=e.start_capture(tmp_path,owner='weaver_test',max_seconds=.1)
    assert a['id']==b['id']
    e.stop_capture()
    assert e.start_capture(tmp_path,owner='weaver_test',max_seconds=.1)['id']==a['id']
    assert len(list(tmp_path.iterdir()))==1
