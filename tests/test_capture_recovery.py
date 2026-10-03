import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import soundfile as sf

from harmonic_shaper.capture import PCMCapture
from harmonic_shaper.capture_recovery import recover_capture, digest


def offer(capture):
    capture.offer(np.full((256,2),.25,dtype='float32'),
                  dict(sample_rate=48000,sample_index=0,capture_file_sample_start=0))


def test_recovery_rejects_live_writer_and_normally_completed_capture(tmp_path):
    capture=PCMCapture(tmp_path,48000);offer(capture)
    with pytest.raises(ValueError,match='active'):recover_capture(capture.folder)
    capture.stop()
    with pytest.raises(ValueError,match='successfully'):recover_capture(capture.folder)


def test_failed_capture_recovers_prefix_without_changing_raw(tmp_path):
    capture=PCMCapture(tmp_path,48000);offer(capture)
    for _ in range(200):
        if capture.written:break
        time.sleep(.005)
    capture.abort('Synthetic stream interruption');capture.stop()
    before={name:digest(capture.folder/name) for name in ('audio.wav','blocks.jsonl','manifest.json')}
    report=recover_capture(capture.folder)
    assert report['status']=='recovered' and report['recovered_samples']==256
    pcm,sr=sf.read(Path(report['directory'])/'audio.wav',dtype='float32',always_2d=True)
    assert sr==48000;np.testing.assert_array_equal(pcm,np.full((256,2),.25,dtype='float32'))
    assert {name:digest(capture.folder/name) for name in before}==before


def test_process_kill_recovers_only_complete_journal_rows(tmp_path):
    script='''
import sys,time
import numpy as np
from harmonic_shaper.capture import PCMCapture
capture=PCMCapture(sys.argv[1],48000)
print(capture.folder,flush=True)
i=0
while True:
 capture.offer(np.full((256,2),.25,dtype='float32'),dict(sample_rate=48000,sample_index=i*256))
 i+=1;time.sleep(.01)
'''
    process=subprocess.Popen([sys.executable,'-u','-c',script,str(tmp_path)],stdout=subprocess.PIPE,text=True)
    try:
        folder=Path(process.stdout.readline().strip())
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if (folder/'blocks.jsonl').exists() and (folder/'blocks.jsonl').stat().st_size:break
            time.sleep(.01)
        assert (folder/'blocks.jsonl').stat().st_size
        process.kill();process.wait(5)
        with (folder/'blocks.jsonl').open('a') as handle:handle.write('{"partial":')
        before={name:digest(folder/name) for name in ('audio.wav','blocks.jsonl')}
        report=recover_capture(folder)
        assert report['status']=='recovered' and report['recovered_samples']>=256
        assert report['prefix_stop_reason']=='Partial journal row'
        pcm,_=sf.read(Path(report['directory'])/'audio.wav',dtype='float32',always_2d=True)
        assert len(pcm)==report['recovered_samples']
        np.testing.assert_array_equal(pcm,np.full(pcm.shape,.25,dtype='float32'))
        assert {name:digest(folder/name) for name in before}==before
    finally:
        if process.poll() is None:process.kill();process.wait(5)


def test_no_confirmed_blocks_are_not_invented(tmp_path):
    capture=PCMCapture(tmp_path,48000);capture.abort('Interrupted before samples');capture.stop()
    with pytest.raises(ValueError,match='No journal-confirmed'):recover_capture(capture.folder)


def test_recovery_api_only_accepts_known_capture_ids(tmp_path):
    from fastapi.testclient import TestClient
    from harmonic_shaper.api import create_app
    from harmonic_shaper.state import VoiceParameterStore
    capture=PCMCapture(tmp_path,48000);offer(capture)
    with TestClient(create_app(VoiceParameterStore(),capture_root=tmp_path)) as client:
        assert client.post('/api/audio/capture/recover',json={'id':str(capture.folder)}).status_code==422
        assert client.post('/api/audio/capture/recover',json={'id':capture.id}).status_code==422
        for _ in range(200):
            if capture.written:break
            time.sleep(.005)
        capture.abort('Interrupted');capture.stop()
        response=client.post('/api/audio/capture/recover',json={'id':capture.id})
        assert response.status_code==200 and response.json()['status']=='recovered'


def test_recovery_retry_reuses_verified_result_and_rejects_tampering(tmp_path):
    capture=PCMCapture(tmp_path,48000);offer(capture)
    for _ in range(200):
        if capture.written:break
        time.sleep(.005)
    capture.abort('Interrupted');capture.stop()
    a=recover_capture(capture.folder);b=recover_capture(capture.folder)
    assert not a['reused'] and b['reused'] and a['directory']==b['directory']
    assert len(list((capture.folder/'recovered').iterdir()))==1
    (Path(a['directory'])/'audio.wav').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):recover_capture(capture.folder)
    assert len(list((capture.folder/'recovered').iterdir()))==1
