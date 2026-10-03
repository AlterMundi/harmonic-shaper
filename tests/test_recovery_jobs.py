import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from harmonic_shaper.api import create_app
from harmonic_shaper.capture import PCMCapture
from harmonic_shaper.recovery_jobs import RecoveryJobs
from harmonic_shaper.state import VoiceParameterStore


def source(root):
    capture=PCMCapture(root,48000)
    capture.offer(np.full((256,2),.25,dtype='float32'),dict(sample_rate=48000,sample_index=0,capture_file_sample_start=0))
    deadline=time.monotonic()+3
    while not capture.written:
        assert time.monotonic()<deadline;time.sleep(.005)
    capture.abort('Synthetic interruption');capture.stop()
    return capture


def test_live_lock_retry_restart_query_and_real_pcm_result(tmp_path,monkeypatch):
    import harmonic_shaper.recovery_jobs as module
    capture=source(tmp_path);original=module.recover_capture
    entered=threading.Event();release=threading.Event();calls=[]
    def paused(folder):
        calls.append(folder);entered.set();assert release.wait(3);return original(folder)
    monkeypatch.setattr(module,'recover_capture',paused)
    service=RecoveryJobs(tmp_path);job_id='a'*32
    try:
        service.start(capture.id,job_id);assert entered.wait(2)
        assert service.read(job_id)['status']=='running'
        restored=RecoveryJobs(tmp_path)
        assert restored.read(job_id)['status']=='running'  # Actual writer lease, no in-memory owner needed.
        assert restored.start(capture.id,job_id)['status']=='running'
        assert len(calls)==1
        with pytest.raises(ValueError,match='another capture'):restored.start('b'*32,job_id)
    finally:
        release.set();service.close()
    saved=RecoveryJobs(tmp_path).read(job_id)
    assert saved['status']=='recovered' and saved['result']['recovered_samples']==256
    assert RecoveryJobs(tmp_path).start(capture.id,job_id)==saved
    assert len(calls)==1 and len(list((capture.folder/'recovered').iterdir()))==1


def test_missing_writer_is_interrupted_not_automatically_restarted(tmp_path):
    service=RecoveryJobs(tmp_path);job_id='a'*32;folder=service.root/job_id;folder.mkdir(parents=True)
    (folder/'job.lock').touch()
    (folder/'job.json').write_text(json.dumps({'schema_version':1,'job_id':job_id,'capture_id':'b'*32,'status':'running'}))
    assert service.read(job_id)['status']=='interrupted'
    assert service.start('b'*32,job_id)['status']=='interrupted'
    assert service.threads=={}


def test_pollable_api_validates_ids_and_recovers_without_second_writer(tmp_path):
    capture=source(tmp_path);body={'id':capture.id,'job_id':'a'*32};url='/api/audio/capture/recovery-jobs'
    with TestClient(create_app(VoiceParameterStore(),capture_root=tmp_path)) as client:
        assert client.get('/api/audio/capture/recovery-contract').json()['pollable_jobs']
        assert client.get(url+'/'+'a'*32).status_code==404
        assert client.post(url,json={**body,'job_id':'../private'}).status_code==422
        assert client.post(url,json=body).status_code==200
        deadline=time.monotonic()+3
        while True:
            r=client.get(url+'/'+body['job_id']).json()
            if r['status'] not in ('queued','running'):break
            assert time.monotonic()<deadline;time.sleep(.005)
        assert r['status']=='recovered'
        assert client.post(url,json=body).json()==r
    with TestClient(create_app(VoiceParameterStore(),capture_root=tmp_path)) as client:
        assert client.get(url+'/'+body['job_id']).json()==r
