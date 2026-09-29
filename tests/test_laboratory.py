from types import SimpleNamespace
import math

import numpy as np
import pytest

from harmonic_shaper.audio_engine import AudioEngine
from harmonic_shaper.audio_levels import soft_limit
from harmonic_shaper.laboratory import LaboratoryInput, LaboratoryConflict
from harmonic_shaper.state import VoiceParameterStore


def control(sequence=0, n=6):
    return dict(schema_version=1,owner="test",sequence=sequence,lease_ms=500,
                voices=[dict(id=i,frequency_hz=40.4*i,gain=.4,phase_deg=i*10.,pan=0.,shape=0.,release_s=.2)
                        for i in range(1,n+1)])


def test_frames_are_atomic_reject_stale_and_preserve_unrelated_voices():
    store=VoiceParameterStore();lab=LaboratoryInput(store)
    lab.apply(control(),now=0.)
    before=store.to_dict()
    bad=control(1);bad['voices'][5]['gain']=float('nan')
    with pytest.raises(ValueError):lab.apply(bad,now=.1)
    assert store.to_dict()==before
    with pytest.raises(LaboratoryConflict):lab.apply(control(),now=.1)
    other=control(1);other['owner']='other'
    with pytest.raises(LaboratoryConflict):lab.apply(other,now=.1)
    store.voice_on(32,999,1000.,.2)
    lab.expire(.6)
    assert set(store.get_snapshot())=={32}
    assert store.get_snapshot()[32].voice_id==999


def test_zero_gain_frame_can_request_immediate_silence():
    store=VoiceParameterStore();lab=LaboratoryInput(store);store.laboratory_input=lab
    lab.apply(control(n=1))
    engine=AudioEngine(store,sample_rate=48000,block_size=256)
    output=np.zeros((256,2),dtype=np.float32)
    engine._audio_callback(output,256,None,None)
    assert np.max(np.abs(output))>0
    stop=control(1,n=1);stop['voices'][0].update(gain=0.,release_s=0.)
    lab.apply(stop)
    engine._audio_callback(output,256,None,None)
    assert np.max(np.abs(output))==0
    assert not engine.voice_frame()['voices']


@pytest.mark.parametrize('count',[1,6,32])
def test_telemetry_reconstructs_unshaped_audio_including_effective_gain_and_phase(count):
    store=VoiceParameterStore();lab=LaboratoryInput(store)
    store.laboratory_input=lab
    lab.apply(control(n=count))
    engine=AudioEngine(store,sample_rate=48000,block_size=256)
    engine._running=True
    output=np.zeros((256,2),dtype=np.float32)
    engine._audio_callback(output,256,SimpleNamespace(outputBufferDacTime=12.),None)
    frame=engine.voice_frame()
    assert frame['sample_index']==0 and len(frame['voices'])==count
    assert frame['control_owner']=='test' and frame['control_sequence']==0
    sampled=frame['control_sampled_monotonic_s']
    assert sampled >= frame['control_applied_monotonic_s']
    reconstructed=np.zeros_like(output)
    t=np.arange(256)/48000
    for voice in frame['voices']:
        wave=np.sin(2*np.pi*voice['frequency_hz']*t+voice['phase_rad'])*voice['gain']
        angle=(voice['pan']+1)*np.pi/4
        reconstructed[:,0]+=wave*np.cos(angle)
        reconstructed[:,1]+=wave*np.sin(angle)
    assert output==pytest.approx(soft_limit(reconstructed),abs=2e-7)
    first_phase=frame['voices'][0]['phase_rad']
    engine._audio_callback(output,256,None,None)
    frame=engine.voice_frame()
    assert frame['sample_index']==256
    assert frame['control_sampled_monotonic_s']==sampled
    assert frame['voices'][0]['phase_rad']==pytest.approx((first_phase+2*np.pi*40.4*256/48000)%(2*np.pi))
    # Snapshot consumers cannot mutate callback state.
    frame['voices'][0]['gain']=100.
    assert engine.voice_frame()['voices'][0]['gain']<1


def test_lease_loss_releases_through_engine_and_telemetry_includes_tails(monkeypatch):
    store=VoiceParameterStore();lab=LaboratoryInput(store);store.laboratory_input=lab
    lab.apply(control(n=1),now=0.)
    engine=AudioEngine(store,sample_rate=48000,block_size=256)
    output=np.zeros((256,2),dtype=np.float32)
    monkeypatch.setattr('harmonic_shaper.audio_engine.time.monotonic',lambda:.01)
    engine._audio_callback(output,256,None,None)
    assert engine.voice_frame()['voices'][0]['envelope']==1
    monkeypatch.setattr('harmonic_shaper.audio_engine.time.monotonic',lambda:.6)
    engine._audio_callback(output,256,None,None)
    frame=engine.voice_frame()
    assert frame['voices'][0]['releasing'] and 0<frame['voices'][0]['envelope']<1
    for _ in range(40):engine._audio_callback(output,256,None,None)
    assert engine.voice_frame()['voices']==[]


def test_http_surface_applies_frames_and_never_fakes_disabled_audio():
    from fastapi.testclient import TestClient
    from harmonic_shaper.api import create_app
    store=VoiceParameterStore()
    with TestClient(create_app(store)) as client:
        response=client.post('/api/laboratory/frame',json=control())
        assert response.status_code==200 and response.json()['applied_sequence']==0
        assert client.post('/api/laboratory/frame',json=control()).status_code==409
        bad=control(1);bad['voices'][0]['phase_deg']='untyped'
        assert client.post('/api/laboratory/frame',json=bad).status_code==422
        assert client.get('/api/audio/voices').status_code==503
