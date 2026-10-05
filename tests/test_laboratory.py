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
        fraction=np.linspace(0.,1.,256)
        gain=voice['gain']+(voice.get('gain_end',voice['gain'])-voice['gain'])*fraction
        wave=np.sin(2*np.pi*voice['frequency_hz']*t+voice['phase_rad']+voice.get('phase_offset_delta_rad',0.)*fraction)*gain
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


def test_lab_gain_phase_and_release_transitions_have_no_boundary_jump():
    store=VoiceParameterStore();lab=LaboratoryInput(store);store.laboratory_input=lab
    engine=AudioEngine(store,sample_rate=48000,block_size=256)
    output=np.zeros((256,2),dtype=np.float32)
    body=control(n=1)
    body['voices'][0].update(frequency_hz=80.,phase_deg=90.,gain=.8,release_s=.012)
    lab.apply(body)
    engine._audio_callback(output,256,None,None)
    assert abs(output[0,0]) < 1e-7  # activation starts at zero, even at peak phase
    previous=output[-1,0]
    for seq,gain,phase in [(1,.1,-90.),(2,.9,150.),(3,0.,150.)]:
        body['sequence']=seq
        body['voices'][0].update(gain=gain,phase_deg=phase)
        lab.apply(body)
        engine._audio_callback(output,256,None,None)
        assert abs(output[0,0]-previous)<.012
        previous=output[-1,0]
    for _ in range(8):
        engine._audio_callback(output,256,None,None)
        assert abs(output[0,0]-previous)<.012
        previous=output[-1,0]
    assert not engine.voice_frame()['voices']
    assert np.max(np.abs(output))==0


def test_updates_of_sustained_voice_preserve_oscillator_and_envelope():
    store=VoiceParameterStore();lab=LaboratoryInput(store);store.laboratory_input=lab
    engine=AudioEngine(store,sample_rate=48000,block_size=256)
    output=np.zeros((256,2),dtype=np.float32)
    state=None
    for sequence in range(60):
        body=control(sequence,n=1)
        body['voices'][0].update(frequency_hz=80.,phase_deg=0.,gain=.2+.1*(sequence%2))
        lab.apply(body)
        previous_phase=engine._voice_state[1]['phase'] if state is not None else 0.
        engine._audio_callback(output,256,None,None)
        if state is None:
            state=engine._voice_state[1]
        assert engine._voice_state[1] is state
        assert state['env']==1.
        assert state['phase']==pytest.approx((previous_phase+2*np.pi*80*256/48000)%(2*np.pi))
        assert not engine.voice_frame()['voices'][0]['releasing']


def test_portaudio_status_is_counted_and_does_not_change_rendered_audio():
    store=VoiceParameterStore();lab=LaboratoryInput(store);store.laboratory_input=lab
    lab.apply(control(n=1),now=0.)
    a=AudioEngine(store,sample_rate=48000,block_size=1024)
    b=AudioEngine(store,sample_rate=48000,block_size=1024)
    a._running=b._running=True
    x=np.zeros((1024,2),dtype=np.float32);y=np.zeros_like(x)
    a._render_into(x,1024,None,SimpleNamespace(output_underflow=True),generated=.01)
    b._render_into(y,1024,None,None,generated=.01)
    assert np.array_equal(x,y)
    old=a.voice_frame()['audio_health'];identity=old['engine_id']
    assert old['output_underflows']==1 and old['status_events']==1
    assert old['last_status_sample_index']==0
    a._render_into(x,1024,None,None,generated=.02)
    assert a.voice_frame()['audio_health']==old
    a._render_into(x,1024,None,SimpleNamespace(output_underflow=False),generated=.03)
    new=a.voice_frame()['audio_health']
    assert new['engine_id']==identity and new['status_events']==2 and new['output_underflows']==1
    assert new['last_status_sample_index']==2048
    old['output_underflows']=500
    assert a.voice_frame()['audio_health']['output_underflows']==1
    assert b.voice_frame()['audio_health']['output_underflows']==0
    fresh=AudioEngine(store)
    assert fresh.voice_frame()['audio_health'] is None
    fresh._running=True
    fresh._render_into(x,1024,None,None,generated=.04)
    assert fresh.voice_frame()['audio_health']['engine_id']!=identity
