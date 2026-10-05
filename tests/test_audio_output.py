import pytest
from fastapi.testclient import TestClient
from harmonic_shaper import audio_engine as module
from harmonic_shaper.audio_engine import AudioEngine, AudioOutputConflict
from harmonic_shaper.state import VoiceParameterStore
from harmonic_shaper.api import create_app


class Devices:
    def __init__(self):
        self.opened=[];self.closed=[];self.fail=None;self.checked=[]
    def query_hostapis(self,index=None):
        values=[{'name':'JACK'},{'name':'native'}]
        return values if index is None else values[index]
    def query_devices(self,device=None,kind=None):
        values=[{'name':'R24','max_output_channels':2,'hostapi':0,'default_samplerate':48000},
                {'name':'Native','max_output_channels':2,'hostapi':1,'default_samplerate':48000}]
        if kind:return values[device or 0]
        return values
    def check_output_settings(self,**kwargs):self.checked.append(kwargs)
    def OutputStream(self,**kwargs):
        owner=self
        class Stream:
            active=True
            def start(self):
                if kwargs['device']==owner.fail:raise RuntimeError('open failed')
                owner.opened.append(kwargs)
            def stop(self):pass
            def close(self):owner.closed.append(kwargs)
        return Stream()


@pytest.fixture
def audio(monkeypatch):
    devices=Devices()
    monkeypatch.setattr(module,'HAS_SOUNDDEVICE',True)
    monkeypatch.setattr(module,'sd',devices)
    engine=AudioEngine(VoiceParameterStore(),sample_rate=48000,device=0,block_size=1024)
    engine.start()
    try:yield engine,devices
    finally:engine.stop()


def request(**patch):return {'expected_revision':0,'device':1,'sample_rate':96000,'block_size':512,**patch}


def test_switch_preflight_jack_rate_clock_and_rollback(audio):
    engine,devices=audio
    old_id=engine._audio_health['engine_id']
    engine._sample_index=1000
    engine._voice_state={1:{'phase':.7}}  # phase accumulator preserved, not new note state
    result=engine.configure_output(request(device=0))
    assert result['requested_sample_rate']==96000 and result['sample_rate']==48000
    assert devices.checked[-1]['samplerate']==48000
    assert result['revision']==1 and result['block_size']==512
    assert engine._audio_health['engine_id']!=old_id and engine._sample_index==0
    assert engine._voice_state[1]['phase']==.7
    with pytest.raises(AudioOutputConflict):engine.configure_output(request())
    devices.fail=1
    with pytest.raises(RuntimeError,match='previous settings restored'):
        engine.configure_output(request(expected_revision=1))
    assert engine.is_running and engine.output_settings()['device']==0
    assert engine.output_settings()['block_size']==512


def test_api_validation_conflicts_and_no_stream_change(audio):
    engine,devices=audio
    with TestClient(create_app(engine._store,engine)) as client:
        settings=client.get('/api/audio/output')
        assert settings.status_code==200 and len(settings.json()['devices'])==2
        for bad in (request(block_size=999),request(sample_rate=True),{**request(),'unexpected':1}):
            assert client.post('/api/audio/output',json=bad).status_code==422
        assert len(devices.opened)==1 and not devices.closed
        engine._record_sink=[]
        assert client.post('/api/audio/output',json=request()).status_code==409
        engine._record_sink=None
        result=client.post('/api/audio/output',json=request())
        assert result.status_code==200 and result.json()['sample_rate']==96000
        assert client.post('/api/audio/output',json=request()).status_code==409


def test_active_voice_and_unsupported_preflight_never_close_current_stream(audio,monkeypatch):
    engine,devices=audio
    engine._store.voice_on(32,999,1000.,.2)
    with pytest.raises(AudioOutputConflict,match='Release all voices'):
        engine.configure_output(request())
    assert len(devices.opened)==1 and not devices.closed
    engine._store.voice_off(999)
    def unsupported(**kwargs):raise RuntimeError('unsupported rate')
    monkeypatch.setattr(devices,'check_output_settings',unsupported)
    with pytest.raises(RuntimeError,match='unsupported rate'):
        engine.configure_output(request())
    assert engine.is_running and len(devices.opened)==1 and not devices.closed


def test_unchanged_settings_do_not_reopen_stream(audio):
    engine,devices=audio
    old_id=engine._audio_health['engine_id']
    result=engine.configure_output(request(device=0,sample_rate=48000,block_size=1024))
    assert result['revision']==0
    assert engine._audio_health['engine_id']==old_id
    assert len(devices.opened)==1 and not devices.closed
