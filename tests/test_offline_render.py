from unittest.mock import patch
import numpy as np
import pytest
from harmonic_shaper.audio_engine import AudioEngine
from harmonic_shaper.laboratory import LaboratoryInput
from harmonic_shaper.state import VoiceParameterStore
from test_laboratory import control

@pytest.mark.parametrize('voices', [1,6,32])
def test_offline_matches_callback_sample_for_sample_with_shape_pan_and_release(voices):
    stores=[VoiceParameterStore(),VoiceParameterStore()]
    engines=[AudioEngine(s, sample_rate=48000, block_size=256) for s in stores]
    for s in stores: s.laboratory_input=LaboratoryInput(s)
    for block in range(150):
        now=block*256/48000
        body=control(block,n=voices)
        for v in body['voices']:
            v.update(shape=.4,pan=(v['id']%3-1)*.7,phase_deg=v['id']*11.)
            if block>=90: v['gain']=0
        for s in stores: s.laboratory_input.apply(body,now=now)
        offline=engines[0].render_block(now=now)
        realtime=np.empty_like(offline)
        with patch('harmonic_shaper.audio_engine.time.monotonic',return_value=now):
            engines[1]._audio_callback(realtime,256,None,None)
        np.testing.assert_array_equal(offline,realtime)
        assert engines[0].voice_frame()==engines[1].voice_frame()
    assert not np.any(offline)


def test_offline_logical_lease_expires_without_wall_clock_and_rejects_bad_time():
    store=VoiceParameterStore(); store.laboratory_input=LaboratoryInput(store)
    store.laboratory_input.apply(control(n=1),now=0)
    engine=AudioEngine(store)
    assert np.max(np.abs(engine.render_block(now=0)))>0
    for i in range(200): engine.render_block(now=.51+i*256/48000)
    assert not engine.voice_frame()['voices']
    with pytest.raises(ValueError): engine.render_block(now=0)
    with pytest.raises(ValueError): engine.render_block(now=float('nan'))
    engine._running=True
    with pytest.raises(RuntimeError): engine.render_block(now=2)
