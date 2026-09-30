"""Recover only the journal-confirmed PCM prefix; never change raw evidence."""
import fcntl
import hashlib
import json
from pathlib import Path
import struct
from uuid import uuid4

import numpy as np
import soundfile as sf


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def data_offset(handle):
    if handle.read(12)[:4]!=b'RIFF':raise ValueError('Not a RIFF capture')
    handle.seek(8)
    if handle.read(4)!=b'WAVE':raise ValueError('Not a WAVE capture')
    fmt=None
    while True:
        header=handle.read(8)
        if len(header)!=8:raise ValueError('No recoverable WAV data chunk')
        kind,size=struct.unpack('<4sI',header)
        if kind==b'data':
            if fmt is None or len(fmt)<16:raise ValueError('Missing WAV format')
            encoding,channels,rate,byte_rate,align,bits=struct.unpack('<HHIIHH',fmt[:16])
            if (encoding,channels,bits,align)!=(3,2,32,8) or rate<=0 or byte_rate!=rate*8:
                raise ValueError('Recovery supports stereo float32 capture only')
            return handle.tell(),rate
        if size>1024*1024:raise ValueError('Unexpected capture WAV header')
        content=handle.read(size)
        if len(content)!=size:raise ValueError('Truncated WAV header')
        if kind==b'fmt ':fmt=content
        if size%2:handle.read(1)


def recover_capture(folder):
    folder=Path(folder)
    if folder.is_symlink():raise ValueError('Capture folder cannot be a symlink')
    folder=folder.resolve()
    for name in ('writer.lock','capture.json','audio.wav','blocks.jsonl'):
        if not (folder/name).is_file() or (folder/name).is_symlink():raise ValueError('Recovery evidence is unavailable')
    with (folder/'writer.lock').open('r+b') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Capture writer is still active') from exc
        try:
            manifest=folder/'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text()).get('status')=='complete':
                raise ValueError('Capture already closed successfully')
            metadata=json.loads((folder/'capture.json').read_text())
            with (folder/'audio.wav').open('rb') as raw:
                offset,rate=data_offset(raw)
                if metadata['sample_rate']!=rate:raise ValueError('Capture sample rate mismatch')
                available=((folder/'audio.wav').stat().st_size-offset)//8
                rows=[];confirmed=0;last=None;reason='end_of_journal'
                with (folder/'blocks.jsonl').open() as journal:
                    for line in journal:
                        try:
                            if not line.endswith('\n'):raise ValueError('Partial journal row')
                            row=json.loads(line);start=row['capture_file_sample_start'];count=row['capture_frames']
                            if type(start) is not int or type(count) is not int or count<=0 or start!=confirmed:
                                raise ValueError('Discontinuous file sample clock')
                            if type(row['sample_index']) is not int or row['sample_index']<0 or row['sample_rate']!=rate or (last is not None and row['sample_index']!=last):
                                raise ValueError('Discontinuous engine sample clock')
                            if confirmed+count>available:raise ValueError('PCM shorter than confirmed journal')
                        except (ValueError,KeyError,TypeError) as exc:reason=str(exc);break
                        confirmed+=count;last=row['sample_index']+count;rows.append(row)
                if not confirmed:raise ValueError('No journal-confirmed complete PCM blocks')
                output=folder/'recovered'/uuid4().hex;output.mkdir(parents=True,mode=0o700)
                report={'schema_version':1,'status':'recovering','capture_id':metadata['id'],
                        'directory':str(output),'source_directory':str(folder),'sample_rate':rate,
                        'recovered_samples':confirmed,'available_samples':available,
                        'unconfirmed_samples':available-confirmed,'prefix_stop_reason':reason,
                        'limits':['Recovered prefix is not a normally completed capture',
                                  'Unjournaled/truncated tail excluded; no missing samples invented']}
                try:
                    raw.seek(offset)
                    with sf.SoundFile(output/'audio.wav',mode='w',samplerate=rate,channels=2,format='WAV',subtype='FLOAT') as audio:
                        remaining=confirmed
                        while remaining:
                            count=min(remaining,65536);chunk=raw.read(count*8)
                            if len(chunk)!=count*8:raise ValueError('PCM changed during recovery')
                            audio.write(np.frombuffer(chunk,dtype='<f4').reshape(-1,2));remaining-=count
                    (output/'blocks.jsonl').write_text(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in rows))
                    report.update(status='recovered',hashes={name:digest(output/name) for name in ('audio.wav','blocks.jsonl')},
                                  source_hashes={name:digest(folder/name) for name in ('audio.wav','blocks.jsonl','capture.json')})
                except Exception as exc:
                    report.update(status='failed',error=str(exc));raise
                finally:
                    temporary=output/'manifest.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False))
                    temporary.replace(output/'manifest.json')
                return report
        finally:fcntl.flock(lock,fcntl.LOCK_UN)
