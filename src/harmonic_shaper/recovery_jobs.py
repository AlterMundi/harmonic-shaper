"""Owner-local recovery receipts; polling never starts or repeats PCM recovery."""
import fcntl
import json
from pathlib import Path
import re
import threading

def recover_capture(folder):
    # Optional PCM dependencies are needed only when explicitly recovering.
    from .capture_recovery import recover_capture as implementation
    return implementation(folder)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{32}', value):
        raise ValueError('Expected a capture/job ID, not a filesystem path')
    return value


def write(folder, report):
    temporary = folder / 'job.tmp'
    if temporary.is_symlink():
        raise ValueError('Regular recovery receipt temporary file required')
    temporary.write_text(json.dumps(report, allow_nan=False, indent=2))
    temporary.replace(folder / 'job.json')


class RecoveryJobs:
    def __init__(self, capture_root):
        self.capture_root = Path(capture_root)
        self.root = self.capture_root / 'recovery-jobs'
        self.lock = threading.RLock()
        self.threads = {}
        self.closed = False

    def read(self, job_id):
        if self.root.is_symlink():
            raise ValueError('Regular recovery jobs root required')
        folder = self.root / identifier(job_id)
        if not folder.exists():
            raise FileNotFoundError('Recovery job unavailable')
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError('Regular recovery job folder required')
        path = folder / 'job.json'
        lease = folder / 'job.lock'
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024*1024 or lease.is_symlink() or not lease.is_file():
            raise ValueError('Recovery receipt unavailable')
        report = json.loads(path.read_text())
        if report.get('job_id') != job_id or report.get('schema_version') != 1:
            raise ValueError('Recovery receipt identity differs')
        identifier(report['capture_id'])
        if report['status'] in ('queued', 'running'):
            with lease.open('rb') as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return report
                # Re-read after taking the lock: the writer may have just finished.
                report = json.loads(path.read_text())
                if report['status'] in ('queued', 'running'):
                    report.update(status='interrupted', error='No live recovery writer')
                    write(folder, report)
        if report['status'] == 'recovered' and (report.get('result', {}).get('status') != 'recovered' or report['result'].get('capture_id') != report['capture_id']):
            raise ValueError('Recovery result identity differs')
        return report

    def start(self, capture_id, job_id):
        identifier(capture_id); identifier(job_id)
        with self.lock:
            if self.root.is_symlink():
                raise ValueError('Regular recovery jobs root required')
            if self.closed:
                raise ValueError('Recovery jobs service is closed')
            folder = self.root / job_id
            if folder.exists() or folder.is_symlink():
                saved = self.read(job_id)
                if saved['capture_id'] != capture_id:
                    raise ValueError('Recovery job ID belongs to another capture')
                return saved
            if any(thread.is_alive() for thread in self.threads.values()):
                raise ValueError('An owned recovery job is already active')
            source = self.capture_root / capture_id
            metadata = source / 'capture.json'
            if source.is_symlink() or not source.is_dir() or metadata.is_symlink() or not metadata.is_file():
                raise ValueError('Known capture required')
            if json.loads(metadata.read_text()).get('id') != capture_id:
                raise ValueError('Capture metadata identity differs')
            folder.mkdir(mode=0o700, parents=True, exist_ok=False)
            handle = (folder / 'job.lock').open('w+b')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = {'schema_version': 1, 'job_id': job_id, 'capture_id': capture_id, 'status': 'queued'}
            try:
                write(folder, report)
                def work():
                    try:
                        report['status'] = 'running'; write(folder, report)
                        result = recover_capture(source)
                        if result.get('capture_id') != capture_id:
                            raise ValueError('Recovered capture identity differs')
                        report.update(status='recovered', result=result); write(folder, report)
                    except Exception as exc:
                        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                        write(folder, report)
                    finally:
                        handle.close()
                thread = threading.Thread(target=work, daemon=True, name='shaper-recovery-job')
                self.threads[job_id] = thread
                thread.start()
            except Exception:
                handle.close()
                raise
            return self.read(job_id)

    def close(self):
        self.closed = True
        for thread in self.threads.values():
            thread.join(timeout=5)
