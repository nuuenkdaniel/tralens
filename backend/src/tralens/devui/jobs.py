"""Durable, bounded private-viewer subprocess supervisor. No automatic retries."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def tree_bytes(root):
    total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in files:
            path=Path(directory)/name
            try:
                if not path.is_symlink():total+=path.stat().st_size
            except FileNotFoundError:pass  # Atomic checkpoint replacement.
    return total


def supervise(command,directory,cancel,*,seconds,storage_bytes,env=None):
    """Kill only our new process group; completed stage artifacts remain in place."""
    from ..translation.workflow import _atomic_json
    state=directory/'worker-state.json'
    started=time.monotonic()
    with (directory/'worker.log').open('xb') as log:
        from ..processes import identity
        token=uuid.uuid4().hex
        env={**(os.environ if env is None else env),'TRALENS_WORKER_TOKEN':token}
        entry='import runpy,sys; runpy.run_path(sys.argv[1])["main"](sys.argv[2:])'
        launcher=[sys.executable,'-c',entry,str(Path(__file__).with_name('process_launcher.py')),*command]
        process=subprocess.Popen(launcher,stdin=subprocess.PIPE,stdout=log,stderr=log,env=env,start_new_session=True)
        evidence={'pid':process.pid,'process_group':process.pid,'status':'running','wall_limit_seconds':seconds,
            'storage_limit_bytes':storage_bytes,'automatic_retries':0,'token':token,
            'owner':identity(os.getpid()),'worker':identity(process.pid)}
        try:
            _atomic_json(state,evidence)
            process.stdin.write(b'G');process.stdin.close()
            failure=None
            while process.poll() is None:
                if cancel.is_set():failure=('cancelled','Job cancelled; completed checkpoints retained.')
                elif time.monotonic()-started>=seconds:failure=('timeout','Job wall-clock deadline reached; completed checkpoints retained.')
                elif tree_bytes(directory)>storage_bytes:failure=('storage_limit','Job storage limit reached; completed checkpoints retained.')
                if failure:break
                cancel.wait(.2)
            if failure:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            else:process.wait()
            status=failure[0] if failure else 'complete' if process.returncode==0 else 'failed'
            evidence.update(status=status,exit_code=process.returncode,seconds=time.monotonic()-started,
                retained_bytes=tree_bytes(directory))
            _atomic_json(state,evidence,replace_owned=True)
            if failure:return {'status':'interrupted' if status=='cancelled' else 'failed',
                'error':{'type':status,'message':failure[1]}}
            if process.returncode:return {'status':'failed','error':{'type':'WorkerExit',
                'message':f'Worker exited {process.returncode}; inspect worker.log and retained stage checkpoints.'}}
            return None
        finally:
            if not process.stdin.closed:process.stdin.close()
            # Descendants (e.g. reconstruction) must not outlive their owned job.
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()


def recover_jobs(root):
    """Never resume jobs; retain an additional restart-recovery record."""
    from ..processes import recover
    from ..translation.workflow import _atomic_json
    blocked=[]
    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir():continue
        path=directory/'worker-state.json'
        if not path.is_file() or path.is_symlink():continue
        try:
            if path.stat().st_size>65536:raise ValueError('oversized worker state')
            state=json.loads(path.read_bytes())
            recovered=directory/'worker-recovery.json'
            if recovered.is_file() and not recovered.is_symlink():
                previous=json.loads(recovered.read_bytes())
                if previous.get('previous_worker')==state:continue
            status=recover(state)
            # A supervisor can die after publishing its terminal state but before
            # its finally block kills descendants. Terminal status is not proof
            # that the owned process group is empty.
            if state.get('status')!='running' and status in ('already_stopped','previous_boot'):continue
            if status in ('legacy_unverifiable','live_supervisor','identity_mismatch'):
                blocked.append({'job':directory.name,'reason':status});continue
            _atomic_json(directory/'worker-recovery.json',{'status':status,'previous_worker':state},replace_owned=True)
        except (ValueError,OSError,KeyError,TypeError) as error:
            blocked.append({'job':directory.name,'reason':str(error)})
    return blocked


def run(config,job,cancel):
    from ..translation.workflow import _atomic_json
    payload={'config':asdict(config),'job':{k:job[k] for k in ('directory','image','options')}}
    payload=json.loads(json.dumps(payload,default=str))
    request=job['directory']/'worker-request.json';_atomic_json(request,payload)
    env={**os.environ,'TRALENS_FRONTIER_PROVIDER':config.frontier_provider,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    entry='import runpy,sys,pathlib; runpy.run_path(sys.argv[1])["main"](pathlib.Path(sys.argv[2]))'
    failure=supervise([sys.executable,'-c',entry,str(Path(__file__).with_name('job_worker.py')),str(request)],
        job['directory'],cancel,seconds=config.job_seconds,storage_bytes=config.job_storage_bytes,env=env)
    if failure:return failure
    result=job['directory']/'worker-result.json'
    if not result.is_file() or result.stat().st_size>16*1024*1024:raise ValueError('worker result unavailable or oversized')
    return json.loads(result.read_bytes())
