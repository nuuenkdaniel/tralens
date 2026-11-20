"""Owned private-viewer process; model initialization occurs only after upload."""
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import sys


def main(request):
    # Linux parent-death handling prevents inference continuing after viewer loss.
    parent=os.getppid()
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if os.getppid()!=parent or parent==1:raise RuntimeError('viewer parent disappeared')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

    resource.setrlimit(resource.RLIMIT_FSIZE,(512*1024**2,512*1024**2))
    doc=json.loads(request.read_bytes())
    if doc['job']['options'].get('reconstruction_policy') != 'sdxl':
        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from tralens.devui.server import Config,execute_job
    from tralens.translation.workflow import _atomic_json
    config=doc['config']
    for key in ('root','model_root','fonts','manga_reread_model','reconstruction_config','text_mask_model','crop_reader_model','sdxl_config'):
        if config.get(key) is not None:config[key]=Path(config[key])
    job=doc['job']
    for key in ('directory','image'):job[key]=Path(job[key])
    result=execute_job(Config(**config),job)
    _atomic_json(job['directory']/'worker-result.json',result)
