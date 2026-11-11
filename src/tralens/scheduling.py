"""One application-owned local inference job at a time, across CLI/viewer processes.

Does not manage, unload, or interrupt models owned by other applications.
"""
from contextlib import contextmanager
import fcntl
from functools import wraps
import os
from pathlib import Path
import threading
import hashlib
import json
import time
import uuid

_mutex=threading.RLock()
_depth=threading.local()


def state_root():
    root=Path(os.environ.get('TRALENS_STATE_ROOT',str(Path.home()/'.cache/tralens'))).expanduser().absolute()
    if any(p.is_symlink() for p in (root,*root.parents)):raise ValueError('inference state root must not use symlinks')
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    return root


@contextmanager
def inference_lease():
    if not _mutex.acquire(blocking=False):raise RuntimeError('Another Tralens inference job is active.')
    stream=None
    try:
        depth=getattr(_depth,'value',0)
        if not depth:
            root=state_root()
            fd=os.open(root/'inference.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
            stream=os.fdopen(fd,'a+')
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise RuntimeError('Another Tralens inference job is active.') from error
        _depth.value=depth+1
        try:yield
        finally:_depth.value=depth
    finally:
        if stream is not None:stream.close()
        _mutex.release()


def serialized(function):
    @wraps(function)
    def run(*args,**kwargs):
        with inference_lease():return function(*args,**kwargs)
    return run


def remote_state_path(endpoint):
    return state_root()/('remote-'+hashlib.sha256(endpoint.encode()).hexdigest()+'.json')


def remote_call(endpoint,call):
    """A missing terminal response is durable uncertainty, not cancellation proof.

    The application lease covers dispatch/response. A crash leaves the intent file
    behind and blocks new remote work even after the kernel releases the lease.
    """
    from .storage import atomic_bytes,sync_directory
    with inference_lease():
        path=remote_state_path(endpoint)
        if path.exists():
            raise RuntimeError('Previous remote request may still be running. Inspect '+str(path)+
                '; confirm server-side completion before using tools/inference_state.py to acknowledge it.')
        intent={'endpoint':endpoint,'request_id':uuid.uuid4().hex,'pid':os.getpid(),'started_unix':time.time(),
            'status':'pending_or_remote_completion_unknown','cancellation_confirmed':False}
        atomic_bytes(path,(json.dumps(intent)+'\n').encode())
        response=call()
        if type(response) is dict and response.get('done') is True:
            path.unlink();sync_directory(path.parent)
        return response


def acknowledge_remote_completion(endpoint,request_id):
    """Operator confirms completion externally; no automatic idle/expiry inference."""
    from .storage import sync_directory
    with inference_lease():
        path=remote_state_path(endpoint)
        data=json.loads(path.read_bytes())
        if data.get('endpoint')!=endpoint or data.get('request_id')!=request_id:
            raise ValueError('remote request changed; acknowledgement rejected')
        destination=path.with_name(path.stem+'-acknowledged-'+request_id+'.json')
        # Retain the incident evidence; do not silently remove the unknown request.
        os.link(path,destination);path.unlink();sync_directory(path.parent)
