"""One application-owned local inference job at a time, across CLI/viewer processes.

Does not manage, unload, or interrupt models owned by other applications.
"""
from contextlib import contextmanager
import fcntl
from functools import wraps
import os
from pathlib import Path
import threading

_mutex=threading.RLock()
_depth=threading.local()


@contextmanager
def inference_lease():
    if not _mutex.acquire(blocking=False):raise RuntimeError('Another Tralens inference job is active.')
    stream=None
    try:
        depth=getattr(_depth,'value',0)
        if not depth:
            root=Path(os.environ.get('TRALENS_STATE_ROOT',str(Path.home()/'.cache/tralens'))).expanduser().absolute()
            if any(p.is_symlink() for p in (root,*root.parents)):raise ValueError('inference state root must not use symlinks')
            root.mkdir(parents=True,exist_ok=True,mode=0o700)
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
