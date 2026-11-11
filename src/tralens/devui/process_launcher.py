"""Wait for durable supervisor ownership before executing any job code."""
import ctypes
import os
import signal
import sys


def main(command):
    parent=os.getppid()
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if parent==1 or os.getppid()!=parent:raise RuntimeError('supervisor disappeared')
    if sys.stdin.buffer.read(1)!=b'G':raise RuntimeError('job was not durably admitted')
    os.execvpe(command[0],command,os.environ)
