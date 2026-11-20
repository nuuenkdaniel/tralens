"""Durable single-file publication; completion manifests are published last."""
import os
from pathlib import Path
import tempfile


def sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def atomic_bytes(path,content,*,replace_owned=False):
    path=Path(path)
    descriptor,temporary=tempfile.mkstemp(prefix='.publish-',dir=path.parent)
    temporary=Path(temporary)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        if replace_owned:os.replace(temporary,path)
        else:os.link(temporary,path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
