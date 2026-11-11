"""Real bounded child processes, no models or network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from tralens.devui.jobs import supervise
from tralens.scheduling import inference_lease


class Isolation(unittest.TestCase):
    def test_deadline_retains_completed_stage_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            script='import pathlib,time;pathlib.Path('+repr(str(root/'completed.txt'))+').write_text("stage done");time.sleep(10)'
            result=supervise([sys.executable,'-c',script],root,threading.Event(),seconds=.4,storage_bytes=1024*1024)
            self.assertEqual(result['error']['type'],'timeout')
            self.assertEqual((root/'completed.txt').read_text(),'stage done')
            state=json.loads((root/'worker-state.json').read_bytes())
            self.assertEqual(state['status'],'timeout')
            self.assertIsNotNone(state['exit_code'])
            with self.assertRaises(ProcessLookupError):os.kill(state['pid'],0)

    def test_cancel_has_no_retry_and_leaves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cancel=threading.Event();cancel.set()
            result=supervise([sys.executable,'-c','import time;time.sleep(10)'],root,cancel,seconds=10,storage_bytes=1024*1024)
            self.assertEqual(result['status'],'interrupted')
            self.assertEqual(json.loads((root/'worker-state.json').read_bytes())['automatic_retries'],0)

    def test_cross_process_application_lease_and_reentrancy(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'TRALENS_STATE_ROOT':directory}):
            script='from tralens.scheduling import inference_lease\nwith inference_lease():print("owned")'
            env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
            with inference_lease():
                with inference_lease():pass
                result=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Another Tralens',result.stderr)
            result=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
