"""Fault injection at actual publication, worker and HTTP boundaries; no models."""
import errno
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from tralens import storage,processes,scheduling
from tralens.devui.jobs import supervise,recover_jobs
from tralens.rendering import reconstruction as rec,reconstruction_worker as worker
from tralens.translation.runtime import OllamaRuntime
from tralens.translation.workflow import _atomic_json


class Hardening(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.env=patch.dict(os.environ,{'TRALENS_STATE_ROOT':str(self.root/'state')})
        self.env.start();self.addCleanup(self.env.stop)

    def test_worker_rejects_decoded_pixel_mask_and_model_drift_before_loading(self):
        rgb=np.full((32,40,3),180,np.uint8);mask=np.zeros((32,40),bool);mask[8:20,10:22]=True
        protected=np.zeros_like(mask);config={'backend':'lama-onnx-512','sha256':rec.MODEL_SHA256,'config_revision':'config'}
        request=rec.request({'size':[40,32],'sha256':'source'},
            {'member_ids':[1],'request_revision':'translation','text':'Exact'},[0,0,40,32],rgb,mask,mask,protected,
            attempt=0,model=rec.model_identity(config))
        def write():
            Image.fromarray(rgb).save(self.root/'input.png')
            for name,value in (('mask',mask),('commit',mask),('protected',protected)):
                Image.fromarray(value.astype(np.uint8)*255).save(self.root/(name+'.png'))
            (self.root/'request.json').write_text(json.dumps({'request':request,'config':config}))
        write();self.assertTrue(np.array_equal(worker.inputs(self.root)[2],rgb))
        session=Mock()
        session.get_inputs.return_value=[SimpleNamespace(name='image',type='tensor(float)',shape=['batch',3,512,512]),
            SimpleNamespace(name='mask',type='tensor(float)',shape=['batch',1,512,512])]
        orientations=iter((1,-1))
        def infer(_,feed):
            direction=next(orientations)
            self.assertEqual(feed['image'].shape,(1,3,512,512))
            native=rgb[:,::direction]
            self.assertTrue(np.array_equal(feed['image'][0,:,:32,:40],native.transpose(2,0,1).astype(np.float32)/255))
            expected=np.pad(mask[:,::direction],((0,480),(0,472)),mode='symmetric')
            self.assertTrue(np.array_equal(feed['mask'][0,0],expected))
            return [np.full((1,3,512,512),123,np.float32)]
        session.run.side_effect=infer;session.get_providers.return_value=['synthetic-test-only']
        runtime=SimpleNamespace(SessionOptions=SimpleNamespace,ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
            InferenceSession=Mock(return_value=session),__version__='synthetic')
        with patch.dict(sys.modules,{'onnxruntime':runtime}),patch.object(worker.resource,'setrlimit'), \
                patch.object(worker,'model_bytes',return_value=b'synthetic-model'):
            worker.main(self.root)
        metadata=json.loads((self.root/'result.json').read_bytes())
        self.assertEqual(metadata['output_pixels_sha256'],rec.pixels_hash(np.full_like(rgb,123)))
        self.assertEqual(session.run.call_count,2)
        self.assertEqual(metadata['views'],['native','horizontal_reflection'])
        for file in ('input','mask','commit','protected'):
            write();image=Image.open(self.root/(file+'.png')).copy();image.putpixel((0,0),(0,0,0) if file=='input' else 255)
            image.save(self.root/(file+'.png'))
            with self.assertRaises(rec.BindingError):worker.inputs(self.root)
        write();doc=json.loads((self.root/'request.json').read_text());doc['config']['sha256']='foreign'
        (self.root/'request.json').write_text(json.dumps(doc))
        with self.assertRaises(rec.BindingError):worker.inputs(self.root)

    def test_disk_full_before_replace_preserves_old_checkpoint(self):
        path=self.root/'checkpoint.json';_atomic_json(path,{'stage':'complete'})
        before=path.read_bytes()
        with patch.object(storage.os,'fsync',side_effect=OSError(errno.ENOSPC,'injected disk full')):
            with self.assertRaises(OSError):_atomic_json(path,{'stage':'new'},replace_owned=True)
        self.assertEqual(path.read_bytes(),before)
        self.assertFalse(list(self.root.glob('.publish-*')))
        with self.assertRaises(FileExistsError):_atomic_json(path,{'overwrite':True})
        self.assertEqual(path.read_bytes(),before)

    def test_no_job_code_before_durable_admission(self):
        target=self.root/'must-not-run'
        command=[sys.executable,'-c',f'from pathlib import Path;Path({str(target)!r}).touch()']
        with patch('tralens.translation.workflow._atomic_json',side_effect=OSError(errno.ENOSPC,'full')):
            with self.assertRaises(OSError):supervise(command,self.root,threading.Event(),seconds=5,storage_bytes=10000)
        self.assertFalse(target.exists())

    def test_process_crash_on_either_side_of_replace_keeps_complete_checkpoint(self):
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
        for after in (False,True):
            with self.subTest(after_replace=after):
                directory=self.root/str(after);directory.mkdir()
                target=directory/'checkpoint.json';ready=directory/'ready'
                _atomic_json(target,{'stage':'old'})
                script=('from pathlib import Path\nimport time\nfrom tralens import storage\n'
                    'from tralens.translation.workflow import _atomic_json\n'
                    f'def pause(*args):\n Path({str(ready)!r}).touch()\n time.sleep(30)\n'
                    +('storage.sync_directory=pause\n' if after else 'storage.os.replace=pause\n')
                    +f'_atomic_json(Path({str(target)!r}),{{"stage":"new"}},replace_owned=True)\n')
                child=subprocess.Popen([sys.executable,'-c',script],env=env)
                try:
                    deadline=time.monotonic()+5
                    while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
                    self.assertTrue(ready.exists());child.kill();child.wait(timeout=3)
                    self.assertEqual(json.loads(target.read_bytes()),{'stage':'new' if after else 'old'})
                finally:
                    if child.poll() is None:child.kill();child.wait()

    def test_terminal_state_does_not_hide_orphan_descendant(self):
        job=self.root/'terminal';job.mkdir();ready=job/'terminal-published'
        descendant=job/'descendant-ready'
        descendant_script=f'from pathlib import Path;import time;Path({str(descendant)!r}).touch();time.sleep(30)'
        command=[sys.executable,'-c',
            'import subprocess,sys,time,pathlib\n'
            f'subprocess.Popen([sys.executable,"-c",{descendant_script!r}])\n'
            f'while not pathlib.Path({str(descendant)!r}).exists():time.sleep(.01)']
        script=('import time,threading\nfrom pathlib import Path\nfrom tralens.devui.jobs import supervise\n'
            'from tralens.translation import workflow\noriginal=workflow._atomic_json\n'
            'def publish(path,value,**kwargs):\n original(path,value,**kwargs)\n'
            f' if path.name=="worker-state.json" and value.get("status")=="complete":\n  Path({str(ready)!r}).touch()\n  time.sleep(30)\n'
            'workflow._atomic_json=publish\n'
            f'supervise({command!r},Path({str(job)!r}),threading.Event(),seconds=10,storage_bytes=1000000)')
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
        parent=subprocess.Popen([sys.executable,'-c',script],env=env)
        try:
            deadline=time.monotonic()+5
            while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(ready.exists());parent.kill();parent.wait(timeout=3)
            self.assertEqual(recover_jobs(self.root),[])
            record=json.loads((job/'worker-recovery.json').read_bytes())
            self.assertEqual(record['previous_worker']['status'],'complete')
            self.assertEqual(record['status'],'owned_group_killed')
        finally:
            if parent.poll() is None:parent.kill();parent.wait()
            state=job/'worker-state.json'
            if state.exists():processes.recover(json.loads(state.read_bytes()))

    def test_stopped_legacy_job_does_not_block_but_live_unowned_group_does(self):
        job=self.root/'legacy';job.mkdir()
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        try:
            _atomic_json(job/'worker-state.json',{'status':'complete','process_group':child.pid})
            self.assertEqual(recover_jobs(self.root),[{'job':'legacy','reason':'legacy_unverifiable'}])
            self.assertIsNone(child.poll());child.kill();child.wait(timeout=3)
            self.assertEqual(recover_jobs(self.root),[])
        finally:
            if child.poll() is None:child.kill();child.wait()

    def test_storage_quota_retains_completed_stage(self):
        script=f'from pathlib import Path;import time;Path({str(self.root/"done")!r}).write_bytes(b"x"*2000);time.sleep(10)'
        report=supervise([sys.executable,'-c',script],self.root,threading.Event(),seconds=3,storage_bytes=1500)
        self.assertEqual(report['error']['type'],'storage_limit')
        self.assertEqual((self.root/'done').stat().st_size,2000)

    def test_restart_kills_owned_descendant_and_never_a_foreign_group(self):
        job=self.root/'a'
        job.mkdir();ready=job/'descendant'
        descendant=f'import os,time,pathlib;pathlib.Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(30)'
        command=[sys.executable,'-c',f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{descendant!r}]);time.sleep(30)']
        script=('from tralens.devui.jobs import supervise\nfrom pathlib import Path\nimport threading\n'
            f'supervise({command!r},Path({str(job)!r}),threading.Event(),seconds=30,storage_bytes=1000000)')
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
        parent=subprocess.Popen([sys.executable,'-c',script],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            deadline=time.monotonic()+5
            while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(ready.exists())
            parent.kill();parent.wait(timeout=3)
            self.assertEqual(recover_jobs(self.root),[])
            record=json.loads((job/'worker-recovery.json').read_bytes())
            self.assertEqual(record['status'],'owned_group_killed')
            deadline=time.monotonic()+3
            while processes.group_members(record['previous_worker']['process_group']) and time.monotonic()<deadline:time.sleep(.02)
            self.assertEqual(processes.group_members(record['previous_worker']['process_group']),[])
        finally:
            if parent.poll() is None:parent.kill();parent.wait()
            state=job/'worker-state.json'
            if state.exists():processes.recover(json.loads(state.read_bytes()))
        stranger=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        try:
            stale={'owner':{**processes.identity(os.getpid()),'start_ticks':-1},'token':'not-the-token','process_group':stranger.pid}
            self.assertEqual(processes.recover(stale),'identity_mismatch')
            self.assertIsNone(stranger.poll())
        finally:stranger.kill();stranger.wait()

    def test_connection_refusal_allows_retry_without_acknowledgement(self):
        endpoint = 'http://127.0.0.1:11434'
        refused = Mock(side_effect=ConnectionRefusedError)
        with self.assertRaises(ConnectionRefusedError):
            scheduling.remote_call(endpoint, refused)
        self.assertFalse(scheduling.remote_state_path(endpoint).exists())
        self.assertEqual(scheduling.remote_call(endpoint, lambda: {'done': True}), {'done': True})

    def test_remote_timeout_blocks_dispatch_across_runtime_instances(self):
        entered=threading.Event();finish=threading.Event();calls=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']));calls.append(1);entered.set()
                finish.wait(3)
                body=b'{"done":true,"message":{"content":"ok"}}'
                try:
                    self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        endpoint=f'http://127.0.0.1:{server.server_port}'
        try:
            with self.assertRaises(TimeoutError):OllamaRuntime(endpoint)._post('/api/chat',{},.1)
            self.assertTrue(entered.is_set())
            with self.assertRaisesRegex(RuntimeError,'may still be running'):
                OllamaRuntime(endpoint)._post('/api/chat',{},1)
            self.assertEqual(len(calls),1)
            path=scheduling.remote_state_path(endpoint);intent=json.loads(path.read_bytes())
            with self.assertRaises(ValueError):scheduling.acknowledge_remote_completion(endpoint,'stale')
            finish.set()  # Synthetic provider completion, not an idle-model-list assumption.
            scheduling.acknowledge_remote_completion(endpoint,intent['request_id'])
            self.assertTrue(OllamaRuntime(endpoint)._post('/api/chat',{},1)['done'])
            self.assertFalse(path.exists())
        finally:finish.set();server.shutdown();server.server_close();thread.join()

    def test_embedding_reports_its_actual_mode_and_finishes_after_history_disk_full(self):
        from tralens.devui.server import App,Config
        app=App(Config(root=self.root,model_root=self.root),runner=lambda *_:{'status':'complete'})
        self.assertEqual(app.public_config()['worker_mode'],'embedded_thread')
        self.assertFalse(app.public_config()['cancellation_available'])
        buffer=io.BytesIO();Image.new('RGB',(30,30),'white').save(buffer,format='PNG')
        original=app._save_job
        def save(job,*,replace_owned=False):
            if replace_owned:raise OSError(errno.ENOSPC,'history full')
            original(job)
        with patch.object(app,'_save_job',side_effect=save):
            key=app.start(buffer.getvalue(),{'mode':'ocr'});self.assertTrue(app.jobs[key]['done'].wait(3))
        self.assertIsNone(app.active);self.assertEqual(app.jobs[key]['status'],'complete')
        self.assertIn('history full',app.jobs[key]['history_warning'])
