"""Local HTTP/upload integration with real pipeline contracts and fake engines."""
from dataclasses import replace
import http.client
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from tralens.devui import server as s


def image_bytes(mode='RGB', size=(100, 100)):
    stream = BytesIO()
    Image.new(mode, size, 'white').save(stream, format='PNG')
    return stream.getvalue()


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = s.Config(root=self.root/'jobs', model_root=self.root/'models')

    def serve(self, app):
        http = s.create_server(app, 0)
        worker = threading.Thread(target=http.serve_forever, daemon=True)
        worker.start()
        def stop():
            http.shutdown()
            http.server_close()
            worker.join(timeout=5)
        self.addCleanup(stop)
        return http

    def request(self, server, method, path, data=None, headers=None):
        client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        try:
            client.request(method, path, body=data, headers=headers or {})
            response = client.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            client.close()

    def test_serving_page_and_config_performs_no_inference(self):
        runner = Mock(side_effect=AssertionError('no inference'))
        app = s.App(self.config, runner=runner)
        http = self.serve(app)
        for path, text in [('/', b'pipeline viewer'), ('/app.js', b'createElement'), ('/style.css', b'columns')]:
            status, headers, data = self.request(http, 'GET', path)
            self.assertEqual(status, 200)
            self.assertIn(text, data)
            self.assertIn('Content-Security-Policy', headers)
        status, _, data = self.request(http, 'GET', '/api/config')
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(data)['render_enabled'])
        self.assertEqual(json.loads(data)['endpoint'], self.config.endpoint)
        self.assertEqual(app.jobs, {})
        runner.assert_not_called()

    def test_http_upload_executes_real_bound_pipeline_without_live_ocr_or_provider(self):
        from tests.test_pipeline import PipelineTests
        from tralens.ocr.automatic_units import propose_page_from_boxes
        from tralens.translation import workflow
        fixture = PipelineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = replace(self.config, fonts=fixture.fonts, model_root=fixture.fixture.weights)
        original = workflow.run_preview
        def factory(image, **kwargs):
            batch = propose_page_from_boxes(fixture.fixture.boxes, image,
                complementary_ownership=True, regroup_companions=True)
            return SimpleNamespace(raw_detections=[fixture.fixture.boxes], automatic_unit_proposals=[batch],
                process_images=Mock(), ocr=SimpleNamespace(close=Mock()))
        def preview(*args, **kwargs):
            kwargs.update(ocr_factory=factory, runtime=SimpleNamespace(generate=fixture.provider))
            return original(*args, **kwargs)
        app = s.App(config)
        http = self.serve(app)
        data = fixture.fixture.image.read_bytes()
        with patch.object(workflow, 'run_preview', side_effect=preview):
            status, _, body = self.request(http, 'POST', '/api/jobs', data,
                {'X-Tralens-Token': app.token, 'X-Tralens-Options': json.dumps(dict(mode='render'))})
            self.assertEqual(status, 202, body)
            identity = json.loads(body)['id']
            self.assertTrue(app.job(identity)['done'].wait(10))
        status, _, body = self.request(http, 'GET', '/api/jobs/'+identity)
        self.assertEqual(status, 200, body)
        view = json.loads(body)
        self.assertEqual(view['status'], 'complete', view.get('error'))
        self.assertEqual(len(view['detections']), 2)
        self.assertEqual([unit['target'] for unit in view['units']], ['Hi!', 'Hi!'])
        self.assertEqual(view['coverage']['rendered'], 2)
        self.assertEqual(view['model_calls'], 2)
        self.assertIn('run/rendering/composite.png', view['artifacts'])
        status, headers, original_data = self.request(http, 'GET', f'/api/jobs/{identity}/artifacts/{view["input"]}')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'image/png')
        self.assertEqual(original_data, data)
        status, headers, _ = self.request(http, 'GET', f'/api/jobs/{identity}/artifacts/run/rendering/review.html')
        self.assertIn('attachment', headers['Content-Disposition'])

    def test_viewer_quality_retry_reaches_real_reread_pipeline_and_displays_selected_source(self):
        import main
        from tests.ocr.test_quality_reread import QualityRereadTests
        from tralens.ocr.automatic_units import propose_page_from_boxes
        from tralens.translation import workflow
        fixture = QualityRereadTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = replace(self.config, fonts=fixture.p.fonts, model_root=fixture.p.fixture.weights,
            manga_reread_model=self.root/'existing-snapshot')
        original = workflow.run_preview
        def factory(image, **kwargs):
            batch = replace(propose_page_from_boxes(fixture.p.fixture.boxes, image,
                complementary_ownership=True, regroup_companions=True), regions=())
            return SimpleNamespace(raw_detections=[fixture.p.fixture.boxes], automatic_unit_proposals=[batch],
                process_images=Mock(), ocr=SimpleNamespace(close=Mock()))
        def preview(*args, **kwargs):
            kwargs.update(ocr_factory=factory, runtime=SimpleNamespace(generate=fixture.p.provider))
            return original(*args, **kwargs)
        app = s.App(config)
        http = self.serve(app)
        with patch.object(workflow, 'run_preview', side_effect=preview):
            first = app.start(fixture.p.fixture.image.read_bytes(), {'mode': 'ocr'})
            self.assertTrue(app.job(first)['done'].wait(5))
        old = {p.relative_to(app.job(first)['directory']): p.read_bytes()
            for p in app.job(first)['directory'].rglob('*') if p.is_file()}
        with patch.object(workflow, 'run_preview', side_effect=preview), \
                patch.object(main, 'quality_refiner', return_value=fixture.refine) as readers:
            status, _, body = self.request(http, 'POST', f'/api/jobs/{first}/retry',
                json.dumps(dict(mode='render', translation_policy='coverage-first',
                    replacement_policy='confidence-gated', quality_reread=True)).encode(),
                {'X-Tralens-Token': app.token, 'Content-Type': 'application/json'})
            self.assertEqual(status, 202, body)
            identity = json.loads(body)['id']
            self.assertTrue(app.job(identity)['done'].wait(10))
        readers.assert_called_once()
        self.assertEqual(readers.call_args.kwargs['manga_model'], config.manga_reread_model)
        view = app.snapshot(identity)
        self.assertEqual(view['status'], 'complete', view['error'])
        self.assertEqual(view['replacement_policy'], 'confidence-gated')
        self.assertTrue(view['quality_reread']['enabled'])
        self.assertEqual(view['retry_of'], first)
        self.assertEqual(view['replacement_confidence']['actual_replacements'], 2)
        self.assertTrue(all(u['source'] == 'こんばんは' and u['raw_source'] == 'こんにちは'
            and u['reread_selected'] for u in view['units']))
        self.assertEqual(old, {p.relative_to(app.job(first)['directory']): p.read_bytes()
            for p in app.job(first)['directory'].rglob('*') if p.is_file()})
        restored = s.App(config, runner=Mock())
        self.assertTrue(restored.snapshot(identity)['options']['quality_reread'])
        self.assertEqual(restored.snapshot(identity)['options']['replacement_policy'], 'confidence-gated')

    def test_busy_job_exposes_real_live_checkpoint_and_rejects_second_upload(self):
        entered, release = threading.Event(), threading.Event()
        def runner(config, job):
            from tralens.translation.workflow import _atomic_json
            directory = job['directory']/'run/preview'
            directory.mkdir()
            _atomic_json(directory/'preview.json', dict(stage='ocr', units=[], attempts=0, ocr=None))
            entered.set()
            release.wait(5)
            return dict(status='complete')
        app = s.App(self.config, runner=runner)
        identity = app.start(image_bytes(), dict(mode='ocr'))
        try:
            self.assertTrue(entered.wait(5))
            self.assertEqual(app.snapshot(identity)['stage'], 'ocr')
            with self.assertRaises(s.RequestError) as caught:
                app.start(image_bytes(), dict(mode='ocr'))
            self.assertEqual(caught.exception.status, 409)
        finally:
            release.set()
            self.assertTrue(app.job(identity)['done'].wait(5))

    def test_worker_error_is_visible_and_preserves_the_upload(self):
        app = s.App(self.config, runner=Mock(side_effect=RuntimeError('test missing local model')))
        original = image_bytes()
        identity = app.start(original, dict(mode='translate'))
        self.assertTrue(app.job(identity)['done'].wait(5))
        view = app.snapshot(identity)
        self.assertEqual(view['status'], 'failed')
        self.assertIn('missing local model', view['error']['message'])
        self.assertEqual(app.artifact(identity, view['input']).read_bytes(), original)
        self.assertIsNone(app.active)

    def test_retry_reuses_upload_in_a_fresh_job_and_keeps_previous_results(self):
        runner = Mock(side_effect=[RuntimeError('first run failed'), dict(status='complete')])
        app = s.App(self.config, runner=runner)
        first = app.start(image_bytes(), dict(mode='ocr', max_attempts=3))
        self.assertTrue(app.job(first)['done'].wait(5))
        previous = app.job(first)
        before = {p.name: p.read_bytes() for p in previous['directory'].iterdir() if p.is_file()}
        second = app.retry(first, dict(max_attempts=7))
        self.assertTrue(app.job(second)['done'].wait(5))
        current = app.job(second)
        self.assertNotEqual(first, second)
        self.assertEqual(current['image'].read_bytes(), previous['image'].read_bytes())
        self.assertEqual(current['options']['mode'], 'ocr')
        self.assertEqual(current['options']['max_attempts'], 7)
        self.assertEqual(current['retry_of'], first)
        self.assertEqual(app.job(first)['status'], 'failed')
        self.assertEqual(before, {p.name: p.read_bytes() for p in previous['directory'].iterdir() if p.is_file()})
        self.assertEqual(runner.call_count, 2)

    def test_http_retry_works_without_upload_and_requires_session_token(self):
        app = s.App(self.config, runner=lambda *args: dict(status='complete'))
        identity = app.start(image_bytes(), dict(mode='ocr'))
        self.assertTrue(app.job(identity)['done'].wait(5))
        http = self.serve(app)
        path = f'/api/jobs/{identity}/retry'
        self.assertEqual(self.request(http, 'POST', path, b'{}')[0], 403)
        status, _, content = self.request(http, 'POST', path, b'{"max_attempts":4}',
            {'X-Tralens-Token': app.token, 'Content-Type': 'application/json'})
        self.assertEqual(status, 202, content)
        new = json.loads(content)['id']
        self.assertTrue(app.job(new)['done'].wait(5))
        self.assertEqual(app.snapshot(new)['retry_of'], identity)
        self.assertEqual(app.snapshot(new)['options']['max_attempts'], 4)
        self.assertEqual(self.request(http, 'POST', '/api/jobs/'+'0'*32+'/retry', b'{}',
            {'X-Tralens-Token': app.token})[0], 404)

    def test_retry_rejects_changed_or_symlinked_upload_without_dispatching(self):
        runner = Mock(return_value=dict(status='complete'))
        app = s.App(self.config, runner=runner)
        identity = app.start(image_bytes(), dict(mode='ocr'))
        self.assertTrue(app.job(identity)['done'].wait(5))
        image = app.job(identity)['image']
        image.write_bytes(image_bytes(size=(99, 99)))
        with self.assertRaisesRegex(s.RequestError, 'changed'):
            app.retry(identity, {})
        self.assertEqual(runner.call_count, 1)
        image.unlink()
        image.symlink_to(self.root/'outside.png')
        with self.assertRaises(s.RequestError):
            app.retry(identity, {})
        self.assertEqual(runner.call_count, 1)

    def test_restart_restores_retry_targets_without_running_models(self):
        app = s.App(self.config, runner=Mock(side_effect=RuntimeError('saved failure')))
        identity = app.start(image_bytes(), dict(mode='ocr', max_attempts=3))
        self.assertTrue(app.job(identity)['done'].wait(5))
        runner = Mock(return_value=dict(status='complete'))
        restored = s.App(self.config, runner=runner)
        self.assertEqual(restored.snapshot(identity)['status'], 'failed')
        self.assertTrue(restored.snapshot(identity)['restored'])
        self.assertEqual(restored.snapshot(identity)['options']['max_attempts'], 3)
        runner.assert_not_called()
        fresh = restored.retry(identity, {})
        self.assertTrue(restored.job(fresh)['done'].wait(5))
        self.assertEqual(restored.job(fresh)['image'].read_bytes(), app.job(identity)['image'].read_bytes())
        runner.assert_called_once()

    def test_legacy_pipeline_runs_are_available_to_retry_after_restart(self):
        directory = self.config.root/('a'*32)
        (directory/'run/preview').mkdir(parents=True)
        data = image_bytes()
        (directory/'input.png').write_bytes(data)
        (directory/'run/pipeline.json').write_text(json.dumps(dict(status='partial', render_policy='conservative',
            source=dict(sha256=s.hashlib.sha256(data).hexdigest(), size=[100, 100]),
            rendering=dict(status='skipped', reason='no_strict_translations'))))
        (directory/'run/preview/preview.json').write_text(json.dumps(dict(status='complete', stage='finished', units=[],
            settings=dict(max_attempts=5, run_local_model=True), ocr=None)))
        runner = Mock()
        app = s.App(self.config, runner=runner)
        runner.assert_not_called()
        job = app.job('a'*32)
        self.assertEqual(job['status'], 'partial')
        self.assertEqual(job['options']['mode'], 'render')
        self.assertEqual(job['options']['render_policy'], 'conservative')
        self.assertEqual(job['options']['max_attempts'], 5)
        self.assertFalse((directory/'viewer.json').exists(), 'restoration must not rewrite old evidence')

    def test_restoration_uses_executed_quality_policy_without_rewriting_legacy_evidence(self):
        directory = self.config.root/('b'*32)
        (directory/'run/preview').mkdir(parents=True)
        data = image_bytes()
        (directory/'input.png').write_bytes(data)
        (directory/'run/pipeline.json').write_text(json.dumps(dict(status='partial',
            source=dict(sha256=s.hashlib.sha256(data).hexdigest(), size=[100, 100]),
            translation_policy='coverage-first',
            replacement_policy=dict(name='confidence-gated'), quality_reread=dict(enabled=True))))
        app = s.App(self.config, runner=Mock())
        snapshot = app.snapshot('b'*32)
        self.assertEqual(snapshot['replacement_policy'], 'confidence-gated')
        self.assertTrue(snapshot['options']['quality_reread'])
        self.assertEqual(snapshot['options']['replacement_policy'], 'confidence-gated')
        self.assertFalse((directory/'viewer.json').exists())

    def test_interrupted_history_is_not_resumed_and_does_not_spend_session_slots(self):
        app = s.App(self.config, runner=lambda *args: dict(status='complete'))
        identity = app.start(image_bytes(), dict(mode='ocr'))
        self.assertTrue(app.job(identity)['done'].wait(5))
        metadata = app.job(identity)['directory']/'viewer.json'
        value = json.loads(metadata.read_text())
        value['status'] = 'running'
        metadata.write_text(json.dumps(value))
        runner = Mock(return_value=dict(status='complete'))
        restored = s.App(self.config, runner=runner)
        self.assertEqual(restored.job(identity)['status'], 'interrupted')
        runner.assert_not_called()
        with patch.object(s, 'MAX_JOBS', 1):
            fresh = restored.retry(identity, {})
            self.assertTrue(restored.job(fresh)['done'].wait(5))
            with self.assertRaises(s.RequestError) as caught:
                restored.retry(identity, {})
            self.assertEqual(caught.exception.status, 429)

    def test_busy_run_cannot_be_retried(self):
        release = threading.Event()
        app = s.App(self.config, runner=lambda *args: (release.wait(5), dict(status='complete'))[1])
        identity = app.start(image_bytes(), dict(mode='ocr'))
        try:
            with self.assertRaises(s.RequestError) as caught:
                app.retry(identity, {})
            self.assertEqual(caught.exception.status, 409)
            self.assertEqual(len(app.jobs), 1)
        finally:
            release.set()
            self.assertTrue(app.job(identity)['done'].wait(5))

    def test_upload_validation_precedes_worker_dispatch(self):
        runner = Mock(side_effect=AssertionError('invalid upload'))
        app = s.App(self.config, runner=runner)
        for content in (b'', b'<svg/>', image_bytes(size=(4097, 2)), image_bytes(mode='RGBA')[:50]):
            with self.subTest(size=len(content)), self.assertRaises(s.RequestError):
                app.start(content, dict(mode='ocr'))
        self.assertEqual(app.jobs, {})
        runner.assert_not_called()

    def test_local_origin_host_token_and_upload_size_guards(self):
        app = s.App(self.config, runner=Mock())
        http = self.serve(app)
        for headers in ({'Host': 'attacker.test'}, {'Origin': 'https://attacker.test'}, {'Sec-Fetch-Site': 'cross-site'}):
            self.assertEqual(self.request(http, 'GET', '/api/config', headers=headers)[0], 403)
        self.assertEqual(self.request(http, 'POST', '/api/jobs', image_bytes())[0], 403)
        self.assertEqual(self.request(http, 'POST', '/api/jobs', b'',
            {'X-Tralens-Token': app.token, 'Content-Length': str(s.MAX_UPLOAD+1)})[0], 413)
        self.assertEqual(app.jobs, {})

    def test_artifacts_do_not_follow_symlinks_or_traverse_paths(self):
        app = s.App(self.config, runner=lambda *args: dict(status='complete'))
        identity = app.start(image_bytes(), dict(mode='ocr'))
        self.assertTrue(app.job(identity)['done'].wait(5))
        directory = app.job(identity)['directory']
        (directory/'outside.txt').symlink_to(self.root/'secret.txt')
        (self.root/'secret.txt').write_text('not served')
        for path in ('../secret.txt', '/etc/passwd', 'outside.txt', 'run/../../secret.txt', 'bad\\path', 'bad\x00path'):
            with self.subTest(path=path), self.assertRaises(s.RequestError):
                app.artifact(identity, path)

    def test_run_options_are_bounded_and_cannot_supply_filesystem_paths(self):
        defaults = s.options({}, render_enabled=True)
        self.assertEqual(defaults['translation_policy'], 'coverage-first')
        self.assertEqual(defaults['max_attempts'], 64)
        self.assertEqual(defaults['replacement_policy'], 'all')
        self.assertFalse(defaults['quality_reread'])
        for value in ([], {'fonts': '/etc/passwd'}, {'mode': 'render'}, {'max_attempts': True},
                      {'max_attempts': 65}, {'allow_fragments': 'yes'}, {'render_policy': []}, {'translation_policy': []}):
            with self.subTest(value=value), self.assertRaises(s.RequestError):
                s.options(value, render_enabled=False)

    def test_quality_options_are_explicit_bounded_and_cannot_select_asset_paths(self):
        runner = Mock(return_value=dict(status='complete'))
        app = s.App(replace(self.config, fonts=self.root/'fonts.json'), runner=runner)
        for value in ({'replacement_policy': []}, {'replacement_policy': 'unknown'}, {'quality_reread': 1},
                      {'quality_reread': True, 'mode': 'ocr'}, {'quality_reread': True, 'translation_policy': 'conservative'},
                      {'replacement_policy': 'confidence-gated', 'mode': 'translate'},
                      {'manga_reread_model': '/etc/passwd'}, {'reread_model': 'unconfigured'}):
            with self.subTest(value=value), self.assertRaises(s.RequestError):
                app.start(image_bytes(), value)
        with self.assertRaisesRegex(s.RequestError, 'existing local snapshot'):
            app.start(image_bytes(), dict(quality_reread=True))
        self.assertFalse(app.jobs)
        runner.assert_not_called()

    def test_configured_readers_do_not_enable_quality_inference_without_run_opt_in(self):
        import main
        config = replace(self.config, fonts=self.root/'fonts.json', manga_reread_model=self.root/'snapshot')
        settings = s.options({}, render_enabled=True)
        app = s.App(config, runner=Mock())
        self.assertTrue(app.public_config()['quality_reread_available'])
        with patch.object(main, 'run_pipeline') as pipeline:
            s.execute_job(config, dict(image=self.root/'image.png', directory=self.root/'job', options=settings))
        self.assertFalse(pipeline.call_args.kwargs['quality_reread'])
        self.assertIsNone(pipeline.call_args.kwargs['manga_reread_model'])
        self.assertEqual(pipeline.call_args.kwargs['replacement_policy'], 'all')

    def test_mode_dispatch_reuses_coordinator_and_preview_workflow(self):
        import main
        from tralens.translation import workflow
        for mode in ('ocr', 'translate', 'render'):
            job = dict(image=self.root/'image.png', directory=self.root/'job',
                options=dict(mode=mode, max_attempts=3, allow_fragments=True, render_policy='conservative', translation_policy='coverage-first'))
            config = replace(self.config, fonts=self.root/'fonts.json')
            with patch.object(main, 'run_pipeline', return_value=dict(status='partial')) as pipeline, \
                    patch.object(workflow, 'run_preview', return_value=dict(status='complete')) as preview:
                s.execute_job(config, job)
            call = pipeline if mode == 'render' else preview
            call.assert_called_once()
            self.assertEqual(call.call_args.kwargs['run_local_model'], mode != 'ocr')
            self.assertTrue(call.call_args.kwargs['run_local_ocr'])
            self.assertEqual(call.call_args.kwargs['max_attempts'], 3)
            self.assertTrue(call.call_args.kwargs['allow_fragment_previews'])
            self.assertEqual(call.call_args.kwargs['translation_policy'], 'coverage-first')
            if mode == 'render':
                preview.assert_not_called()
                self.assertEqual(call.call_args.kwargs['fonts'], config.fonts)
            else:
                pipeline.assert_not_called()

    def test_frontend_respects_ollama_host_and_explicit_endpoint_override(self):
        from tools.dev_frontend import configured_endpoint
        for host, expected in (
            ('', 'http://127.0.0.1:11434'),
            ('100.100.1.2', 'http://100.100.1.2:11434'),
            ('localhost:11435', 'http://127.0.0.1:11435'),
            ('::1', 'http://[::1]:11434'),
            ('[::1]:11435', 'http://[::1]:11435'),
            ('http://10.0.0.2:11434/', 'http://10.0.0.2:11434'),
        ):
            with self.subTest(host=host), patch.dict(os.environ, {'OLLAMA_HOST': host}):
                self.assertEqual(configured_endpoint(), expected)
                self.assertEqual(configured_endpoint('http://127.0.0.1:11436'), 'http://127.0.0.1:11436')

    def test_invalid_environment_endpoint_is_not_silently_used_or_replaced(self):
        from tools.dev_frontend import configured_endpoint
        for host in ('https://example.com', '8.8.8.8', 'localhost/path', 'http://user:pass@localhost:11434'):
            with self.subTest(host=host), patch.dict(os.environ, {'OLLAMA_HOST': host}):
                with self.assertRaises(ValueError):
                    configured_endpoint()
                self.assertEqual(configured_endpoint('http://127.0.0.1:11434'), 'http://127.0.0.1:11434')

    def test_import_and_help_are_engine_free(self):
        script = '''
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'paddle', 'paddleocr', 'ollama', 'torch', 'transformers'}:
            raise AssertionError('engine import: '+fullname)
sys.meta_path.insert(0, Guard())
from tralens.devui.server import App
from tools.dev_frontend import main
main(['--help'])
'''
        result = subprocess.run([sys.executable, '-c', script, str(Path(__file__).resolve().parents[1]/'src')],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--fonts', result.stdout)

    def test_stage_view_states_never_invent_an_unpublished_page(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('optional local evidence unavailable; Node.js for frontend state checks')
        script = r'''
const assert = require('node:assert/strict');
const {previewStates, progressView, outputReady, missingOutputMessage, qualityNotice} = require(process.argv[1]);
const base = {status: 'running', stage: 'ocr', preview_stage: 'ocr', options: {mode: 'render'},
  detections: [], units: [], artifacts: [], rendering_coverage: null, error: null};
assert.equal(previewStates(null, true).original.available, true);
assert.equal(previewStates(null, true).final.available, false);
assert.equal(previewStates(base, true).ocr.state, 'running');
assert.equal(previewStates(base, true).translation.available, false);
const translating = {...base, stage: 'translation', preview_stage: 'translation',
  detections: [{source_id: 1}], units: [{status: 'preview', target: 'Hello'}]};
assert.equal(previewStates(translating).translation.state, 'running');
assert.equal(progressView(translating), 'translation');
// The PNG may still be being written: the rendering manifest must be present.
const pending = {...translating, artifacts: ['run/rendering/composite.png']};
assert.equal(outputReady(pending, 'final'), false);
const finished = {...pending, status: 'partial', stage: 'finished', preview_stage: 'finished',
  rendering_coverage: {rendered: 1, saved_units: 2}};
assert.equal(outputReady(finished, 'final'), true);
assert.equal(progressView(finished), 'final');
assert.equal(previewStates(finished).final.label, '1 replacement');
assert.match(qualityNotice(finished, true).copy, /broad replacement/);
assert.equal(qualityNotice(finished, true).action, 'Retry with quality checks');
assert.equal(qualityNotice(finished, false).action, 'Retry with artwork protection');
const protectedRun = {...finished, replacement_policy: 'confidence-gated', quality_reread: {enabled: true},
  replacement_confidence: {actual_replacements: 1, bound_translations_without_replacement: 1}};
assert.match(qualityNotice(protectedRun).copy, /paired rereads/);
assert.match(qualityNotice(protectedRun).copy, /1 translations kept without repainting/);
assert.equal(qualityNotice(protectedRun).action, null);
assert.equal(qualityNotice(base), null);
const skipped = {...finished, artifacts: [], rendering_coverage: null, units: [],
  rendering_skipped: 'no_strict_translations'};
assert.equal(previewStates(skipped).final.state, 'skipped');
assert.equal(progressView(skipped), 'translation');
assert.match(missingOutputMessage(skipped, 'final').copy, /No complete, source-bound translations/);
const ocrOnly = {...finished, artifacts: [], rendering_coverage: null, options: {mode: 'ocr'}};
assert.equal(previewStates(ocrOnly).translation.state, 'skipped');
assert.equal(previewStates(ocrOnly).clean.state, 'skipped');
const failed = {...base, status: 'failed', error: {message: 'OCR failed'}};
assert.equal(previewStates(failed).ocr.state, 'failed');
assert.equal(previewStates(failed).translation.state, 'skipped');
assert.match(missingOutputMessage(failed, 'final').copy, /run stopped/);
const startupFailed = {...failed, preview_stage: null};
assert.equal(previewStates(startupFailed).ocr.state, 'skipped');
assert.equal(previewStates(startupFailed).ocr.label, 'Setup failed');
assert.equal(previewStates(startupFailed).translation.state, 'skipped');
'''
        result = subprocess.run([node, '-e', script, str(s.STATIC/'app.js')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
