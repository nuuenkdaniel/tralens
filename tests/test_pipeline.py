"""Coordinator integration: real synthetic OCR proposals/translation/rendering, fake engines only."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from tests.translation import test_workflow as workflow_tests
from tests.rendering.font_fixture import synthetic_font
from tralens.rendering import automatic
from tralens.translation.runtime import Generation, GenerationError
import main


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = workflow_tests.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.font = self.root/'synthetic.ttf'
        self.font.write_bytes(synthetic_font())
        self.license = self.root/'license.txt'
        self.license.write_bytes(b'Synthetic fixture license evidence, not a legal certification.')
        self.fonts = self.root/'fonts.json'
        self.fonts.write_text(json.dumps({'schema_version': 1, 'fonts': [{
            'id': 'dialogue', 'role': 'dialogue', 'weight': 'regular', 'path': str(self.font),
            'sha256': self.sha(self.font), 'license': 'synthetic-test-only',
            'license_path': str(self.license), 'license_sha256': self.sha(self.license)}]}))
        self.provider = Mock(return_value=Generation('Hi!', '{"done":true,"message":{"content":"Hi!"}}'))

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_pipeline(self, name='job', **options):
        kwargs = dict(fonts=self.fonts, device='cpu', model_root=self.fixture.weights,
            run_local_ocr=True, run_local_model=True, ocr_factory=self.fixture.factory,
            runtime=SimpleNamespace(generate=self.provider), render_policy='conservative',
            removal_policy='legacy',visibility_policy='replacement-only')
        kwargs.update(options)
        with patch('socket.socket', side_effect=AssertionError('no live network')):
            return main.run_pipeline(self.fixture.image, self.root/name, **kwargs)

    def test_full_synthetic_pipeline_uses_bound_preview_and_preserves_pixels(self):
        source = self.fixture.image.read_bytes()
        render = automatic.run_automatic
        def compose(**kwargs):
            self.fixture.engine.ocr.close.assert_called_once()
            self.assertEqual(self.provider.call_count, 2)
            self.assertEqual(kwargs['preview'], self.root/'job/preview/preview.json')
            return render(**kwargs)
        with patch.object(automatic, 'run_automatic', side_effect=compose) as call:
            report = self.run_pipeline()
        call.assert_called_once()
        self.assertEqual((report['status'], report['stage']), ('complete', 'finished'), report)
        self.assertEqual(report['coverage']['rendered'], 2)
        self.assertEqual(report['preview']['ocr_cleanup'], 'ocr_closed_before_translation')
        self.assertFalse(report['approved_export'])
        self.assertFalse(report['full_page_translation_verified'])
        self.assertEqual(self.fixture.image.read_bytes(), source)
        job = self.root/'job'
        self.assertEqual(json.loads((job/'pipeline.json').read_text()), report)
        self.assertEqual(report['preview']['sha256'], self.sha(job/report['preview']['path']))
        self.assertEqual(report['rendering']['manifest_sha256'], self.sha(job/report['rendering']['manifest']))
        evidence = job/'rendering/evidence-preview.json'
        self.assertEqual(evidence.read_bytes(), (job/'preview/preview.json').read_bytes())
        with Image.open(job/'rendering/source.png') as src, Image.open(job/'rendering/clean.png') as clean, \
                Image.open(job/'rendering/composite.png') as comp, Image.open(job/'rendering/erase-mask.png') as erase, \
                Image.open(job/'rendering/alpha-mask.png') as alpha:
            self.assertTrue(np.array_equal(np.asarray(src)[np.asarray(erase) == 0], np.asarray(clean)[np.asarray(erase) == 0]))
            self.assertTrue(np.array_equal(np.asarray(clean)[np.asarray(alpha) == 0], np.asarray(comp)[np.asarray(alpha) == 0]))

    def test_translation_budget_limits_are_reported_not_full_page_success(self):
        report = self.run_pipeline(max_attempts=1)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['coverage']['strict_translations'], 1)
        self.assertEqual(report['coverage']['translation_budget_exhausted'], 1)
        self.assertEqual(report['coverage']['rendered'], 1)
        self.assertTrue(report['coverage']['partial_output'])
        self.provider.assert_called_once()

    def test_valid_detector_page_edge_does_not_fail_unrelated_replacements(self):
        from tralens.ocr.types import Text_Box
        from tralens.ocr.automatic_units import propose_page_from_boxes
        with Image.open(self.fixture.image) as source:
            width,height=source.size
        boxes=[Text_Box(b.paddle_box_id,.95,b.paddle_poly,b.paddle_text) for b in self.fixture.boxes]
        boxes+=[Text_Box(2,.3,[[width-20,height-15],[width,height-15],
            [width,height],[width-20,height]],'あ')]
        self.fixture.engine.raw_detections=[boxes]
        self.fixture.engine.automatic_unit_proposals=[propose_page_from_boxes(boxes,self.fixture.image,
            complementary_ownership=True,regroup_companions=True)]
        before=self.fixture.image.read_bytes()
        result=self.run_pipeline(render_policy='best-effort',replacement_policy='confidence-gated')
        self.assertEqual(result['status'],'partial',result)
        self.assertIsNone(result['error'])
        self.assertEqual(result['coverage']['rendered'],2)
        self.assertEqual(self.fixture.image.read_bytes(),before)

    def test_invalid_text_mask_checkpoint_is_rejected_before_ocr_or_translation(self):
        invalid = self.root/'invalid.onnx'
        invalid.write_bytes(b'not the approved model')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.run_pipeline(text_mask_model=invalid)
        self.provider.assert_not_called()
        self.fixture.engine.ocr.close.assert_not_called()
        self.assertFalse((self.root/'job').exists())

    def test_best_effort_recovers_layout_failure_without_more_translation_calls(self):
        report = self.run_pipeline(render_policy='best-effort', min_size=144, max_size=144)
        self.assertEqual(report['coverage']['rendered'], 2)
        self.assertEqual(report['render_policy'], 'best-effort')
        self.assertEqual(report['rendering']['quality_summary']['best_effort_rendered'], 2)
        self.assertEqual(self.provider.call_count, 2)
        self.assertFalse(report['approved_export'])

    def test_invalid_render_policy_is_rejected_before_upstream_calls(self):
        with self.assertRaisesRegex(ValueError, 'policy'):
            self.run_pipeline(render_policy='unknown')
        self.provider.assert_not_called()
        self.assertFalse((self.root/'job').exists())

    def test_restoration_is_automatic_in_the_main_pipeline_and_can_be_disabled(self):
        from tralens.ocr.automatic_units import propose_page_from_boxes
        with Image.open(self.fixture.image) as opened:
            pixels = np.asarray(opened).copy()
        pixels[np.all(pixels == 255, axis=2)] = [215, 195, 165]
        Image.fromarray(pixels).save(self.fixture.image)
        batch = propose_page_from_boxes(self.fixture.boxes, str(self.fixture.image),
            complementary_ownership=True, regroup_companions=True)
        self.fixture.engine.automatic_unit_proposals = [batch]
        for method, rendered in (('flat', 0), ('auto', 2)):
            self.fixture.engine.ocr.close.reset_mock()
            report = self.run_pipeline(method, background_method=method)
            self.assertEqual(report['coverage']['rendered'], rendered)
            self.assertEqual(report['background_method'], method)
            self.assertEqual(report['status'], 'complete' if rendered == 2 else 'partial')

    def test_render_evaluation_cap_retains_other_translation(self):
        report = self.run_pipeline(max_units=1)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['coverage']['strict_translations'], 2)
        self.assertEqual(report['coverage']['rendered'], 1)
        self.assertEqual(report['coverage']['render_skip_reasons'], {'evaluation_cap': 1})

    def test_typographic_rejection_does_not_erase_or_claim_success(self):
        report = self.run_pipeline(min_size=144, max_size=144)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['coverage']['rendered'], 0)
        manifest = json.loads((self.root/'job/rendering/manifest.json').read_text())
        self.assertEqual(manifest['pixel_audit']['clean_changed_pixels'], 0)
        self.assertEqual(sum(report['coverage']['render_skip_reasons'].values()), 2)

    def test_new_defaults_remove_source_even_when_requested_font_cannot_fit(self):
        report=main.run_pipeline(self.fixture.image,self.root/'removal-defaults',fonts=self.fonts,
            device='cpu',model_root=self.fixture.weights,run_local_ocr=True,run_local_model=True,
            ocr_factory=self.fixture.factory,runtime=SimpleNamespace(generate=self.provider),
            min_size=144,max_size=144)
        self.assertEqual(report['removal_policy'],'aggressive')
        self.assertEqual(report['visibility_policy'],'always-show')
        manifest=json.loads((self.root/'removal-defaults/rendering/manifest.json').read_bytes())
        self.assertGreater(manifest['pixel_audit']['clean_changed_pixels'],0)
        self.assertEqual(manifest['pixel_audit']['outside_cleanup_changed_pixels'],0)
        self.assertEqual(manifest['coverage']['rendered'],0)
        self.assertEqual(manifest['removal_summary']['removed'],2)

    def test_no_strict_results_skips_rendering_and_keeps_preview(self):
        for name, budget in (('zero', 0), ('failed-results', 2)):
            self.fixture.engine.ocr.close.reset_mock()
            self.provider.side_effect = GenerationError('synthetic provider failure', None)
            with patch.object(automatic, 'run_automatic', side_effect=AssertionError('no strict results')):
                report = self.run_pipeline(name, max_attempts=budget)
            self.assertEqual(report['status'], 'partial')
            self.assertEqual(report['rendering']['reason'], 'no_strict_translations')
            self.assertEqual(report['coverage']['rendered'], 0)
            self.assertFalse((self.root/name/'rendering').exists())
            self.assertTrue((self.root/name/'preview/preview.json').is_file())

    def test_artwork_retention_requires_an_explicit_nontext_decision(self):
        outcome = {'reasons':['visual_non_text']}
        row = {'status':'abstained', 'result':{'outcomes':[outcome]}}
        self.assertTrue(automatic.retained_artwork(row))
        self.assertFalse(automatic.retained_artwork({**row, 'status':'failed'}))
        self.assertFalse(automatic.retained_artwork({**row, 'luna_review_error':'provider timeout'}))
        self.assertFalse(automatic.retained_artwork({**row, 'luna_first_decision':{'category':'unreadable'}}))
        outcome['reasons'] = ('visual_non_text',)  # Live dataclass before the JSON checkpoint round trip.
        self.assertTrue(automatic.retained_artwork(row))
        outcome['reasons'] = ['unreadable']
        self.assertFalse(automatic.retained_artwork(row))

    def test_empty_ocr_completes_with_pixel_identical_composite(self):
        self.fixture.engine.raw_detections = [[]]
        self.fixture.engine.automatic_unit_proposals = [None]
        report = self.run_pipeline()
        self.assertEqual(report['status'], 'complete')
        from PIL import Image
        with Image.open(self.fixture.image) as source, Image.open(self.root/'job'/report['rendering']['composite']) as composite:
            self.assertEqual(source.convert('RGB').tobytes(),composite.convert('RGB').tobytes())
        self.assertEqual(report['coverage']['saved_units'], 0)
        self.provider.assert_not_called()

    def test_structurally_blocked_units_and_fragment_previews_are_never_promoted(self):
        self.fixture.engine.automatic_unit_proposals = [replace(self.fixture.batch, regions=())]
        with patch.object(automatic, 'run_automatic', side_effect=AssertionError('fragments cannot render')):
            report = self.run_pipeline(allow_fragment_previews=True, translation_policy='conservative')
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['coverage']['ocr_blocked_units'], 2)
        self.assertEqual(report['coverage']['ocr_blocking_reasons']['unknown_region_ownership'], 2)
        self.assertEqual(report['coverage']['strict_translations'], 0)
        self.assertEqual(report['preview']['fragment_previews'], 2)
        self.assertEqual(report['rendering']['reason'], 'no_strict_translations')

    def test_ocr_failure_is_checkpointed_and_does_not_render(self):
        self.fixture.engine.process_images.side_effect = RuntimeError('synthetic OCR failure')
        with patch.object(automatic, 'run_automatic', side_effect=AssertionError('failed OCR')):
            report = self.run_pipeline()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['rendering']['reason'], 'upstream_failed_or_interrupted')
        self.assertEqual(report['error']['type'], 'RuntimeError')
        self.provider.assert_not_called()
        self.fixture.engine.ocr.close.assert_called_once()

    def test_partial_translation_failure_still_renders_valid_strict_results(self):
        self.provider.side_effect = [self.provider.return_value, GenerationError('synthetic failure', None)]
        report = self.run_pipeline()
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['preview']['status'], 'complete_with_failures')
        self.assertEqual(report['coverage']['rendered'], 1)

    def test_interruption_never_starts_renderer(self):
        self.provider.side_effect = KeyboardInterrupt()
        with patch.object(automatic, 'run_automatic', side_effect=AssertionError('interrupted')):
            report = self.run_pipeline()
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(report['preview']['attempts'], 1)
        self.assertFalse((self.root/'job/rendering').exists())

    def test_render_failure_retains_preview_and_records_failed_stage(self):
        with patch.object(automatic, 'run_automatic', side_effect=ValueError('synthetic render failure')):
            report = self.run_pipeline()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['error']['stage'], 'rendering')
        self.assertEqual(report['rendering']['status'], 'failed')
        self.assertTrue((self.root/'job/preview/preview.json').is_file())
        self.assertEqual(json.loads((self.root/'job/pipeline.json').read_text()), report)

    def test_source_change_during_translation_fails_closed(self):
        def change(**kwargs):
            self.fixture.image.write_bytes(b'externally changed')
            return self.provider.return_value
        self.provider.side_effect = change
        with patch.object(automatic, 'run_automatic', side_effect=AssertionError('stale source')):
            report = self.run_pipeline()
        self.assertEqual(report['status'], 'failed')
        self.assertFalse((self.root/'job/rendering').exists())

    def test_font_manifest_drift_during_translation_fails_closed(self):
        def change(**kwargs):
            self.fonts.write_text(self.fonts.read_text()+' ')
            return self.provider.return_value
        self.provider.side_effect = change
        with patch.object(automatic, 'run_automatic', side_effect=AssertionError('changed fonts')):
            report = self.run_pipeline()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('changed after preflight', report['error']['message'])

    def test_renderer_must_bind_the_exact_preflight_and_saved_preview(self):
        render = automatic.run_automatic
        def drift(**kwargs):
            result = render(**kwargs)
            result['fonts_sha256'] = '0'*64
            return result
        with patch.object(automatic, 'run_automatic', side_effect=drift):
            report = self.run_pipeline()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('original pipeline inputs', report['error']['message'])
        self.assertNotIn('composite', report['rendering'])

    def test_font_license_changed_during_translation_blocks_rendering(self):
        def change(**kwargs):
            self.license.write_bytes(b'changed license evidence')
            return self.provider.return_value
        self.provider.side_effect = change
        with patch.object(automatic, 'run_automatic', side_effect=AssertionError('changed license')):
            report = self.run_pipeline()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('hash', report['error']['message'])

    def test_unsupported_source_rejected_before_ocr_or_output_reservation(self):
        for kind in ('size', 'mode', 'profile', 'orientation'):
            options = {}
            image = Image.new('RGB', (4097, 2) if kind == 'size' else (180, 130), 'white')
            if kind == 'mode':
                image = image.convert('RGBA')
                image.putpixel((0, 0), (255, 255, 255, 254))
            if kind == 'profile':
                options['icc_profile'] = b'synthetic unsupported profile'
            if kind == 'orientation':
                exif = Image.Exif()
                exif[274] = 6
                options['exif'] = exif
            image.save(self.fixture.image, **options)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.run_pipeline()
            self.fixture.factory.assert_not_called()
            self.provider.assert_not_called()
            self.assertFalse((self.root/'job').exists())

    def test_font_license_raqm_and_size_options_rejected_before_ocr(self):
        for options in ({'min_size': 7}, {'min_size': 30, 'max_size': 20}, {'max_units': 33}, {'font_id': 'foreign'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_pipeline(**options)
        with patch.object(automatic.n.features, 'check_feature', return_value=False):
            with self.assertRaisesRegex(ValueError, 'RAQM'):
                self.run_pipeline()
        self.license.write_bytes(b'changed license evidence')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.run_pipeline()
        self.fixture.factory.assert_not_called()
        self.provider.assert_not_called()
        self.assertFalse((self.root/'job').exists())

    def test_existing_output_and_symlink_inputs_or_ancestors_never_overwritten(self):
        destination = self.root/'job'
        destination.mkdir()
        keep = destination/'keep.txt'
        keep.write_text('user data')
        with self.assertRaises(ValueError):
            self.run_pipeline()
        self.assertEqual(keep.read_text(), 'user data')
        (self.root/'link').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_pipeline('link/other')
        font_link = self.root/'fonts-link.json'
        font_link.symlink_to(self.fonts)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_pipeline('other', fonts=font_link)
        self.fixture.factory.assert_not_called()

    def test_explicit_model_and_ocr_optins_required_by_api(self):
        for options in ({'run_local_ocr': False}, {'run_local_model': False}):
            with self.assertRaisesRegex(ValueError, 'opt-ins'):
                self.run_pipeline(**options)
        self.fixture.factory.assert_not_called()

    def test_malformed_font_manifest_cli_is_a_clean_preflight_rejection(self):
        self.fonts.write_text('{}')
        args = ['--image', str(self.fixture.image), '--output', str(self.root/'job'), '--device', 'cpu',
            '--model-root', str(self.fixture.weights), '--run-local-ocr', '--run-local-model',
            '--render', '--fonts', str(self.fonts)]
        with patch('tralens.translation.workflow.run_preview', side_effect=AssertionError('preflight first')):
            with self.assertRaises(SystemExit) as error:
                main.main(args)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root/'job').exists())

    def test_real_cli_dispatch_in_subprocess_without_engines_or_network(self):
        # The actual CLI, preview workflow and renderer execute. Only the OCR
        # engine and provider are replaced, using the exact synthetic fixture.
        code = '''
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'paddleocr', 'paddle', 'ollama', 'torch', 'transformers'}:
            raise AssertionError('forbidden engine import: '+fullname)
sys.meta_path.insert(0, Guard())
from tests.test_pipeline import PipelineTests
from tralens.translation import workflow
from unittest.mock import patch
import main
test = PipelineTests()
test.setUp()
original = workflow.run_preview
def preview(*args, **kwargs):
    kwargs.update(ocr_factory=test.fixture.factory, runtime=type('Provider', (), {'generate': test.provider})())
    return original(*args, **kwargs)
try:
    args = ['--image', str(test.fixture.image), '--output', str(test.root/'job'),
        '--device', 'cpu', '--model-root', str(test.fixture.weights), '--run-local-ocr',
        '--run-local-model', '--render', '--fonts', str(test.fonts), '--max-attempts', sys.argv[2]]
    with patch.object(workflow, 'run_preview', side_effect=preview), patch('socket.socket', side_effect=AssertionError('network')):
        result = main.main(args)
    assert result == int(sys.argv[3]), result
    assert (test.root/'job/pipeline.json').is_file()
finally:
    test.doCleanups()
'''
        root = Path(__file__).resolve().parents[1]
        for budget, expected in (('2', '0'), ('1', '1'), ('0', '1')):
            run = subprocess.run([sys.executable, '-c', code, str(root/'src'), budget, expected], cwd=root,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout+'\n'+run.stderr)


class PipelineCLITests(unittest.TestCase):
    def test_render_optins_reject_before_importing_stages(self):
        code = '''
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tralens', 'PIL', 'cv2', 'paddleocr', 'ollama'}:
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Guard())
import main
base = ['--image', 'unused', '--output', 'unused', '--device', 'cpu', '--model-root', 'unused', '--run-local-ocr']
for extra in (['--render'], ['--render', '--fonts', 'unused'], ['--render', '--run-local-model'], ['--fonts', 'unused']):
    try: main.main(base+extra)
    except SystemExit as error: assert error.code == 2, error.code
    else: raise AssertionError('expected opt-in rejection')
'''
        source = str(Path(__file__).resolve().parents[1]/'src')
        run = subprocess.run([sys.executable, '-I', '-c', code, source], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout+'\n'+run.stderr)

    def test_cli_forwards_options_and_distinguishes_outcomes(self):
        args = ['--image', 'input.png', '--output', 'fresh', '--device', 'cpu', '--model-root', 'local',
            '--run-local-ocr', '--run-local-model', '--render', '--fonts', 'fonts.json',
            '--font-id', 'dialogue', '--min-size', '12', '--max-size', '24', '--max-units', '5', '--max-attempts', '20']
        for status, expected in (('complete', 0), ('partial', 1), ('failed', 1), ('interrupted', 130)):
            report = {'status': status, 'coverage': None, 'rendering': {}, 'error': None}
            with patch.object(main, 'run_pipeline', return_value=report) as run:
                self.assertEqual(main.main(args), expected)
            kwargs = run.call_args.kwargs
            self.assertEqual((kwargs['font_id'], kwargs['min_size'], kwargs['max_size'], kwargs['max_units']),
                             ('dialogue', 12, 24, 5))
            self.assertEqual(kwargs['max_attempts'], 20)


if __name__ == '__main__':
    unittest.main()
