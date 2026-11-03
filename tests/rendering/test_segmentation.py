"""Opt-in automatic model routing with fake inference; no weight dependency."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from tralens.rendering import automatic as a, natural as n, restoration as r, segmentation as s
from . import test_restoration as fixtures


class ModelTests(unittest.TestCase):
    def test_wrong_checkpoint_is_rejected_before_loading_or_networking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.onnx'
            path.write_bytes(b'not a model')
            with patch.object(s.cv2.dnn, 'readNetFromONNX') as load:
                with self.assertRaisesRegex(ValueError, 'checkpoint'):
                    s.model_spec(path)
            load.assert_not_called()

    def test_buffer_load_native_transform_and_range_validation(self):
        content = b'mock'
        digest = n.digest(content)
        net = Mock()
        net.getUnconnectedOutLayersNames.return_value = ['seg']
        net.forward.return_value = [np.full((1, 1, 1024, 1024), .5, np.float32)]
        spec = dict(path='/unused/explicit.onnx', sha256=digest)
        with patch.object(s, 'MODEL_BYTES', 4), patch.object(s, 'MODEL_SHA256', digest), \
                patch.object(n, 'read_explicit', return_value=content), \
                patch.object(s.cv2.dnn, 'readNetFromONNX', return_value=net) as load:
            model = s.TextMask(Image.new('RGB', (180, 130), 'white'), spec)
            self.assertEqual(model.raw.shape, (130, 180))
            self.assertTrue((model.raw == 127).all())
            self.assertEqual(model.evidence['resized_size'], [180, 130])
            self.assertEqual(model.evidence['padding_right_bottom'], [844, 894])
            self.assertEqual(model.evidence['forwards'], 1)
            self.assertEqual(load.call_args.args[0].tobytes(), content)
            net.setPreferableTarget.assert_called_with(cv2.dnn.DNN_TARGET_CPU)
            net.forward.return_value = [np.full((1, 1, 1024, 1024), np.nan, np.float32)]
            with self.assertRaisesRegex(ValueError, 'probability'):
                s.TextMask(Image.new('RGB', (180, 130), 'white'), spec)

    def test_checkpoint_drift_stops_before_the_backend(self):
        with patch.object(n, 'read_explicit', return_value=b'changed'), \
                patch.object(s.cv2.dnn, 'readNetFromONNX') as load:
            with self.assertRaisesRegex(ValueError, 'changed'):
                s.TextMask(Image.new('RGB', (180, 130)), dict(path='/unused', sha256='0'*64))
            load.assert_not_called()


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixtures.AutomaticRestorationTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.fixture = self.helper.fixture
        self.spec = dict(path='/explicit/model.onnx', sha256='test-only')

    def run_render(self, name='semantic', **kwargs):
        return self.helper.run_render(name, text_mask_model=Path('/explicit/model.onnx'), **kwargs)

    def model(self):
        source = np.array(Image.open(self.fixture.image).convert('RGB'))
        # A fake segmentation signal, not mask-quality benchmark evidence.
        raw = np.where(np.max(source, axis=2) < 100, 245, 0).astype(np.uint8)
        return SimpleNamespace(raw=raw, evidence=dict(forwards=1, model='fake-test-only'))

    def forced_fallback(self):
        return patch.multiple(a, derive_plan=Mock(side_effect=a.AutomaticSkip('flat_test', 'forced flat failure')))

    def test_cheap_route_does_not_initialize_model_even_when_configured(self):
        with patch.object(s, 'model_spec', return_value=self.spec), \
                patch.object(s, 'TextMask', side_effect=AssertionError('unnecessary model work')):
            report = self.run_render()
        self.assertGreater(report['coverage']['rendered'], 0)
        self.assertNotIn('text_mask_inference', report)
        self.assertEqual(report['background_summary']['semantic_mask_rendered'], 0)

    def test_automatic_native_fallback_is_cached_and_wording_stays_exact(self):
        self.helper.recolor()
        with self.forced_fallback(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('classical_test', 'forced restoration failure')), \
                patch.object(s, 'model_spec', return_value=self.spec), patch.object(s, 'TextMask', return_value=self.model()) as load:
            report = self.run_render()
        load.assert_called_once()
        self.assertEqual(report['coverage']['rendered'], 2)
        self.assertEqual(report['background_summary']['semantic_mask_rendered'], 2)
        self.assertEqual(report['background_summary']['flat_rendered'], 2)
        for row in report['units']:
            self.assertEqual(row['binding']['text'], 'Hi!')
            self.assertEqual(row['layout']['original_text'], 'Hi!')
            self.assertEqual(row['cleanup_method'], 'flat_fill_semantic')
        n.verify_pixel_audit(report['pixel_audit'])
        self.assertTrue((self.fixture.root/'semantic/text-probability.png').is_file())

    def test_backend_failure_is_not_retried_for_every_unit_and_does_not_erase(self):
        self.helper.recolor()
        with self.forced_fallback(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('classical_test', 'forced restoration failure')), \
                patch.object(s, 'model_spec', return_value=self.spec), patch.object(s, 'TextMask', side_effect=ValueError('bad backend')) as load:
            report = self.run_render()
        load.assert_called_once()
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)
        self.assertTrue(all(row['background_attempts'][-1]['reason_code'] == 'text_mask_backend' for row in report['units']))

    def test_layout_failure_rolls_back_a_successfully_derived_semantic_mask(self):
        self.helper.recolor()
        with self.forced_fallback(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('classical_test', 'forced restoration failure')), \
                patch.object(s, 'model_spec', return_value=self.spec), patch.object(s, 'TextMask', return_value=self.model()), \
                patch.object(a.typography, 'layout_dialogue', side_effect=ValueError('no fit')):
            report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)

    def test_periodic_routing_records_its_method_and_uses_mask_only_compositing(self):
        self.helper.recolor()
        derive_native = s.derive

        def periodic_plan(image, detections, members, raw, *, surface):
            if surface == 'flat':
                raise a.AutomaticSkip('flat_test', 'test periodic route')
            # Routing-only double: primitive periodic recovery has separate
            # known-reference tests; this test does not claim texture quality.
            plan = derive_native(image, detections, members, raw, surface='flat')
            plan.restored_pixels = np.tile(plan.background, (int(plan.erase.sum()), 1)).astype(np.uint8)
            plan.background = None
            plan.evidence['surface'] = dict(method='phase_locked_tile', test_double=True)
            return plan

        with self.forced_fallback(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('classical_test', 'forced')), \
                patch.object(s, 'model_spec', return_value=self.spec), patch.object(s, 'TextMask', return_value=self.model()), \
                patch.object(s, 'derive', side_effect=periodic_plan):
            report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], 2)
        self.assertEqual(report['background_summary']['periodic_rendered'], 2)
        self.assertTrue(all(row['cleanup_method'] == 'phase_locked_tile' for row in report['units']))
        n.verify_pixel_audit(report['pixel_audit'])

    def test_flat_only_mode_does_not_attempt_periodic_repair(self):
        self.helper.recolor()
        with self.forced_fallback(), patch.object(s, 'model_spec', return_value=self.spec), \
                patch.object(s, 'TextMask', return_value=self.model()), \
                patch.object(s, 'derive', side_effect=a.AutomaticSkip('unsupported', 'test')) as derive:
            report = self.run_render(background_method='flat')
        self.assertTrue(all(call.kwargs['surface'] == 'flat' for call in derive.call_args_list))
        self.assertEqual(report['coverage']['rendered'], 0)


if __name__ == '__main__':
    unittest.main()
