"""Demand planning and real coordinator with injected readers/providers only."""
import copy
import json
from unittest import TestCase
from unittest.mock import patch

from tralens.rendering import automatic
from tralens.ocr import source_localization as crops


class DemandTests(TestCase):
    def setUp(self):
        from tests.rendering.test_localized_strokes import LocalizationPipelineTests
        self.fixture = LocalizationPipelineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.p = self.fixture.pipeline
        self.f = self.fixture.f

    def run_case(self, name='demand'):
        result = self.fixture.run_case(name, True)
        self.assertIsNone(result['error'], result)
        folder = self.p.root/name
        native = json.loads((folder/'rendering-native/manifest.json').read_bytes())
        preview = json.loads((folder/'preview/preview.json').read_bytes())
        return result, folder, preview, native

    def test_only_unresolved_crop_runs_after_native_and_translation_once(self):
        calls = []
        predict = self.f.engine.ocr.predict.side_effect
        def read(*args, **kwargs):
            calls.append(self.p.provider.call_count)
            self.assertTrue((self.p.root/'demand/rendering-native/manifest.json').is_file())
            return predict(*args, **kwargs)
        self.f.engine.ocr.predict.side_effect = read
        result, folder, preview, native = self.run_case()
        plan = native['localization_request_plan']
        self.assertEqual(len(plan['requests']), 1)
        placed = {r['unit_id'] for r in native['units'] if r['status'] == 'automatic_composed'}
        self.assertFalse(set(plan['requests']) & placed)
        self.assertEqual(calls, [2])
        self.assertEqual(self.p.provider.call_count, 2)
        self.f.engine.process_images.assert_called_once()
        self.assertNotIn('source_localization', preview['ocr'])
        derived = json.loads((folder/'localization/preview.json').read_bytes())
        self.assertEqual(derived['units'], preview['units'])
        self.assertEqual(derived['ocr']['raw_detections'], preview['ocr']['raw_detections'])
        self.assertEqual(result['source_localization']['predictions_consumed'], 1)
        self.assertEqual(result['source_localization']['calls'], 1)
        self.assertGreaterEqual(result['source_localization']['execution_seconds'], 0)

    def test_empty_plan_does_not_initialize_second_reader(self):
        self.fixture.native = automatic.derive_plan
        _, folder, _, native = self.run_case()
        self.assertEqual(native['localization_request_plan']['requests'], [])
        self.f.factory.assert_called_once()
        self.f.engine.ocr.predict.assert_not_called()
        self.assertTrue((folder/'rendering/composite.png').is_file())

    def test_budget_is_assigned_before_inference_and_misses_remain_visible(self):
        with patch('tralens.rendering.localized_strokes.MAX_RETRIES', 0):
            result, _, _, native = self.run_case()
        self.assertEqual(native['localization_request_plan']['requests'], [])
        self.assertGreater(result['source_localization']['budget_missed'], 0)
        self.f.engine.ocr.predict.assert_not_called()

    def test_model_free_failure_prevents_crop(self):
        with patch('tralens.rendering.localized_strokes.prerequisites',
                   side_effect=automatic.AutomaticSkip('localized_open', 'synthetic open region')):
            _, _, _, native = self.run_case()
        self.assertTrue(any(r['reason'] == 'localized_open' for r in native['localization_request_plan']['records']))
        self.f.engine.ocr.predict.assert_not_called()

    def test_failed_crop_preserves_primary_and_partial_status(self):
        self.f.engine.ocr.predict.side_effect = RuntimeError('synthetic crop failure')
        result, folder, _, native = self.run_case()
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['source_localization']['failed'], 1)
        final = json.loads((folder/'rendering/manifest.json').read_bytes())
        self.assertEqual(final['coverage']['rendered'], native['coverage']['rendered'])
        self.assertEqual((folder/'rendering/composite.png').read_bytes(), (folder/'rendering-native/composite.png').read_bytes())
        self.assertEqual(self.p.provider.call_count, 2)

    def test_optional_reader_initialization_failure_keeps_native_output(self):
        self.f.factory.side_effect = [self.f.engine, RuntimeError('synthetic crop-only initialization failure')]
        result, folder, _, _ = self.run_case()
        self.assertEqual(result['status'], 'partial')
        self.assertIn('initialization failure', result['source_localization']['error'])
        self.assertEqual((folder/'rendering/composite.png').read_bytes(),
                         (folder/'rendering-native/composite.png').read_bytes())
        self.f.engine.ocr.predict.assert_not_called()
        self.assertEqual(self.p.provider.call_count, 2)

    def test_stale_font_plan_and_exact_request_coverage_reject(self):
        _, folder, _, _ = self.run_case()
        data = json.loads((folder/'localization/preview.json').read_bytes())
        for index, change in enumerate(('font', 'style', 'requests', 'source')):
            forged = copy.deepcopy(data)
            plan = forged['ocr']['source_localization']['request_plan']
            if change == 'font':plan['fonts_sha256'] = '0'*64
            elif change == 'style':plan['style']['max_size'] -= 1
            elif change == 'requests':plan['requests'] = []
            else:plan['source']['sha256'] = '0'*64
            path = folder/f'forged-{index}.json';path.write_text(json.dumps(forged))
            with self.assertRaises(ValueError):
                automatic.run_automatic(preview=path, image=self.f.image, fonts=self.p.fonts,
                    output=folder/f'bad-{index}', replacement_policy='confidence-gated', cleanup_policy='localized')

    def test_stale_crop_plan_rejects_before_reader_or_output(self):
        _, folder, preview, native = self.run_case()
        plan = native['localization_request_plan']
        for key in ('preview_revision', 'geometry_revision'):
            bad = copy.deepcopy(plan);bad[key] = '0'*64
            with self.assertRaises(ValueError):
                crops.execute_requests(preview, bad, image=self.f.image, output=folder/key,
                    model_identity=self.f.engine.paddle_provenance, reader=self.f.engine.ocr)
            self.assertFalse((folder/key).exists())
        self.assertEqual(self.f.engine.ocr.predict.call_count, 1)

    def test_replay_uses_no_reader_and_keeps_provenance(self):
        _, folder, preview, native = self.run_case()
        saved = json.loads((folder/'localization/preview.json').read_bytes())['ocr']['source_localization']
        derived = crops.execute_requests(preview, native['localization_request_plan'], image=self.f.image,
            output=folder/'replay', model_identity=saved['model_identity'], replay=saved)
        stage = derived['ocr']['source_localization']
        self.assertEqual((stage['calls'], stage['reused_predictions']), (0, 1))
        self.assertTrue(any('replay_origin_revision' in r for r in stage['records']))
        self.assertEqual(self.f.engine.ocr.predict.call_count, 1)
        # Missing saved evidence must not silently initiate inference.
        saved['records'] = []
        derived = crops.execute_requests(preview, native['localization_request_plan'], image=self.f.image,
            output=folder/'missing', model_identity=saved['model_identity'], replay=saved)
        self.assertEqual(derived['ocr']['source_localization']['calls'], 0)
        self.assertTrue(any(r['reason'] == 'saved_prediction_unavailable'
                            for r in derived['ocr']['source_localization']['records']))
