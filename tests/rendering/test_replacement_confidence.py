"""Replacement-only triage: no production reference words, masks or model calls."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import test_pipeline as pipeline_fixtures
from tests.rendering import test_natural as fixtures
from tralens.rendering import automatic as a, best_effort as b, natural as n, replacement_confidence as rc, restoration as r
from tralens.translation.contracts import Reading


class ReplacementConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CommandTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.unit = next(u for u in n.automatic_units(self.f.preview['ocr']['batches'][0]) if 'main' in u.member_ids)

    def unit_with_scores(self, scores):
        raw = json.loads(self.unit.selected.evidence_json)
        rows = []
        for index, score in enumerate(scores):
            item = copy.deepcopy(raw[0])
            item.update(source_id='line-' + str(index), confidence=score)
            rows.append(item)
        ids = tuple(row['source_id'] for row in rows)
        reading = Reading(''.join(row['text'] for row in rows), 'raw_main_reading', n.canonical(rows))
        return replace(self.unit, member_ids=ids, order=ids, readings=(reading,), selected_revision=reading.revision)

    def test_weakest_line_not_average_and_exact_boundary(self):
        unit = self.unit_with_scores([1, 1, .69, 1])
        result = rc.assess_source(unit, 'Hello', .7)
        self.assertFalse(result['allowed'])
        self.assertEqual(result['minimum_main_line_score'], .69)
        self.assertEqual(result['reason_code'], 'replacement_low_ocr_confidence')
        self.assertEqual(result['source_reading_revision'], unit.selected_revision)
        unit = self.unit_with_scores([.7, .999])
        self.assertTrue(rc.assess_source(unit, 'Hello', .7)['allowed'])
        self.assertFalse(rc.settings('confidence-gated')['score_is_accuracy_probability'])

    def test_alternatives_missing_invalid_or_foreign_scores_do_not_inherit_confidence(self):
        for score in (True, -1, 1.01, '0.99', None):
            with self.subTest(score=score):
                result = rc.assess_source(self.unit_with_scores([score]), 'Hello', .7)
                self.assertEqual(result['reason_code'], 'replacement_confidence_unavailable')
                self.assertIsNone(result['minimum_main_line_score'])
        reading = replace(self.unit.selected, origin='automatic_manga_ocr')
        unit = replace(self.unit, readings=(reading,), selected_revision=reading.revision)
        self.assertEqual(rc.assess_source(unit, 'Hello', .7)['reason_code'], 'replacement_confidence_unavailable')
        raw = json.loads(self.unit.selected.evidence_json)
        for change in ({'source_id': 1}, {'text': 'different'}, {'page': {}}, {'status': 'rejected'}):
            reading = replace(self.unit.selected, evidence_json=n.canonical([{**raw[0], **change}]))
            unit = replace(self.unit, readings=(reading,), selected_revision=reading.revision)
            self.assertFalse(rc.assess_source(unit, 'Hello', .7)['allowed'])

    def test_identity_is_exact_not_normalized_or_a_claim_of_correct_ocr(self):
        unit = self.unit_with_scores([.1])
        result = rc.assess_source(unit, unit.selected.text, .7)
        self.assertFalse(result['allowed'])
        self.assertEqual(result['reason_code'], 'replacement_unchanged_text')
        self.assertEqual(result['minimum_main_line_score'], .1)
        high = self.unit_with_scores([.99])
        self.assertTrue(rc.assess_source(high, high.selected.text + ' ', .7)['allowed'])

    def test_configuration_rejects_bad_numbers_and_inactive_threshold(self):
        for score in (True, float('nan'), float('inf'), -0.01, 1.01, '0.7'):
            with self.subTest(score=score), self.assertRaises(ValueError):
                rc.settings('confidence-gated', score)
        with self.assertRaises(ValueError):
            rc.settings('all', .7)
        with self.assertRaises(ValueError):
            rc.settings('unknown')
        self.assertIsNone(rc.settings())
        self.assertEqual(rc.settings('confidence-gated', 0)['minimum_main_line_score'], 0)
        self.assertEqual(rc.settings('confidence-gated', 1)['minimum_main_line_score'], 1)

    def run_render(self, name, **kwargs):
        return a.run_automatic(preview=self.f.root / 'preview.json', image=self.f.image,
            fonts=self.f.root / 'fonts.json', output=self.f.root / name, **kwargs)

    def test_default_and_explicit_all_keep_every_png_identical(self):
        self.run_render('default')
        report = self.run_render('all', replacement_policy='all')
        self.assertNotIn('replacement_policy', report)
        self.assertNotIn('replacement_confidence', report)
        for path in (self.f.root / 'default').glob('*.png'):
            self.assertEqual(path.read_bytes(), (self.f.root / 'all' / path.name).read_bytes())

    def test_low_source_score_keeps_bound_translation_without_allocating_cleanup(self):
        before = (self.f.root / 'preview.json').read_bytes()
        with patch.object(a, 'OriginalComponents', side_effect=AssertionError('no geometry for deferred input')):
            report = self.run_render('low', replacement_policy='confidence-gated', min_ocr_confidence=1)
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['replacement_confidence']['low_ocr_confidence_deferred'], 1)
        self.assertEqual(report['replacement_confidence']['bound_translation_results_retained'], 1)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)
        self.assertEqual((self.f.root / 'preview.json').read_bytes(), before)
        row = next(row for row in report['units'] if 'binding' in row)
        self.assertEqual(row['binding']['text'], 'Hi!\nAV?')
        self.assertIn('EXPERIMENTAL CONFIDENCE GATE', (self.f.root / 'low/review.html').read_text())

    def test_unchanged_text_preserves_original_without_needing_target_font_glyphs(self):
        record = self.f.selected_record
        text = self.unit.selected.text
        record['result']['translations'] = ((self.unit.unit_id, text),)
        record['result']['outcomes'] = ({**record['result']['outcomes'][0], 'translation': text},)
        self.f.write_inputs()
        with patch.object(a, 'derive_plan', side_effect=AssertionError('no erase for identity target')):
            report = self.run_render('identity', replacement_policy='confidence-gated')
        self.assertEqual(report['replacement_confidence']['unchanged_text_preserved'], 1)
        self.assertEqual(report['coverage']['rendered'], 0)
        for name in ('clean.png', 'composite.png'):
            with Image.open(self.f.root / 'identity' / name) as result, Image.open(self.f.image) as source:
                self.assertTrue(np.array_equal(np.asarray(result), np.asarray(source)))

    def test_high_score_never_authorizes_coarse_footprint_after_native_failures(self):
        with patch.object(a, 'derive_plan', side_effect=a.AutomaticSkip('uncertain', 'no native support')), \
                patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('uncertain', 'no native support')), \
                patch.object(b, 'native_plan', side_effect=ValueError('ambiguous original components')), \
                patch.object(b, 'derive_plan', side_effect=AssertionError('coarse erasure must not run')):
            report = self.run_render('native-only', replacement_policy='confidence-gated')
        row = next(row for row in report['units'] if 'binding' in row)
        self.assertTrue(row['replacement_confidence']['allowed'])
        self.assertEqual(row['reason_code'], 'replacement_cleanup_confidence')
        self.assertEqual(report['replacement_confidence']['cleanup_support_deferred'], 1)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)

    def test_emergency_small_type_is_not_a_confident_replacement(self):
        report = self.run_render('small', min_size=144, max_size=144, replacement_policy='confidence-gated')
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['replacement_confidence']['readability_deferred'], 1)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)

    def test_successful_native_candidate_retains_exact_target_and_pixel_audits(self):
        before = self.f.image.read_bytes()
        report = self.run_render('native', replacement_policy='confidence-gated')
        self.assertEqual(report['coverage']['rendered'], 1)
        row = next(row for row in report['units'] if row['status'] == 'automatic_composed')
        self.assertEqual(row['binding']['text'], row['layout']['original_text'])
        self.assertFalse(report['replacement_confidence']['semantic_quality_verified'])
        n.verify_pixel_audit(report['pixel_audit'])
        self.assertEqual(self.f.image.read_bytes(), before)

    def test_shared_blank_layout_can_be_trimmed_without_changing_any_stroke(self):
        with Image.open(self.f.image) as source:
            plan = a.derive_plan(source, json.loads(n.canonical(self.f.preview['ocr']['raw_detections'][0])), list(self.unit.member_ids))
        reserved = np.zeros_like(plan.erase)
        candidates = plan.layout & ~plan.erase
        y, x = np.argwhere(candidates)[0]
        reserved[y, x] = True
        trimmed = a.reserve_layout(plan, reserved)
        self.assertFalse((trimmed.layout & reserved).any())
        self.assertTrue(trimmed.obstacles[y, x])
        self.assertTrue(np.array_equal(plan.erase, trimmed.erase))
        self.assertTrue(np.array_equal(plan.foreground, trimmed.foreground))
        self.assertEqual(trimmed.evidence['layout_reservation']['cleanup_pixels_discarded'], 0)

    def test_reservation_never_clips_erase_to_force_a_fit(self):
        with Image.open(self.f.image) as source:
            plan = a.derive_plan(source, json.loads(n.canonical(self.f.preview['ocr']['raw_detections'][0])), list(self.unit.member_ids))
        with self.assertRaises(a.AutomaticSkip):
            a.reserve_layout(plan, plan.erase.copy())


class ConfidencePipelineTests(unittest.TestCase):
    def setUp(self):
        self.p = pipeline_fixtures.PipelineTests()
        self.p.setUp()
        self.addCleanup(self.p.doCleanups)

    def test_gate_does_not_reduce_translation_calls_or_modify_coverage_requests(self):
        self.p.fixture.engine.automatic_unit_proposals = [replace(self.p.fixture.batch, regions=())]
        result = self.p.run_pipeline(render_policy='best-effort', replacement_policy='confidence-gated', min_ocr_confidence=1)
        self.assertEqual(result['coverage']['total_translations'], 2)
        self.assertEqual(self.p.provider.call_count, 2)
        self.assertEqual(result['coverage']['coverage_translations'], 2)
        self.assertEqual(result['coverage']['rendered'], 0)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['rendering']['replacement_confidence']['bound_translation_results_retained'], 2)
        preview = json.loads((self.p.root / 'job/preview/preview.json').read_bytes())
        self.assertTrue(all(record['unit']['blocked_reasons'] for record in preview['units']))
        self.assertTrue(all(record['result'] for record in preview['units']))

    def test_invalid_gate_is_rejected_before_upstream_inference(self):
        for options in ({'replacement_policy': 'invalid'}, {'min_ocr_confidence': .7},
                        {'replacement_policy': 'confidence-gated', 'min_ocr_confidence': float('nan')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.p.run_pipeline(**options)
            self.p.provider.assert_not_called()
            self.p.fixture.factory.assert_not_called()
            self.assertFalse((self.p.root / 'job').exists())


if __name__ == '__main__':
    unittest.main()
