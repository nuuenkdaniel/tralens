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
    def test_letter_sequence_requires_regular_printed_positions(self):
        def label(i,word,x,y):
            return dict(source_id=i,text=word,confidence=.98,polygon=[[x,y],[x+10,y],[x+10,y+12],[x,y+12]])
        seq=[label(3,'K',20,20),label(8,'L',60,23),label(4,'M',102,27)]
        self.assertTrue(all(rc.printed_sequence(i,seq) for i in (3,8,4)))
        self.assertFalse(rc.printed_sequence(3,seq[:2]))
        self.assertFalse(rc.printed_sequence(3,seq[:2]+[label(4,'M',96,57)]))

    def test_circled_letter_needs_enclosing_ink_in_the_original(self):
        from PIL import Image,ImageDraw
        art = Image.new('RGB',(90,64),'white')
        pen = ImageDraw.Draw(art)
        box = [(22,14),(46,14),(46,38),(22,38)]
        pen.text((28,19),'R',fill='black')
        self.assertFalse(rc.circled_letter(art,box,'R'))
        pen.ellipse((22,14,46,38),outline='black',width=2)
        self.assertTrue(rc.circled_letter(art,box,'R'))
        self.assertFalse(rc.circled_letter(art,box,'あ'))


    def test_weak_latin_over_hatching_requires_parallel_marks_outside_the_box(self):
        from PIL import ImageDraw
        image=Image.new('RGB',(60,80),'white')
        draw=ImageDraw.Draw(image)
        for y in (20,29,39):
            for dy in (0,3,6):
                draw.line((21,y+dy,25,y+dy+2),fill='black')
        line=dict(confidence=.39,text='lli',polygon=[[18,16],[29,16],[29,50],[18,50]])
        self.assertFalse(rc.pencil_hatching(image,line))
        for y in (12,22,32):
            draw.line((33,y,37,y+2),fill='black')
        self.assertTrue(rc.pencil_hatching(image,line))
        self.assertFalse(rc.pencil_hatching(image,dict(line,confidence=.95)))

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

    def test_unsupported_transcript_and_printed_literals(self):
        def check(source, target, **kwargs):
            reading = Reading(source, 'raw_main_reading', self.unit.selected.evidence_json)
            unit = replace(self.unit, readings=(reading,), selected_revision=reading.revision)
            return rc.original_lettering_reason(unit, target, [], (0,0,20,20), (100,100), **kwargs)
        self.assertEqual(check('……', 'Something is wrong.')[0], 'uncorroborated_source')
        self.assertEqual(check('……','...',reviewed_source='…')[0], 'replacement_unchanged_text')
        self.assertIsNone(check('助けて！', 'Help!'))
        self.assertEqual(check('foo_1', 'What happened to the screen?')[0], 'uncorroborated_source')
        self.assertIsNone(check('foo_1', 'What happened to the screen?', reviewed_source='画面どうしたの？'))
        self.assertEqual(check('case_1.pdf','Case one')[0], 'replacement_unchanged_text')
        from tralens.ocr.types import printed_latin_lines
        split = [dict(text=part,confidence=score,status='accepted') for part,score in [('ca',.99),('talog',.92)]]
        self.assertTrue(printed_latin_lines('catalog',split))
        self.assertFalse(printed_latin_lines('同log',split))
        self.assertFalse(printed_latin_lines('catalog',[split[0],dict(split[1],confidence=.65)]))
        self.assertEqual(check('$ ls -la | less', '$ ls -l all less')[0], 'replacement_unchanged_text')
        self.assertEqual(check('s/犬[1-9','s/dog[1-9]')[0], 'replacement_unchanged_text')
        self.assertEqual(check('s/質問/問題/g','replace the question')[0], 'replacement_unchanged_text')
        self.assertIsNone(check('質問して！','Ask!'))
        self.assertIsNone(check('参照 https://example.org', 'See https://example.org.'))
        self.assertEqual(check('参照 https://example.org', 'See https://example.org.uk')[0], 'uncorroborated_source')
        self.assertEqual(check('参照 https://exaωple.org', 'See https://exaωple.org')[0], 'uncorroborated_source')

    def test_preserve_existing_printed_labels_only_with_complete_raw_word_evidence(self):
        unit = self.unit_with_scores([.99, .99])
        rows = json.loads(unit.selected.evidence_json)
        rows[0].update(text='Open', status='accepted')
        rows[1].update(text='Closed', status='accepted')

        def selected():
            reading = Reading('OpenClosed', 'raw_main_reading', n.canonical(rows))
            return replace(unit, readings=(reading,), selected_revision=reading.revision)

        self.assertTrue(rc.retains_printed_words(selected(), 'Open, Closed.'))
        for changed in ('Open!', 'Open, closed.', 'Open and closed.', 'OPEN/CLOSED', 'Closed/Open', 'Open'):
            self.assertFalse(rc.retains_printed_words(selected(), changed))
        rows[0]['confidence'] = .8
        self.assertFalse(rc.retains_printed_words(selected(), 'Open/Closed'))

    def test_stacked_romanization_belongs_to_same_label(self):
        reading = Reading('青木花', 'raw_main_reading', self.unit.selected.evidence_json)
        unit = replace(self.unit, readings=(reading,), selected_revision=reading.revision)
        def row(key, text, x, y):
            return dict(source_id=key, confidence=.99, text=text,
                        polygon=[[x,y],[x+32,y],[x+32,y+14],[x,y+14]])
        def reason(rows,target='Aoki Hana'):
            return rc.original_lettering_reason(unit,target,rows,(30,40,140,90),(500,500))
        first,second=row(1,'Aoki',52,85),row(2,'Hana',53,99)
        self.assertEqual(reason([second,first],'青木花 / Aoki Hana')[0],'replacement_unchanged_text')
        self.assertIsNone(reason([second,first],'青木花 / Different Person'))
        self.assertEqual(reason([second,first])[0], 'replacement_unchanged_text')
        self.assertIsNone(reason([first,row(2,'Hana',99,99)]))
        self.assertIsNone(reason([first,dict(second,confidence=.82)]))

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
