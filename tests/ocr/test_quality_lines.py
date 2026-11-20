"""Portable mixed raw/paired fallback; no saved transcripts or live models."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image, ImageDraw

from tralens.ocr import quality_lines as lines, quality_reread as q
from tralens.ocr.automatic_units import propose_page_from_boxes
from tralens.ocr.types import Text_Box
from tralens.translation.contracts import automatic_units, canonical, Reading
from tralens.translation.runtime import Generation
from tralens.translation import coverage
from tralens.rendering import replacement_confidence as rc


class MixedLineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = Image.new('RGB', (240, 220), 'white')
        draw = ImageDraw.Draw(self.image)
        draw.rectangle((10, 10, 225, 205), outline='black', width=2)
        boxes = []
        for index, y in enumerate((40, 80, 120)):
            boxes.append(Text_Box(index, .6 if index == 1 else .91,
                [[40, y], [180, y], [180, y+30], [40, y+30]], 'あい' if index == 1 else 'いう'))
            for x in (45, 80, 115):
                draw.rectangle((x, y+5, x+8, y+22), fill='black')
        self.path = self.root/'input.png'
        self.image.save(self.path)
        self.content = self.path.read_bytes()
        self.boxes = boxes
        self.batch = propose_page_from_boxes(boxes, self.path, complementary_ownership=True, regroup_companions=True)
        self.batch = replace(self.batch, regions=())
        self.assertEqual(len(self.batch.proposals), 1)
        self.engine = SimpleNamespace(automatic_unit_proposals=[self.batch])
        self.vision = Mock(side_effect=self.read)
        self.manga = Mock(return_value='あお')

    def read(self, payload):
        if 'lines' in payload['format']['properties']:
            return Generation(canonical({'lines': {'0':'いう', '1':'あい', '2':'いう'}, 'uncertain':True, 'notes':'fixture primary rejected'}), '{}')
        return Generation(canonical({'text':'あお', 'uncertain':False}), '{"done":true}')

    def run_refine(self, **kwargs):
        output = self.root/'run'
        output.mkdir()
        return q.refine(self.engine, self.image, self.content, vision=self.vision, manga=self.manga,
                        model_identity='synthetic independent readers', output=output, **kwargs)

    def selected(self):
        batch = self.engine.automatic_unit_proposals[0]
        unit = automatic_units(batch, self.engine.automatic_recognition_results[0])[0]
        return unit, q.preferred_reading(unit)

    def test_only_weak_line_is_reread_and_raw_scores_are_not_laundered(self):
        before = self.image.tobytes()
        stage = self.run_refine()
        unit, candidate = self.selected()
        self.assertEqual(candidate.text, 'いうあおいう')
        self.assertEqual(unit.selected.text, 'いうあいいう')
        self.assertEqual(stage['primary_vision_calls'], 1)
        self.assertEqual(stage['fallback_vision_calls'], 1)
        self.assertEqual(stage['vision_calls'], 2)
        self.assertEqual(stage['line_calls'], 1)
        derived = coverage.derive(unit, candidate.revision)
        allowed = rc.assess_source(derived, 'Fixture target.', .7)
        self.assertTrue(allowed['allowed'])
        self.assertEqual([r['source_id'] for r in allowed['line_scores']], [0, 2])
        self.assertFalse(rc.assess_source(derived, 'Fixture target.', .95)['allowed'])
        self.assertEqual(self.image.tobytes(), before)
        raw = json.loads(canonical([asdict(r) for r in self.batch.detections]))
        data = json.loads(candidate.evidence_json)
        q.validate_pixels(data, self.image, raw)

    def test_context_retry_only_after_disagreement_and_counts_all_attempts(self):
        def read(payload):
            if 'lines' in payload['format']['properties']:
                return self.read(payload)
            context = len(payload['messages'][1]['images']) == 2
            return Generation(canonical({'text':'あお' if context else 'あう', 'uncertain':False}), '{}')
        self.vision.side_effect = read
        stage = self.run_refine()
        self.assertIsNotNone(self.selected()[1])
        self.assertEqual(stage['fallback_vision_calls'], 2)
        self.assertEqual(stage['line_calls'], len(lines.ANGLES)+1)
        self.assertEqual(self.engine.automatic_recognition_results[0][0].fallback_attempts, 2)

    def test_manga_rotation_corroborates_without_rewriting_exact_visual_string(self):
        self.manga.side_effect = ('違う', 'あお')
        stage = self.run_refine()
        self.assertEqual(stage['line_calls'], 2)
        self.assertEqual(self.selected()[1].text, 'いうあおいう')
        self.assertEqual(stage['line_fallbacks'][0]['lines'][1]['trials'][0]['readings'][1]['angle'], -4)

    def test_uncertainty_disagreement_and_budget_never_select_a_partial_block(self):
        self.manga.return_value = '不一致'
        stage = self.run_refine(max_fallback_calls=1)
        self.assertIsNone(self.selected()[1])
        self.assertEqual(stage['fallback_vision_calls'], 1)
        self.assertFalse(stage['line_fallbacks'][0]['preferred'])
        self.assertEqual(stage['line_fallbacks'][0]['status'], 'uncorroborated_line')

    def test_pixel_payload_raw_text_and_confidence_drift_reject(self):
        self.run_refine()
        unit, candidate = self.selected()
        proof = json.loads(candidate.evidence_json)['quality_reread']
        for mutate in (lambda p:p['lines'][0].update(text='forged'), lambda p:p['lines'][0].update(confidence=.99),
                       lambda p:p['lines'][1]['trials'][0]['readings'][0].update(text='different')):
            changed = copy.deepcopy(proof)
            mutate(changed)
            with self.assertRaises(ValueError):lines.validate_reading(unit, changed)
        raw = json.loads(canonical([asdict(r) for r in self.batch.detections]))
        for mutate in (lambda p:p['lines'][1]['binding'].update(native_pixels_sha256='0'*64),
                       lambda p:p['lines'][1]['trials'][0].update(payload_sha256='0'*64),
                       lambda p:p['lines'][1]['trials'][0]['readings'][0].update(angle=90)):
            changed = copy.deepcopy(proof)
            mutate(changed)
            with self.assertRaises(ValueError):lines.validate_pixels(changed, self.image, raw)

    def test_empty_or_ambiguous_components_cannot_form_a_presentation(self):
        raw = json.loads(canonical([asdict(r) for r in self.batch.detections]))
        with self.assertRaises(ValueError):lines.OriginalLines(Image.new('RGB', self.image.size, 'white'), raw).line(1)
        raw.append({**raw[1], 'source_id':'duplicate-overlap'})
        with self.assertRaises(ValueError):lines.OriginalLines(self.image, raw).line(1)

    def test_primary_paired_lines_do_not_become_artificial_sentence_breaks(self):
        self.vision.side_effect=lambda payload: Generation(canonical({'lines':{'0':'いう','1':'あお','2':'いう'},
            'uncertain':False,'notes':'synthetic independent line readings'}),'{}')
        self.manga.side_effect=lambda image,source_box_id,crop_bounds: 'あお' if source_box_id==1 else 'いう'
        stage=self.run_refine()
        unit,candidate=self.selected()
        self.assertEqual(candidate.text,'いうあおいう')
        self.assertEqual(stage['line_calls'],3)
        data=json.loads(candidate.evidence_json)
        self.assertEqual(data['quality_reread']['visual_line_texts'],['いう','あお','いう'])
        self.assertEqual(data['quality_reread']['line_join'],'')
        for mutate in (lambda p:p['quality_reread'].update(line_join='\n'),
                       lambda p:p['quality_reread']['visual_line_texts'].__setitem__(1,'別の文字')):
            changed=copy.deepcopy(data);mutate(changed)
            reading=Reading(changed['text'],q.ENGINE,canonical(changed))
            with self.assertRaises(ValueError):
                q.preferred_reading(replace(unit,readings=(unit.selected,reading)))
        # Older saved candidates remain valid evidence with their exact old text.
        legacy=copy.deepcopy(data)
        legacy['quality_reread'].pop('line_join');legacy['quality_reread'].pop('visual_line_texts')
        legacy['text']='いう\nあお\nいう';legacy['quality_reread']['verdict']['text']=legacy['text']
        reading=Reading(legacy['text'],q.ENGINE,canonical(legacy))
        restored=replace(unit,readings=(unit.selected,reading))
        self.assertEqual(q.preferred_reading(restored).text,legacy['text'])

    def high_score_audit_fixture(self):
        boxes=[Text_Box(b.paddle_box_id,.97,b.paddle_poly,'AーBの' if b.paddle_box_id==1 else b.paddle_text)
               for b in self.boxes]
        self.batch=replace(propose_page_from_boxes(boxes,self.path,complementary_ownership=True,regroup_companions=True),regions=())
        self.engine.automatic_unit_proposals=(self.batch,)
        self.vision.side_effect=lambda payload: self.read(payload) if 'lines' in payload['format']['properties'] else Generation(
            canonical({'text':'ＡｌＢの','uncertain':False}),'{}')
        self.manga.return_value='AlBの'

    def test_high_raw_score_does_not_skip_a_paired_mixed_script_glyph_audit(self):
        self.high_score_audit_fixture()
        stage=self.run_refine()
        unit,candidate=self.selected()
        self.assertEqual(stage['primary_vision_calls'],1)
        self.assertEqual(stage['fallback_vision_calls'],1)
        self.assertEqual(candidate.text,'いうＡｌＢのいう')
        proof=json.loads(candidate.evidence_json)['quality_reread']
        self.assertEqual(proof['script_audit_ids'],[1])
        self.assertEqual(proof['lines'][1]['kind'],'paired_isolated_line')
        assessment=rc.assess_source(coverage.derive(unit,candidate.revision),'Synthetic target.',.95)
        self.assertTrue(assessment['allowed'])
        self.assertEqual([line['source_id'] for line in assessment['line_scores']],[0,2])
        self.assertEqual(unit.selected.text,'いうAーBのいう')
        changed=copy.deepcopy(proof);changed['script_audit_ids']=[]
        with self.assertRaises(ValueError):lines.validate_reading(unit,changed)

    def test_high_score_audit_zero_budget_never_uses_a_dictionary_correction(self):
        self.high_score_audit_fixture()
        self.manga.return_value='independent disagreement'
        stage=self.run_refine(max_line_attempts=0)
        self.assertIsNone(self.selected()[1])
        self.assertEqual(stage['line_calls'],0)
        self.assertLessEqual(stage['vision_calls'],12)
        self.manga.assert_not_called()

    def test_high_score_audit_disagreement_retains_only_the_original_input(self):
        self.high_score_audit_fixture()
        self.manga.return_value='independent disagreement'
        stage=self.run_refine()
        self.assertIsNone(self.selected()[1])
        self.assertEqual(self.selected()[0].selected.text,'いうAーBのいう')
        self.assertEqual(stage['line_calls'],2*len(lines.ANGLES))
        self.assertLessEqual(stage['vision_calls'],12)

    def test_mixed_script_routing_hint_is_not_a_dictionary_or_auto_correction(self):
        self.assertTrue(q.suspect_latin_mark('AーBの'))
        self.assertTrue(q.suspect_latin_mark('ＡーＢ'))
        for text in ('キャラクター','A-B','A ー B','Aー','ーB','普通の日本語'):
            self.assertFalse(q.suspect_latin_mark(text))

    def test_full_pipeline_and_offline_review_keep_mixed_reading_binding(self):
        import main
        from tests.test_pipeline import PipelineTests
        from tools.translation_review import run as review
        fixture = PipelineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.engine.raw_detections = [self.boxes]
        self.engine.process_images = Mock()
        self.engine.ocr = SimpleNamespace(close=Mock())
        def refine(engine, *, output):
            q.refine(engine, self.image, self.content, vision=self.vision, manga=self.manga,
                     model_identity='synthetic independent readers', output=output)
        result = main.run_pipeline(self.path, self.root/'pipeline', fonts=fixture.fonts, device='cpu',
            model_root=fixture.fixture.weights, run_local_ocr=True, run_local_model=True,
            ocr_factory=Mock(return_value=self.engine), runtime=SimpleNamespace(generate=fixture.provider),
            render_policy='best-effort', replacement_policy='confidence-gated', ocr_refine=refine)
        self.assertEqual(result['coverage']['rendered'], 1, result)
        report = review(self.root/'pipeline/preview/preview.json', self.path, self.root/'offline-review')
        self.assertEqual(report['raw_detections'], 3)
        self.assertEqual(self.path.read_bytes(), self.content)


if __name__ == '__main__':
    unittest.main()
