"""Paired OCR/grouping hypotheses with fake readers and real source bindings."""
import copy
from dataclasses import asdict, replace
from io import BytesIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from tests import test_pipeline as pipeline_fixtures
from tests.ocr import test_automatic_recognition as fixtures
from tralens.ocr import quality_reread as q
from tralens.ocr.automatic_units import propose_page_from_boxes
from tralens.ocr.types import Text_Box
from tralens.translation import coverage
from tralens.translation.contracts import automatic_units, canonical, Reading, flatten_candidates
from tralens.translation.runtime import Generation, GenerationError
from tralens.rendering import natural as n, replacement_confidence as rc


class QualityRereadTests(unittest.TestCase):
    def setUp(self):
        self.p = pipeline_fixtures.PipelineTests()
        self.p.setUp()
        self.addCleanup(self.p.doCleanups)
        f = self.p.fixture
        boxes = [Text_Box(b.paddle_box_id, .6, b.paddle_poly, b.paddle_text) for b in f.boxes]
        batch = propose_page_from_boxes(boxes, f.image, complementary_ownership=True, regroup_companions=True)
        batch = replace(batch, regions=())
        f.boxes, f.batch = boxes, batch
        f.engine.raw_detections = [boxes]
        f.engine.automatic_unit_proposals = [batch]
        self.before = copy.deepcopy(asdict(batch))
        self.vision = Mock(return_value=Generation(canonical({'lines': {'0': 'こんばんは'}, 'uncertain': False, 'notes': 'Fixture only.'}), '{"done":true}'))
        self.manga = Mock(return_value='こんばんは')

    def refine(self, engine, *, output, **kwargs):
        with Image.open(self.p.fixture.image) as opened:
            return q.refine(engine, opened.convert('RGB'), self.p.fixture.image.read_bytes(),
                vision=self.vision, manga=self.manga, model_identity='two explicit synthetic readers', output=output, **kwargs)

    def run_pipeline(self, **kwargs):
        return self.p.run_pipeline(render_policy='best-effort', replacement_policy='confidence-gated',
                                   ocr_refine=self.refine, **kwargs)

    def test_actual_pipeline_translates_and_renders_exact_corrobated_candidate_without_changing_raw(self):
        result = self.run_pipeline()
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['coverage']['total_translations'], 2)
        self.assertEqual(result['coverage']['rendered'], 2)
        self.assertEqual(self.p.provider.call_count, 2)
        saved = json.loads((self.p.root/'job/preview/preview.json').read_bytes())
        self.assertEqual(saved['ocr']['original_unit_proposals'][0], json.loads(canonical(self.before)))
        self.assertEqual(saved['ocr']['batches'][0]['detections'], self.before['detections'] if isinstance(self.before['detections'], list) else json.loads(canonical(self.before['detections'])))
        for record in saved['units']:
            self.assertEqual(record['unit']['selected_revision'], record['coverage']['source_reading_revision'])
            self.assertNotEqual(record['translation_input']['selected_revision'], record['unit']['selected_revision'])
            unit = next(u for u in automatic_units(saved['ocr']['batches'][0], saved['ocr']['automatic_recognition_results'][0]) if u.unit_id == record['unit']['unit_id'])
            candidate = q.preferred_reading(unit)
            self.assertEqual(candidate.text, 'こんばんは')
            self.assertEqual(unit.selected.text, 'こんにちは')
            self.assertEqual(coverage.bound_result(record, unit)['text'], 'Hi!')
            derived = coverage.derive(unit, candidate.revision)
            assessment = rc.assess_source(derived, 'Hi!', .99)
            self.assertTrue(assessment['allowed'])
            self.assertIsNone(assessment['minimum_main_line_score'])
        # Renderer rechecks candidate view pixels against the explicit original.
        manifest = json.loads((self.p.root/'job/rendering/manifest.json').read_bytes())
        n.verify_pixel_audit(manifest['pixel_audit'])
        self.assertTrue(all(r['layout']['original_text'] == 'Hi!' for r in manifest['units']))
        from tools.translation_review import run
        review = run(self.p.root/'job/preview/preview.json', self.p.fixture.image, self.p.root/'offline-review')
        self.assertEqual(review['raw_detections'], 2)

    def test_disagreement_uncertainty_and_line_count_never_select_visual_fluency(self):
        for index, verdict in enumerate(({'text':'こんばんは', 'uncertain':True, 'notes':'ambiguous'},
                                          {'text':'こんばんは\n余計', 'uncertain':False, 'notes':'wrong line count'},
                                          {'text':'こんばんは', 'uncertain':False, 'notes':'disagreement'})):
            self.p.fixture.engine.automatic_unit_proposals = [self.p.fixture.batch]
            self.p.fixture.engine.__dict__.pop('quality_rereads', None)
            output = self.p.root/('probe-'+str(index));output.mkdir()
            answer = {'lines': {str(i): text for i, text in enumerate(verdict['text'].splitlines())},
                      'uncertain': verdict['uncertain'], 'notes': verdict['notes']}
            self.vision.return_value = Generation(canonical(answer), '{}')
            self.manga.return_value = '違う'
            stage = self.refine(self.p.fixture.engine, output=output)
            units = automatic_units(self.p.fixture.engine.automatic_unit_proposals[0], self.p.fixture.engine.automatic_recognition_results[0])
            self.assertTrue(all(q.preferred_reading(unit) is None for unit in units))
            self.assertTrue(all(record['preferred'] is False for record in stage['records']))
            self.assertEqual(stage['vision_calls'], 2)

    def test_budget_failures_count_and_do_not_discard_original_units(self):
        self.vision.side_effect = RuntimeError('fixture provider failed')
        output = self.p.root/'failed';output.mkdir()
        stage = self.refine(self.p.fixture.engine, output=output, max_units=1)
        self.assertEqual(stage['vision_calls'], 1)
        self.assertEqual(stage['line_calls'], 0)
        self.assertEqual(stage['records'][0]['status'], 'failed')
        self.assertEqual(len(self.p.fixture.engine.automatic_unit_proposals[0].proposals), 2)
        with self.assertRaises(ValueError):
            self.refine(self.p.fixture.engine, output=output)

    def test_truncated_model_output_is_retained_and_never_selected(self):
        failed = '{"done":true,"done_reason":"length","message":{"content":"partial"}}'
        self.vision.side_effect = GenerationError('truncated', failed)
        output = self.p.root/'truncated';output.mkdir()
        stage = self.refine(self.p.fixture.engine, output=output, max_units=1)
        self.assertEqual(stage['records'][0]['status'], 'failed')
        self.assertTrue(stage['records'][0]['failed_response_sha256'])
        self.assertEqual(next(output.glob('*-failed-response.json')).read_text(), failed)
        self.assertFalse(stage['records'][0]['preferred'])

    def test_empty_but_correctly_bound_enclosure_does_not_call_either_decoder(self):
        f = self.p.fixture
        with Image.open(f.image) as source:
            image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        for box in f.boxes:
            draw.rectangle(box.bbox, fill='white')
        image.save(f.image)
        f.batch = replace(propose_page_from_boxes(f.boxes, f.image), regions=())
        f.engine.automatic_unit_proposals = [f.batch]
        output = self.p.root/'empty';output.mkdir()
        stage = self.refine(f.engine, output=output)
        self.assertEqual(stage['vision_calls'], 0)
        self.vision.assert_not_called()
        self.manga.assert_not_called()

    def test_presentation_resource_failure_defers_only_reread_not_original_units(self):
        output = self.p.root/'over-budget';output.mkdir()
        with patch.object(q, 'vision_payload', return_value=({}, b'x'*(4*1024*1024+1))):
            stage = self.refine(self.p.fixture.engine, output=output)
        self.assertEqual(stage['vision_calls'], 0)
        self.assertEqual(stage['line_calls'], 0)
        self.assertLessEqual(stage['artifact_bytes'], 64*1024*1024)
        self.assertTrue(all('presentation_failed' in result.reason for result in self.p.fixture.engine.automatic_recognition_results[0]))
        self.assertEqual(len(self.p.fixture.engine.automatic_unit_proposals[0].proposals), 2)
        self.vision.assert_not_called()

    def test_line_and_block_pixel_or_preferred_evidence_drift_is_rejected(self):
        output = self.p.root/'probe';output.mkdir()
        self.refine(self.p.fixture.engine, output=output)
        batch = self.p.fixture.engine.automatic_unit_proposals[0]
        candidates = flatten_candidates(self.p.fixture.engine.automatic_recognition_results[0])
        units = automatic_units(batch, candidates)
        raw = json.loads(canonical([asdict(d) for d in batch.detections]))
        with Image.open(self.p.fixture.image) as opened:
            image = opened.convert('RGB')
        for mutate in (lambda c:c['quality_reread'].update(native_pixels_sha256='0'*64),
                       lambda c:c['quality_reread']['line_evidence'][0]['readings'][0].update(bounds=[0,0,100000,100000]),
                       lambda c:c['quality_reread']['line_evidence'][0]['readings'][0].update(pixels_sha256='0'*64)):
            candidate = copy.deepcopy(candidates[0]);mutate(candidate)
            with self.assertRaises(ValueError):q.validate_pixels(candidate, image, raw)
        candidate = copy.deepcopy(candidates[0]);candidate['quality_reread']['line_evidence'][0]['readings'][0]['text']='invented'
        unit = units[0]
        original_reading = next(r for r in unit.readings if r.origin == q.ENGINE)
        modified = replace(original_reading, evidence_json=canonical(candidate))
        with self.assertRaises(ValueError):
            q.preferred_reading(replace(unit, readings=(unit.selected, modified)))

    def test_empty_original_crop_cannot_be_promoted_from_decoder_hallucination(self):
        with Image.open(self.p.fixture.image) as opened:
            image = opened.convert('RGB')
        # Unit-level original source binding rejects a supplied blank raster.
        with self.assertRaises(ValueError):
            q.refine(self.p.fixture.engine, Image.new('RGB', image.size, 'white'), self.p.fixture.image.read_bytes(),
                vision=self.vision, manga=self.manga, model_identity='fixture', output=self.p.root)
        self.vision.assert_not_called()


class TerminalRowTests(unittest.TestCase):
    def fixture(self):
        image = Image.new('RGB', (260, 240), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 235, 220), outline='black', width=3)
        boxes = []
        for index, (x,y,r,b) in enumerate(((55,40,190,75),(55,80,190,115),(55,120,190,155),(100,154,135,174))):
            boxes.append(Text_Box(index,.6,[[x,y],[r,y],[r,b],[x,b]],'あいう' if index<3 else 'で'))
            draw.rectangle((x+5,y+5,x+12,y+12),fill='black')
        page = fixtures.AutomaticRecognitionTests().fixture()[2].page
        stream=BytesIO();image.save(stream,format='PNG');content=stream.getvalue()
        from tralens.ocr.recognition_units import snapshot_detections, image_fingerprint
        from tralens.ocr.automatic_units import propose_automatic_units
        page=replace(page,image_sha256=__import__('hashlib').sha256(content).hexdigest(),pixels_sha256=image_fingerprint(image))
        batch=propose_automatic_units(snapshot_detections(boxes,page),image)
        self.assertEqual([p.main_ids for p in batch.proposals],[(0,1,2),(3,)])
        return image,batch

    def test_short_terminal_row_same_original_enclosure_joins_once_and_retains_original(self):
        image,batch=self.fixture();before=asdict(batch)
        updated,evidence=q.terminal_groups(batch,q.enclosure_map(image,batch.detections))
        self.assertEqual(len(updated.proposals),1)
        self.assertEqual(updated.proposals[0].main_ids,(0,1,2,3))
        self.assertEqual(asdict(batch),before)
        self.assertEqual(len(evidence),1)
        self.assertIn(q.TERMINAL_REASON,updated.proposals[0].structural_reasons)

    def test_different_enclosure_or_ambiguous_host_is_not_a_terminal_merge(self):
        image,batch=self.fixture();owners=q.enclosure_map(image,batch.detections)
        owners[3]=(999,.99)
        updated,evidence=q.terminal_groups(batch,owners)
        self.assertEqual(updated,batch)
        self.assertFalse(evidence)


if __name__ == '__main__':unittest.main()
