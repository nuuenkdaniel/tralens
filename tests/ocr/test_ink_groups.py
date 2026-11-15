"""Original-ink joins, retained partitions and saved-pixel handoffs; no live models."""
from copy import deepcopy
from dataclasses import asdict, replace
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from tralens.ocr import ink_groups as g
from tralens.ocr.automatic_recognition import AutomaticRecognitionResult
from tralens.ocr.automatic_units import propose_automatic_units, crop_conflicts
from tralens.ocr.recognition_units import PageProvenance, snapshot_detections, image_fingerprint
from tralens.ocr.types import Text_Box
from tralens.translation.contracts import automatic_units, canonical, revision


class InkGroupTests(unittest.TestCase):
    def fixture(self, *, third=False, divider=False, empty=False, path='synthetic.png'):
        image = Image.new('RGB', (320, 240), 'white');draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 300, 220), outline='black', width=3)
        boxes = []
        locations = [(180, 50, 150), (150, 50, 150), (120, 50, 102), (120, 104, 150)]
        if third:locations.append((210, 50, 150))
        for identity, (x, y, bottom) in enumerate(locations):
            boxes.append(Text_Box(identity, .6, [[x-4,y],[x+30,y],[x+30,bottom],[x-4,bottom]], 'あいう'))
            if not empty:
                for top in range(y+5, bottom-14, 24):
                    draw.rectangle((x+5, top, x+24, top+13), fill='black')
        if divider:draw.line((149, 23, 149, 180), fill='black', width=2)
        data = BytesIO();image.save(data, format='PNG');original = data.getvalue()
        page = PageProvenance('synthetic', str(path), hashlib.sha256(original).hexdigest(), '0'*64, image_fingerprint(image))
        batch = propose_automatic_units(snapshot_detections(boxes, page), image)
        template = batch.proposals[0]
        def proposal(ids):
            points = [p for i in ids for p in batch.detections[i].polygon];xs, ys = zip(*points)
            bounds = (min(xs), min(ys), max(xs), max(ys))
            identity = {'policy': batch.policy, 'page': asdict(page), 'main_ids': ids, 'members': ids,
                'member_polygons': [(i, batch.detections[i].polygon) for i in ids], 'annotation_parents': (), 'direction': 'vertical'}
            name = 'proposal-'+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
            conflicts = crop_conflicts(bounds, ids, batch.detections)
            return replace(template, proposal_id=name, main_ids=ids, member_ids=ids, annotation_ids=(),
                annotation_parents=(), direction='vertical', crop_bounds=bounds, crop_conflicts=conflicts,
                crop_safe=not conflicts, structural_reasons=('automatic_not_human_reviewed', 'unknown_region_ownership'),
                word_reasons=(), region_ids=(), grouping_method='synthetic_partition')
        proposals = [proposal((0, 1)), proposal((2, 3))]
        if third:proposals.append(proposal((4,)))
        return image, original, replace(batch, proposals=tuple(proposals), regions=()), boxes

    def test_overlapping_detector_margins_and_stacked_pieces_use_original_seam(self):
        image, original, batch, _ = self.fixture();before = asdict(batch);pixels = image.tobytes()
        updated, evidence = g.plan(batch, image)
        self.assertEqual(len(updated['proposals']), 1)
        self.assertEqual(updated['proposals'][0]['main_ids'], [0,1,2,3])
        self.assertEqual(evidence['records'][0]['columns'][-1]['ids'], [2,3])
        self.assertEqual(evidence['records'][0]['seams'][0]['white_fraction'], 1)
        self.assertEqual(updated['detections'], g.snapshot(batch)['detections'])
        self.assertEqual(updated['proposals'][0]['crop_bounds'], [116,50,210,150])
        self.assertFalse(evidence['records'][0]['erase_permission'])
        self.assertEqual(asdict(batch), before);self.assertEqual(image.tobytes(), pixels)
        reversed_batch = replace(batch, proposals=tuple(reversed(batch.proposals)))
        other, _ = g.plan(reversed_batch, image)
        self.assertEqual(other['proposals'][0]['proposal_id'], updated['proposals'][0]['proposal_id'])

    def test_blank_ink_separator_unknown_orientation_and_ambiguity_do_not_join(self):
        for options in ({'empty':True}, {'divider':True}, {'third':True}):
            image, _, batch, _ = self.fixture(**options)
            updated, evidence = g.plan(batch, image)
            self.assertEqual(updated, g.snapshot(batch), options)
            self.assertFalse(evidence['records'])
        image, _, batch, _ = self.fixture()
        unknown = replace(batch, proposals=(batch.proposals[0], replace(batch.proposals[1], direction='unknown')))
        self.assertEqual(g.plan(unknown, image)[0], g.snapshot(unknown))

    def test_interleaved_membership_and_duplicate_source_are_not_continuations(self):
        image, _, batch, _ = self.fixture()
        a,b = batch.proposals
        interleaved = replace(batch, proposals=(replace(a, main_ids=(0,2), member_ids=(0,2)),
            replace(b, main_ids=(1,3), member_ids=(1,3))))
        self.assertFalse(g.plan(interleaved, image)[1]['records'])
        duplicated = replace(batch, proposals=(a, replace(b, main_ids=(1,2,3), member_ids=(1,2,3))))
        with self.assertRaisesRegex(ValueError, 'duplicate'):g.plan(duplicated, image)

    def test_raw_strings_and_scores_do_not_choose_a_group(self):
        from tralens.ocr.automatic_units import detection_evidence_sha256
        image, _, batch, _ = self.fixture()
        raw = tuple(replace(d, text='arbitrary', confidence=.99) for d in batch.detections)
        changed = replace(batch, detections=raw, evidence_sha256=detection_evidence_sha256(raw))
        old = g.plan(batch, image)[1]['records'];new = g.plan(changed, image)[1]['records']
        self.assertEqual(old, new)

    def test_existing_corner_conflict_survives_json_roundtrip(self):
        from tralens.ocr.automatic_units import detection_evidence_sha256
        image, original, batch, _ = self.fixture()
        corner = replace(batch.detections[0], source_id='corner', polygon=((204,145),(216,145),(216,158),(204,158)))
        detections = batch.detections+(corner,)
        batch = replace(batch, detections=detections, evidence_sha256=detection_evidence_sha256(detections))
        engine = SimpleNamespace(automatic_unit_proposals=(batch,))
        g.refine(engine,image,original)
        current = g.snapshot(engine.automatic_unit_proposals[0])
        self.assertEqual(len(current['proposals']),1)
        self.assertEqual(current['proposals'][0]['crop_conflicts'][0]['source_id'],'corner')
        ocr = json.loads(canonical({'batches':[current], 'ink_column_groups':engine.ink_column_groups}))
        g.validate_saved(ocr,image)

    def test_refiner_retains_prior_partition_results_and_validates_saved_geometry(self):
        image, original, batch, _ = self.fixture()
        results = tuple(AutomaticRecognitionResult(p.proposal_id, reason='retained failed trial', error='synthetic') for p in batch.proposals)
        engine = SimpleNamespace(automatic_unit_proposals=(batch,), automatic_recognition_results=(results,))
        prior = canonical([asdict(r) for r in results]);g.refine(engine, image, original)
        ocr = {'batches':[g.snapshot(engine.automatic_unit_proposals[0])], 'ink_column_groups':deepcopy(engine.ink_column_groups)}
        g.validate_saved(ocr, image)
        self.assertEqual(canonical(ocr['ink_column_groups']['previous_results']), prior)
        self.assertEqual(ocr['ink_column_groups']['previous_batch'], g.snapshot(batch))
        self.assertEqual(len(engine.automatic_recognition_results[0]), 1)
        with self.assertRaisesRegex(ValueError, 'already attempted'):g.refine(engine, image, original)
        for kind in ('seam', 'line', 'prior', 'output', 'missing'):
            changed = deepcopy(ocr);stage = changed['ink_column_groups']
            if kind == 'seam':stage['records'][0]['seams'][0]['original_rgb_sha256'] = '0'*64
            elif kind == 'line':stage['records'][0]['lines'][0]['ink_bounds'][0] += 1
            elif kind == 'prior':stage['previous_results'][0]['reason'] = 'changed'
            elif kind == 'output':changed['batches'][0]['proposals'][0]['main_ids'].reverse()
            else:changed.pop('ink_column_groups')
            with self.subTest(kind=kind),self.assertRaises(ValueError):g.validate_saved(changed, image)
        changed = image.copy();changed.putpixel((179,70), (0,0,0))
        with self.assertRaisesRegex(ValueError, 'drift'):g.validate_saved(ocr, changed)

    def test_actual_coordinator_translates_group_once_and_renderer_rechecks_saved_seam(self):
        import main
        from tests.test_pipeline import PipelineTests
        from tralens.rendering.automatic import run_automatic
        fixture = PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        path = fixture.root/'ink-source.png'
        image, original, batch, boxes = self.fixture(path=path);path.write_bytes(original)
        engine = SimpleNamespace(automatic_unit_proposals=(batch,), raw_detections=[boxes], image_path=str(path),
            process_images=Mock(), ocr=SimpleNamespace(close=Mock()))
        reader = Mock(return_value='あいうあいうあいうあいう');reader.identity={'synthetic':True}
        with patch('tralens.ocr.hayai_adapter.verify_snapshot', return_value=fixture.root), \
                patch('tralens.ocr.hayai_adapter.HayaiReader', return_value=reader):
            result = main.run_pipeline(path, fixture.root/'grouped', fonts=fixture.fonts, device='cpu',
                model_root=fixture.fixture.weights, run_local_ocr=True, run_local_model=True,
                ocr_factory=Mock(return_value=engine), runtime=SimpleNamespace(generate=fixture.provider),
                crop_reader_model=fixture.root, ink_column_groups=True, render_policy='best-effort')
        self.assertEqual(result['coverage']['rendered'], 1, result)
        self.assertEqual(fixture.provider.call_count, 1);reader.close.assert_called_once()
        preview = fixture.root/'grouped/preview/preview.json';saved = json.loads(preview.read_bytes())
        self.assertEqual(len(saved['ocr']['ink_column_groups']['previous_batch']['proposals']), 2)
        self.assertEqual(saved['units'][0]['unit']['order'], [0,1,2,3])
        saved['ocr']['ink_column_groups']['records'][0]['seams'][0]['original_rgb_sha256'] = '0'*64
        bad = fixture.root/'bad.json';bad.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'grouping geometry/pixels drift'):
            run_automatic(preview=bad, image=path, fonts=fixture.fonts, output=fixture.root/'bad-render')

    def test_missing_grouping_proof_stops_before_translation_dispatch(self):
        import main
        from tests.test_pipeline import PipelineTests
        fixture=PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        path=fixture.root/'missing-proof.png'
        image,original,batch,boxes=self.fixture(path=path);path.write_bytes(original)
        engine=SimpleNamespace(automatic_unit_proposals=(batch,),raw_detections=[boxes],image_path=str(path),
            process_images=Mock(),ocr=SimpleNamespace(close=Mock()))
        def incomplete_refiner(engine,*,output):
            g.refine(engine,image,original)
            del engine.ink_column_groups
        report=main.run_pipeline(path,fixture.root/'missing-proof',fonts=fixture.fonts,device='cpu',
            model_root=fixture.fixture.weights,run_local_ocr=True,run_local_model=True,
            ocr_factory=Mock(return_value=engine),runtime=SimpleNamespace(generate=fixture.provider),ocr_refine=incomplete_refiner)
        self.assertEqual(report['status'],'failed',report)
        saved=json.loads((fixture.root/'missing-proof/preview/preview.json').read_bytes())
        self.assertIn('missing saved ink grouping evidence',saved['error']['message'])
        fixture.provider.assert_not_called()


if __name__ == '__main__':unittest.main()
