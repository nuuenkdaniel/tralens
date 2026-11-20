"""Bounded inward-boundary repairs and read-only multi-page geometry evaluation."""
from io import BytesIO
from pathlib import Path
from collections import Counter
import json
import unittest

import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import natural as n
from tralens.rendering import preparation as a
from tests.local_evidence import local_evidence
from .test_preparation import raster
from . import test_natural as fixtures


class InwardBoundaryTests(unittest.TestCase):
    def test_border_connected_low_contrast_fringe_can_be_excluded_without_glyph_loss(self):
        for color in ((255, 255, 255), (241, 239, 233)):
            image, detections = raster(color)
            ImageDraw.Draw(image).line((17, 40, 22, 40), fill=tuple(v-3 for v in color))
            result = a.propose_geometry(image, detections, ['main'])
            self.assertEqual(result['status'], 'proposed', result)
            repair = result['measured']['additional_boundary_inset']
            self.assertGreater(repair['inset_pixels'], 0)
            self.assertLessEqual(repair['inset_pixels'], 3)
            self.assertEqual(repair['required_pixels_discarded'], 0)
            geometry = result['geometry']
            masks = [n.union_mask(geometry[k], image.size) for k in ('interior', 'cleanup_permission', 'protected')]
            clean, glyph, background = n.cleanup(image, *masks)
            self.assertEqual(background, list(color))
            self.assertTrue(np.array_equal(np.asarray(image)[~glyph], np.asarray(clean)[~glyph]))
            self.assertEqual(clean.getpixel((21, 40)), image.getpixel((21, 40)))

    def test_isolated_edge_mark_and_interior_contamination_are_not_hidden(self):
        for kind in ('isolated', 'interior', 'gradient', 'annotation'):
            image, detections = raster()
            draw = ImageDraw.Draw(image)
            if kind == 'isolated':
                draw.point((21, 40), fill=(252, 252, 252))
            elif kind == 'interior':
                draw.line((17, 40, 35, 40), fill=(252, 252, 252))
            elif kind == 'gradient':
                for x in range(20, 40):
                    draw.line((x, 20, x, 100), fill=(200+x,)*3)
            else:
                draw.line((17, 40, 22, 40), fill=(252, 252, 252))
                detections.append({'source_id': 'annotation', 'polygon': fixtures.rect(20, 38, 24, 44)})
            result = a.propose_geometry(image, detections, ['main'])
            self.assertEqual(result['status'], 'rejected', (kind, result))

    def test_inset_cannot_cut_narrow_clearance_punctuation_or_owned_collar(self):
        image, detections = raster()
        ImageDraw.Draw(image).line((17, 40, 22, 40), fill=(252, 252, 252))
        ImageDraw.Draw(image).rectangle((23, 60, 25, 65), fill='black')
        detections[0]['polygon'] = fixtures.rect(21, 35, 70, 95)
        result = a.propose_geometry(image, detections, ['main'])
        self.assertEqual(result['status'], 'rejected', result)

    def test_near_background_antialias_bridge_is_evidence_not_variance_bypass(self):
        image, detections = raster()
        draw = ImageDraw.Draw(image)
        draw.line((17, 40, 19, 40), fill=(254, 254, 254))
        draw.line((20, 40, 22, 40), fill=(252, 252, 252))
        result = a.propose_geometry(image, detections, ['main'])
        self.assertEqual(result['status'], 'proposed', result)
        evidence = result['measured']['additional_boundary_inset']
        self.assertEqual(evidence['original_cleanup_diagnostics']['maximum_channel_delta'], 2)
        self.assertEqual(evidence['boundary_connection_minimum_delta'], 1)

    def test_neck_probe_distinguishes_a_thin_corridor_from_shared_speech_space(self):
        image = Image.new('RGB', (220, 140), 'white')
        shape = Image.new('L', image.size)
        draw = ImageDraw.Draw(shape)
        draw.rectangle((15, 15, 85, 125), fill=255)
        draw.rectangle((135, 15, 205, 125), fill=255)
        draw.rectangle((80, 67, 140, 73), fill=255)
        envelope = np.asarray(shape) != 0
        pixels = np.asarray(image).copy()
        pixels[envelope & ~a.erode(envelope, 2)] = 0
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 48, 80), fill='black')
        draw.rectangle((160, 40, 168, 80), fill='black')
        first = n.union_mask([fixtures.rect(35, 35, 55, 90)], image.size)
        second = n.union_mask([fixtures.rect(155, 35, 175, 90)], image.size)
        context = a.OriginalRegions(image)
        evidence = a.separator_evidence(context, a.erode(envelope, 2), first, second, 2)
        self.assertTrue(evidence['separating_clearance_found'])
        self.assertIn('no separator selected', evidence['scope'])
        whole = np.zeros_like(envelope)
        whole[17:124, 17:204] = True
        wide = Image.new('RGB', image.size, 'white')
        draw = ImageDraw.Draw(wide)
        draw.rectangle((15, 15, 205, 125), outline='black', width=2)
        draw.rectangle((40, 40, 48, 80), fill='black')
        draw.rectangle((160, 40, 168, 80), fill='black')
        evidence = a.separator_evidence(a.OriginalRegions(wide), whole, first, second, 2)
        self.assertFalse(evidence['separating_clearance_found'])
        self.assertTrue(all(not row['distinct_complete_ink_cores'] for row in evidence['trials']))

    def test_repaired_source_bound_proposal_confirms_and_composes_without_locality_loss(self):
        from dataclasses import asdict
        from tralens.ocr.automatic_units import propose_page_from_boxes
        from tralens.ocr.types import Text_Box
        from tralens.translation.contracts import TranslationRequest, TranslationResult, UnitOutcome
        fixture = fixtures.CommandTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with Image.open(fixture.image) as original:
            image = original.copy()
        ImageDraw.Draw(image).line((7, 20, 12, 20), fill=(252, 252, 252))
        image.save(fixture.image)
        raw = fixture.preview['ocr']['raw_detections'][0]
        boxes = [Text_Box(d['source_id'], d['confidence'], d['polygon'], d['text']) for d in raw]
        batch = asdict(propose_page_from_boxes(boxes, str(fixture.image)))
        units = n.automatic_units(batch)
        records = []
        for unit in units:
            request = TranslationRequest((unit,), (unit.unit_id,))
            outcome = UnitOutcome(unit.unit_id, 'preview', 'Hi!\nAV?', (), 'unverified', unit.selected_revision)
            result = TranslationResult(request.revision, 'preview', ((unit.unit_id, outcome.translation),), outcomes=(outcome,))
            records.append({'unit': asdict(unit), 'request': asdict(request), 'unit_revision': unit.revision,
                'request_revision': request.revision, 'status': 'preview', 'binding_current': True,
                'input_kind': 'strict_utterance', 'result': asdict(result)})
        detections = [{**{k: v for k, v in d.items() if k not in ('page', 'metadata_json')}, 'ocr_metadata': {}}
                      for d in batch['detections']]
        content = fixture.image.read_bytes()
        source = {'path': str(fixture.image), 'sha256': n.digest(content), 'size': list(image.size),
                  'pixels_sha256': n.digest(f'RGB:{image.width}:{image.height}:'.encode()+image.tobytes())}
        preview = {'schema_version': 1, 'preview_only': True, 'input': source,
                   'ocr': {'batches': [batch], 'raw_detections': [detections]}, 'units': records}
        path = fixture.root/'repaired-preview.json'
        path.write_text(n.canonical(preview))
        uid = next(u.unit_id for u in units if 'main' in u.member_ids)
        inputs = dict(preview=path, image=fixture.image, fonts=fixture.root/'fonts.json')
        proposed = a.prepare(**inputs, unit_ids=[uid], font_id='dialogue', min_size=14, max_size=28,
                             output=fixture.root/'proposals')
        row = proposed['units'][0]
        self.assertEqual(row['status'], 'proposed', row)
        self.assertTrue(row['measured']['additional_boundary_inset']['geometry_inset_applied'])
        selected = fixture.root/'confirmed.json'
        a.confirm_preparation(**inputs, proposal_bundle=fixture.root/'proposals', proposal_ids=[row['proposal_id']],
            operator='synthetic-test-operator', intent='Synthetic pixel geometry inspection only', output=selected)
        rendered = n.run('compose', **inputs, selection=selected, output=fixture.root/'composed')
        self.assertEqual(rendered['units'][0]['status'], 'candidate_composed')
        self.assertEqual(rendered['pixel_audit']['outside_cleanup_changed_pixels'], 0)
        self.assertEqual(rendered['pixel_audit']['outside_alpha_changed_pixels'], 0)
        self.assertEqual(fixture.image.read_bytes(), content)


class SavedCoverageEvidence(unittest.TestCase):
    @local_evidence(files=('test-results/render-preparation-parent-v2/proposals.json',
        'test-results/workflow-fragments-priority-parent-v1/preview.json',
        'test-results/render-parent-input-v1/fonts.json', 'images/ubunchu-ja-test/ubunchu01_04.png'))
    def test_current_four_units_readonly_coverage_and_separator_evidence(self):
        root = Path(__file__).resolve().parents[2]
        bundle = n.document(n.read_explicit(root/'test-results/render-preparation-parent-v2/proposals.json'))
        preview_bytes = n.read_explicit(root/'test-results/workflow-fragments-priority-parent-v1/preview.json')
        image_path = root/'images/ubunchu-ja-test/ubunchu01_04.png'
        image_bytes = n.read_explicit(image_path)
        fonts_bytes = n.read_explicit(root/'test-results/render-parent-input-v1/fonts.json')
        self.assertEqual(n.digest(image_bytes), bundle['source']['sha256'])
        self.assertEqual(n.digest(preview_bytes), bundle['preview_sha256'])
        self.assertEqual(n.digest(fonts_bytes), bundle['fonts_sha256'])
        preview = n.document(preview_bytes)
        with Image.open(BytesIO(image_bytes)) as original:
            rgb = original.convert('RGB')
        context = a.OriginalRegions(rgb)
        results = []
        for row in bundle['units']:
            binding = row['selection_binding']
            contract = {'schema_version': 1, 'purpose': 'candidate_only', 'source': bundle['source'],
                'preview_sha256': n.digest(preview_bytes), 'fonts_sha256': n.digest(fonts_bytes),
                'units': [{**binding, 'geometry_intent': 'Read-only source-pixel evaluation; no confirmation.'}]}
            n.bind(preview, image_path, rgb, image_bytes, preview_bytes, contract, fonts_bytes)
            result = a.propose_geometry(rgb, preview['ocr']['raw_detections'][0], binding['member_ids'],
                                        bundle['request']['settings'], context=context)
            results.append({'unit_id': row['unit_id'], 'members': binding['member_ids'], 'status': result['status'],
                'reason': result.get('reason'), 'boundary_inset': result['measured'].get('additional_boundary_inset'),
                'separator': result['measured'].get('separator_evidence')})
            if result['status'] == 'proposed':
                geometry = result['geometry']
                clean, glyph, _ = n.cleanup(rgb, *[n.union_mask(geometry[key], rgb.size)
                                                 for key in ('interior', 'cleanup_permission', 'protected')])
                self.assertTrue(np.array_equal(np.asarray(rgb)[~glyph], np.asarray(clean)[~glyph]))
        print('READ_ONLY_COVERAGE='+json.dumps(results, sort_keys=True))
        self.assertGreaterEqual(sum(r['status'] == 'proposed' for r in results), 2)

    def evaluate_page(self, image_name):
        root = Path(__file__).resolve().parents[2]
        artifact = root/'test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json'
        document = n.document(n.read_explicit(artifact))
        image_path = root/'images/ubunchu-ja-test'/image_name
        pages = [p for p in document['pages'] if Path(p['page']['image_path']).name == image_name]
        self.assertEqual(len(pages), 1)
        page = pages[0]
        content = n.read_explicit(image_path)
        self.assertEqual(n.digest(content), page['page']['image_sha256'])
        with Image.open(BytesIO(content)) as opened:
            rgb = opened.convert('RGB')
        self.assertEqual(n.digest(f'RGB:{rgb.width}:{rgb.height}:'.encode()+rgb.tobytes()), page['page']['pixels_sha256'])
        units = n.automatic_units(page['batch'])  # Existing immutable raw/proposal hash checks, no translation.
        self.assertEqual(len(units), len(page['batch']['proposals']))
        detections = page['batch']['detections']
        budget = a.MAX_PREFLIGHT_PIXELS//(rgb.width*rgb.height*max(1, len(detections)))
        count = min(16, len(units), 2*budget, 2*n.MAX_UNIT_PIXELS//(rgb.width*rgb.height))
        self.assertGreater(count, 0)
        context = a.OriginalRegions(rgb)
        results = []
        for unit in units[:count]:
            result = a.propose_geometry(rgb, detections, list(unit.member_ids), context=context)
            results.append({'members': list(unit.member_ids), 'status': result['status'], 'reason': result.get('reason')})
            if result['status'] == 'proposed':
                geometry = result['geometry']
                clean, glyph, _ = n.cleanup(rgb, *[n.union_mask(geometry[key], rgb.size)
                                                 for key in ('interior', 'cleanup_permission', 'protected')])
                self.assertTrue(np.array_equal(np.asarray(rgb)[~glyph], np.asarray(clean)[~glyph]))
        print('READ_ONLY_ADDITIONAL_GEOMETRY='+json.dumps({'image': image_name, 'scope': 'geometry-only; no strict translated preview, selection or confirmation manufactured',
            'artifact_sha256': n.digest(n.read_explicit(artifact)), 'maximum_evaluation_batches': 2,
            'evaluated_prefix': count, 'saved_units': len(units),
            'counts': dict(Counter(r['status'] for r in results)), 'units': results}, sort_keys=True))

    @local_evidence(files=('test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json',
                          'images/ubunchu-ja-test/ubunchu01_02.png'))
    def test_additional_original_page_01_02_geometry_only(self):
        self.evaluate_page('ubunchu01_02.png')

    @local_evidence(files=('test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json',
                          'images/ubunchu-ja-test/ubunchu02_04.png'))
    def test_additional_original_page_02_04_geometry_only(self):
        self.evaluate_page('ubunchu02_04.png')
