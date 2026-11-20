"""Original-pixel synthetic feasibility and explicit preparation handoff."""
from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import natural as n
from . import test_natural as fixtures


def raster(color='white', opened=False):
    image = Image.new('RGB', (180, 150), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 10, 140, 135), outline='black', width=2)
    if opened:
        draw.rectangle((15, 50, 17, 65), fill=color)
    for y in (40, 60, 80):
        draw.rectangle((55, y, 64, y+8), fill='black')
    detections = [{'source_id': 'main', 'polygon': fixtures.rect(50, 35, 70, 95)}]
    return image, detections


class GeometryFeasibilityTests(unittest.TestCase):
    def test_closed_white_offwhite_geometry_passes_unchanged_cleanup(self):
        from tralens.rendering.preparation import propose_geometry
        for color in ('white', (241, 239, 233)):
            image, detections = raster(color)
            result = propose_geometry(image, detections, ['main'])
            self.assertEqual(result['status'], 'proposed', result)
            geometry = result['geometry']
            masks = {key: n.union_mask(geometry[field], image.size) for key, field in
                     (('interior', 'interior'), ('layout', 'layout'), ('permission', 'cleanup_permission'), ('protected', 'protected'))}
            clean, glyph, _ = n.cleanup(image, masks['interior'], masks['permission'], masks['protected'])
            self.assertTrue(np.array_equal(np.asarray(image)[~glyph], np.asarray(clean)[~glyph]))
            self.assertFalse((masks['permission'] & ~masks['interior']).any())
            self.assertFalse((masks['layout'] & ~masks['interior']).any())

    def test_open_frame_and_nonmember_ink_reject(self):
        from tralens.rendering.preparation import propose_geometry
        image, detections = raster(opened=True)
        self.assertEqual(propose_geometry(image, detections, ['main'])['status'], 'rejected')
        image, detections = raster()
        ImageDraw.Draw(image).rectangle((90, 40, 94, 65), fill='black')
        result = propose_geometry(image, detections, ['main'])
        self.assertEqual(result['status'], 'rejected')
        detections.append({'source_id': 8, 'polygon': fixtures.rect(87, 35, 97, 70)})
        self.assertEqual(propose_geometry(image, detections, ['main'])['status'], 'rejected')

    def test_border_neighborhood_and_gradient_reject(self):
        from tralens.rendering.preparation import propose_geometry
        image, detections = raster()
        ImageDraw.Draw(image).rectangle((16, 40, 55, 45), fill='black')
        self.assertEqual(propose_geometry(image, detections, ['main'])['status'], 'rejected')
        image, detections = raster()
        for x in range(20, 135):
            ImageDraw.Draw(image).line((x, 15, x, 30), fill=(200+x//5,)*3)
        result = propose_geometry(image, detections, ['main'])
        self.assertEqual(result['status'], 'rejected')

    def test_ellipse_and_glyph_counter_are_supported_without_hull_erasure(self):
        from tralens.rendering.preparation import propose_geometry
        image, detections = raster()
        image.paste('white', (0, 0, *image.size))
        draw = ImageDraw.Draw(image)
        draw.ellipse((15, 10, 140, 135), outline='black', width=2)
        draw.ellipse((54, 40, 65, 55), outline='black', width=2)
        draw.rectangle((55, 70, 64, 80), fill='black')
        result = propose_geometry(image, detections, ['main'])
        self.assertEqual(result['status'], 'proposed', result)
        interior = n.union_mask(result['geometry']['interior'], image.size)
        self.assertFalse(interior[10, 15])
        self.assertTrue(interior[47, 59])  # Accounted-for white glyph counter.

    def test_disconnected_punctuation_is_rejected_not_eroded_or_ignored(self):
        from tralens.rendering.preparation import propose_geometry
        image, detections = raster()
        image.putpixel((60, 99), (0, 0, 0))
        result = propose_geometry(image, detections, ['main'])
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('unexplained ink', result['reason'])

    def test_shared_balloon_and_furigana_nonmember_are_rejected(self):
        from tralens.rendering.preparation import propose_geometry
        image, detections = raster()
        draw = ImageDraw.Draw(image)
        draw.rectangle((78, 40, 80, 52), fill='black')
        detections.append({'source_id': 'annotation', 'polygon': fixtures.rect(76, 38, 82, 55)})
        for members in (['main'], ['annotation']):
            self.assertEqual(propose_geometry(image, detections, members)['status'], 'rejected')

    def test_textured_interior_and_clipped_whole_component_reject(self):
        from tralens.rendering.preparation import propose_geometry
        image, detections = raster()
        for y in range(20, 120, 3):
            ImageDraw.Draw(image).line((85, y, 120, y), fill=(210, 210, 210))
        self.assertEqual(propose_geometry(image, detections, ['main'])['status'], 'rejected')
        image, detections = raster()
        ImageDraw.Draw(image).rectangle((60, 40, 90, 43), fill='black')
        result = propose_geometry(image, detections, ['main'])
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('whole ink component', result['reason'])

    def test_eight_border_intrusions_are_excluded_by_original_geometry_not_whitening(self):
        from tralens.rendering.preparation import propose_geometry
        image, detections = raster()
        points = [(138, 131), (137, 132), (138, 132), (18, 133), (19, 133), (136, 133), (137, 133), (138, 133)]
        for point in points:
            image.putpixel(point, (0, 0, 0))
        # An over-wide hand rectangle fails exactly on these eight original
        # border pixels; preparation must not turn them into permitted glyphs.
        interior = n.union_mask([fixtures.rect(18, 20, 138, 133)], image.size)
        permission = n.union_mask([fixtures.rect(50, 35, 70, 95)], image.size)
        with self.assertRaises(n.CleanupBlocked) as caught:
            n.cleanup(image, interior, permission, np.zeros_like(interior))
        self.assertEqual(caught.exception.diagnostics['offending_count'], 8)
        result = propose_geometry(image, detections, ['main'])
        self.assertEqual(result['status'], 'proposed', result)
        geometry = result['geometry']
        actual = n.union_mask(geometry['interior'], image.size)
        permission = n.union_mask(geometry['cleanup_permission'], image.size)
        clean, glyph, _ = n.cleanup(image, actual, permission, np.zeros_like(actual))
        for x, y in points:
            self.assertFalse(actual[y, x])
            self.assertFalse(glyph[y, x])
            self.assertEqual(clean.getpixel((x, y)), image.getpixel((x, y)))

    def test_touching_twin_lobes_and_unrepresentable_holes_reject(self):
        from tralens.rendering.preparation import propose_geometry, polygons_inside, Unsupported
        image = Image.new('RGB', (240, 150), 'white')
        shape = Image.new('L', image.size)
        draw = ImageDraw.Draw(shape)
        draw.ellipse((10, 10, 125, 135), fill=255)
        draw.ellipse((110, 10, 225, 135), fill=255)
        mask = np.asarray(shape) != 0
        inner = n.cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0
        pixels = np.asarray(image).copy()
        pixels[mask & ~inner] = 0
        image = Image.fromarray(pixels)
        for offset in (0, 100):
            ImageDraw.Draw(image).rectangle((55+offset, 40, 64+offset, 70), fill='black')
        dets = [{'source_id': i, 'polygon': fixtures.rect(50+offset, 35, 70+offset, 95)} for i, offset in enumerate((0, 100))]
        result = propose_geometry(image, dets, [0, 1])
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('non-single-span', result['reason'])
        ring = np.zeros((50, 50), bool)
        ring[5:45, 5:45] = True
        ring[15:35, 15:35] = False
        with self.assertRaises(Unsupported):
            polygons_inside(ring, ring)

    def test_distinct_interiors_with_touching_original_outlines_are_rejected(self):
        from tralens.rendering.preparation import propose_geometry
        image = Image.new('RGB', (200, 130), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 80, 110), outline='black', width=2)
        draw.rectangle((80, 10, 150, 110), outline='black', width=2)
        draw.rectangle((30, 40, 38, 80), fill='black')
        draw.rectangle((110, 40, 118, 80), fill='black')
        detections = [{'source_id': name, 'polygon': fixtures.rect(x, 30, x+20, 90)}
                      for name, x in (('left', 25), ('right', 105))]
        for name in ('left', 'right'):
            result = propose_geometry(image, detections, [name])
            self.assertEqual(result['status'], 'rejected', result)
            self.assertIn('outline-connected', result['reason'])


class PreparationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CommandTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.uid = self.f.selection['units'][0]['unit_id']

    def inputs(self):
        return dict(preview=self.f.root/'preview.json', image=self.f.image, fonts=self.f.root/'fonts.json')

    def prepare(self, name='proposals', unit_ids=None):
        from tralens.rendering.preparation import prepare
        return prepare(**self.inputs(), unit_ids=[self.uid] if unit_ids is None else unit_ids, font_id='dialogue',
                       min_size=14, max_size=28, output=self.f.root/name)

    def confirm(self, report, name='confirmed.json', proposal_ids=None, **changes):
        from tralens.rendering.preparation import confirm_preparation
        args = dict(self.inputs(), proposal_bundle=self.f.root/'proposals',
            proposal_ids=[report['units'][0]['proposal_id']] if proposal_ids is None else proposal_ids,
            operator='synthetic-test-operator', intent='Synthetic original-raster geometry inspection only', output=self.f.root/name)
        args.update(changes)
        return confirm_preparation(**args)

    def test_proposal_is_not_selection_and_confirmed_selection_composes_exactly_locally(self):
        before = self.f.image.read_bytes()
        report = self.prepare()
        self.assertEqual(report['purpose'], 'flat_geometry_proposals')
        self.assertEqual(report['units'][0]['status'], 'proposed', report)
        self.assertFalse((self.f.root/'proposals/clean.png').exists())
        self.assertFalse((self.f.root/'proposals/composite.png').exists())
        with self.assertRaisesRegex(ValueError, 'candidate-only'):
            n.run('compose', **self.inputs(), selection=self.f.root/'proposals/proposals.json', output=self.f.root/'direct')
        selection = self.confirm(report)
        self.assertEqual(selection['purpose'], 'candidate_only')
        self.assertFalse(selection['preparation_confirmation']['identity_verified'])
        self.assertEqual(selection['preparation_confirmation']['linguistic_approval'], 'not_assessed')
        self.assertNotIn('reviewed', selection['preparation_confirmation'])
        self.assertNotIn('human_reviewed', selection['units'][0])
        result = n.run('compose', **self.inputs(), selection=self.f.root/'confirmed.json', output=self.f.root/'composed')
        self.assertEqual(result['units'][0]['status'], 'candidate_composed', result['units'])
        self.assertEqual(self.f.image.read_bytes(), before)
        self.assertEqual(result['pixel_audit']['outside_cleanup_changed_pixels'], 0)
        self.assertEqual(result['pixel_audit']['outside_alpha_changed_pixels'], 0)
        self.assertEqual(result['pixel_audit']['protected_composite_changed_pixels'], 0)
        with Image.open(self.f.image) as original, Image.open(self.f.root/'composed/clean.png') as clean, \
                Image.open(self.f.root/'composed/unit-000-glyph-cleanup.png') as glyph:
            self.assertTrue(np.array_equal(np.asarray(original)[np.asarray(glyph) == 0], np.asarray(clean)[np.asarray(glyph) == 0]))

    def test_explicit_subset_only_and_unsupported_records_are_preserved(self):
        ids = [r['unit']['unit_id'] for r in self.f.preview['units']]
        report = self.prepare(unit_ids=ids)
        self.assertEqual(len(report['units']), 2)
        ready = [r for r in report['units'] if r['status'] == 'proposed']
        unsupported = [r for r in report['units'] if r['status'] == 'unsupported']
        self.assertEqual((len(ready), len(unsupported)), (1, 1))
        selected = self.confirm(report, proposal_ids=[ready[0]['proposal_id']])
        self.assertEqual([u['unit_id'] for u in selected['units']], [self.uid])
        for chosen in ([], ['all'], [ready[0]['proposal_id']]*2, [unsupported[0]['proposal_id']], ['geometry-'+'0'*32]):
            with self.subTest(chosen=chosen), self.assertRaises(ValueError):
                self.confirm(report, proposal_ids=chosen, name='invalid.json')
        with self.assertRaises(ValueError):
            self.confirm(report, name='empty-operator.json', operator='')

    def test_result_and_request_drift_never_create_confirmable_geometry(self):
        for field in ('request_revision', 'unit_revision'):
            original = self.f.selected_record[field]
            self.f.selected_record[field] = '0'*64
            self.f.write_inputs()
            report = self.prepare(name=field)
            self.assertEqual(report['units'][0]['status'], 'rejected')
            self.assertFalse(report['units'][0]['binding_validated'])
            self.f.selected_record[field] = original
        self.f.selected_record['result']['outcomes'][0]['reading_revision'] = '0'*64
        self.f.write_inputs()
        report = self.prepare(name='result-drift')
        self.assertEqual(report['units'][0]['status'], 'rejected')

    def test_source_pixel_font_and_license_drift_reject(self):
        report = self.prepare()
        for target in (self.f.image, self.f.root/'fonts.json', self.f.font, self.f.root/'synthetic-license.txt', self.f.root/'preview.json'):
            before = target.read_bytes()
            target.write_bytes(before+b' ')
            with self.subTest(target=target.name), self.assertRaises(ValueError):
                self.confirm(report)
            target.write_bytes(before)
        self.f.preview['input']['pixels_sha256'] = '0'*64
        self.f.write_inputs()
        with self.assertRaisesRegex(ValueError, 'source binding'):
            self.prepare(name='pixel-drift')

    def test_tampered_geometry_overlay_report_and_requests_reject(self):
        report = self.prepare()
        root = self.f.root/'proposals'
        for name in ('unit-000-overlay.png', 'review.html', 'evidence-request.json', 'evidence-preview.json'):
            target = root/name
            before = target.read_bytes()
            target.write_bytes(before+b' ')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'artifact drift'):
                self.confirm(report)
            target.write_bytes(before)
        target = root/'proposals.json'
        doc = json.loads(target.read_text())
        doc['units'][0]['geometry']['interior'][0][0][0] += 1
        target.write_text(n.canonical(doc))
        with self.assertRaisesRegex(ValueError, 'manifest drift'):
            self.confirm(report)

    def test_determinism_fresh_outputs_symlinks_and_resource_bounds(self):
        from tralens.rendering import preparation as a
        one, two = self.prepare(), self.prepare(name='again')
        self.assertEqual(one, two)
        self.assertEqual((self.f.root/'proposals/proposals.json').read_bytes(), (self.f.root/'again/proposals.json').read_bytes())
        with self.assertRaises(ValueError):
            self.prepare()
        (self.f.root/'link').symlink_to(self.f.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.prepare(name='link/new')
        with self.assertRaises(ValueError):
            self.confirm(one, name='proposals/selection.json')
        with patch.object(a, 'MAX_PIXELS', 10):
            with self.assertRaisesRegex(ValueError, 'pixel/side budget'):
                self.prepare(name='large')
        with self.assertRaises(ValueError):
            self.prepare(name='too-many', unit_ids=['x'+str(i) for i in range(17)])

    def test_actual_synthetic_cli_prepare_confirm_compose_and_offline_imports(self):
        tool = Path(__file__).resolve().parents[2]/'tools/render_preview.py'
        common = [item for key, value in self.inputs().items() for item in ('--'+key, str(value))]
        def cli(*args):
            result = subprocess.run([sys.executable, str(tool), *args, *common], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        cli('prepare', '--unit-id', self.uid, '--font-id', 'dialogue', '--min-size', '14', '--max-size', '28', '--output', str(self.f.root/'proposals'))
        report = json.loads((self.f.root/'proposals/proposals.json').read_text())
        cli('confirm-preparation', '--proposal-bundle', str(self.f.root/'proposals'), '--proposal-id', report['units'][0]['proposal_id'],
            '--operator', 'synthetic-test-operator', '--intent', 'Synthetic geometry inspection', '--output', str(self.f.root/'confirmed.json'))
        cli('compose', '--selection', str(self.f.root/'confirmed.json'), '--output', str(self.f.root/'composed'))
        result = json.loads((self.f.root/'composed/manifest.json').read_text())
        self.assertEqual(result['units'][0]['status'], 'candidate_composed')
        script = '''
import importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'paddleocr','paddle','ollama','transformers','torch','requests','httpx'} or fullname == 'tralens.ocr.detection':
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Guard())
from tralens.rendering.preparation import prepare, confirm_preparation
prepare(preview=sys.argv[2],image=sys.argv[3],fonts=sys.argv[4],unit_ids=[sys.argv[5]],font_id='dialogue',min_size=14,max_size=28,output=sys.argv[6])
'''
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, '-c', script, str(root/'src'), str(self.inputs()['preview']), str(self.f.image),
            str(self.inputs()['fonts']), self.uid, str(self.f.root/'guarded')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_multiunit_collision_rejects_both_and_fabricated_flags_cannot_confirm(self):
        from tralens.rendering import preparation as a
        from tralens.translation.contracts import TranslationResult, UnitOutcome
        from dataclasses import asdict
        other = next(r for r in self.f.preview['units'] if r is not self.f.selected_record)
        uid = other['unit']['unit_id']
        outcome = UnitOutcome(uid, 'preview', 'Hi!', (), 'unverified', other['unit']['selected_revision'])
        result = TranslationResult(other['request_revision'], 'preview', ((uid, 'Hi!'),), outcomes=(outcome,))
        other.update(status='preview', input_kind='strict_utterance', result=asdict(result))
        self.f.write_inputs()
        with Image.open(self.f.image) as source:
            raw = json.loads(n.canonical(self.f.preview['ocr']['raw_detections'][0]))
            original = a.propose_geometry(source.convert('RGB'), raw, ['main'])
        with patch.object(a, 'propose_geometry', return_value=original):
            report = self.prepare(unit_ids=[self.uid, uid])
        self.assertTrue(all(r['status'] == 'rejected' and 'collision' in r['reason'] for r in report['units']))
        path = self.f.root/'proposals/proposals.json'
        doc = json.loads(path.read_text())
        doc['operator_confirmation'] = {'reviewed': True, 'operator': 'not a real approval'}
        path.write_text(n.canonical(doc))
        with self.assertRaisesRegex(ValueError, 'manifest drift'):
            self.confirm(report)

    def test_component_member_and_preflight_work_budgets_reject_before_exports(self):
        from tralens.rendering import preparation as a
        with patch.object(a, 'MAX_PREFLIGHT_PIXELS', 1):
            with self.assertRaisesRegex(ValueError, 'raster work budget'):
                self.prepare(name='work-budget')
        self.assertFalse((self.f.root/'work-budget').exists())
        image = Image.new('RGB', (200, 200), 'white')
        pixels = np.asarray(image).copy()
        pixels[::2, ::2] = 0
        with self.assertRaisesRegex(ValueError, 'component budget'):
            a.OriginalRegions(Image.fromarray(pixels))
        image, detections = raster()
        many = [{'source_id': i, 'polygon': fixtures.rect(50, 35, 70, 95)} for i in range(33)]
        with self.assertRaisesRegex(ValueError, 'member mask raster budget'):
            a.propose_geometry(image, many, list(range(33)))

    def test_report_is_static_escaped_and_artifact_symlinks_cannot_confirm(self):
        report = self.prepare()
        root = self.f.root/'proposals'
        page = (root/'review.html').read_text()
        self.assertIn('Content-Security-Policy', page)
        self.assertIn('max-width:600px', page)
        self.assertNotIn('<script', page)
        self.assertNotIn('src="http', page)
        path = root/'unit-000-overlay.png'
        content = path.read_bytes()
        path.unlink()
        target = self.f.root/'overlay.png'
        target.write_bytes(content)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.confirm(report)

    def test_blocked_fragments_and_visual_alternates_remain_unsupported_evidence(self):
        self.f.selected_record.update(status='blocked', input_kind='strict_utterance',
            fragment_previews=[{'status': 'fragment_preview', 'text': 'not a selection'}],
            visual_alternates=[{'status': 'alternate_preview', 'text': 'not a selection'}])
        self.f.write_inputs()
        before = (self.f.root/'preview.json').read_bytes()
        report = self.prepare()
        self.assertEqual(report['units'][0]['status'], 'unsupported')
        self.assertFalse(report['units'][0]['binding_validated'])
        self.assertEqual((self.f.root/'proposals/evidence-preview.json').read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'unsupported proposals'):
            self.confirm(report)


if __name__ == '__main__':
    unittest.main()
