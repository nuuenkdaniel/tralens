"""Offline synthetic raster checks for the candidate renderer."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tralens.rendering import natural as n
from tralens.rendering.natural import polygon_mask, layout
from tralens.translation.contracts import (TranslationRequest, TranslationResult, UnitOutcome,
                                         automatic_units, canonical, revision)
from PIL import Image, ImageDraw, ImageFont
from .font_fixture import synthetic_font


def rect(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class NaturalTests(unittest.TestCase):
    def test_mask_bounds_reject_boolean_and_outside(self):
        for points in ([[True, 1], [9, 1], [5, 9]],
                       [[0, 0], [10, 0], [5, 9]]):
            with self.assertRaises(ValueError):
                polygon_mask(points, (10, 10))

    def test_missing_coverage_blocks_before_drawing(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            layout("Hello", b"", frozenset(), Image.new("L", (20, 20), 255),
                   {"min_size": 12, "max_size": 20, "outline": 0})

    def test_self_intersection_and_empty_polygon_rejected(self):
        for p in ([], [[0, 0], [9, 9], [0, 9], [9, 0]], [[1, 1]]*3):
            with self.assertRaises(ValueError):
                polygon_mask(p, (10, 10))

    def test_unknown_font_coverage_not_certified(self):
        with self.assertRaises(ValueError):
            n.font_coverage(b'not a font')

    def test_duplicate_nonfinite_and_deep_json_rejected(self):
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'['*40+b'0'+b']'*40):
            with self.assertRaises(ValueError):
                n.document(data)

    def test_white_offwhite_exact_outside_and_protected(self):
        for color in ((255, 255, 255), (239, 237, 231)):
            image = Image.new('RGB', (60, 60), color)
            ImageDraw.Draw(image).rectangle((25, 25, 30, 30), fill='black')
            interior = n.union_mask([rect(5, 5, 54, 54)], image.size)
            permission = n.union_mask([rect(20, 20, 35, 35)], image.size)
            protected = ~interior
            clean, glyph, sampled = n.cleanup(image, interior, permission, protected)
            self.assertEqual(sampled, list(color))
            self.assertTrue(np.all(np.asarray(clean)[glyph] == color))
            self.assertTrue(np.array_equal(np.asarray(clean)[~glyph], np.asarray(image)[~glyph]))
            self.assertTrue(np.array_equal(np.asarray(clean)[protected], np.asarray(image)[protected]))
            self.assertEqual(image.getpixel((25, 25)), (0, 0, 0))

    def test_texture_gradient_neighbor_and_boundary_block_cleanup(self):
        for kind in ('texture', 'gradient', 'neighbor', 'boundary', 'protected'):
            image = Image.new('RGB', (60, 60), 'white')
            draw = ImageDraw.Draw(image)
            draw.rectangle((25, 25, 30, 30), fill='black')
            interior = n.union_mask([rect(5, 5, 54, 54)], image.size)
            permission = n.union_mask([rect(20, 20, 35, 35)], image.size)
            protected = np.zeros((60, 60), dtype=bool)
            if kind == 'texture':
                for y in range(6, 54, 3):
                    draw.line((6, y, 18, y), fill=(230, 230, 230))
            if kind == 'gradient':
                for x in range(5, 20):
                    draw.line((x, 5, x, 54), fill=(220+x,)*3)
            if kind == 'neighbor':
                draw.rectangle((8, 8, 12, 12), fill='black')
            if kind == 'boundary':
                draw.rectangle((20, 20, 25, 25), fill='black')
            if kind == 'protected':
                protected[25, 25] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                n.cleanup(image, interior, permission, protected)


class RasterTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_font()
        self.coverage = n.font_coverage(self.data)
        self.style = {'min_size': 14, 'max_size': 28, 'outline': 1}

    def test_shaped_kerning_ink_outline_newline_punctuation_deterministic(self):
        mask = polygon_mask(rect(10, 10, 589, 189), (600, 200))
        text = 'AV, really?!\nYes -- of course.'
        a, p = layout(text, self.data, self.coverage, mask, self.style)
        b, q = layout(text, self.data, self.coverage, mask, self.style)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(p, q)
        self.assertEqual(p['draw_text'], text)
        self.assertEqual(a.getchannel('A').getbbox(), tuple(p['ink_bounds']))
        self.assertFalse(((np.asarray(a.getchannel('A')) != 0) & (np.asarray(mask) == 0)).any())
        from io import BytesIO
        font = ImageFont.truetype(BytesIO(self.data), p['size'], layout_engine=ImageFont.Layout.RAQM)
        draw = ImageDraw.Draw(Image.new('L', (1, 1)))
        self.assertLess(draw.textlength('AV', font=font), draw.textlength('A', font=font)+draw.textlength('V', font=font))
        ink = p['ink_bounds']
        self.assertLessEqual(abs((ink[0]+ink[2])-(10+590)), 2)
        self.assertLessEqual(abs((ink[1]+ink[3])-(10+190)), 2)

    def test_irregular_mask_and_overflow_and_missingglyph(self):
        mask = polygon_mask([[100, 0], [199, 100], [100, 199], [0, 100]], (200, 200))
        layer, p = layout('Hello world! A new line.', self.data, self.coverage, mask, self.style)
        self.assertIn('world!', p['draw_text'])
        self.assertFalse(((np.asarray(layer.getchannel('A')) != 0) & (np.asarray(mask) == 0)).any())
        with self.assertRaisesRegex(ValueError, 'overflow'):
            layout('Unbreakableword'*30, self.data, self.coverage, Image.new('L', (20, 20), 255), self.style)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            layout('Hello', self.data, self.coverage-{ord('H')}, mask, self.style)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.font = self.root/'synthetic.ttf'
        self.font.write_bytes(synthetic_font())
        self.image = self.root/'original.png'
        # Real deterministic proposal generation on a synthetic raster only.
        from tralens.ocr.automatic_units import propose_page_from_boxes
        from tralens.ocr.types import Text_Box
        image = Image.new('RGB', (180, 130), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 5, 75, 110), outline='black', width=2)
        draw.rectangle((100, 5, 170, 110), outline='black', width=2)
        for y in (26, 46, 66):
            draw.rectangle((28, y, 37, y+9), fill='black')
            draw.rectangle((123, y, 132, y+9), fill='black')
        image.save(self.image)
        box = Text_Box('main', .99, rect(25, 20, 45, 90), 'こんにちは')
        neighbor = Text_Box(7, .99, rect(120, 20, 140, 90), 'こんにちは')
        batch = asdict(propose_page_from_boxes([box, neighbor], str(self.image)))
        unit = next(u for u in automatic_units(batch) if 'main' in u.member_ids)
        self.assertEqual(unit.disposition, 'preview', unit.blocking_reasons)
        request = TranslationRequest((unit,), (unit.unit_id,))
        outcome = UnitOutcome(unit.unit_id, 'preview', 'Hi!\nAV?', (), 'unverified', unit.selected_revision)
        result = TranslationResult(request.revision, 'preview', ((unit.unit_id, outcome.translation),), outcomes=(outcome,))
        raw = [{k: v for k, v in d.items() if k not in ('page', 'metadata_json')} for d in batch['detections']]
        for d in raw:
            d['ocr_metadata'] = {}
        source = {'path': str(self.image), 'sha256': n.digest(self.image.read_bytes()),
                  'pixels_sha256': n.digest(b'RGB:180:130:'+image.tobytes()), 'size': [180, 130]}
        self.preview = {'schema_version': 3, 'preview_only': True, 'input': source,
            'ocr': {'batches': [batch], 'raw_detections': [raw]}, 'units': [{
                'unit': asdict(unit), 'request': asdict(request), 'unit_revision': unit.revision,
                'request_revision': request.revision, 'status': 'preview', 'binding_current': True,
                'input_kind': 'strict_utterance', 'result': asdict(result)}]}
        self.selected_record = self.preview['units'][0]
        other = next(u for u in automatic_units(batch) if 'main' not in u.member_ids)
        other_request = TranslationRequest((other,), (other.unit_id,))
        self.preview['units'].append({'unit': asdict(other), 'request': asdict(other_request),
            'unit_revision': other.revision, 'request_revision': other_request.revision,
            'status': 'not_selected', 'binding_current': True, 'result': None})
        order = [u.unit_id for u in automatic_units(batch)]
        self.preview['units'].sort(key=lambda r: order.index(r['unit']['unit_id']))
        # Synthetic license evidence tests hash binding, not a redistribution claim.
        license_path = self.root/'synthetic-license.txt'
        license_path.write_bytes(b'Synthetic test-only license evidence; font not redistributed.')
        self.fonts = {'schema_version': 1, 'fonts': [{'id': 'dialogue', 'role': 'dialogue', 'weight': 'regular',
            'path': str(self.font), 'sha256': n.digest(self.font.read_bytes()), 'license': 'synthetic-test-only',
            'license_path': str(license_path), 'license_sha256': n.digest(license_path.read_bytes())}]}
        self.selection = {'schema_version': 1, 'purpose': 'candidate_only', 'source': source,
            'preview_sha256': '', 'fonts_sha256': '', 'units': [{
                'unit_id': unit.unit_id, 'unit_revision': unit.revision, 'request_revision': request.revision,
                'result_sha256': revision(asdict(result)), 'member_ids': list(unit.member_ids), 'order': list(unit.order),
                'text': outcome.translation, 'font_id': 'dialogue', 'geometry_intent': 'Synthetic flat-interior fixture',
                'interior': [rect(10, 10, 70, 105)], 'layout': [rect(12, 12, 68, 103)],
                'cleanup_permission': [rect(23, 20, 43, 85)],
                'protected': [rect(0, 0, 179, 8)], 'style': {'min_size': 14, 'max_size': 28, 'outline': 1}}]}
        self.write_inputs()

    def write_inputs(self):
        for name, doc in (('preview', self.preview), ('fonts', self.fonts)):
            (self.root/(name+'.json')).write_text(canonical(doc))
        self.selection['preview_sha256'] = n.digest((self.root/'preview.json').read_bytes())
        self.selection['fonts_sha256'] = n.digest((self.root/'fonts.json').read_bytes())
        (self.root/'selection.json').write_text(canonical(self.selection))

    def run_render(self, mode='compose', name='out'):
        return n.run(mode, preview=self.root/'preview.json', image=self.image,
            selection=self.root/'selection.json', fonts=self.root/'fonts.json', output=self.root/name)

    def test_compose_real_raster_exact_outside_clean_protected_and_source_immutable(self):
        original = self.image.read_bytes()
        with patch('socket.socket', side_effect=AssertionError('no networking')):
            report = self.run_render()
        self.assertEqual(report['units'][0]['status'], 'candidate_composed')
        self.assertFalse(report['approved_export'])
        self.assertEqual(self.image.read_bytes(), original)
        with Image.open(self.image) as im, Image.open(self.root/'out/clean.png') as clean, \
                Image.open(self.root/'out/composite.png') as composite, \
                Image.open(self.root/'out/unit-000-glyph-cleanup.png') as glyph, \
                Image.open(self.root/'out/unit-000-text.png') as layer:
            src, plate, comp = map(np.asarray, (im, clean, composite))
            g = np.asarray(glyph) != 0
            alpha = np.asarray(layer.getchannel('A')) != 0
            self.assertTrue(np.array_equal(src[~g], plate[~g]))
            self.assertTrue(np.array_equal(plate[~alpha], comp[~alpha]))
            self.assertTrue(np.array_equal(src[:9], comp[:9]))
        for name, sha in report['artifacts'].items():
            self.assertEqual(n.digest((self.root/'out'/name).read_bytes()), sha)

    def test_plan_never_erases(self):
        report = self.run_render('plan')
        self.assertEqual(report['units'][0]['status'], 'candidate_layout_no_erase')
        with Image.open(self.image) as source, Image.open(self.root/'out/clean.png') as clean:
            self.assertEqual(source.tobytes(), clean.tobytes())
        self.assertFalse((self.root/'out/unit-000-glyph-cleanup.png').exists())

    def test_actual_cli_compose_plan_and_help_import_offline(self):
        tool = Path(__file__).resolve().parents[2]/'tools/render_preview.py'
        for mode in ('compose', 'plan'):
            process = subprocess.run([sys.executable, str(tool), mode, '--preview', str(self.root/'preview.json'),
                '--image', str(self.image), '--selection', str(self.root/'selection.json'), '--fonts', str(self.root/'fonts.json'),
                '--output', str(self.root/mode)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((self.root/mode/'manifest.json').read_text())
            self.assertNotEqual(report['units'][0]['status'], 'blocked')
        process = subprocess.run([sys.executable, str(tool), '--help'], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('--selection', process.stdout)
        spec = importlib.util.spec_from_file_location('render_preview_test', tool)
        module = importlib.util.module_from_spec(spec)
        with patch('socket.socket', side_effect=AssertionError('no network')):
            spec.loader.exec_module(module)

    def test_stale_source_preview_font_result_and_typed_member_order_rejected(self):
        original = copy.deepcopy(self.selection)
        mutations = [('source', None), ('preview_sha256', '0'*64), ('fonts_sha256', '0'*64),
                     ('result_sha256', '0'*64), ('member_ids', [True]), ('order', ['foreign']), ('text', 'changed')]
        for key, value in mutations:
            self.selection = copy.deepcopy(original)
            if key in ('source', 'preview_sha256', 'fonts_sha256'):
                self.selection[key] = value
            else:
                self.selection['units'][0][key] = value
            (self.root/'selection.json').write_text(canonical(self.selection))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_render(name=key)

    def test_font_and_license_hashes_rejected(self):
        for key in ('sha256', 'license_sha256'):
            before = self.fonts['fonts'][0][key]
            self.fonts['fonts'][0][key] = '0'*64
            self.write_inputs()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'hash'):
                self.run_render(name=key)
            self.fonts['fonts'][0][key] = before

    def test_fragment_alternate_blocked_not_promoted(self):
        for kind in ('unverified_text_fragment', 'unverified_visual_alternate', 'blocked'):
            record = self.selected_record
            record['input_kind'] = kind
            self.write_inputs()
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'selection unsupported'):
                self.run_render(name=kind)

    def test_collision_symlink_and_arbitrary_sidecar_geometry_rejected(self):
        (self.root/'out').mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.run_render()
        (self.root/'link').symlink_to(self.root/'out', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_render(name='link')
        self.selection['units'][0]['layout'] = '/must/not/be/opened.json'
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'mask polygon'):
            self.run_render(name='sidecar')

    def test_overlapping_units_neighbor_and_outofbounds_geometry_rejected(self):
        self.selection['units'][0]['layout'] = [rect(0, 0, 300, 200)]
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.run_render()

    def test_neighbor_annotation_geometry_blocks_before_cleanup(self):
        self.selection['units'][0]['interior'] = [rect(10, 10, 160, 105)]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'neighbor/annotation'):
            self.run_render()
        self.assertFalse((self.root/'out').exists())

    def test_selected_region_collision_rejected(self):
        other = next(r for r in self.preview['units'] if r is not self.selected_record)
        uid = other['unit']['unit_id']
        reading = other['unit']['selected_revision']
        outcome = UnitOutcome(uid, 'preview', 'Hi!', (), 'unverified', reading)
        result = TranslationResult(other['request_revision'], 'preview', ((uid, 'Hi!'),), outcomes=(outcome,))
        other.update(status='preview', input_kind='strict_utterance', result=asdict(result))
        second = copy.deepcopy(self.selection['units'][0])
        second.update(unit_id=uid, unit_revision=other['unit_revision'], request_revision=other['request_revision'],
                      member_ids=other['unit']['member_ids'], order=other['unit']['order'], text='Hi!',
                      result_sha256=revision(asdict(result)))
        self.selection['units'].append(second)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'regions collide'):
            self.run_render()

    def test_embedded_profile_and_orientation_rejected(self):
        with Image.open(self.image) as opened:
            image = opened.copy()
        image.save(self.image, icc_profile=b'synthetic profile bytes, never converted')
        with self.assertRaisesRegex(ValueError, 'color profile'):
            self.run_render()
        exif = Image.Exif()
        exif[274] = 6
        image.save(self.image, exif=exif)
        with self.assertRaisesRegex(ValueError, 'orientation'):
            self.run_render()

    def test_rgb_transparent_color_is_not_silently_dropped(self):
        with Image.open(self.image) as opened:
            image = opened.convert('RGB')
        image.save(self.image, transparency=(255, 255, 255))
        with self.assertRaisesRegex(ValueError, 'transparency'):
            self.run_render()

    def test_import_and_cli_do_not_load_ocr_or_providers(self):
        tool = Path(__file__).resolve().parents[2]/'tools/render_preview.py'
        # A real subprocess renders a synthetic candidate with forbidden engine
        # and provider imports intercepted before loading the CLI/module.
        script = '''
import importlib.abc, importlib.util, pathlib, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'paddleocr','paddle','ollama','transformers','torch','requests','httpx'} or fullname == 'tralens.ocr.detection':
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Guard())
spec = importlib.util.spec_from_file_location('render_cli', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main(sys.argv[2:])
'''
        result = subprocess.run([sys.executable, '-c', script, str(tool), 'compose',
            '--preview', str(self.root/'preview.json'), '--image', str(self.image),
            '--selection', str(self.root/'selection.json'), '--fonts', str(self.root/'fonts.json'),
            '--output', str(self.root/'guarded')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_file_mutation_and_stale_request_rejected(self):
        before = self.image.read_bytes()
        with Image.open(self.image) as original:
            altered = original.copy()
        altered.putpixel((179, 129), (0, 0, 0))
        altered.save(self.image)
        with self.assertRaisesRegex(ValueError, 'stale source'):
            self.run_render()
        self.image.write_bytes(before)
        self.selected_record['request_revision'] = '0'*64
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'stale request'):
            self.run_render()

    def test_symlink_input_and_symlink_parent_rejected(self):
        link = self.root/'preview-link.json'
        link.symlink_to(self.root/'preview.json')
        with self.assertRaisesRegex(ValueError, 'symlink input'):
            n.run('plan', preview=link, image=self.image, selection=self.root/'selection.json',
                  fonts=self.root/'fonts.json', output=self.root/'out')
        (self.root/'parent-link').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink output'):
            self.run_render(name='parent-link/out')

    def test_source_mode_profile_and_mask_protection_rejected(self):
        self.selection['units'][0]['protected'] = [rect(20, 20, 40, 40)]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'protected'):
            self.run_render()
        image = Image.new('RGBA', (180, 130), 'white')
        image.save(self.image)
        with self.assertRaisesRegex(ValueError, 'source mode'):
            self.run_render()

    def test_engine_coverage_and_overflow_stay_reviewable(self):
        self.selection['units'][0]['style'] = {'min_size': 144, 'max_size': 144, 'outline': 1}
        self.write_inputs()
        report = self.run_render()
        self.assertEqual(report['units'][0]['status'], 'blocked')
        self.assertIn('overflow', report['units'][0]['reason'])
        with patch.object(n.features, 'check_feature', return_value=False):
            report = self.run_render(name='no-raqm')
        self.assertIn('RAQM', report['units'][0]['reason'])

    def test_cleanup_failure_retains_blocked_and_does_not_composite_layer(self):
        self.selection['units'][0]['cleanup_permission'] = [rect(25, 25, 30, 50)]
        self.write_inputs()
        report = self.run_render()
        self.assertEqual(report['units'][0]['status'], 'blocked')
        with Image.open(self.image) as source, Image.open(self.root/'out/composite.png') as comp:
            self.assertEqual(source.tobytes(), comp.tobytes())
