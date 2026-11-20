"""Source-only cleanup diagnostics and paired fitting regressions; no page assets."""
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tralens.rendering import natural as n
from .font_fixture import synthetic_font
from . import test_natural as fixtures


class CleanupDiagnosticTests(unittest.TestCase):
    def fixture(self, color=(245, 243, 239)):
        image = Image.new('RGB', (80, 80), color)
        ImageDraw.Draw(image).rectangle((30, 30, 38, 38), fill='black')
        interior = n.union_mask([fixtures.rect(5, 5, 74, 74)], image.size)
        permission = n.union_mask([fixtures.rect(25, 25, 45, 45)], image.size)
        protected = np.zeros((80, 80), dtype=bool)
        return image, interior, permission, protected

    def blocked(self, args):
        with self.assertRaises(n.CleanupBlocked) as caught:
            n.cleanup(*args)
        return caught.exception.diagnostics

    def test_sparse_contamination_reports_exact_pixels_and_supported_geometry_fix(self):
        image, interior, permission, protected = self.fixture()
        points = [(5, 5), (6, 5), (73, 73)]
        for point in points:
            image.putpixel(point, (0, 0, 0))
        diagnostic = self.blocked((image, interior, permission, protected))
        self.assertEqual(diagnostic['code'], 'nonflat_sample')
        self.assertEqual(diagnostic['background_estimate_rgb'], [245, 243, 239])
        self.assertEqual(diagnostic['sample_count'], int((interior & ~permission).sum()))
        self.assertEqual(diagnostic['offending_count'], 3)
        self.assertEqual(diagnostic['offending_coordinates'], [list(p) for p in points])
        self.assertEqual(diagnostic['offending_bounds'], [5, 5, 74, 74])
        self.assertFalse(diagnostic['coordinates_truncated'])
        self.assertEqual(diagnostic['semantic_classification'], 'not_inferred')
        # Synthetic, explicit revised interior excludes only the offending edge.
        revised = n.union_mask([fixtures.rect(7, 7, 72, 72)], image.size)
        clean, glyph, color = n.cleanup(image, revised, permission, protected)
        self.assertEqual(color, [245, 243, 239])
        self.assertTrue(np.array_equal(np.asarray(image)[~glyph], np.asarray(clean)[~glyph]))
        self.assertTrue(np.array_equal(np.asarray(image)[protected], np.asarray(clean)[protected]))

    def test_texture_and_gradient_share_measured_nonflat_code_and_bounded_coordinates(self):
        for kind in ('texture', 'gradient'):
            image, interior, permission, protected = self.fixture()
            draw = ImageDraw.Draw(image)
            for y in range(5, 75):
                draw.line((5, y, 20, y), fill=(190+(y % 2)*40,)*3 if kind == 'texture' else (180+y,)*3)
            diagnostic = self.blocked((image, interior, permission, protected))
            self.assertEqual(diagnostic['code'], 'nonflat_sample')
            self.assertGreater(diagnostic['offending_count'], 32)
            self.assertEqual(len(diagnostic['offending_coordinates']), 32)
            self.assertTrue(diagnostic['coordinates_truncated'])
            self.assertLess(len(json.dumps(diagnostic)), 4096)
            self.assertEqual(diagnostic['semantic_classification'], 'not_inferred')

    def test_insufficient_empty_samples_and_boundary_touch_are_distinct(self):
        image, interior, permission, protected = self.fixture()
        all_permission = interior.copy()
        d = self.blocked((image, interior, all_permission, protected))
        self.assertEqual(d['code'], 'insufficient_sample')
        self.assertEqual(d['sample_count'], 0)
        self.assertIsNone(d['background_estimate_rgb'])
        self.assertEqual(d['offending_count'], 0)
        image.putpixel((25, 31), (0, 0, 0))
        d = self.blocked((image, interior, permission, protected))
        self.assertEqual(d['code'], 'boundary_touch')
        self.assertGreater(d['offending_count'], 0)
        self.assertIn('source', d['next_safe_action'].lower())

    def test_small_sample_estimate_dark_background_and_excessive_area_stay_blocked(self):
        image, interior, permission, protected = self.fixture()
        permission = interior.copy()
        permission[5, 5:35] = False
        d = self.blocked((image, interior, permission, protected))
        self.assertEqual(d['code'], 'insufficient_sample')
        self.assertEqual(d['sample_count'], 30)
        self.assertEqual(d['background_estimate_rgb'], [245, 243, 239])
        image, interior, permission, protected = self.fixture(color=(100, 100, 100))
        self.assertEqual(self.blocked((image, interior, permission, protected))['code'], 'unsupported_dark_background')
        image, interior, _, protected = self.fixture()
        permission = n.union_mask([fixtures.rect(10, 10, 68, 68)], image.size)
        ImageDraw.Draw(image).rectangle((12, 12, 66, 66), fill='black')
        d = self.blocked((image, interior, permission, protected))
        self.assertEqual(d['code'], 'excessive_ink_area')
        self.assertGreater(d['glyph_count'], d['interior_count']//3)

    def test_unpermitted_ink_is_not_mislabelled_as_semantic_neighbor(self):
        image, interior, permission, protected = self.fixture()
        # Protected ink is excluded from background samples, but never gains
        # erase permission. The pixels do not prove whether it is text or art.
        protected[10, 10] = True
        image.putpixel((10, 10), (0, 0, 0))
        d = self.blocked((image, interior, permission, protected))
        self.assertEqual(d['code'], 'unpermitted_ink')
        self.assertEqual(d['offending_coordinates'], [[10, 10]])
        self.assertEqual(d['semantic_classification'], 'not_inferred')


class FittingEvidenceTests(unittest.TestCase):
    def test_existing_spaces_do_not_reject_a_complete_ink_fit(self):
        data = synthetic_font()
        font = ImageFont.truetype(BytesIO(data), 28, layout_engine=ImageFont.Layout.RAQM)
        text = 'ABCDE FGH'
        reference = Image.new('RGBA', (400, 200))
        ImageDraw.Draw(reference).multiline_text((100, 50), 'ABCDE \nFGH', font=font, fill='black',
            align='center', spacing=5, direction='ltr', language='en')
        ink = reference.getchannel('A').getbbox()
        width = max(ink[2]-ink[0]+2, math.ceil(ImageDraw.Draw(reference).textlength('ABCDE', font=font)))
        # Complete independently rendered ink fits this narrow region. The
        # original retained trailing space has advance but no visible pixels.
        layer, p = n.layout(text, data, n.font_coverage(data), Image.new('L', (width, 120), 255),
                            {'min_size': 28, 'max_size': 28, 'outline': 0})
        self.assertEqual(p['draw_text'], 'ABCDE \nFGH')
        self.assertEqual(p['original_text'], text)
        self.assertEqual(p['size'], 28)
        actual = layer.crop(layer.getchannel('A').getbbox())
        self.assertEqual(actual.size, (ink[2]-ink[0], ink[3]-ink[1]))
        self.assertEqual(actual.tobytes(), reference.crop(ink).tobytes())

    def test_narrow_unbreakable_word_still_limits_size_and_never_gets_rewritten(self):
        data = synthetic_font()
        cov = n.font_coverage(data)
        text = 'ABCDE FGH'
        style = {'min_size': 14, 'max_size': 28, 'outline': 0}
        _, narrow = n.layout(text, data, cov, Image.new('L', (54, 180), 255), style)
        _, wider = n.layout(text, data, cov, Image.new('L', (100, 180), 255), style)
        self.assertLess(narrow['size'], wider['size'])
        self.assertEqual(wider['size'], 28)
        for result in (narrow, wider):
            self.assertEqual(result['draw_text'].replace('\n', ''), text)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            n.layout(text, data, cov, Image.new('L', (54, 180), 255), {**style, 'min_size': 28})

    def test_longer_dialogue_height_limit_remains_a_real_mask_constraint(self):
        data = synthetic_font()
        cov = n.font_coverage(data)
        text = 'Go on to do it as we go'
        style = {'min_size': 14, 'max_size': 28, 'outline': 0}
        _, short = n.layout(text, data, cov, Image.new('L', (100, 70), 255), style)
        layer, tall = n.layout(text, data, cov, Image.new('L', (100, 150), 255), style)
        self.assertLess(short['size'], tall['size'])
        self.assertEqual(tall['size'], 28)
        for result in (short, tall):
            self.assertEqual(result['original_text'], text)
            self.assertEqual(result['draw_text'].replace('\n', ''), text)
        self.assertLessEqual(layer.getchannel('A').getbbox()[3], 150)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            n.layout(text, data, cov, Image.new('L', (100, 70), 255), {**style, 'min_size': 28})


class DiagnosticWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CommandTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def inputs(self):
        f = self.f
        return dict(preview=f.root/'preview.json', image=f.image, selection=f.root/'selection.json', fonts=f.root/'fonts.json')

    def test_blocked_manifest_report_and_recomputed_diagnostics_are_bound(self):
        from tralens.rendering.review import verify_candidate, specimen_html
        # Include a known synthetic balloon border as sample contamination.
        self.f.selection['units'][0]['interior'] = [fixtures.rect(7, 10, 75, 105)]
        self.f.write_inputs()
        report = self.f.run_render()
        row = report['units'][0]
        self.assertEqual(row['status'], 'blocked')
        self.assertEqual(row['cleanup_diagnostics']['code'], 'nonflat_sample')
        root = self.f.root/'out'
        html = (root/'review.html').read_text()
        self.assertIn('Next safe action', html)
        self.assertIn('Offending pixels', html)
        with Image.open(self.f.image) as original, Image.open(root/'composite.png') as composite:
            self.assertEqual(original.tobytes(), composite.tobytes())
        verify_candidate(candidate=root, **self.inputs())
        row['cleanup_diagnostics']['offending_count'] += 1
        page = specimen_html(report).encode()
        (root/'review.html').write_bytes(page)
        report['artifacts']['review.html'] = n.digest(page)
        (root/'manifest.json').write_text(n.canonical(report))
        with self.assertRaisesRegex(ValueError, 'cleanup diagnostics drift'):
            verify_candidate(candidate=root, **self.inputs())

    def test_successful_review_rejects_invented_rehashed_cleanup_diagnostics(self):
        from tralens.rendering.review import record_review, verify_review, specimen_html
        report = self.f.run_render()
        root = self.f.root/'out'
        record = self.f.root/'review.json'
        record_review(candidate=root, unit_id=report['units'][0]['unit_id'], state='rework',
            operator='synthetic-test-operator', note='Synthetic aesthetic test', output=record, **self.inputs())
        report['units'][0]['cleanup_diagnostics'] = {'code': 'invented', 'message': 'invented',
            'background_estimate_rgb': [255, 255, 255], 'sample_count': 1, 'offending_count': 1,
            'offending_bounds': [0, 0, 1, 1], 'next_safe_action': '<script>untrusted</script>'}
        page = specimen_html(report).encode()
        self.assertNotIn(b'<script>', page)
        (root/'review.html').write_bytes(page)
        report['artifacts']['review.html'] = n.digest(page)
        (root/'manifest.json').write_text(n.canonical(report))
        with self.assertRaisesRegex(ValueError, 'cleanup diagnostics drift'):
            verify_review(candidate=root, review=record, **self.inputs())

    def test_actual_offline_cli_persists_actionable_block_and_original_pixels(self):
        self.f.selection['units'][0]['interior'] = [fixtures.rect(7, 10, 75, 105)]
        self.f.write_inputs()
        tool = Path(__file__).resolve().parents[2]/'tools/render_preview.py'
        args = [item for key, path in self.inputs().items() for item in ('--'+key, str(path))]
        process = subprocess.run([sys.executable, str(tool), 'compose', *args, '--output', str(self.f.root/'cli')],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((self.f.root/'cli/manifest.json').read_text())
        self.assertEqual(report['units'][0]['cleanup_diagnostics']['code'], 'nonflat_sample')
        self.assertEqual(report['units'][0]['status'], 'blocked')
        self.assertEqual(report['pixel_audit']['composite_source_changed_pixels'], 0)
        self.assertEqual(report['pixel_audit']['cleanup_mask_pixels'], 0)
