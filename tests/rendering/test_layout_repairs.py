"""Bounded layout/locality regressions, using only original synthetic font/raster data."""
from io import BytesIO
import struct
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tralens.rendering import natural as n
from .font_fixture import synthetic_font
from . import test_natural as fixtures


class LayoutRepairTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_font()
        self.coverage = n.font_coverage(self.data)

    def test_hyphenated_utterance_uses_larger_two_lines_in_tall_region(self):
        text = 'Mira-kun!?'
        layer, p = n.layout(text, self.data, self.coverage, Image.new('L', (110, 170), 255),
                            {'min_size': 14, 'max_size': 28, 'outline': 1})
        self.assertEqual(p['size'], 28)
        self.assertEqual(p['draw_text'], 'Mira-\nkun!?')
        self.assertEqual(p['original_text'], text)
        self.assertEqual(p['draw_text'].replace('\n', ''), text)
        self.assertEqual(p['inserted_break_offsets'], [5])
        self.assertGreater(np.count_nonzero(np.asarray(layer.getchannel('A'))), 0)

    def test_balanced_candidates_preserve_spaces_and_explicit_newlines(self):
        text = 'One two three four\nFive-six!'
        candidates = n.line_candidates(text, lambda s: len(s)*10, 110)
        self.assertTrue(any('One two \nthree four' in s for s, _ in candidates))
        for shaped, offsets in candidates:
            rebuilt = text
            for offset in reversed(offsets):
                rebuilt = rebuilt[:offset]+'\n'+rebuilt[offset:]
            self.assertEqual(shaped, rebuilt)
            self.assertIn('four\nFive', shaped)

    def test_punctuation_and_nonbreaking_hyphen_are_not_break_sites(self):
        for text in ('No!?', '?!', 'A--B', 'A-!?', '-Hello', 'Foo\u2011bar!?'):
            with self.subTest(text=text):
                self.assertEqual(n.break_positions(text), [0, len(text)])
        self.assertEqual(n.break_positions('Foo-bar!?'), [0, 4, 9])
        layer, p = n.layout('No!?', self.data, self.coverage, Image.new('L', (100, 170), 255),
                            {'min_size': 14, 'max_size': 28, 'outline': 0})
        self.assertEqual(p['draw_text'], 'No!?')

    def test_preplacement_clipping_cannot_pass_when_metrics_underestimate_ink(self):
        # Simulate a bbox/ink discrepancy while keeping the actual RAQM glyph
        # drawing unchanged. The old page-first algorithm accepted cropped ink.
        original = ImageDraw.ImageDraw.multiline_textbbox
        font = ImageFont.truetype(BytesIO(self.data), 28, layout_engine=ImageFont.Layout.RAQM)
        box = ImageDraw.Draw(Image.new('L', (1, 1))).multiline_textbbox((0, 0), 'ABC', font=font, stroke_width=1,
                                                                 direction='ltr', language='en', align='center')
        page_width = int(box[2]-box[0])-6
        def narrow(draw, *args, **kwargs):
            box = original(draw, *args, **kwargs)
            return box[0]+6, box[1], box[2]-6, box[3]
        with patch.object(ImageDraw.ImageDraw, 'multiline_textbbox', narrow):
            with self.assertRaisesRegex(ValueError, 'overflow'):
                n.layout('ABC', self.data, self.coverage, Image.new('L', (page_width, 80), 255),
                         {'min_size': 28, 'max_size': 28, 'outline': 1})

    def test_page_edge_layer_matches_complete_independent_padded_raster(self):
        mask = Image.new('L', (56, 65), 255)
        layer, p = n.layout('AV!\nYo?', self.data, self.coverage, mask,
                            {'min_size': 24, 'max_size': 24, 'outline': 2})
        font = ImageFont.truetype(BytesIO(self.data), p['size'], layout_engine=ImageFont.Layout.RAQM)
        reference = Image.new('RGBA', (400, 400))
        ImageDraw.Draw(reference).multiline_text((150, 150), p['draw_text'], font=font, fill='black',
            stroke_width=2, stroke_fill='white', align='center', spacing=p['spacing'], direction='ltr', language='en')
        reference = reference.crop(reference.getchannel('A').getbbox())
        actual = layer.crop(layer.getchannel('A').getbbox())
        self.assertEqual(actual.tobytes(), reference.tobytes())
        self.assertEqual(actual.size, reference.size)
        self.assertEqual(p['local_alpha_pixels'], np.count_nonzero(np.asarray(layer.getchannel('A'))))

    def test_local_canvas_budget_checked_before_allocation(self):
        with patch.object(n, 'MAX_LOCAL_PIXELS', 10), patch.object(n.Image, 'new', wraps=Image.new) as allocate:
            with self.assertRaisesRegex(ValueError, 'local raster budget'):
                n.layout('Hello', self.data, self.coverage, Image.new('L', (200, 100), 255),
                         {'min_size': 28, 'max_size': 28, 'outline': 1})
            self.assertFalse(any(call.args[:2] == ('RGBA', (200, 100)) for call in allocate.call_args_list))

    def test_cmap_expansion_budget_checked_before_populating_set(self):
        # Compact cmap12 groups alias a glyph range over many Unicode blocks.
        groups = b''.join(struct.pack('>III', 1000+i*96, 1094+i*96, 1) for i in range(1000))
        cmap = struct.pack('>HHHHI', 0, 1, 3, 10, 12)+struct.pack('>HHIII', 12, 0, 16+len(groups), 0, 1000)+groups
        maxp = struct.pack('>IH', 0x10000, 96)
        header = struct.pack('>IHHHH', 0x10000, 2, 32, 1, 0)
        data = header+struct.pack('>4sIII', b'cmap', 0, 44, len(cmap))+struct.pack('>4sIII', b'maxp', 0, 44+len(cmap), len(maxp))+cmap+maxp
        with self.assertRaisesRegex(ValueError, 'coverage.*budget'):
            n.font_coverage(data)

    def test_break_site_budget_and_full_page_budget_fail_before_big_allocation(self):
        with self.assertRaisesRegex(ValueError, 'break site budget'):
            n.line_candidates('a-b-'*100, len, 100)
        mask = Image.new('L', (100, 100), 255)
        with patch.object(n, 'MAX_PAGE_PIXELS', 10), patch.object(n.Image, 'new', wraps=Image.new) as allocate:
            with self.assertRaisesRegex(ValueError, 'page pixel budget'):
                n.layout('Hi!', self.data, self.coverage, mask, {'min_size': 14, 'max_size': 28, 'outline': 0})
            allocate.assert_not_called()


class LocalityRepairTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CommandTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def assert_original_outputs(self):
        root = self.fixture.root
        with Image.open(self.fixture.image) as src, Image.open(root/'out/clean.png') as clean, \
                Image.open(root/'out/composite.png') as composite:
            self.assertEqual(src.tobytes(), clean.tobytes())
            self.assertEqual(src.tobytes(), composite.tobytes())

    def test_persisted_counts_match_actual_pixels(self):
        f = self.fixture
        report = f.run_render()
        from json import loads
        saved = loads((f.root/'out/manifest.json').read_text())
        with Image.open(f.image) as src, Image.open(f.root/'out/clean.png') as clean, \
                Image.open(f.root/'out/composite.png') as comp, Image.open(f.root/'out/unit-000-text.png') as layer, \
                Image.open(f.root/'out/unit-000-glyph-cleanup.png') as glyph, \
                Image.open(f.root/'out/unit-000-protected.png') as protected:
            g = np.asarray(glyph) != 0
            a = np.asarray(layer.getchannel('A')) != 0
            p = np.asarray(protected) != 0
            expected = n.pixel_audit(src, clean, comp, g, a, p)
            # Independent exact checks of the totals, not merely helper equality.
            self.assertEqual(expected['outside_cleanup_unchanged_pixels'], int(g.size-g.sum()))
            self.assertEqual(expected['protected_unchanged_pixels'], int(p.sum()))
            self.assertEqual(expected['outside_alpha_unchanged_pixels'], int(a.size-a.sum()))
            self.assertEqual(expected['untouched_source_pixels'], int(np.count_nonzero(np.all(np.asarray(src) == np.asarray(comp), axis=2))))
        self.assertEqual(saved['pixel_audit'], expected)
        self.assertEqual(saved['units'][0]['pixel_audit'], expected)
        self.assertEqual(report['units'][0]['outside_layout_alpha_pixels'], 0)
        self.assertEqual(report['units'][0]['outside_cleanup_permission_pixels'], 0)

    def test_runtime_cleanup_violation_blocks_without_committing_pixels(self):
        original = n.cleanup
        def bad_cleanup(*args):
            plate, glyph, color = original(*args)
            plate.putpixel((0, 0), (255, 0, 0))
            return plate, glyph, color
        with patch.object(n, 'cleanup', bad_cleanup):
            report = self.fixture.run_render()
        row = report['units'][0]
        self.assertEqual(row['status'], 'blocked')
        self.assertEqual(row['pixel_audit']['outside_cleanup_changed_pixels'], 1)
        self.assertEqual(row['pixel_audit']['protected_clean_changed_pixels'], 1)
        self.assert_original_outputs()

    def test_runtime_composition_violation_blocks_without_committing_pixels(self):
        original = Image.alpha_composite
        def bad_composite(*args):
            result = original(*args)
            result.putpixel((179, 129), (255, 0, 0, 255))
            return result
        with patch.object(n.Image, 'alpha_composite', bad_composite):
            report = self.fixture.run_render()
        row = report['units'][0]
        self.assertEqual(row['status'], 'blocked')
        self.assertEqual(row['pixel_audit']['outside_alpha_changed_pixels'], 1)
        self.assert_original_outputs()

    def test_global_locality_violation_retains_rejected_counts_and_rolls_back(self):
        original = Image.Image.alpha_composite
        def bad_global(image, *args, **kwargs):
            original(image, *args, **kwargs)
            image.putpixel((179, 129), (255, 0, 0, 255))
        with patch.object(Image.Image, 'alpha_composite', bad_global):
            report = self.fixture.run_render()
        self.assertEqual(report['units'][0]['status'], 'blocked')
        self.assertEqual(report['units'][0]['failure_kind'], 'runtime_locality')
        self.assertEqual(report['rejected_pixel_audit']['outside_alpha_changed_pixels'], 1)
        self.assertEqual(report['pixel_audit']['untouched_source_pixels'], 180*130)
        self.assert_original_outputs()

    def test_font_and_local_raster_resource_failures_are_blocked_in_manifest(self):
        with patch.object(n, 'MAX_FONT_BYTES', 1):
            report = self.fixture.run_render()
        self.assertEqual(report['units'][0]['status'], 'blocked')
        self.assertEqual(report['units'][0]['failure_kind'], 'resource_limit')
        self.assert_original_outputs()
        with patch.object(n, 'MAX_LOCAL_PIXELS', 10):
            report = self.fixture.run_render(name='local-budget')
        self.assertEqual(report['units'][0]['failure_kind'], 'resource_limit')
        self.assertIn('local raster budget', report['units'][0]['reason'])

    def test_plan_persists_zero_cleanup_and_outside_alpha_counts(self):
        report = self.fixture.run_render('plan')
        audit = report['pixel_audit']
        self.assertEqual(audit['clean_changed_pixels'], 0)
        self.assertEqual(audit['outside_alpha_changed_pixels'], 0)
        self.assertEqual(audit['outside_cleanup_unchanged_pixels'], 180*130)

    def test_actual_cli_composes_hyphenated_candidate_and_pixel_audit(self):
        f = self.fixture
        text = 'Mira-kun!?'
        result = f.selected_record['result']
        result['translations'] = [[f.selection['units'][0]['unit_id'], text]]
        result['outcomes'][0]['translation'] = text
        f.selection['units'][0].update(text=text, result_sha256=n.revision(result))
        f.write_inputs()
        tool = Path(__file__).resolve().parents[2]/'tools/render_preview.py'
        process = subprocess.run([sys.executable, str(tool), 'compose', '--preview', str(f.root/'preview.json'),
            '--image', str(f.image), '--selection', str(f.root/'selection.json'), '--fonts', str(f.root/'fonts.json'),
            '--output', str(f.root/'out')], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((f.root/'out/manifest.json').read_text())
        row = report['units'][0]
        self.assertEqual(row['status'], 'candidate_composed')
        self.assertEqual(row['layout']['draw_text'], 'Mira-\nkun!?')
        self.assertEqual(row['layout']['original_text'], text)
        self.assertEqual(report['pixel_audit']['outside_alpha_changed_pixels'], 0)
        self.assertTrue((f.root/'out/review.html').is_file())


if __name__ == '__main__':
    unittest.main()
