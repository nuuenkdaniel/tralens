"""Comfortable automatic fitting, exact wording and shared bounded RAQM work."""
from io import BytesIO
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tests.local_evidence import local_evidence
from tralens.rendering import automatic as a, natural as n, typography as t
from .font_fixture import synthetic_font
from . import test_natural as fixtures


class TypographyTests(unittest.TestCase):
    def test_balloon_interior_box_ignores_connected_gutter_and_avoids_art(self):
        paper = np.zeros((100,140),bool)
        paper[30:80,20:115] = True
        paper[4:40,92:95] = True
        paper[30:39,20:34] = False
        box = t.interior_box(paper)
        left,top,right,bottom = box.getbbox()
        self.assertTrue((np.asarray(box,dtype=bool)<=paper).all())
        self.assertGreater(right-left,65)
        self.assertGreater(bottom-top,31)
        self.assertGreaterEqual(top,30)


    def setUp(self):
        self.data = synthetic_font()
        self.coverage = n.font_coverage(self.data)
        self.style = {'min_size': 14, 'max_size': 28, 'outline': 0}

    def layout(self, text, mask=None, style=None):
        return t.layout_dialogue(text, self.data, self.coverage,
            Image.new('L', (180, 160), 255) if mask is None else mask,
            self.style if style is None else style)

    def test_rotated_sound_can_share_empty_tile_corners_without_overlapping_ink(self):
        from tralens.rendering.outlined_sound_cleanup import source_centered_lettering
        blocked = np.zeros((130,180),bool)
        frame = (30,24,148,104)
        args = ('Clap',self.data,self.coverage,frame)
        style = {**self.style,'rotation':20,'outline':2,'fill':[255]*3,'outline_fill':[0]*3}
        first,xy,proof = source_centered_lettering(*args,blocked,style)
        alpha = np.asarray(first.getchannel('A'))!=0
        x,y=xy
        blocked[y,x]=True
        second,new_xy,_ = source_centered_lettering(*args,blocked,style)
        self.assertEqual(xy,new_xy)
        self.assertEqual(first.tobytes(),second.tobytes())
        self.assertEqual(proof['original_text'],'Clap')
        blocked[y:y+first.height,x:x+first.width] |= alpha
        self.assertIsNone(source_centered_lettering(*args,blocked,style))

    def test_stutter_prefix_stays_with_its_word_and_ordinary_compounds_can_break(self):
        self.assertEqual(n.break_positions('N-no.'),[0,5])
        self.assertNotIn(3,n.break_positions("Th-that's new."))
        self.assertIn(6,n.break_positions('paper-thin'))
        _,meta,_ = self.layout('W-well, yeah.',Image.new('L',(100,130),255))
        self.assertNotIn('W-\n',meta['draw_text'])

    def test_thought_keeps_three_beats_and_exact_characters(self):
        source = 'Oh! Too late. They know...'
        _, proof, _ = self.layout(source,Image.new('L',(320,140),255),{**self.style,'preset':'thought'})
        self.assertEqual(proof['draw_text'],'Oh! \nToo late. \nThey know...')
        self.assertEqual(proof['draw_text'].replace('\n',''),source)
        self.assertGreaterEqual(proof['size'],20)

    def test_compact_hyphenated_name_prefers_an_intact_readable_line(self):
        text = 'Nina-san'
        _,meta,_ = self.layout(text,Image.new('L',(115,100),255))
        self.assertEqual(meta['draw_text'],text)
        _,narrow,_ = self.layout(text,Image.new('L',(64,100),255))
        self.assertIn('\n',narrow['draw_text'])
        self.assertGreaterEqual(narrow['size'],self.style['min_size'])

    def test_comfortable_fit_is_inset_and_does_not_always_maximize_font(self):
        mask = Image.new('L', (180, 160), 255)
        layer, meta, content = self.layout('Hello there!', mask)
        self.assertEqual(meta['size'], 25)
        typography = meta['typography']
        self.assertEqual(typography['preferred_size'], 25)
        self.assertGreaterEqual(typography['margin_pixels'], 2)
        self.assertLessEqual(typography['margin_pixels'], 8)
        alpha = np.asarray(layer.getchannel('A')) != 0
        self.assertFalse((alpha & (np.asarray(content) == 0)).any())
        margin = typography['margin_pixels']
        self.assertFalse(alpha[:margin].any())
        self.assertFalse(alpha[-margin:].any())
        self.assertFalse(alpha[:, :margin].any())
        self.assertFalse(alpha[:, -margin:].any())
        self.assertLessEqual(len(typography['probes']), t.MAX_PROBES)

    def test_short_word_orphans_and_tiny_last_line_are_layout_penalties_only(self):
        bad = t.line_quality("I don't need it.", "I\ndon't\nneed\nit.", [5, 40, 35, 10])
        good = t.line_quality("I don't need it.", "I don't\nneed it.", [40, 39])
        self.assertEqual(bad['short_word_orphans'], 2)
        self.assertEqual(good['short_word_orphans'], 0)
        self.assertGreater(bad['cost'], good['cost'])
        widow = t.line_quality('Please bring everyone here now.', 'Please bring\neveryone here\nnow.', [100, 105, 30])
        self.assertEqual(widow['tiny_final_line'], 1)
        # Short utterances and explicit line breaks remain intentional inputs.
        for original, shaped in (("I'm sorry.", "I'm\nsorry."), ('I\nam here.', 'I\nam here.')):
            quality = t.line_quality(original, shaped, [10, 50])
            self.assertEqual(quality['short_word_orphans'], 0)

    def test_actual_narrow_fit_avoids_four_stacked_words_with_a_smaller_legal_font(self):
        mask = Image.new('L', (105, 160), 255)
        _, old = n.layout("I don't need it.", self.data, self.coverage, mask, self.style)
        _, new, _ = self.layout("I don't need it.", mask)
        self.assertEqual(len(old['draw_text'].split('\n')), 4)
        self.assertEqual(len(new['draw_text'].split('\n')), 2)
        self.assertGreaterEqual(new['size'], self.style['min_size'])
        self.assertLess(new['size'], old['size'])

    def test_exact_spaces_punctuation_hyphens_and_newlines_survive(self):
        for text in ('One two three four!', 'Mira-kun!?', 'One  two\nFive-six!', ' Really? Yes! '):
            layer, meta, _ = self.layout(text)
            rebuilt = text
            for offset in reversed(meta['inserted_break_offsets']):
                rebuilt = rebuilt[:offset]+'\n'+rebuilt[offset:]
            self.assertEqual(rebuilt, meta['draw_text'])
            self.assertEqual(meta['original_text'], text)
            self.assertEqual(meta['line_break_policy'], 'comfortable_legal_breaks_v1')
            self.assertTrue(layer.getchannel('A').getbbox())

    def test_irregular_holes_never_become_text_permission(self):
        mask = Image.new('L', (220, 180), 255)
        ImageDraw.Draw(mask).rectangle((0, 0, 50, 70), fill=0)
        layer, _, content = self.layout('A safe\nnew line.', mask)
        allowed = np.asarray(mask) != 0
        inset = np.asarray(content) != 0
        self.assertFalse((inset & ~allowed).any())
        self.assertFalse(((np.asarray(layer.getchannel('A')) != 0) & ~inset).any())

    def test_minimum_size_and_missing_glyphs_remain_hard_failures(self):
        with self.assertRaisesRegex(ValueError, 'overflow'):
            self.layout('Unbreakableword'*10, Image.new('L', (35, 40), 255))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            t.layout_dialogue('Hello', self.data, self.coverage-{ord('H')}, Image.new('L', (200, 120), 255), self.style)
        with self.assertRaisesRegex(ValueError, 'comfortable text margins'):
            self.layout('Hi', Image.new('L', (3, 3), 255))
        _, meta, _ = self.layout('Hi!', style={'min_size': 28, 'max_size': 28, 'outline': 0})
        self.assertEqual(meta['size'], 28)
        self.assertEqual(len(meta['typography']['probes']), 1)

    def test_final_alpha_is_complete_independent_raqm_raster(self):
        layer, meta, _ = self.layout('AV!\nYes?')
        font = ImageFont.truetype(BytesIO(self.data), meta['size'], layout_engine=ImageFont.Layout.RAQM)
        reference = Image.new('RGBA', (400, 400))
        ImageDraw.Draw(reference).multiline_text((100, 100), meta['draw_text'], font=font,
            fill='black', align='center', spacing=meta['spacing'], direction='ltr', language='en')
        reference = reference.crop(reference.getchannel('A').getbbox())
        actual = layer.crop(layer.getchannel('A').getbbox())
        self.assertEqual(actual.size, reference.size)
        self.assertEqual(actual.tobytes(), reference.tobytes())
        self.assertEqual(meta['local_alpha_pixels'], int(np.count_nonzero(np.asarray(layer.getchannel('A')))))

    def test_probes_share_one_budget_and_report_all_spent_work(self):
        original = n.layout
        works = []
        def counted(*args, **kwargs):
            works.append(kwargs['work'])
            return original(*args, **kwargs)
        with patch.object(n, 'layout', side_effect=counted):
            _, meta, _ = self.layout('Hello there!')
        self.assertGreater(len(works), 1)
        self.assertTrue(all(work is works[0] for work in works))
        self.assertEqual(meta['rasterized_pixels'], works[0].rasterized_pixels)
        self.assertEqual(meta['search_work'], works[0].search)
        self.assertGreater(meta['search_work']['measures'], 0)

    def test_resource_failure_after_a_fitting_probe_cannot_reset_limits_or_sneak_through(self):
        original = n.layout
        calls = 0
        def limited(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise n.ResourceLimit('synthetic shared work exhaustion')
            return original(*args, **kwargs)
        with patch.object(n, 'layout', side_effect=limited):
            with self.assertRaises(n.ResourceLimit):
                self.layout('Hello there!')
        self.assertEqual(calls, 2)

    def test_shared_raster_and_measurement_limits_apply_across_calls(self):
        work = n.LayoutWork(rasterized_pixels=64_000_000)
        with self.assertRaisesRegex(n.ResourceLimit, 'cumulative local raster budget'):
            n.layout('Hi!', self.data, self.coverage, Image.new('L', (200, 100), 255), self.style, work=work)
        work = n.LayoutWork(search={'measures': 65536, 'transitions': 0})
        with self.assertRaisesRegex(n.ResourceLimit, 'measurement budget'):
            n.layout('Hi!', self.data, self.coverage, Image.new('L', (200, 100), 255), self.style, work=work)

    def test_deterministic_layout_and_default_manual_policy_unchanged(self):
        one, meta, mask = self.layout('Hello there!')
        two, again, other = self.layout('Hello there!')
        self.assertEqual(one.tobytes(), two.tobytes())
        self.assertEqual(meta, again)
        self.assertEqual(mask.tobytes(), other.tobytes())
        _, legacy = n.layout('Hello there!', self.data, self.coverage, Image.new('L', (180, 160), 255), self.style)
        self.assertNotIn('typography', legacy)
        self.assertEqual(legacy['line_break_policy'], 'balanced_legal_hyphen_visible_space_v2')
        self.assertEqual(legacy['size'], 28)


class TypographyContainmentTests(unittest.TestCase):
    def test_runtime_margin_violation_blocks_unit_without_erasing_source(self):
        fixture = fixtures.CommandTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = t.layout_dialogue
        def outside_margin(text, data, coverage, mask, style):
            layer, meta, content = original(text, data, coverage, mask, style)
            padding = (np.asarray(mask) != 0) & (np.asarray(content) == 0)
            y, x = np.argwhere(padding)[0]
            layer.putpixel((int(x), int(y)), (0, 0, 0, 255))
            return layer, meta, content
        with patch.object(t, 'layout_dialogue', side_effect=outside_margin):
            report = a.run_automatic(preview=fixture.root/'preview.json', image=fixture.image,
                fonts=fixture.root/'fonts.json', output=fixture.root/'render', min_size=14, max_size=28, render_policy='conservative')
        self.assertEqual(report['coverage']['rendered'], 0)
        selected = next(row for row in report['units'] if row['machine_selected'])
        self.assertEqual(selected['reason_code'], 'typography_containment')
        self.assertEqual(selected['outside_content_alpha_pixels'], 1)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)
        with Image.open(fixture.image) as source, Image.open(fixture.root/'render/clean.png') as clean:
            self.assertEqual(source.tobytes(), clean.tobytes())


class AdditionalPageTypographyTests(unittest.TestCase):
    def evaluate_page(self, image_name):
        """Geometry + a literal QA string, never a manufactured translation."""
        root = Path(__file__).resolve().parents[2]
        artifact = root/'test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json'
        image_path = root/'images/ubunchu-ja-test'/image_name
        data = n.read_explicit(artifact)
        pages = [page for page in n.document(data)['pages'] if Path(page['page']['image_path']).name == image_name]
        self.assertEqual(len(pages), 1)
        page = pages[0]
        content = n.read_explicit(image_path)
        self.assertEqual(page['page']['image_path'], str(image_path))
        self.assertEqual(page['page']['image_sha256'], n.digest(content))
        with Image.open(BytesIO(content)) as opened:
            rgb = opened.convert('RGB')
        pixels = np.asarray(rgb).copy()
        self.assertEqual(page['page']['pixels_sha256'], n.digest(f'RGB:{rgb.width}:{rgb.height}:'.encode()+rgb.tobytes()))
        self.assertEqual(page['page'], page['batch']['page'])
        units = n.automatic_units(page['batch'])
        raw = page['batch']['detections']
        count = min(a.MAX_SLOTS, n.MAX_UNIT_PIXELS//(rgb.width*rgb.height), len(units))
        self.assertGreater(count, 0)
        context = a.OriginalComponents(rgb)
        fonts = n.document(n.read_explicit(root/'test-results/render-parent-input-v1/fonts.json'))
        font_data, coverage = n.load_fonts(fonts)['dialogue-bold']
        rows = []
        safe_masks = 0
        for unit in units[:count]:
            row = {'unit_id': unit.unit_id, 'members': list(unit.member_ids), 'erase_applied': False}
            rows.append(row)
            try:
                plan = a.derive_plan(rgb, raw, list(unit.member_ids), context=context)
            except a.AutomaticSkip as error:
                row.update(status='geometry_skipped', reason_code=error.code)
                continue
            safe_masks += 1
            try:
                layer, meta, content_mask = t.layout_dialogue('Sample dialogue.', font_data, coverage,
                    Image.fromarray(plan.layout.astype(np.uint8)*255), {'min_size': 16, 'max_size': 28, 'outline': 0})
            except n.ResourceLimit:
                raise
            except ValueError as error:
                self.assertIn('overflow:', str(error))
                row.update(status='sample_overflow', reason_code='typography_overflow')
                continue
            alpha = np.asarray(layer.getchannel('A')) != 0
            self.assertFalse((alpha & ~plan.layout).any())
            self.assertFalse((alpha & plan.obstacles).any())
            self.assertFalse((alpha & (np.asarray(content_mask) == 0)).any())
            self.assertGreaterEqual(meta['size'], 16)
            self.assertLessEqual(meta['size'], 28)
            rebuilt = 'Sample dialogue.'
            for offset in reversed(meta['inserted_break_offsets']):
                rebuilt = rebuilt[:offset]+'\n'+rebuilt[offset:]
            self.assertEqual(meta['draw_text'], rebuilt)
            row.update(status='sample_fit', size=meta['size'], draw_text=meta['draw_text'],
                ink_bounds=list(layer.getchannel('A').getbbox()), typography=meta['typography'])
        self.assertTrue(np.array_equal(pixels, np.asarray(rgb)))
        self.assertEqual(n.read_explicit(image_path), content)
        self.assertEqual(n.read_explicit(artifact), data)
        counts = dict(Counter(row['status'] for row in rows))
        report = {'image': image_name, 'scope': 'geometry/typesetting QA only; sample is not a translation',
            'sample_text': 'Sample dialogue.', 'source_sha256': n.digest(content), 'artifact_sha256': n.digest(data),
            'evaluated_units': count, 'saved_units': len(units), 'safe_masks': safe_masks,
            'counts': counts, 'units': rows}
        print('ADDITIONAL_PAGE_TYPOGRAPHY='+json.dumps({k: v for k, v in report.items() if k != 'units'}, sort_keys=True))
        return report

    @local_evidence(files=('test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json',
        'images/ubunchu-ja-test/ubunchu01_02.png', 'test-results/render-parent-input-v1/fonts.json'))
    def test_page_01_02_geometry_and_sample_fit_without_translation_or_erasure(self):
        report = self.evaluate_page('ubunchu01_02.png')
        self.assertEqual(report['evaluated_units'], 15)
        self.assertEqual(report['safe_masks'], 6)
        self.assertEqual(report['counts'], {'geometry_skipped': 9, 'sample_fit': 5, 'sample_overflow': 1})

    @local_evidence(files=('test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json',
        'images/ubunchu-ja-test/ubunchu02_04.png', 'test-results/render-parent-input-v1/fonts.json'))
    def test_page_02_04_geometry_and_sample_fit_without_translation_or_erasure(self):
        report = self.evaluate_page('ubunchu02_04.png')
        self.assertEqual(report['evaluated_units'], 14)
        self.assertEqual(report['safe_masks'], 8)
        self.assertEqual(report['counts'], {'geometry_skipped': 6, 'sample_fit': 8})


class SavedTypographyTests(unittest.TestCase):
    @local_evidence(files=('test-results/automatic-irregular-render-v1/manifest.json',
        'test-results/workflow-fragments-priority-parent-v1/preview.json',
        'images/ubunchu-ja-test/ubunchu01_04.png', 'test-results/render-parent-input-v1/fonts.json'))
    def test_saved_page_preserves_six_cleanups_and_improves_short_word_wrapping(self):
        root = Path(__file__).resolve().parents[2]
        baseline = root/'test-results/automatic-irregular-render-v1'
        before = n.document(n.read_explicit(baseline/'manifest.json'))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'render'
            after = a.run_automatic(preview=root/'test-results/workflow-fragments-priority-parent-v1/preview.json',
                image=root/'images/ubunchu-ja-test/ubunchu01_04.png',
                fonts=root/'test-results/render-parent-input-v1/fonts.json', output=output, render_policy='conservative')
            self.assertEqual(after['coverage']['rendered'], 6)
            self.assertEqual({key: after['coverage'][key] for key in before['coverage']}, before['coverage'])
            self.assertEqual(after['coverage']['strict_rendered'], 6)
            self.assertEqual(after['coverage']['coverage_rendered'], 0)
            for key in ('source', 'preview_sha256', 'fonts_sha256'):
                self.assertEqual(after[key], before[key])
            for name in ('source.png', 'clean.png', 'erase-mask.png'):
                self.assertEqual(n.digest(n.read_explicit(baseline/name)), before['artifacts'][name])
                with Image.open(baseline/name) as old, Image.open(output/name) as new:
                    self.assertTrue(np.array_equal(np.asarray(old), np.asarray(new)), name)
            for index, (old, new) in enumerate(zip(before['units'], after['units'])):
                self.assertEqual(old['status'], new['status'])
                self.assertEqual(old['reason_code'], new['reason_code'])
                if new['status'] != 'automatic_composed':
                    continue
                meta = new['layout']
                self.assertGreaterEqual(meta['size'], after['style']['min_size'])
                self.assertLessEqual(meta['size'], after['style']['max_size'])
                self.assertLessEqual(len(meta['typography']['probes']), t.MAX_PROBES)
                self.assertLessEqual(meta['rasterized_pixels'], 64_000_000)
                with Image.open(output/f'unit-{index:03d}-content-mask.png') as mask, \
                        Image.open(output/f'unit-{index:03d}-text.png') as text:
                    self.assertFalse(((np.asarray(text.getchannel('A')) != 0) & (np.asarray(mask) == 0)).any())
                if new['binding']['text'] == "I don't need it.":
                    self.assertEqual(len(old['layout']['draw_text'].split('\n')), 4)
                    self.assertEqual(len(meta['draw_text'].split('\n')), 2)
                    chosen = meta['typography']['probes'][meta['typography']['chosen_probe']]
                    self.assertEqual(chosen['line_quality']['short_word_orphans'], 0)
            with Image.open(output/'source.png') as source, Image.open(output/'clean.png') as clean, \
                    Image.open(output/'composite.png') as composite, Image.open(output/'erase-mask.png') as erase, \
                    Image.open(output/'alpha-mask.png') as alpha:
                self.assertTrue(np.array_equal(np.asarray(source)[np.asarray(erase) == 0], np.asarray(clean)[np.asarray(erase) == 0]))
                self.assertTrue(np.array_equal(np.asarray(clean)[np.asarray(alpha) == 0], np.asarray(composite)[np.asarray(alpha) == 0]))
            self.assertLessEqual(len(after['artifacts'])+1, 256)
            self.assertFalse(after['approved_export'])


if __name__ == '__main__':
    unittest.main()
