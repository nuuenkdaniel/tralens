"""Coverage-first fallbacks; originals/locality/bindings remain hard requirements."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import automatic as a, best_effort as b, natural as n, restoration as r, segmentation as s
from . import test_restoration as fixtures


class BestEffortPrimitiveTests(unittest.TestCase):
    def test_uncertain_footprint_is_explicit_and_neighbor_pixels_stay_exact(self):
        yy, xx = np.mgrid[:180, :240]
        pixels = np.repeat(np.where((xx+yy) % 6 < 2, 110, 225)[..., None], 3, axis=2).astype(np.uint8)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.rectangle((65, 55, 80, 100), fill='black')
        draw.line((30, 80, 130, 80), fill='black', width=3)
        raw = [dict(source_id='main', polygon=[[60, 50], [90, 50], [90, 110], [60, 110]]),
               dict(source_id='neighbor', polygon=[[87, 60], [100, 60], [100, 105], [87, 105]])]
        original = np.array(image)
        plan = b.derive_plan(image, raw, ['main'])
        result = original.copy()
        a.apply_cleanup(result, plan)
        self.assertFalse(plan.evidence['foreground_is_text_certified'])
        self.assertEqual(plan.evidence['artwork_preservation_inside_mask'], 'not_guaranteed')
        self.assertGreater(plan.evidence['protected_source_pixels'], 0)
        self.assertTrue(np.array_equal(result[~plan.erase], original[~plan.erase]))
        self.assertTrue(np.array_equal(result[60:106, 87:101], original[60:106, 87:101]))
        self.assertTrue(np.array_equal(original, np.array(image)))
        self.assertFalse((plan.erase & plan.obstacles).any())

    def test_coincident_neighbor_cannot_be_erased_as_a_best_effort_shortcut(self):
        image = Image.new('RGB', (150, 150), 'white')
        points = [[40, 40], [70, 40], [70, 100], [40, 100]]
        raw = [dict(source_id='main', polygon=points), dict(source_id='other', polygon=points)]
        # One-pixel expansion cannot manufacture independent target pixels.
        with self.assertRaises(a.AutomaticSkip) as skipped:
            b.derive_plan(image, raw, ['main'])
        self.assertEqual(skipped.exception.code, 'best_effort_neighbor_conflict')

    def test_large_context_and_foreign_or_stale_geometry_are_not_waived(self):
        image = Image.new('RGB', (1200, 1200), 'white')
        raw = [dict(source_id=1, polygon=[[10, 10], [1100, 10], [1100, 1100], [10, 1100]])]
        with self.assertRaises(n.ResourceLimit):
            b.derive_plan(image, raw, [1])
        with self.assertRaisesRegex(ValueError, 'foreign'):
            b.derive_plan(image, raw, [2])
        with self.assertRaisesRegex(ValueError, 'typed'):
            b.derive_plan(image, raw, [True])
        context = r.OriginalGeometry(image, raw)
        image.putpixel((0, 0), (0, 0, 0))
        with self.assertRaisesRegex(ValueError, 'pixels changed'):
            b.derive_plan(image, raw, [1], context=context)


class BestEffortWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixtures.AutomaticRestorationTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.f = self.helper.fixture

    def run_render(self, name='best', **options):
        f = self.f
        return a.run_automatic(preview=f.root/'preview.json', image=f.image, fonts=f.root/'fonts.json',
            output=f.root/name, **options)

    def force_failures(self):
        return patch.object(a, 'derive_plan', side_effect=a.AutomaticSkip('uncertain_mask', 'forced uncertain mask'))

    def test_default_replaces_instead_of_skipping_after_all_cheap_methods_fail(self):
        self.helper.recolor()
        original = self.f.image.read_bytes()
        with self.force_failures(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('complex', 'forced complex background')):
            report = self.run_render()
        self.assertEqual(report['render_policy'], 'best-effort')
        self.assertEqual(report['coverage']['rendered'], 2)
        self.assertEqual(report['quality_summary']['best_effort_rendered'], 2)
        self.assertTrue(report['quality_summary']['has_degraded_output'])
        self.assertEqual(report['background_summary']['best_effort_cleanup_rendered'], 2)
        self.assertEqual(self.f.image.read_bytes(), original)
        for row in report['units']:
            self.assertEqual(row['layout']['original_text'], row['binding']['text'])
            self.assertEqual(row['render_quality'], 'best_effort')
            self.assertFalse(row['best_effort']['reused_cleanup'])
            self.assertFalse(row['mask_evidence']['foreground_is_text_certified'])
            self.assertTrue(row['best_effort']['warnings'])
        self.assertIn('BEST-EFFORT OUTPUT', (self.f.root/'best/review.html').read_text())
        n.verify_pixel_audit(report['pixel_audit'])
        for name, digest in report['artifacts'].items():
            self.assertEqual(n.digest((self.f.root/'best'/name).read_bytes()), digest)

    def test_conservative_switch_retains_the_previous_no_erasure_behavior(self):
        self.helper.recolor()
        with self.force_failures(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('complex', 'forced')):
            report = self.run_render(render_policy='conservative')
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['quality_summary']['best_effort_rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)

    def test_fully_opaque_alpha_sources_keep_original_bytes_and_canonical_rgb_binding(self):
        for mode in ('LA', 'RGBA'):
            self.helper.recolor(source_mode=mode)
            original = self.f.image.read_bytes()
            preflight = a.preflight_automatic(image=self.f.image, fonts=self.f.root/'fonts.json')
            report = self.run_render(mode)
            self.assertEqual(report['coverage']['rendered'], 2)
            self.assertEqual(report['source'], preflight['source'])
            self.assertEqual((self.f.root/mode/'original-file.bin').read_bytes(), original)
            self.assertEqual(self.f.image.read_bytes(), original)
            n.verify_pixel_audit(report['pixel_audit'])
            # Return to RGB before the fixture's next source recoloring.
            Image.open(self.f.image).convert('RGB').save(self.f.image)

    def test_one_nonopaque_pixel_or_rgb_transparency_key_is_not_silently_discarded(self):
        for kind in ('alpha', 'transparent_color'):
            image = Image.new('RGBA' if kind == 'alpha' else 'RGB', (180, 130), 'white')
            options = {}
            if kind == 'alpha':
                image.putpixel((179, 129), (255, 255, 255, 254))
            else:
                options['transparency'] = (255, 255, 255)
            image.save(self.f.image, **options)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'transparency'):
                a.preflight_automatic(image=self.f.image, fonts=self.f.root/'fonts.json')

    def test_typesetting_failure_reuses_cleanup_and_reports_smaller_type(self):
        self.helper.recolor()
        report = self.run_render(min_size=144, max_size=144)
        self.assertEqual(report['coverage']['rendered'], 2)
        for row in report['units']:
            self.assertTrue(row['best_effort']['reused_cleanup'])
            self.assertTrue(row['layout']['typography']['below_requested_minimum'])
            self.assertGreaterEqual(row['layout']['size'], 8)
            self.assertEqual(row['layout']['original_text'], 'Hi!')

    def test_emergency_breaks_preserve_every_character_and_original_newline(self):
        fonts = n.load_fonts(self.f.fonts)
        data, coverage = fonts['dialogue']
        text = 'Supercalifragilisticexpialidocious'*2+'!\nNext  line?'
        layer, meta, content = b.fit(text, data, coverage, Image.new('L', (90, 180), 255),
            dict(min_size=16, max_size=28, outline=1))
        self.assertEqual(meta['original_text'], text)
        self.assertEqual(b._insert(text, meta['inserted_break_offsets']), meta['draw_text'])
        self.assertEqual(meta['typography']['probes'][-1]['method'], 'emergency_character_wrap')
        self.assertEqual(layer.size, content.size)

    def test_bounded_measurement_cache_preserves_pixels_offsets_and_logical_budgets(self):
        data, coverage = n.load_fonts(self.f.fonts)['dialogue']
        text = 'Repeat this exact sentence. '*5
        mask = Image.new('L', (320, 180), 255)
        style = dict(min_size=16, max_size=28, outline=1)
        original = ImageDraw.ImageDraw.textlength
        with patch.object(ImageDraw.ImageDraw, 'textlength', autospec=True, side_effect=original) as cached:
            layer, meta = n.layout(text, data, coverage, mask, style)
        with patch.object(n, 'lru_cache', side_effect=lambda **kwargs: lambda function: function), \
                patch.object(ImageDraw.ImageDraw, 'textlength', autospec=True, side_effect=original) as uncached:
            reference, expected = n.layout(text, data, coverage, mask, style)
        self.assertEqual(layer.tobytes(), reference.tobytes())
        self.assertEqual(meta, expected)
        self.assertLess(cached.call_count, .75*uncached.call_count)

    def test_combining_marks_are_not_split_from_their_base_during_emergency_wrap(self):
        data, _ = n.load_fonts(self.f.fonts)['dialogue']
        text = 'a\u0301b\u0301c\u0301d\u0301'*8
        shaped, offsets = b.emergency_wrap(text, data, 40)
        self.assertEqual(b._insert(text, offsets), shaped)
        self.assertTrue(all(text[offset] != '\u0301' for offset in offsets))

    def test_backend_failure_still_places_text_using_a_labelled_color_backing(self):
        self.helper.recolor()
        with self.force_failures(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('complex', 'forced')), \
                patch.object(b, 'native_plan', side_effect=ValueError('native hint unavailable')), \
                patch.object(b.cv2, 'inpaint', side_effect=cv2.error('backend unavailable')):
            report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], 2)
        self.assertTrue(all(row['cleanup_method'] == 'best_effort_color_backing' for row in report['units']))
        n.verify_pixel_audit(report['pixel_audit'])

    def test_existing_good_units_remain_pixel_identical_and_are_not_downgraded(self):
        self.helper.recolor()
        before = self.run_render('conservative', render_policy='conservative')
        after = self.run_render('best')
        self.assertEqual(before['coverage'], after['coverage'])
        self.assertEqual(after['quality_summary']['best_effort_rendered'], 0)
        for name in before['artifacts']:
            if name.endswith('.png'):
                self.assertEqual((self.f.root/'conservative'/name).read_bytes(), (self.f.root/'best'/name).read_bytes())

    def test_dense_page_publishes_maximum_slots_and_explicitly_reports_remaining_units(self):
        from tralens.ocr.automatic_units import propose_page_from_boxes
        from tralens.ocr.types import Text_Box
        from tralens.translation.contracts import TranslationRequest, TranslationResult, UnitOutcome
        with Image.open(self.f.image) as opened:
            tile = opened.convert('RGB')
        image = Image.new('RGB', (tile.width*6, tile.height*4), 'white')
        boxes = []
        for row in range(4):
            for col in range(6):
                x, y = col*tile.width, row*tile.height
                image.paste(tile, (x, y))
                for detection in self.f.preview['ocr']['raw_detections'][0]:
                    boxes.append(Text_Box(len(boxes), .99,
                        [[px+x, py+y] for px, py in detection['polygon']], detection['text']))
        image.save(self.f.image)
        batch = asdict(propose_page_from_boxes(boxes, str(self.f.image)))
        units = n.automatic_units(batch)
        self.assertEqual(len(units), 48)
        records = []
        for unit in units:
            self.assertEqual(unit.disposition, 'preview', unit.blocking_reasons)
            request = TranslationRequest((unit,), (unit.unit_id,))
            outcome = UnitOutcome(unit.unit_id, 'preview', 'Hi!', (), 'unverified', unit.selected_revision)
            result = TranslationResult(request.revision, 'preview', ((unit.unit_id, 'Hi!'),), outcomes=(outcome,))
            records.append(dict(unit=asdict(unit), unit_revision=unit.revision, request=asdict(request),
                request_revision=request.revision, status='preview', binding_current=True,
                input_kind='strict_utterance', result=asdict(result)))
        raw = [{**{k: v for k, v in d.items() if k not in ('page', 'metadata_json')}, 'ocr_metadata': {}}
               for d in batch['detections']]
        preview = dict(schema_version=1, preview_only=True, units=records,
            input=dict(path=str(self.f.image), sha256=n.digest(self.f.image.read_bytes()), size=list(image.size),
                pixels_sha256=n.digest(f'RGB:{image.width}:{image.height}:'.encode()+image.tobytes())),
            ocr=dict(batches=[batch], raw_detections=[raw]))
        (self.f.root/'preview.json').write_text(n.canonical(preview))
        report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], a.MAX_SLOTS)
        self.assertEqual(report['coverage']['strict_eligible'], 48)
        self.assertEqual(sum(row['reason_code'] == 'resource_budget' for row in report['units']), 16)
        self.assertTrue(report['coverage']['partial_output'])
        self.assertLessEqual(len(report['artifacts'])+1, report['limits']['artifact_count'])
        n.verify_pixel_audit(report['pixel_audit'])

    def test_missing_translation_and_resource_caps_are_not_bypassed(self):
        self.helper.recolor()
        with patch.object(a, 'MAX_SLOTS', 0), patch.object(b, 'derive_plan', side_effect=AssertionError('no budget')):
            report = self.run_render('capped')
        self.assertEqual(report['coverage']['rendered'], 0)
        for row in self.f.preview['units']:
            row['status'] = 'blocked'
        (self.f.root/'preview.json').write_text(n.canonical(self.f.preview))
        with patch.object(b, 'derive_plan', side_effect=AssertionError('no strict translation')):
            report = self.run_render('untranslated')
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)

    def test_untranslated_page_does_not_allocate_component_geometry(self):
        self.helper.recolor()
        for row in self.f.preview['units']:
            row['status'] = 'blocked'
        (self.f.root/'preview.json').write_text(n.canonical(self.f.preview))
        with patch.object(a, 'OriginalComponents', side_effect=AssertionError('no renderable target')):
            report = self.run_render()
        self.assertEqual(report['coverage']['attempted'], 0)

    def test_missing_glyphs_fail_before_cleanup_or_component_allocation(self):
        self.helper.recolor()
        for row in self.f.preview['units']:
            uid = row['unit']['unit_id']
            row['result']['translations'] = ((uid, 'caf\u00e9'),)
            row['result']['outcomes'][0]['translation'] = 'caf\u00e9'
        (self.f.root/'preview.json').write_text(n.canonical(self.f.preview))
        with patch.object(a, 'OriginalComponents', side_effect=AssertionError('missing target glyph')):
            report = self.run_render()
        self.assertEqual(report['coverage']['strict_eligible'], 2)
        self.assertEqual(report['coverage']['attempted'], 0)
        self.assertTrue(all('missing glyph' in row['reason'] for row in report['units']))

    def test_typography_resource_failure_is_not_mislabeled_as_input_drift(self):
        self.helper.recolor()
        with patch.object(a.typography, 'layout_dialogue', side_effect=n.ResourceLimit('test raster budget')):
            report = self.run_render(render_policy='conservative')
        self.assertTrue(all(row['reason_code'] == 'resource_budget' for row in report['units']))

    def test_first_candidate_locality_failure_cannot_be_retried_as_artistic_uncertainty(self):
        self.helper.recolor()
        original = Image.alpha_composite
        calls = 0
        def corrupt_first(*args):
            nonlocal calls
            result = original(*args)
            calls += 1
            if calls == 1:
                result.putpixel((0, 0), (255, 0, 0, 255))
            return result
        with patch.object(Image, 'alpha_composite', side_effect=corrupt_first):
            report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], 1)
        self.assertEqual(report['units'][0]['reason_code'], 'runtime_locality')
        self.assertNotIn('best_effort_attempts', report['units'][0])
        n.verify_pixel_audit(report['pixel_audit'])

    def test_fallback_locality_failure_stops_remaining_fallback_routes(self):
        self.helper.recolor()
        original = Image.alpha_composite
        def corrupt(*args):
            result = original(*args)
            result.putpixel((0, 0), (255, 0, 0, 255))
            return result
        with self.force_failures(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('complex', 'forced')), \
                patch.object(Image, 'alpha_composite', side_effect=corrupt), \
                patch.object(b, 'derive_plan', side_effect=AssertionError('integrity failures cannot reach footprint')):
            report = self.run_render()
        self.assertTrue(all(row['reason_code'] == 'runtime_locality' for row in report['units']))

    def test_best_effort_does_not_publish_a_compositor_locality_failure(self):
        self.helper.recolor()
        original = Image.alpha_composite
        def corrupt(*args):
            image = original(*args)
            image.putpixel((0, 0), (255, 0, 0, 255))
            return image
        with self.force_failures(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('complex', 'forced')), \
                patch.object(Image, 'alpha_composite', side_effect=corrupt):
            report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)

    def test_replaced_semantic_plan_is_not_misreported_as_a_learned_mask(self):
        self.helper.recolor()
        image = Image.open(self.f.image).convert('RGB')
        raw = np.where(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY) < 180, 245, 0).astype(np.uint8)
        original = s.derive

        def tiny_layout(rgb, detections, members, probability, *, surface):
            plan = original(rgb, detections, members, probability, surface='flat')
            plan.layout[:] = False
            plan.layout[30, 30] = True
            return plan

        with self.force_failures(), patch.object(r, 'derive_plan', side_effect=r.RestorationSkip('complex', 'forced')), \
                patch.object(s, 'model_spec', return_value=dict(path='test-only', sha256='test-only')), \
                patch.object(s, 'TextMask', return_value=SimpleNamespace(raw=raw, evidence=dict(forwards=1))), \
                patch.object(s, 'derive', side_effect=tiny_layout):
            report = self.run_render(text_mask_model=Path('/test-only'))
        self.assertEqual(report['coverage']['rendered'], 2)
        self.assertEqual(report['background_summary']['semantic_mask_rendered'], 0)
        self.assertTrue(all('text_mask_method' not in row for row in report['units']))

    def test_cli_defaults_to_best_effort_and_exposes_the_conservative_switch(self):
        self.helper.recolor()
        tool = Path(__file__).resolve().parents[2]/'tools/render_preview.py'
        for mode, count in (('default', 2), ('conservative', 0)):
            command = [sys.executable, str(tool), 'auto-compose', '--preview', str(self.f.root/'preview.json'),
                '--image', str(self.f.image), '--fonts', str(self.f.root/'fonts.json'), '--background-method', 'flat',
                '--output', str(self.f.root/mode)]
            if mode == 'conservative':
                command += ['--render-policy', mode]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((self.f.root/mode/'manifest.json').read_text())
            self.assertEqual(report['coverage']['rendered'], count)


if __name__ == '__main__':
    unittest.main()
