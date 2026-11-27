"""Automatic-only coverage binding, strict priority and compact retained masks."""
import copy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests import test_pipeline as fixtures
from tralens.rendering import automatic as a, natural as n
from tralens.translation.contracts import automatic_units, canonical


class CoverageRenderingTests(unittest.TestCase):
    def setUp(self):
        self.p = fixtures.PipelineTests()
        self.p.setUp()
        self.addCleanup(self.p.doCleanups)
        self.f = self.p.fixture

    def make_coverage(self, mixed=False):
        batch = self.f.batch
        batch = (replace(batch, proposals=(replace(batch.proposals[0], crop_safe=False), batch.proposals[1]))
                 if mixed else replace(batch, regions=()))
        self.f.engine.automatic_unit_proposals = [batch]

    def render(self, name, **kwargs):
        return a.run_automatic(preview=self.p.root/'job/preview/preview.json', image=self.f.image,
            fonts=self.p.fonts, output=self.p.root/name, **kwargs)

    def test_pipeline_renders_coverage_only_inputs_with_separate_truthful_counts(self):
        self.make_coverage()
        before = self.f.image.read_bytes()
        report = self.p.run_pipeline(render_policy='best-effort')
        self.assertEqual(report['coverage']['strict_translations'], 0)
        self.assertEqual(report['coverage']['coverage_translations'], 2)
        self.assertEqual(report['coverage']['total_translations'], 2)
        self.assertEqual(report['coverage']['rendered'], 2, report)
        self.assertEqual(report['coverage']['coverage_rendered'], 2)
        rc = report['rendering']['coverage']
        self.assertIsNone(rc['strict_coverage_fraction'])
        self.assertEqual(rc['total_coverage_fraction'], 1)
        manifest = json.loads((self.p.root/'job/rendering/manifest.json').read_text())
        self.assertTrue(all(r['render_quality'] == 'best_effort' for r in manifest['units']))
        self.assertTrue(all(r['source_uncertainty']['blocked_reasons'] for r in manifest['units']))
        self.assertEqual(manifest['pixel_audit']['outside_alpha_changed_pixels'], 0)
        self.assertEqual(manifest['pixel_audit']['outside_cleanup_changed_pixels'], 0)
        self.assertEqual(self.f.image.read_bytes(), before)
        self.assertFalse(report['full_page_translation_verified'])

    def test_conservative_and_manual_bindings_never_promote_coverage(self):
        self.make_coverage()
        report = self.p.run_pipeline(render_policy='conservative')
        self.assertEqual(report['coverage']['coverage_translations'], 2)
        self.assertEqual(report['coverage']['rendered'], 0)
        result = self.render('conservative', render_policy='conservative')
        self.assertTrue(all(r['reason_code'] == 'coverage_requires_best_effort' for r in result['units']))
        preview = json.loads((self.p.root/'job/preview/preview.json').read_text())
        unit = automatic_units(preview['ocr']['batches'][0])[0]
        with self.assertRaises(ValueError):
            n.strict_result_binding(preview['units'][0], unit)
        from tralens.rendering import preparation
        with Image.open(self.f.image) as image, self.assertRaises(ValueError):
            preparation._binding(preview['units'][0], preview, image, preview['input'],
                [(self.p.root/'job/preview/preview.json').read_bytes(), self.f.image.read_bytes(), self.p.fonts.read_bytes()],
                'dialogue', {'min_size': 16, 'max_size': 28, 'outline': 0})

    def test_strict_budget_priority_and_strict_rasters_remain_exact(self):
        self.make_coverage(mixed=True)
        report = self.p.run_pipeline(render_policy='best-effort')
        self.assertEqual((report['coverage']['strict_translations'], report['coverage']['coverage_translations']), (1, 1))
        self.assertEqual(report['coverage']['rendered'], 2)
        preview_path = self.p.root/'job/preview/preview.json'
        full = json.loads(preview_path.read_text())
        base = copy.deepcopy(full)
        base['units'][0].update(status='budget_exhausted', result=None)
        preview_path.write_text(canonical(base))
        baseline = self.render('baseline')
        preview_path.write_text(canonical(full))
        strict = next(i for i, r in enumerate(full['units']) if r['input_kind'] == 'strict_utterance')
        for suffix in ('text', 'proposed-erase', 'applied-erase', 'layout-mask', 'content-mask'):
            name = f'unit-{strict:03d}-{suffix}.png'
            self.assertEqual((self.p.root/'job/rendering'/name).read_bytes(), (self.p.root/'baseline'/name).read_bytes())
        with patch.object(a, 'MAX_SLOTS', 1):
            limited = self.render('limited')
        self.assertEqual(limited['coverage']['strict_rendered'], 1)
        self.assertEqual(limited['coverage']['coverage_rendered'], 0)
        self.assertEqual(limited['units'][0]['reason_code'], 'resource_budget')

    def test_uncertain_region_cannot_displace_even_fallback_strict_placement(self):
        self.make_coverage(mixed=True)
        self.p.run_pipeline(render_policy='conservative')
        # Force the uncertain candidate to propose the STRICT unit's cleanup.
        with Image.open(self.f.image) as image:
            preview = json.loads((self.p.root/'job/preview/preview.json').read_text())
            strict_ids = preview['units'][1]['unit']['member_ids']
            plan = a.derive_plan(image, preview['ocr']['raw_detections'][0], strict_ids)
        with patch.object(a, 'derive_plan', return_value=plan):
            rendered = self.render('collision', min_size=144, max_size=144)
        self.assertEqual(rendered['coverage']['strict_rendered'], 1)
        self.assertEqual(rendered['units'][1]['render_quality'], 'best_effort')
        self.assertNotEqual(rendered['units'][1]['reason_code'], 'region_collision')
        # The coverage route may find its own bounded fallback, but cannot edit
        # any reserved strict layout/erase/alpha pixels.
        self.assertEqual(rendered['pixel_audit']['outside_alpha_changed_pixels'], 0)
        if rendered['units'][0]['status'] == 'automatic_composed':
            root = self.p.root/'collision'
            with Image.open(root/'unit-000-applied-erase.png') as erase, Image.open(root/'unit-000-text.png') as text, \
                    Image.open(root/'unit-001-layout-mask.png') as layout:
                self.assertFalse(((np.asarray(erase) != 0) & (np.asarray(layout) != 0)).any())
                self.assertFalse(((np.asarray(text.getchannel('A')) != 0) & (np.asarray(layout) != 0)).any())

    def test_tampered_coverage_binding_retains_original_and_does_not_attempt_cleanup(self):
        self.make_coverage()
        self.p.run_pipeline(render_policy='conservative')
        preview_path = self.p.root/'job/preview/preview.json'
        original = json.loads(preview_path.read_text())
        mutations = [('coverage', {}), ('translation_input', {}), ('request_revision', '0'*64)]
        for index, (key, value) in enumerate(mutations):
            doc = copy.deepcopy(original)
            for row in doc['units']:
                row[key] = value
            preview_path.write_text(canonical(doc))
            with patch.object(a, 'derive_plan', side_effect=AssertionError('do not plan changed evidence')):
                result = self.render('tampered-'+str(index))
            self.assertEqual(result['coverage']['attempted'], 0)
            self.assertEqual(result['coverage']['rendered'], 0)
            self.assertEqual(result['pixel_audit']['clean_changed_pixels'], 0)

    def test_strict_fallback_raster_budget_is_reserved_before_any_coverage_work(self):
        from tralens.rendering import best_effort
        self.make_coverage(mixed=True)
        self.p.run_pipeline(render_policy='conservative')
        with Image.open(self.f.image) as image:
            pixels = image.width*image.height
            preview = json.loads((self.p.root/'job/preview/preview.json').read_text())
            raw = preview['ocr']['raw_detections'][0]
            geometry_work = a.OriginalComponents(image).bind_geometry(raw)
            plan = a.derive_plan(image, raw, preview['units'][1]['unit']['member_ids'])
        plan.evidence.update(cleanup_method='best_effort_native_fill', warnings=['Synthetic native hint'])
        # Reserve the actual geometry strips and the two strict page rasters.
        with patch.object(a, 'MAX_WORK', geometry_work+2*pixels), \
                patch.object(a, 'derive_plan', side_effect=a.AutomaticSkip('component_spill', 'synthetic primary failure')), \
                patch.object(best_effort, 'native_plan', return_value=plan):
            result = self.render('raster-priority', background_method='flat')
        self.assertEqual(result['coverage']['strict_rendered'], 1, result['units'])
        self.assertEqual(result['coverage']['coverage_rendered'], 0)
        self.assertEqual(result['units'][0]['reason_code'], 'resource_budget')
        self.assertEqual(result['coverage']['attempted'], 1)

    def test_dense_page_can_attempt_thirty_two_units_without_old_slot_or_mask_limit(self):
        from PIL import ImageDraw
        from tralens.ocr.types import Text_Box
        from tralens.ocr.automatic_units import propose_page_from_boxes
        image = Image.new('RGB', (1800, 1800), 'white')
        draw = ImageDraw.Draw(image)
        boxes = []
        for row in range(4):
            for column in range(8):
                x, y = 10+column*220, 10+row*440
                draw.rectangle((x, y, x+190, y+350), outline='black', width=2)
                for dy in (60, 90, 120):
                    draw.rectangle((x+75, y+dy, x+84, y+dy+9), fill='black')
                boxes.append(Text_Box(len(boxes), .99, [[x+70, y+50], [x+95, y+50],
                    [x+95, y+150], [x+70, y+150]], 'こんにちは'))
        image.save(self.f.image)
        batch = propose_page_from_boxes(boxes, str(self.f.image), complementary_ownership=True, regroup_companions=True)
        self.f.engine.raw_detections = [boxes]
        self.f.engine.automatic_unit_proposals = [replace(batch, regions=())]
        report = self.p.run_pipeline(render_policy='best-effort', max_attempts=64)
        self.assertEqual(report['coverage']['coverage_translations'], 32)
        self.assertEqual(report['coverage']['rendered'], 32, report)
        manifest = json.loads((self.p.root/'job/rendering/manifest.json').read_text())
        self.assertEqual(manifest['coverage']['attempted'], 32)
        self.assertGreater(len(manifest['artifacts']), 256)
        masks = manifest['retained_masks']
        self.assertEqual(masks['boolean_mask_bytes']*8, masks['equivalent_unpacked_mask_bytes'])
        self.assertLess(masks['boolean_mask_bytes'], 64*1024*1024)
        self.assertEqual(manifest['pixel_audit']['outside_cleanup_changed_pixels'], 0)


class PackedMaskTests(unittest.TestCase):
    def test_non_byte_aligned_masks_roundtrip_exactly_at_one_bit_per_pixel(self):
        generator = np.random.default_rng(7)
        masks = [generator.integers(0, 2, (131, 179), dtype=np.uint8).astype(bool) for _ in range(4)]
        samples = generator.integers(0, 256, (int(masks[1].sum()), 3), dtype=np.uint8)
        plan = a.AutomaticPlan(*masks, [255, 255, 255], {'test': True}, samples)
        packed = a.PackedPlan(plan)
        raw_bytes = sum(mask.nbytes for mask in masks)
        packed_bytes = sum(mask.nbytes for mask in packed.masks.values())
        self.assertLessEqual(packed_bytes, (raw_bytes+31)//8)
        for name, expected in zip(('foreground', 'erase', 'layout', 'obstacles'), masks):
            self.assertEqual(getattr(packed, name).dtype, bool)
            self.assertTrue(np.array_equal(getattr(packed, name), expected))
        self.assertIs(packed.restored_pixels, samples)
        one, two = np.zeros((131, 179, 3), np.uint8), np.zeros((131, 179, 3), np.uint8)
        a.apply_cleanup(one, plan)
        a.apply_cleanup(two, packed)
        self.assertTrue(np.array_equal(one, two))
