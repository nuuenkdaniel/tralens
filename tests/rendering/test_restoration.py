"""Known-background restoration and automatic fallback; no models, downloads or OCR inference."""
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import automatic as a, natural as n, restoration as r
from tralens.translation.contracts import TranslationRequest, TranslationResult, UnitOutcome
from . import test_natural as fixtures


def sample(kind):
    yy, xx = np.mgrid[0:180, 0:280]
    base = np.array([40, 65, 90] if kind == 'dark' else [215, 195, 165], float)
    pixels = np.zeros((180, 280, 3))+base
    if kind == 'gradient':
        pixels += xx[:, :, None]*np.array([.08, .12, .15])+yy[:, :, None]*np.array([.04, .03, .02])
    if kind == 'texture':
        pixels += np.random.default_rng(2).integers(-3, 4, pixels.shape)
    if kind == 'dense':
        pixels += np.where((xx+yy) % 2, 20, -20)[:, :, None]
    clean = pixels.clip(0, 255).astype(np.uint8)
    image = Image.fromarray(clean)
    draw = ImageDraw.Draw(image)
    for y in (50, 80, 110):
        draw.rectangle((55, y, 64, y+8), fill='white' if kind == 'dark' else 'black')
    detections = [{'source_id': 'main', 'polygon': fixtures.rect(50, 40, 70, 135)}]
    return image, clean, detections


class RestorationTests(unittest.TestCase):
    def test_color_dark_gradient_and_mild_texture_are_restored_not_whitened(self):
        for kind in ('color', 'dark', 'gradient', 'texture'):
            with self.subTest(kind=kind):
                image, known_background, detections = sample(kind)
                before = np.asarray(image).copy()
                with self.assertRaises(a.AutomaticSkip):
                    a.derive_plan(image, detections, ['main'])
                plan = r.derive_plan(image, detections, ['main'])
                restored = before.copy()
                a.apply_cleanup(restored, plan)
                self.assertIsNone(plan.background)
                self.assertEqual(plan.evidence['method'], 'telea' if kind == 'texture' else 'navier_stokes')
                self.assertTrue(np.array_equal(restored[~plan.erase], before[~plan.erase]))
                self.assertTrue(np.array_equal(before, np.asarray(image)))
                error = np.abs(restored.astype(float)-known_background)[plan.erase]
                white_error = np.abs(255-known_background.astype(float))[plan.erase]
                self.assertLess(float(error.mean()), 4)
                self.assertLess(float(error.max()), 8)
                self.assertLess(float(error.mean()), float(white_error.mean())/5)
                self.assertFalse((plan.erase & plan.obstacles).any())
                self.assertIn('unverified', plan.evidence['restoration_quality'])

    def test_antialiased_foreground_is_connected_to_high_contrast_core(self):
        image, clean, detections = sample('gradient')
        for y in (50, 80, 110):
            image.putpixel((54, y), tuple(int(v*.65) for v in clean[y, 54]))
        plan = r.derive_plan(image, detections, ['main'])
        self.assertTrue(plan.foreground[50, 54])
        self.assertTrue(plan.erase[50, 54])

    def test_sparse_connected_glyph_uses_ink_occupancy_not_empty_bbox_area(self):
        _, clean, _ = sample('color')
        detections = [{'source_id': 'main', 'polygon': fixtures.rect(47, 47, 83, 78)}]
        image = Image.fromarray(clean)
        ImageDraw.Draw(image).rectangle((50, 50, 80, 75), outline='black', width=3)
        plan = r.derive_plan(image, detections, ['main'])
        self.assertGreater(31*26, 37*32//2)
        self.assertLess(int(plan.foreground.sum()), 37*32//2)
        dense = Image.fromarray(clean)
        ImageDraw.Draw(dense).rectangle((50, 50, 80, 75), fill='black')
        with self.assertRaises(r.RestorationSkip) as skipped:
            r.derive_plan(dense, detections, ['main'])
        self.assertEqual(skipped.exception.code, 'restoration_component_extent')

    def test_dense_texture_border_contacts_unknown_marks_and_nonmembers_reject(self):
        for kind in ('dense', 'border', 'unknown', 'nonmember'):
            image, _, detections = sample('dense' if kind == 'dense' else 'color')
            if kind == 'border':
                ImageDraw.Draw(image).line((64, 54, 110, 54), fill='black', width=2)
            elif kind == 'unknown':
                image.putpixel((68, 125), (195, 175, 145))
            elif kind == 'nonmember':
                detections.append({'source_id': 'annotation', 'polygon': detections[0]['polygon']})
            before = np.asarray(image).copy()
            with self.subTest(kind=kind), self.assertRaises(r.RestorationSkip):
                r.derive_plan(image, detections, ['main'])
            self.assertTrue(np.array_equal(before, np.asarray(image)))

    def test_geometry_and_pixel_drift_or_frame_clipping_cannot_restore(self):
        image, _, detections = sample('color')
        context = r.OriginalGeometry(image, detections)
        image.putpixel((0, 0), (0, 0, 0))
        with self.assertRaisesRegex(ValueError, 'pixels changed'):
            r.derive_plan(image, detections, ['main'], context=context)
        image, _, detections = sample('color')
        context = r.OriginalGeometry(image, detections)
        detections[0]['polygon'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'geometry changed'):
            r.derive_plan(image, detections, ['main'], context=context)
        image, _, detections = sample('color')
        detections[0]['polygon'] = fixtures.rect(0, 0, 70, 135)
        with self.assertRaises(r.RestorationSkip) as skipped:
            r.derive_plan(image, detections, ['main'])
        self.assertEqual(skipped.exception.code, 'restoration_frame')

    def test_backend_mask_types_bounds_and_locality_are_hard_requirements(self):
        image, _, _ = sample('color')
        mask = Image.new('L', image.size)
        ImageDraw.Draw(mask).rectangle((54, 49, 65, 59), fill=255)
        for method in ('telea', 'navier_stokes'):
            restored = r.inpaint(image, mask, method)
            self.assertTrue(np.array_equal(np.asarray(restored)[np.asarray(mask) == 0],
                                           np.asarray(image)[np.asarray(mask) == 0]))
        mask.putpixel((55, 50), 128)
        with self.assertRaisesRegex(ValueError, 'binary'):
            r.inpaint(image, mask, 'telea')
        mask.putpixel((55, 50), 255)
        mask.putpixel((0, 0), 255)
        with self.assertRaisesRegex(ValueError, 'frame'):
            r.inpaint(image, mask, 'telea')
        mask.putpixel((0, 0), 0)
        original = cv2.inpaint
        def corrupt(*args):
            result = original(*args)
            result[0, 0] = 0
            return result
        with patch.object(r.cv2, 'inpaint', side_effect=corrupt):
            with self.assertRaises(r.RestorationSkip) as skipped:
                r.inpaint(image, mask, 'telea')
        self.assertEqual(skipped.exception.code, 'inpaint_locality')

    def test_local_budget_precedes_inpainting(self):
        image = Image.new('RGB', (1025, 8), 'red')
        mask = Image.new('L', image.size)
        mask.putpixel((5, 4), 255)
        with patch.object(r.cv2, 'inpaint', side_effect=AssertionError('must not allocate backend work')):
            with self.assertRaises(n.ResourceLimit):
                r.inpaint(image, mask, 'telea')

    def test_second_classical_method_is_tried_after_model_mismatch(self):
        image, _, detections = sample('gradient')
        original = r.inpaint
        methods = []
        def first_bad(rgb, mask, method):
            methods.append(method)
            if len(methods) == 1:
                return rgb.copy()  # Original lettering fails the background model check.
            return original(rgb, mask, method)
        with patch.object(r, 'inpaint', side_effect=first_bad):
            plan = r.derive_plan(image, detections, ['main'])
        self.assertEqual(methods, ['navier_stokes', 'telea'])
        self.assertEqual(plan.evidence['method'], 'telea')
        self.assertFalse(plan.evidence['attempts'][0]['model_consistent'])


class AutomaticRestorationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CommandTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def recolor(self, gradient=False, dark=False):
        """Preserve synthetic source shapes and regenerate genuine bound contracts."""
        f = self.fixture
        from tralens.ocr.automatic_units import propose_page_from_boxes
        from tralens.ocr.types import Text_Box
        with Image.open(f.image) as source:
            original = np.asarray(source).copy()
        yy, xx = np.mgrid[0:original.shape[0], 0:original.shape[1]]
        background = np.zeros_like(original, dtype=float)+([40, 65, 90] if dark else [215, 195, 165])
        if gradient:
            background += xx[:, :, None]*[.08, .12, .15]+yy[:, :, None]*[.04, .03, .02]
        pixels = background.clip(0, 255).astype(np.uint8)
        pixels[np.any(original != 255, axis=2)] = 255-original[np.any(original != 255, axis=2)] if dark else original[np.any(original != 255, axis=2)]
        image = Image.fromarray(pixels)
        image.save(f.image)
        raw = f.preview['ocr']['raw_detections'][0]
        batch = asdict(propose_page_from_boxes([Text_Box(d['source_id'], d['confidence'], d['polygon'], d['text']) for d in raw], str(f.image)))
        if dark:
            # Test-only structural ground truth: these two original synthetic
            # enclosures and memberships did not change when their colors did.
            # This does not teach production OCR to infer dark balloon ownership.
            batch['regions'] = f.preview['ocr']['batches'][0]['regions']
            for proposal in batch['proposals']:
                proposal['structural_reasons'] = [reason for reason in proposal['structural_reasons']
                    if reason not in ('unknown_region_ownership', 'mixed_region_ownership')]
        units = n.automatic_units(batch)
        records = []
        for unit in units:
            self.assertEqual(unit.disposition, 'preview', unit.blocking_reasons)
            request = TranslationRequest((unit,), (unit.unit_id,))
            outcome = UnitOutcome(unit.unit_id, 'preview', 'Hi!', (), 'unverified', unit.selected_revision)
            result = TranslationResult(request.revision, 'preview', ((unit.unit_id, 'Hi!'),), outcomes=(outcome,))
            records.append({'unit': asdict(unit), 'unit_revision': unit.revision, 'request': asdict(request),
                'request_revision': request.revision, 'status': 'preview', 'binding_current': True,
                'input_kind': 'strict_utterance', 'result': asdict(result)})
        f.preview = {'schema_version': 1, 'preview_only': True,
            'input': {'path': str(f.image), 'sha256': n.digest(f.image.read_bytes()), 'size': list(image.size),
                'pixels_sha256': n.digest(f'RGB:{image.width}:{image.height}:'.encode()+image.tobytes())},
            'ocr': {'batches': [batch], 'raw_detections': [[{**{k: v for k, v in d.items() if k not in ('page', 'metadata_json')},
                'ocr_metadata': {}} for d in batch['detections']]]}, 'units': records}
        (f.root/'preview.json').write_text(n.canonical(f.preview))

    def run_render(self, name='auto', **options):
        f = self.fixture
        options.setdefault('render_policy', 'conservative')
        return a.run_automatic(preview=f.root/'preview.json', image=f.image, fonts=f.root/'fonts.json',
            output=f.root/name, min_size=14, max_size=28, **options)

    def test_flat_first_restoration_fallback_and_flat_only_switch(self):
        with patch.object(r, 'derive_plan', side_effect=AssertionError('white cleanup must not invoke restoration')):
            white = self.run_render('white')
        self.assertEqual(white['background_summary']['restored_rendered'], 0)
        self.recolor(gradient=True)
        original = self.fixture.image.read_bytes()
        flat = self.run_render('flat', background_method='flat')
        self.assertEqual(flat['coverage']['rendered'], 0)
        restored = self.run_render()
        self.assertEqual(restored['coverage']['rendered'], 2)
        self.assertEqual(restored['background_summary']['restored_rendered'], 2)
        self.assertTrue(all(row['background_attempts'][0]['status'] == 'skipped' for row in restored['units']))
        self.assertTrue(all(row['cleanup_method'].startswith('inpaint_') for row in restored['units']))
        self.assertFalse(restored['approved_export'])
        self.assertEqual(self.fixture.image.read_bytes(), original)
        n.verify_pixel_audit(restored['pixel_audit'])
        for name, digest in restored['artifacts'].items():
            self.assertEqual(n.digest(n.read_explicit(self.fixture.root/'auto'/name)), digest)

    def test_no_translation_or_fragment_is_promoted_by_restoration(self):
        self.recolor()
        for row in self.fixture.preview['units']:
            row['input_kind'] = 'unverified_text_fragment'
        (self.fixture.root/'preview.json').write_text(n.canonical(self.fixture.preview))
        with patch.object(r, 'derive_plan', side_effect=AssertionError('strict binding required first')):
            report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)

    def test_bound_dark_background_uses_white_text_and_a_dark_outline(self):
        self.recolor(dark=True)
        report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], 2)
        self.assertEqual(report['background_summary']['restored_rendered'], 2)
        for row in report['units']:
            self.assertEqual(row['text_contrast']['fill'], [255, 255, 255])
            self.assertEqual(row['layout']['resolved_style']['fill'], [255, 255, 255])
            self.assertEqual(row['layout']['resolved_style']['outline_fill'], [0, 0, 0])

    def test_typography_failure_does_not_apply_successful_inpainting(self):
        self.recolor()
        with patch.object(a.typography, 'layout_dialogue', side_effect=ValueError('synthetic overflow')):
            report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)

    def test_backend_locality_failure_retains_original_page_and_attempt_reason(self):
        self.recolor()
        original = cv2.inpaint
        def corrupt(*args):
            result = original(*args)
            result[0, 0] = 0
            return result
        with patch.object(r.cv2, 'inpaint', side_effect=corrupt):
            report = self.run_render()
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)
        self.assertEqual(report['background_summary']['restoration_attempted'], 2)
        for row in report['units']:
            self.assertEqual(row['background_attempts'][-1]['reason_code'], 'inpaint_locality')

    def test_cli_background_method_is_explicitly_disableable(self):
        self.recolor()
        f = self.fixture
        tool = Path(__file__).resolve().parents[2]/'tools/render_preview.py'
        for method, count in (('flat', 0), ('auto', 2)):
            output = f.root/method
            process = subprocess.run([sys.executable, str(tool), 'auto-compose', '--preview', str(f.root/'preview.json'),
                '--image', str(f.image), '--fonts', str(f.root/'fonts.json'), '--output', str(output),
                '--background-method', method, '--min-size', '14', '--render-policy', 'conservative'], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((output/'manifest.json').read_text())
            self.assertEqual(report['coverage']['rendered'], count)


if __name__ == '__main__':
    unittest.main()
