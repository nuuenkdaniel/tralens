"""Irregular flat whitespace: keep full source clearance, never trim ink or paint boxes."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tests.local_evidence import local_evidence
from .font_fixture import synthetic_font
from .test_natural import rect
from tralens.rendering import automatic as a, natural as n


def sloped_balloon(color=(255, 255, 255)):
    image = Image.new('RGB', (160, 150), color)
    draw = ImageDraw.Draw(image)
    draw.polygon([(10, 10), (140, 10), (140, 135), (95, 135), (10, 50)], outline='black', width=2)
    for x, y in ((50, 26), (80, 80)):
        draw.rectangle((x, y, x+9, y+9), fill='black')
    # Sloped source geometry covers both glyphs but not the diagonal balloon edge.
    detections = [{'source_id': 'main', 'polygon': [[40, 20], [100, 20], [100, 100], [80, 100]]}]
    return image, detections


class SourceLocalLayoutTests(unittest.TestCase):
    def test_rectangular_path_keeps_exact_previous_mask(self):
        safe = np.ones((100, 100), bool)
        seed = np.zeros_like(safe)
        seed[40:60, 40:60] = True
        window = [20, 20, 80, 80]
        old, old_bounds, old_steps = a.grow_rectangle(safe, seed, window)
        mask, bounds, steps, evidence = a.source_local_layout(safe, seed, window)
        self.assertTrue(np.array_equal(mask, old))
        self.assertEqual((bounds, steps), (old_bounds, old_steps))
        self.assertEqual(evidence['method'], 'source_local_rectangle')

    def test_obstructed_empty_corner_not_actual_source_seed_uses_safe_mask(self):
        safe = np.ones((120, 120), bool)
        seed = np.zeros_like(safe)
        seed[30:40, 30:40] = seed[75:85, 75:85] = True
        safe[75:95, 20:35] = False
        # An interior obstacle stays a hole, not a whitened/filled polygon.
        safe[50:55, 50:55] = False
        before = safe.copy(), seed.copy()
        with self.assertRaises(a.AutomaticSkip):
            a.grow_rectangle(safe, seed, [20, 20, 100, 100])
        mask, bounds, _, evidence = a.source_local_layout(safe, seed, [20, 20, 100, 100])
        self.assertEqual(evidence['method'], 'source_local_connected_safe_mask')
        self.assertEqual(evidence['connectivity'], 4)
        self.assertEqual(evidence['required_seed_pixels_discarded'], 0)
        self.assertTrue((mask[seed]).all())
        self.assertFalse((mask & ~safe).any())
        self.assertFalse(mask[50:55, 50:55].any())
        self.assertEqual(bounds, [20, 20, 100, 100])
        self.assertTrue(np.array_equal(before[0], safe))
        self.assertTrue(np.array_equal(before[1], seed))

    def test_actual_seed_obstruction_is_not_excised_or_retried(self):
        safe = np.ones((100, 100), bool)
        seed = np.zeros_like(safe)
        seed[30:60, 30:60] = True
        safe[35, 35] = False
        with patch.object(a.cv2, 'connectedComponentsWithStats', side_effect=AssertionError('no fallback for actual contact')):
            with self.assertRaises(a.AutomaticSkip) as skipped:
                a.source_local_layout(safe, seed, [10, 10, 90, 90])
        self.assertEqual(skipped.exception.code, 'layout_seed_obstructed')

    def test_full_source_seed_outside_window_is_not_silently_trimmed(self):
        safe = np.ones((100, 100), bool)
        seed = np.zeros_like(safe)
        seed[20:30, 20:30] = seed[70:80, 70:80] = True
        safe[70:80, 20:30] = False
        with self.assertRaises(a.AutomaticSkip) as skipped:
            a.source_local_layout(safe, seed, [25, 25, 90, 90])
        self.assertEqual(skipped.exception.code, 'layout_window')

    def test_disconnected_and_diagonally_touching_seed_pockets_still_reject(self):
        for diagonal in (False, True):
            safe = np.zeros((100, 100), bool)
            safe[10:40, 10:40] = True
            safe[40:70, 40:70] = True
            if not diagonal:
                safe[40:70, 40:70] = False
                safe[50:80, 50:80] = True
            seed = np.zeros_like(safe)
            seed[20:25, 20:25] = seed[55:60, 55:60] = True
            with self.subTest(diagonal=diagonal), self.assertRaises(a.AutomaticSkip) as skipped:
                a.source_local_layout(safe, seed, [0, 0, 100, 100])
            self.assertEqual(skipped.exception.code, 'layout_seed_disconnected')

    def test_unanchored_whitespace_pocket_does_not_become_layout_permission(self):
        safe = np.ones((100, 120), bool)
        seed = np.zeros_like(safe)
        seed[20:30, 20:30] = seed[60:70, 45:55] = True
        safe[60:80, 10:25] = False
        safe[:, 70:75] = False
        mask, _, _, evidence = a.source_local_layout(safe, seed, [10, 10, 110, 90])
        self.assertEqual(evidence['method'], 'source_local_connected_safe_mask')
        self.assertTrue(mask[seed].all())
        self.assertFalse(mask[:, 75:].any())

    def test_component_budget_is_not_bypassed_by_fallback(self):
        safe = np.ones((100, 100), bool)
        seed = np.zeros_like(safe)
        seed[20:30, 20:30] = seed[60:70, 60:70] = True
        safe[60:70, 20:30] = False
        real = cv2.connectedComponentsWithStats
        def excessive(*args, **kwargs):
            _, labels, stats, centers = real(*args, **kwargs)
            return 8193, labels, stats, centers
        with patch.object(a.cv2, 'connectedComponentsWithStats', side_effect=excessive):
            with self.assertRaises(n.ResourceLimit):
                a.source_local_layout(safe, seed, [10, 10, 90, 90])


class IrregularPlanTests(unittest.TestCase):
    def test_complete_plan_and_raqm_fit_preserve_source_border_and_art(self):
        for color in ((255, 255, 255), (241, 239, 233)):
            with self.subTest(color=color):
                image, detections = sloped_balloon(color)
                source = np.asarray(image).copy()
                plan = a.derive_plan(image, detections, ['main'])
                self.assertEqual(plan.background, list(color))
                self.assertEqual(plan.evidence['layout_derivation']['method'], 'source_local_connected_safe_mask')
                self.assertGreater(plan.evidence['layout_derivation']['rectangular_seed_rejection']['evidence']
                                   ['obstructed_seed']['offending_count'], 0)
                self.assertFalse((plan.layout & plan.obstacles).any())
                self.assertFalse((a.p.dilate(plan.foreground, 2) & ~plan.layout).any())
                data = synthetic_font()
                layer, provenance = n.layout('Hi!\nAV?', data, n.font_coverage(data),
                    Image.fromarray(plan.layout.astype(np.uint8)*255), {'min_size': 14, 'max_size': 24, 'outline': 0})
                self.assertGreaterEqual(provenance['size'], 14)
                alpha = np.asarray(layer.getchannel('A')) != 0
                self.assertFalse((alpha & ~plan.layout).any())
                clean = source.copy()
                clean[plan.erase] = plan.background
                plate = Image.fromarray(clean)
                composite = Image.alpha_composite(plate.convert('RGBA'), layer).convert('RGB')
                audit = n.pixel_audit(image, plate, composite, plan.erase, alpha, plan.obstacles)
                n.verify_pixel_audit(audit)
                self.assertEqual(audit['outside_cleanup_changed_pixels'], 0)
                self.assertEqual(audit['protected_composite_changed_pixels'], 0)
                self.assertTrue(np.array_equal(source, np.asarray(image)))

    def test_nonmember_geometry_in_empty_corner_stays_excluded(self):
        image, detections = sloped_balloon()
        # Unrelated geometry outside the member footprint remains an obstacle,
        # even when it contains no dark pixels and lies inside the local window.
        detections.append({'source_id': 'neighbor', 'polygon': rect(110, 30, 115, 70)})
        plan = a.derive_plan(image, detections, ['main'])
        forbidden = a.p.dilate(np.asarray(n.polygon_mask(detections[1]['polygon'], image.size)) != 0, 2)
        self.assertFalse((plan.layout & forbidden).any())
        self.assertFalse((plan.erase & forbidden).any())

    def test_irregular_layout_never_bypasses_flatness_ink_or_nonmember_checks(self):
        for kind in ('nonflat_collar', 'colored_ink', 'nonmember', 'border_contact'):
            image, detections = sloped_balloon()
            draw = ImageDraw.Draw(image)
            if kind == 'nonflat_collar':
                image.putpixel((47, 26), (220, 220, 220))
            elif kind == 'colored_ink':
                image.putpixel((51, 27), (100, 0, 0))
            elif kind == 'nonmember':
                detections.append({'source_id': 'annotation', 'polygon': rect(50, 26, 53, 29)})
            else:
                draw.line((89, 85, 140, 85), fill='black', width=2)
            with self.subTest(kind=kind), patch.object(a, 'source_local_layout',
                    side_effect=AssertionError('cleanup must reject before layout')):
                with self.assertRaises(a.AutomaticSkip):
                    a.derive_plan(image, detections, ['main'])


class SavedIrregularLayoutTests(unittest.TestCase):
    @local_evidence(files=('test-results/workflow-fragments-priority-parent-v1/preview.json',
        'images/ubunchu-ja-test/ubunchu01_04.png', 'test-results/render-parent-input-v1/fonts.json',
        'test-results/automatic-tight-render-parent-v1/manifest.json'))
    def test_saved_page_retains_six_candidates_without_changing_previous_erase_geometry(self):
        root = Path(__file__).resolve().parents[2]
        baseline = root/'test-results/automatic-tight-render-parent-v1'
        before = n.document(n.read_explicit(baseline/'manifest.json'))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'render'
            after = a.run_automatic(preview=root/'test-results/workflow-fragments-priority-parent-v1/preview.json',
                image=root/'images/ubunchu-ja-test/ubunchu01_04.png',
                fonts=root/'test-results/render-parent-input-v1/fonts.json', output=output, render_policy='conservative')
            self.assertEqual(before['coverage']['rendered'], 5)
            self.assertEqual(after['coverage']['rendered'], 6)
            for key in ('source', 'preview_sha256', 'fonts_sha256'):
                self.assertEqual(before[key], after[key])
            for index, (old, new) in enumerate(zip(before['units'], after['units'])):
                self.assertEqual(old['unit_id'], new['unit_id'])
                if old['status'] == 'automatic_composed':
                    self.assertEqual(new['status'], 'automatic_composed')
                    # Typography evolves independently of the v1/v2 erase and
                    # source-local layout permissions; do not freeze old text.
                    for name in ('applied-erase', 'foreground', 'layout-mask'):
                        artifact = f'unit-{index:03d}-{name}.png'
                        self.assertEqual(n.digest(n.read_explicit(baseline/artifact)), before['artifacts'][artifact])
                        with Image.open(baseline/f'unit-{index:03d}-{name}.png') as old_image, \
                                Image.open(output/f'unit-{index:03d}-{name}.png') as new_image:
                            self.assertTrue(np.array_equal(np.asarray(old_image), np.asarray(new_image)), (index, name))
                if old.get('reason_code') in ('component_spill', 'unassigned_contrast'):
                    self.assertEqual(new['reason_code'], old['reason_code'])
                    self.assertEqual(new['status'], 'skipped')
            gained = [row for old, row in zip(before['units'], after['units'])
                      if old['status'] != 'automatic_composed' and row['status'] == 'automatic_composed']
            self.assertEqual(len(gained), 1)
            self.assertEqual(gained[0]['mask_evidence']['layout_derivation']['method'], 'source_local_connected_safe_mask')
            self.assertEqual(gained[0]['mask_evidence']['layout_derivation']['required_seed_pixels_discarded'], 0)
            self.assertGreaterEqual(gained[0]['layout']['size'], after['style']['min_size'])
            n.verify_pixel_audit(after['pixel_audit'])
            for name, digest in after['artifacts'].items():
                self.assertEqual(n.digest(n.read_explicit(output/name)), digest)
            print('IRREGULAR_SAVED_COVERAGE='+json.dumps(after['coverage'], sort_keys=True))


if __name__ == '__main__':
    unittest.main()
