"""No model downloads/inference. Trials must preserve source/art evidence."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools import background_trials as b
from tralens.rendering import automatic as a


def periodic(size=(160, 180), px=7, py=11):
    yy, xx = np.mgrid[:size[0], :size[1]]
    pixels = np.full((*size, 3), 235, np.uint8)
    pixels[(((xx+3) % px-px//2)**2+((yy-4) % py-py//2)**2) <= 2] = [50, 65, 80]
    return pixels


def sample():
    image = Image.new('RGB', (220, 180), 'white')
    draw = ImageDraw.Draw(image)
    for y in (50, 80, 110):
        draw.rectangle((80, y, 90, y+9), fill='black')
    detections = [dict(source_id='main', polygon=[[75, 42], [96, 42], [96, 128], [75, 128]])]
    probability = np.zeros((180, 220), np.uint8)
    probability[45:125, 77:95] = 240
    return image, detections, probability


class BackgroundTrialTests(unittest.TestCase):
    def test_periodic_tile_is_native_phase_exact_and_ignores_target_pixels(self):
        clean = periodic()
        known = np.ones(clean.shape[:2], bool)
        known[50:110, 55:125] = False
        first = clean.copy()
        first[~known] = 0
        second = clean.copy()
        second[~known] = 130
        predicted, evidence = b.periodic_surface(first, known)
        other, _ = b.periodic_surface(second, known)
        self.assertTrue(np.array_equal(predicted, clean))
        self.assertTrue(np.array_equal(predicted, other))
        self.assertEqual(evidence['period_xy'], [7, 11])
        self.assertEqual(evidence['known_max_error'], 0)
        self.assertEqual(evidence['outliers_discarded'], 0)

    def test_periodic_model_rejects_known_crossing_line_and_phase_break(self):
        for kind in ('line', 'phase'):
            clean = periodic()
            known = np.ones(clean.shape[:2], bool)
            known[60:100, 60:120] = False
            if kind == 'line':
                clean[20:140, 30:33] = 0
            else:
                clean[:, 90:] = np.roll(clean[:, 90:], 2, axis=0)
            with self.subTest(kind=kind), self.assertRaises(a.AutomaticSkip):
                b.periodic_surface(clean, known)

    def test_flat_is_not_mislabeled_as_periodic(self):
        pixels = np.full((100, 120, 3), 230, np.uint8)
        with self.assertRaises(a.AutomaticSkip) as skipped:
            b.periodic_surface(pixels, np.ones((100, 120), bool))
        self.assertEqual(skipped.exception.code, 'periodic_flat')

    def test_flat_source_bound_native_plan_does_not_paint_its_rectangle(self):
        image, detections, probability = sample()
        original = np.array(image)
        plan = b.derive(image, detections, ['main'], probability)
        pixels = original.copy()
        a.apply_cleanup(pixels, plan)
        self.assertTrue((pixels == 255).all())
        self.assertTrue(np.array_equal(pixels[~plan.erase], original[~plan.erase]))
        self.assertLess(int(plan.erase.sum()), 22*87)
        self.assertTrue(np.array_equal(original, np.array(image)))

    def test_small_faint_native_antialias_is_included_whole(self):
        image, detections, probability = sample()
        image.putpixel((78, 52), (250, 250, 250))
        probability[52, 78] = 0
        plan = b.derive(image, detections, ['main'], probability)
        self.assertTrue(plan.foreground[52, 78])
        self.assertGreater(plan.evidence['faint_components'], 0)

    def test_nearby_nonmember_and_touching_art_are_not_clipped_to_get_a_mask(self):
        for kind in ('nonmember', 'crossing', 'spilling'):
            image, detections, probability = sample()
            if kind == 'nonmember':
                detections.append(dict(source_id='neighbor', polygon=detections[0]['polygon']))
            elif kind == 'crossing':
                ImageDraw.Draw(image).line((84, 10, 84, 165), fill='black', width=3)
            else:
                ImageDraw.Draw(image).line((84, 54, 155, 54), fill='black', width=3)
                probability[50:57, 80:158] = 240
            original = np.array(image)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                b.derive(image, detections, ['main'], probability)
            self.assertTrue(np.array_equal(original, np.array(image)))

    def test_high_model_probability_does_not_authorize_a_gray_art_interior(self):
        image, detections, probability = sample()
        ImageDraw.Draw(image).ellipse((83, 51, 94, 73), outline=(80, 80, 80), width=2)
        probability[np.all(np.array(image) == 80, axis=2)] = 254
        with self.assertRaises(a.AutomaticSkip) as skipped:
            b.derive(image, detections, ['main'], probability)
        self.assertEqual(skipped.exception.code, 'trial_multiple_ink_interiors')

    def test_periodic_low_noise_is_measured_not_discarded(self):
        pixels = periodic()
        noise = np.random.default_rng(41).integers(-1, 2, pixels.shape[:2])
        pixels = np.clip(pixels.astype(np.int16)+noise[..., None], 0, 255).astype(np.uint8)
        known = np.ones(pixels.shape[:2], bool)
        known[50:110, 55:125] = False
        predicted, evidence = b.periodic_surface(pixels, known)
        self.assertLessEqual(evidence['known_max_error'], 2)
        self.assertLessEqual(evidence['known_mae'], 1)
        self.assertEqual(evidence['outliers_discarded'], 0)
        self.assertLessEqual(int(np.abs(predicted.astype(int)-pixels.astype(int)).max()), 2)

    def test_type_shape_foreign_members_and_budgets_fail_before_cleanup(self):
        image, detections, probability = sample()
        for ids, raw in ((['foreign'], probability), (['main'], probability.astype(float)),
                         (['main'], probability[:-1]), ([True], probability)):
            with self.assertRaises(ValueError):
                b.derive(image, detections, ids, raw)
        with self.assertRaises(ValueError):
            b.periodic_surface(np.zeros((1024, 1024, 3), np.uint8), np.ones((1024, 1024), bool))


if __name__ == '__main__':
    unittest.main()
