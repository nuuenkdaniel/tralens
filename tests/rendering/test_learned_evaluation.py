"""Offline experiment invariants; no model weights, runtime, or local pages needed."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools import freeze_learned_evaluation as f
from tools import learned_restoration_experiment as e
from tools import summarize_learned_evaluation as s
from .font_fixture import synthetic_font


class ExperimentInvariantTests(unittest.TestCase):
    def test_native_mapping_removes_right_bottom_padding_before_resize(self):
        values = np.ones((1024, 1024), np.float32)
        values[:800, :500] = .5
        mapped = e.native_raw(values, (100, 160), (500, 800))
        self.assertEqual(mapped.shape, (160, 100))
        self.assertEqual(mapped.dtype, np.uint8)
        self.assertTrue((mapped == 127).all())

    def test_invalid_mask_shape_probability_or_transform_rejects(self):
        for values, resized in ((np.zeros((10, 10)), (10, 10)),
                                (np.full((1024, 1024), np.nan), (512, 1024)),
                                (np.full((1024, 1024), 2), (512, 1024)),
                                (np.zeros((1024, 1024)), (0, 1024))):
            with self.assertRaises(AssertionError):
                e.native_raw(values, (100, 100), resized)

    def test_mask_metrics_do_not_hide_empty_predictions_or_false_positives(self):
        truth = np.array([[True, True], [False, False]])
        candidate = np.array([[True, False], [True, False]])
        self.assertEqual(e.binary_metrics(candidate, truth),
            dict(tp=1, fp=1, fn=1, precision=.5, recall=.5, iou=1/3))
        empty = e.binary_metrics(np.zeros_like(truth), truth)
        self.assertIsNone(empty['precision'])
        self.assertEqual(empty['recall'], 0)

    def test_model_cannot_authorize_its_own_difference_support(self):
        original = np.full((10, 12, 3), 125, np.uint8)
        proposed = np.zeros_like(original)
        allowed = np.zeros((10, 12), bool)
        allowed[3:5, 4:6] = True
        result = e.mask_copy(original, proposed, allowed)
        self.assertTrue((result[allowed] == 0).all())
        self.assertTrue(np.array_equal(result[~allowed], original[~allowed]))
        self.assertTrue((original == 125).all())
        with self.assertRaises(AssertionError):
            e.mask_copy(original, proposed, allowed.astype(np.uint8)*255)

    def test_reference_error_only_measures_the_fixed_mask(self):
        reference = np.full((3, 3, 3), 50, np.uint8)
        selected = np.zeros((3, 3), bool)
        selected[1, 1] = True
        candidate = reference.copy()
        candidate[~selected] = 0
        metrics = e.metrics_against_reference(candidate, reference, selected)
        self.assertTrue(metrics['exact_reconstruction'])
        self.assertIsNone(metrics['masked_psnr_db'])
        candidate[selected] = 60
        self.assertEqual(e.metrics_against_reference(candidate, reference, selected)['masked_mae'], 10)

    def test_padding_never_exposes_reflected_hidden_text_as_known_context(self):
        selected = np.zeros((40, 60), bool)
        selected[15:25, 20:40] = True
        first = np.full((40, 60, 3), 255, np.uint8)
        first[selected] = 0
        second = first.copy()
        second[selected] = 75
        image_a, mask_a = e.pad_lama_context(first, selected)
        image_b, mask_b = e.pad_lama_context(second, selected)
        self.assertEqual(image_a.shape, (512, 512, 3))
        self.assertTrue(np.array_equal(mask_a, mask_b))
        self.assertTrue(np.array_equal(mask_a[:40, :60], selected))
        self.assertGreater(int(mask_a.sum()), int(selected.sum()))
        # Source pixels hidden by the original mask cannot affect KNOWN input,
        # even when the model-space pad is many times larger than the crop.
        self.assertTrue(np.array_equal(image_a[~mask_a], image_b[~mask_b]))
        image_a[mask_a] = 0
        image_b[mask_b] = 0
        self.assertTrue(np.array_equal(image_a, image_b))

    def test_fixture_freeze_is_repeatable_and_checks_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'images/ubunchu-ja-test').mkdir(parents=True)
            path = root/'images/ubunchu-ja-test/ubunchu01_04.png'
            Image.new('RGB', (384, 384), 'white').save(path)
            image_sha = e.sha(path)
            ids = [36, 37, 38, 39, 40]
            detections = [dict(source_id=i, polygon=f.polygon((50+i, 40, 70+i, 80))) for i in ids]
            proposals = dict(pages=[dict(page=dict(image_path=str(path), image_sha256=image_sha),
                batch=dict(detections=detections, proposals=[dict(member_ids=[36]) for _ in range(6)]))])
            saved = dict(input=dict(sha256=image_sha), ocr=dict(raw_detections=[detections]))
            for name, value in (('automatic-unit-diagnosis-parent-v3/automatic-proposals.json', proposals),
                                ('workflow-fragments-priority-parent-v1/preview.json', saved)):
                destination = root/'test-results'/name
                destination.parent.mkdir(parents=True)
                f.save_json(destination, value)
            font = root/'font.ttf'
            font.write_bytes(synthetic_font())
            fonts = root/'fonts.json'
            f.save_json(fonts, dict(fonts=[dict(path=str(font), sha256=e.sha(font))]))
            first, first_hash = f.freeze(root=root, output=root/'first', fonts=fonts)
            second, second_hash = f.freeze(root=root, output=root/'second', fonts=fonts)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first, second)
            self.assertEqual(len(first['cases']), 24)
            self.assertEqual(sum(c['split'] == 'held_out' for c in first['cases']), 8)
            self.assertEqual(len(e.load_inputs(root, root/'first')['cases']), 24)
            with self.assertRaises(FileExistsError):
                f.freeze(root=root, output=root/'first', fonts=fonts)
            path.write_bytes(b'changed source')
            with self.assertRaisesRegex(AssertionError, 'source image drift'):
                e.load_inputs(root, root/'first')

    def test_summary_keeps_mask_counts_separate_from_restoration_and_flags_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks, restoration = root/'masks', root/'restoration'
            masks.mkdir()
            restoration.mkdir()
            candidate = dict(nonmember_polygon_pixels=0, ink_metrics=dict(tp=3, fp=1, fn=2))
            row = dict(id='sample', split='development', kind='synthetic_known_overlay',
                candidates=dict(ctd_raw=candidate, ctd_refined_source_bound=candidate))
            f.save_json(masks/'report.json', dict(frozen_manifest_sha256='same', cases=[row]))
            measurements = dict(telea=dict(status='completed', masked_mae=3, composited_outside_changed_pixels=0),
                navier_stokes=dict(status='completed', masked_mae=2, composited_outside_changed_pixels=0),
                lama=dict(status='completed', masked_mae=1, composited_outside_changed_pixels=0))
            fixed = dict(frozen_manifest_sha256='same', cases=[dict(id='sample', split='development',
                kind='synthetic_known_overlay', margin=16, category='dots', candidates=measurements)], skipped=[])
            f.save_json(restoration/'report.json', fixed)
            summary = s.summarize(masks, restoration, root/'review')
            self.assertEqual(summary['mask_splits']['development']['synthetic_cases'], 1)
            self.assertEqual(summary['mask_splits']['development']['current_synthetic_plans'], 0)
            self.assertEqual(summary['mask_splits']['development']['learned']['ctd_raw']['synthetic_micro_ink_precision'], .75)
            self.assertEqual(summary['restoration_splits']['development']['lama_lower_mae_than_best_classical'], 1)
            self.assertEqual(summary['outside_mask_changed_pixels'], 0)
            self.assertTrue(summary['no_production_adoption'])
            self.assertIn('not a fresh confirmatory', summary['corrected_rerun_caveat'])
            self.assertTrue((root/'review/index.html').is_file())
            fixed['frozen_manifest_sha256'] = 'different'
            f.save_json(restoration/'report.json', fixed)
            with self.assertRaises(AssertionError):
                s.summarize(masks, restoration, root/'bad-review')


if __name__ == '__main__':
    unittest.main()
