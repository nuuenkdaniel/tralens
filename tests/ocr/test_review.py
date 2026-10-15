"""Deterministic callbacks only: these tests do not run live Manga OCR."""
import socket
import unittest
from unittest.mock import Mock, patch

import numpy as np

from .test_grouping import box, ocr_module as m
from tools.diagnostics import debug_color


class ReviewTests(unittest.TestCase):
    def setUp(self):
        for target, name in ((socket.socket, "connect"), (socket, "create_connection"),
                             (socket, "getaddrinfo"), (m, "create_paddle_ocr")):
            guard = patch.object(target, name, side_effect=AssertionError("live operation forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        self.ocr = m.OCR.__new__(m.OCR)
        self.ocr.image_path = "synthetic-only"
        self.ocr.media_type = m.MediaType.MANGA
        self.ocr.review_policy = m.OCRReviewPolicy()
        self.ocr.manga_recognizer = Mock(return_value="候補")
        # Every pixel varies, so preprocessing/rotation/channel mistakes are visible.
        self.image = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)

    def review(self, boxes, **kwargs):
        return self.ocr.review_boxes(boxes, self.image, regions={}, **kwargs)

    def test_python310_string_enums_and_explicit_mode(self):
        for enum, value in ((m.MediaType, "manga"), (m.OCRStatus, "accepted"),
                            (m.OCREngine, "paddle")):
            self.assertIsInstance(enum(value), str)
            self.assertEqual(enum(value), value)
        self.assertRaises(ValueError, m.MediaType, "automatically-guessed")

    def test_same_box_original_rgb_crop_clips_and_records_id(self):
        b = box(27, (-2, 4, 12, 25), "Ａ 12", .4)
        self.ocr.manga_recognizer.return_value = "A12"
        original = b.poly.copy()
        self.review([b])
        crop = self.ocr.manga_recognizer.call_args.args[0]
        self.assertEqual(crop.mode, "RGB")
        self.assertEqual(crop.size, (14, 25))
        self.assertTrue(np.array_equal(np.asarray(crop), self.image[2:27, 0:14, ::-1]))
        self.assertEqual(self.ocr.manga_recognizer.call_args.kwargs,
                         {"source_box_id": 27, "crop_bounds": (0, 2, 14, 27)})
        self.assertEqual(b.ocr_metadata["source_box_id"], 27)
        self.assertEqual(b.ocr_metadata["crop_bounds"], (0, 2, 14, 27))
        self.assertEqual(b.status, m.OCRStatus.ACCEPTED)
        self.assertEqual(b.selected_engine, m.OCREngine.PADDLE)
        self.assertEqual(b.selected_text, "Ａ 12")
        self.assertEqual(b.paddle_text, "Ａ 12")
        self.assertEqual(b.confidence, .4)
        self.assertEqual(b.paddle_confidence, .4)
        self.assertEqual(b.paddle_box_id, 27)
        self.assertTrue(np.array_equal(b.paddle_poly, original))
        self.assertTrue(np.array_equal(b.poly, original))

    def test_padding_avoids_known_neighbor(self):
        b = box(10, (10, 10, 20, 40), "text", .4)
        neighbor = box(11, (21, 10, 31, 40), "accepted", .99)
        self.review([b, neighbor])
        self.assertEqual(b.ocr_metadata["crop_bounds"], (8, 8, 21, 42))
        crop = self.ocr.manga_recognizer.call_args.args[0]
        self.assertTrue(np.array_equal(np.asarray(crop), self.image[8:42, 8:21, ::-1]))
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)

    def test_original_polygon_precision_and_fractional_crop_bounds(self):
        poly = np.array([[10.0000001, 10.2], [20.0000001, 10.2],
                           [20.0000001, 40.1], [10.0000001, 40.1]], dtype=np.float64)
        b = m.Text_Box(5, .4, poly, "text")
        self.review([b])
        self.assertEqual(b.paddle_poly.dtype, poly.dtype)
        self.assertTrue(np.array_equal(b.paddle_poly, poly))
        self.assertEqual(b.ocr_metadata["crop_bounds"], (8, 8, 23, 43))

    def test_overlapping_neighbor_is_not_recognized(self):
        b = box(10, (10, 10, 20, 40), "text", .4)
        self.review([b, box(11, (19, 10, 31, 40), "accepted", .99)])
        self.ocr.manga_recognizer.assert_not_called()
        self.assertEqual(b.reason, "neighbor_overlap")
        self.assertEqual(b.status, m.OCRStatus.REVIEW)

    def test_right_bottom_clipping(self):
        b = box(1, (70, 45, 85, 65), "text", .4)
        self.review([b])
        self.assertEqual(b.ocr_metadata["crop_bounds"], (68, 43, 80, 60))

    def test_rotated_bbox_overlap_without_polygon_or_crop_overlap(self):
        source = m.Text_Box(1, .4, [(10, 25), (25, 10), (40, 25), (25, 40)], "source")
        neighbor = m.Text_Box(2, .99, [(35, 60), (60, 35), (60, 60)], "separate")
        self.assertLess(neighbor.bbox[0], source.bbox[2])
        self.assertLess(neighbor.bbox[1], source.bbox[3])
        self.review([source, neighbor])
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
        self.assertEqual(source.ocr_metadata["crop_bounds"], (8, 8, 42, 42))
        crop = self.ocr.manga_recognizer.call_args.args[0]
        self.assertTrue(np.array_equal(np.asarray(crop), self.image[8:42, 8:42, ::-1]))
        self.assertEqual(source.ocr_metadata["crop_isolation"]["neighbors"][0]["decision"],
                         "outside_crop")

    def test_disjoint_polygons_still_block_rectangular_crop_contamination(self):
        source = m.Text_Box(1, .4, [(10, 25), (25, 10), (40, 25), (25, 40)], "source")
        neighbor = m.Text_Box(2, .99, [(10, 10), (18, 10), (10, 18)], "annotation")
        shared, _ = m.cv2.intersectConvexConvex(source.poly, neighbor.poly)
        self.assertEqual(shared, 0)
        self.review([source, neighbor])
        self.ocr.manga_recognizer.assert_not_called()
        self.assertEqual(source.reason, "crop_contamination")
        self.assertEqual(source.status, m.OCRStatus.REVIEW)

    def test_rotated_neighbor_only_in_padding_trims_without_cutting_source(self):
        source = m.Text_Box(1, .4, [(10, 25), (25, 10), (40, 25), (25, 40)], "source")
        neighbor = m.Text_Box(2, .99, [(34, 48), (48, 34), (48, 48)], "separate")
        self.review([source, neighbor])
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
        bounds = source.ocr_metadata["crop_bounds"]
        self.assertEqual(bounds, (8, 8, 40, 42))
        a, b, c, d = bounds
        rectangle = np.array([(a, b), (c, b), (c, d), (a, d)], dtype=np.float32)
        contamination, _ = m.cv2.intersectConvexConvex(rectangle, neighbor.poly)
        self.assertEqual(contamination, 0)
        crop = self.ocr.manga_recognizer.call_args.args[0]
        self.assertTrue(np.array_equal(np.asarray(crop), self.image[b:d, a:c, ::-1]))

    def test_tiny_contained_text_fragment_not_blanket_veto_at_any_confidence(self):
        for score in (.01, .99):
            with self.subTest(neighbor_confidence=score):
                self.ocr.manga_recognizer.reset_mock()
                self.ocr.manga_recognizer.return_value = "AXC"
                source = box(1, (10, 10, 30, 50), "ABC", .4)
                fragment = box(2, (15, 20, 18, 24), "B", score)
                self.review([source, fragment])
                self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
                self.assertEqual(source.ocr_metadata["crop_bounds"], (8, 8, 32, 52))
                self.assertEqual(source.ocr_metadata["crop_isolation"]["neighbors"][0]["decision"],
                                 "contained_text_fragment")
                self.assertEqual(source.reason, "engine_disagreement")
                self.assertEqual(source.status, m.OCRStatus.REVIEW)
                self.assertEqual(source.paddle_text, "ABC")
                self.assertEqual(source.selected_text, "ABC")
                self.assertEqual(source.confidence, .4)

    def test_near_duplicate_requires_geometry_and_full_text_agreement(self):
        source = box(1, (10, 10, 30, 50), "same", .4)
        duplicate = box(2, (10.2, 10.1, 30.2, 50.1), "same", .99)
        self.review([source, duplicate])
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
        self.assertEqual(source.ocr_metadata["crop_isolation"]["neighbors"][0]["decision"],
                         "near_duplicate")
        source = box(3, (10, 10, 30, 50), "same", .4)
        conflict = box(4, (10.2, 10.1, 30.2, 50.1), "different", .99)
        self.ocr.manga_recognizer.reset_mock()
        self.review([source, conflict])
        self.ocr.manga_recognizer.assert_not_called()
        self.assertEqual(source.status, m.OCRStatus.REVIEW)

    def test_tiny_different_annotation_is_ambiguous_not_assumed_noise(self):
        source = box(1, (10, 10, 30, 50), "ABC", .4)
        annotation = box(2, (15, 20, 18, 24), "XYZ", .01)
        self.review([source, annotation])
        self.ocr.manga_recognizer.assert_not_called()
        self.assertEqual(source.status, m.OCRStatus.REVIEW)
        self.assertEqual(source.reason, "neighbor_overlap")

    def test_substantial_contained_neighbor_and_fragment_budget_remain_blocked(self):
        source = box(1, (10, 10, 30, 50), "ABC", .4)
        neighbor = box(2, (12, 12, 28, 32), "B", .99)
        self.review([source, neighbor])
        self.ocr.manga_recognizer.assert_not_called()
        self.assertEqual(source.status, m.OCRStatus.REVIEW)
        source = box(1, (10, 10, 30, 50), "ABC", .4)
        fragments = [box(i+2, (12, 14+7*i, 16, 20+7*i), "B", .99) for i in range(4)]
        self.review([source] + fragments)
        self.ocr.manga_recognizer.assert_not_called()
        self.assertEqual(source.status, m.OCRStatus.REVIEW)

    def test_partial_large_neighbor_conflict_independent_of_confidence(self):
        for score in (.01, .99):
            source = box(1, (10, 10, 30, 50), "source", .4)
            neighbor = box(2, (29, 5, 70, 55), "separate", score)
            crop, _, error = self.ocr._review_crop(source, [source, neighbor], self.image, 2)
            self.assertIsNone(crop)
            self.assertEqual(error, "neighbor_overlap")
        self.ocr.manga_recognizer.assert_not_called()

    def test_rotated_polygon_clips_to_visible_footprint_with_original_pixels(self):
        source = m.Text_Box(9, .4, [(-20, 10), (10, 10), (-20, 40)], "source")
        outside = box(10, (-10, 15, -5, 20), "separate", .99)
        self.review([source, outside])
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
        self.assertEqual(source.ocr_metadata["crop_bounds"], (0, 8, 12, 22))
        crop = self.ocr.manga_recognizer.call_args.args[0]
        self.assertTrue(np.array_equal(np.asarray(crop), self.image[8:22, 0:12, ::-1]))
        self.assertEqual(self.ocr.manga_recognizer.call_args.kwargs["source_box_id"], 9)
        self.assertEqual(source.paddle_poly.tolist(), [[-20, 10], [10, 10], [-20, 40]])

    def test_outside_polygon_with_intersecting_bbox_is_not_a_valid_crop(self):
        source = m.Text_Box(1, .4, [(-20, 5), (5, -20), (-20, -20)], "source")
        self.review([source])
        self.ocr.manga_recognizer.assert_not_called()
        self.assertEqual(source.reason, "invalid_crop")

    def test_invalid_source_polygons_do_not_invoke_recognizer(self):
        polygons = [np.array([[10, 10], [20, 10], [float("nan"), 30]]),
                    np.array([[10, 10], [20, 10], [15, 15], [20, 30], [10, 30]]),
                    np.array([[10, 10], [20, 30], [10, 30], [20, 10]]),
                    np.array([10, 20, 30]),
                    np.array([["bad", "polygon"], ["bad", "polygon"], ["bad", "polygon"]])]
        for polygon in polygons:
            source = box(1, (10, 10, 20, 40), "source", .4)
            source.paddle_poly = polygon
            self.review([source])
            self.assertIn(source.reason, ("invalid_polygon", "invalid_crop"))
            if source.reason == "invalid_polygon":
                self.assertEqual(source.status, m.OCRStatus.REVIEW)
        self.ocr.manga_recognizer.assert_not_called()

    def test_invalid_neighbor_is_review_ambiguity_not_silently_ignored(self):
        source = box(1, (10, 10, 30, 50), "source", .4)
        neighbor = box(2, (15, 20, 18, 24), "annotation", .99)
        neighbor.paddle_poly = np.array([[15, 20], [18, 24], [15, 24], [18, 20]])
        self.review([source, neighbor])
        self.ocr.manga_recognizer.assert_not_called()
        self.assertEqual(source.reason, "invalid_neighbor_polygon")
        self.assertEqual(source.status, m.OCRStatus.REVIEW)

    def test_crop_isolation_is_independent_of_polygon_winding(self):
        source = m.Text_Box(1, .4, [(25, 40), (40, 25), (25, 10), (10, 25)], "source")
        neighbor = m.Text_Box(2, .99, [(60, 60), (60, 35), (35, 60)], "separate")
        self.review([source, neighbor])
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
        self.assertEqual(source.ocr_metadata["crop_bounds"], (8, 8, 42, 42))

    def test_disagreement_retains_candidates_and_spans(self):
        b = box(1, (10, 10, 20, 40), "猫だ", .4)
        self.ocr.manga_recognizer.return_value = "犬だ"
        self.review([b])
        self.assertEqual(b.status, m.OCRStatus.REVIEW)
        self.assertEqual(b.reason, "engine_disagreement")
        self.assertEqual(b.text, "猫だ")
        self.assertEqual(b.ocr_metadata["candidates"], {"paddle": "猫だ", "manga": "犬だ"})
        self.assertEqual(b.ocr_metadata["differing_spans"], [
            {"operation": "replace", "paddle_span": (0, 1), "manga_span": (0, 1),
             "paddle_text": "猫", "manga_text": "犬"}])

    def test_blank_recovery_uses_real_enclosed_region_but_remains_review(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        m.cv2.rectangle(image, (10, 10), (80, 90), (0, 0, 0), 3)
        b = box(7, (30, 25, 40, 70), "", .1)
        self.ocr.review_boxes([b], image)
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
        self.assertIsNotNone(b.ocr_metadata["region_evidence"])
        self.assertEqual(b.selected_text, "候補")
        self.assertEqual(b.paddle_text, "")
        self.assertEqual(b.selected_engine, m.OCREngine.MANGA)
        self.assertEqual(b.status, m.OCRStatus.REVIEW)
        self.assertEqual(b.reason, "blank_paddle_recovery_requires_review")
        self.assertEqual(b.confidence, .1)

    def test_blank_without_region_is_retained_rejected(self):
        b = box(1, (10, 10, 20, 40), "", .1)
        result = self.review([b])
        self.assertIs(result[0], b)
        self.assertEqual(b.status, m.OCRStatus.REJECTED)
        self.ocr.manga_recognizer.assert_not_called()

    def test_blank_manga_candidate_stays_review(self):
        b = box(1, (10, 10, 20, 40), "ASCII123", .4)
        self.ocr.manga_recognizer.return_value = " "
        self.review([b])
        self.assertEqual(b.status, m.OCRStatus.REVIEW)
        self.assertEqual(b.reason, "blank_manga_candidate")

    def test_ascii_digits_are_valid_manga_candidates(self):
        b = box(1, (10, 10, 20, 40), "123", .4)
        self.ocr.manga_recognizer.return_value = "１２３"
        self.review([b])
        self.assertEqual(b.status, m.OCRStatus.ACCEPTED)

    def test_invalid_empty_and_outside_crops_never_invoke_callback(self):
        for bounds in ((10, 10, 10, 40), (10, 10, 20, 10), (90, 10, 100, 40),
                       (-20, 10, -10, 40)):
            with self.subTest(bounds=bounds):
                b = box(1, bounds, "text", .4)
                self.review([b])
                self.assertEqual(b.reason, "invalid_crop")
                self.assertEqual(b.status, m.OCRStatus.REJECTED)
        self.ocr.manga_recognizer.assert_not_called()

    def test_degenerate_polygon_with_nonempty_bbox_does_not_invoke_callback(self):
        b = m.Text_Box(1, .4, [(10, 10), (15, 20), (20, 30), (15, 20)], "text")
        self.review([b])
        self.assertEqual(b.status, m.OCRStatus.REJECTED)
        self.assertEqual(b.reason, "invalid_crop")
        self.ocr.manga_recognizer.assert_not_called()

    def test_nonmanga_and_accepted_gating(self):
        b = box(1, (10, 10, 20, 40), "text", .4)
        self.review([b], media_type=m.MediaType.OTHER)
        accepted = box(2, (10, 10, 20, 40), "good", .99)
        self.review([accepted])
        self.ocr.manga_recognizer.assert_not_called()
        self.assertEqual(b.status, m.OCRStatus.REVIEW)
        self.assertEqual(accepted.status, m.OCRStatus.ACCEPTED)

    def test_configurable_rejection_and_blank_eligibility(self):
        self.ocr.review_policy = m.OCRReviewPolicy(reject_low_confidence=True)
        low = box(1, (10, 10, 20, 40), "text", .4)
        self.review([low])
        self.assertEqual(low.status, m.OCRStatus.REJECTED)
        self.ocr.review_policy = m.OCRReviewPolicy(blank_region_review=False)
        blank = box(2, (10, 10, 20, 40), "", .1)
        self.ocr.review_boxes([blank], self.image, regions={0: (9, .99)})
        self.assertEqual(blank.status, m.OCRStatus.REJECTED)
        self.ocr.manga_recognizer.assert_not_called()
        self.assertRaises(ValueError, m.OCRReviewPolicy, crop_padding=5)

    def test_attempt_bound_and_error_propagation(self):
        b = box(1, (10, 10, 20, 40), "text", .4)
        self.ocr.manga_recognizer.side_effect = RuntimeError("test-only failure")
        with self.assertRaisesRegex(RuntimeError, "test-only failure"):
            self.review([b])
        self.review([b])
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
        self.assertEqual(b.ocr_metadata["fallback_attempts"], 1)
        self.assertEqual(b.reason, "manga_recognition_failed")

    def test_successful_attempt_bound(self):
        b = box(1, (10, 10, 20, 40), "text", .4)
        self.review([b])
        self.review([b])
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
        self.assertEqual(b.ocr_metadata["fallback_attempts"], 1)

    def test_recognizer_must_return_text_not_confidence(self):
        b = box(1, (10, 10, 20, 40), "text", .4)
        self.ocr.manga_recognizer.return_value = ("candidate", .99)
        self.assertRaises(TypeError, self.review, [b])
        self.assertEqual(b.confidence, .4)
        self.assertEqual(b.status, m.OCRStatus.REVIEW)

    def test_process_default_manga_retains_raw_and_nonmanga_dispatch(self):
        boxes = [box(1, (10, 10, 20, 40), "text", .4),
                 box(2, (30, 10, 40, 40), "good", .99)]
        self.ocr.predict = lambda: [boxes]
        with patch.object(m.cv2, "imread", return_value=self.image):
            groups = self.ocr.process_images()
        self.assertIs(self.ocr.raw_detections[0], boxes)
        self.assertCountEqual([b.box_id for g in groups[0] for b in g.group], [1, 2])
        self.assertEqual(self.ocr.manga_recognizer.call_count, 1)
        self.ocr.predict = lambda: [[box(3, (10, 10, 20, 40), "text", .4)]]
        self.ocr._group_boxes_manga = Mock(side_effect=AssertionError("wrong media dispatch"))
        self.ocr.manga_recognizer.reset_mock()
        with patch.object(m.cv2, "imread", return_value=self.image):
            self.ocr.process_images(media_type=m.MediaType.OTHER)
        self.ocr.manga_recognizer.assert_not_called()

    def test_missing_optional_dependency_fails_before_paddle_initialization(self):
        missing = ModuleNotFoundError("No module named 'manga_ocr'", name="manga_ocr")
        with patch.object(m.importlib, "import_module", side_effect=missing) as importer:
            with self.assertRaises(ModuleNotFoundError) as error:
                m.OCR("not-opened", enable_manga_ocr=True)
        importer.assert_called_once_with("tralens.ocr.manga_adapter")
        m.create_paddle_ocr.assert_not_called()
        self.assertEqual(error.exception.name, "manga_ocr")

    def test_debug_colors(self):
        b = box(1, (10, 10, 20, 40))
        self.assertFalse(hasattr(b, "debug_color"))
        self.assertFalse(hasattr(m.OCR, "visualize_groups"))
        self.assertFalse(hasattr(m.OCR, "visualize_groups_polygon"))
        for status, color in ((m.OCRStatus.ACCEPTED, "green"),
                              (m.OCRStatus.REVIEW, "orange"), (m.OCRStatus.REJECTED, "red")):
            b.set_status(status, "test")
            self.assertEqual(debug_color(b), color)


if __name__ == "__main__":
    unittest.main()
