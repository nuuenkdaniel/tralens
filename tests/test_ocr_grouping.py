"""Offline grouping tests: no Paddle initialization or external resources."""
import importlib.util
import os
import socket
import unittest
from pathlib import Path
from unittest.mock import patch


# Disable PaddleX's import-time host check without reading environment values.
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

SPEC = importlib.util.spec_from_file_location(
    "tralens_ocr_offline", Path(__file__).resolve().parents[1] / "src" / "ocr.py")
ocr_module = importlib.util.module_from_spec(SPEC)
with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), \
     patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")):
    SPEC.loader.exec_module(ocr_module)
OCR = ocr_module.OCR


def box(identifier, bounds, text="text", confidence=.99):
    x1, y1, x2, y2 = bounds
    return ocr_module.Text_Box(identifier, confidence,
                               [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], text)


class GroupingTests(unittest.TestCase):
    def setUp(self):
        constructor = patch.object(ocr_module, "PaddleOCR",
                                   side_effect=AssertionError("OCR initialization forbidden"))
        constructor.start()
        self.addCleanup(constructor.stop)
        self.ocr = OCR.__new__(OCR)
        self.ocr.image_path = "not-read-by-geometric-tests"
        self.ocr._manga_regions = lambda boxes: ({}, "synthetic_no_region")

    def group(self, boxes):
        groups = self.ocr._group_boxes_manga(boxes)
        actual = [id(b) for g in groups for b in g.group]
        self.assertCountEqual(actual, [id(b) for b in boxes])
        self.assertEqual(len(actual), len(set(actual)))
        indices = [i for g in groups for i in g.metadata["source_indices"]]
        self.assertCountEqual(indices, range(len(boxes)))
        return groups

    def test_vertical_columns_merge_and_read_right_to_left(self):
        boxes = [box(0, (10, 10, 20, 90)), box(1, (28, 12, 38, 92)),
                 box(2, (46, 11, 56, 91))]
        groups = self.group(boxes)
        self.assertEqual(len(groups), 1)
        self.assertEqual([b.box_id for b in groups[0].group], [2, 1, 0])
        self.assertEqual(groups[0].direction, "vertical")
        self.assertEqual(groups[0].method, "geometry_fallback")
        reversed_groups = self.group(list(reversed(boxes)))
        self.assertEqual([b.box_id for b in reversed_groups[0].group], [2, 1, 0])

    def test_nearby_separate_dialogue_with_offset_alignment(self):
        boxes = [box(0, (10, 10, 20, 80)), box(1, (28, 10, 38, 80)),
                 box(2, (45, 45, 55, 115)), box(3, (63, 45, 73, 115))]
        groups = self.group(boxes)
        self.assertEqual([{b.box_id for b in g.group} for g in groups],
                         [{0, 1}, {2, 3}])

    def test_horizontal_narration_top_to_bottom(self):
        boxes = [box(2, (11, 42, 121, 52)), box(0, (10, 10, 120, 20)),
                 box(1, (12, 26, 122, 36))]
        groups = self.group(boxes)
        self.assertEqual(len(groups), 1)
        self.assertEqual([b.box_id for b in groups[0].group], [0, 1, 2])
        self.assertEqual(groups[0].direction, "horizontal")

    def test_horizontal_fragments_left_to_right_despite_jitter(self):
        groups = self.group([box(1, (55, 9, 95, 19)),
                             box(0, (10, 10, 50, 20))])
        self.assertEqual(len(groups), 1)
        self.assertEqual([b.box_id for b in groups[0].group], [0, 1])

    def test_direction_scale_and_unknown_boxes_stay_separate(self):
        groups = self.group([box(0, (10, 10, 20, 80)),
                             box(1, (22, 10, 92, 20)),
                             box(2, (22, 25, 47, 125)),
                             box(3, (100, 100, 110, 110), text=""),
                             box(4, (120, 100, 120, 110))])
        self.assertEqual(len(groups), 5)
        self.assertTrue(all(g.status == "singleton" for g in groups))

    def test_complete_link_does_not_bridge_long_chain(self):
        boxes = [box(i, (10 + 18*i, 10, 20 + 18*i, 90)) for i in range(7)]
        groups = self.group(boxes)
        self.assertGreater(len(groups), 1)
        for group in groups:
            self.assertLessEqual(max(b.bbox[0] for b in group.group)
                                 - min(b.bbox[2] for b in group.group), 40)

    def test_trusted_regions_separate_even_aligned_nearby_columns(self):
        boxes = [box(i, (10 + 18*i, 10, 20 + 18*i, 90)) for i in range(4)]
        self.ocr._manga_regions = lambda boxes: (
            {0: (7, 1.), 1: (7, .95), 2: (8, 1.), 3: (8, 1.)}, "synthetic")
        groups = self.group(boxes)
        self.assertEqual([{b.box_id for b in g.group} for g in groups],
                         [{2, 3}, {0, 1}])
        self.assertTrue(all(g.method == "region_geometry" for g in groups))

    def test_unassigned_box_cannot_bridge_known_regions(self):
        boxes = [box(i, (10 + 18*i, 10, 20 + 18*i, 90)) for i in range(3)]
        self.ocr._manga_regions = lambda boxes: (
            {0: (7, .95), 2: (8, .95)}, "synthetic")
        groups = self.group(boxes)
        self.assertEqual(len(groups), 2)
        self.assertIn("mixed", [g.method for g in groups])

    def test_same_region_does_not_override_distant_geometry(self):
        self.ocr._manga_regions = lambda boxes: (
            {0: (7, 1.), 1: (7, 1.)}, "synthetic")
        self.assertEqual(len(self.group([box(0, (10, 10, 20, 90)),
                                        box(1, (150, 10, 160, 90))])), 2)

    def test_empty_and_process_dispatch_without_constructor(self):
        self.assertEqual(self.group([]), [])
        boxes = [box(0, (10, 10, 20, 90))]
        self.ocr.predict = lambda: [boxes, []]
        self.ocr._group_boxes_proximity = lambda _: self.fail("legacy dispatch")
        groups = self.ocr.process_images()
        self.assertEqual(groups[0][0].group, boxes)
        self.assertEqual(groups[1], [])

    def test_missing_image_assigns_every_box_via_real_fallback(self):
        del self.ocr._manga_regions
        boxes = [box(0, (10, 10, 20, 90)), box(1, (28, 10, 38, 90)),
                 box(2, (150, 100, 160, 110), text="")]
        with patch.object(ocr_module.cv2, "imread", return_value=None):
            groups = self.group(boxes)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(g.method == "geometry_fallback" for g in groups))
        self.assertTrue(all(g.metadata["region_status"] == "image_unavailable"
                            for g in groups))


class RegionMaskTests(unittest.TestCase):
    cv2, np = ocr_module.cv2, ocr_module.np

    def setUp(self):
        constructor = patch.object(ocr_module, "PaddleOCR",
                                   side_effect=AssertionError("OCR initialization forbidden"))
        constructor.start()
        self.addCleanup(constructor.stop)
        self.ocr = OCR.__new__(OCR)
        self.ocr.image_path = "synthetic-image"
        self.image = self.np.full((180, 240, 3), 255, dtype=self.np.uint8)

    def regions(self, boxes):
        for b in boxes:
            b.poly = self.np.array(b.poly, dtype=self.np.float32)
        with patch.object(self.cv2, "imread", return_value=self.image):
            return self.ocr._manga_regions(boxes)

    def test_closed_bubbles_and_ocr_ink_removal(self):
        self.cv2.rectangle(self.image, (10, 10), (90, 140), (0, 0, 0), 3)
        self.cv2.rectangle(self.image, (110, 10), (190, 140), (0, 0, 0), 3)
        boxes = [box(0, (30, 30, 40, 100)), box(1, (55, 30, 65, 100)),
                 box(2, (130, 30, 140, 100))]
        for b in boxes:
            self.cv2.rectangle(self.image, b.bbox[:2], b.bbox[2:], (0, 0, 0), -1)
        regions, status = self.regions(boxes)
        self.assertEqual(status, "region_analysis_complete")
        self.assertEqual(set(regions), {0, 1, 2})
        self.assertEqual(regions[0][0], regions[1][0])
        self.assertNotEqual(regions[0][0], regions[2][0])

    def test_open_background_low_confidence_and_blank_use_fallback(self):
        boxes = [box(0, (30, 30, 40, 100))]
        self.assertEqual(self.regions(boxes)[0], {})
        self.cv2.rectangle(self.image, (10, 10), (90, 140), (0, 0, 0), 3)
        boxes = [box(1, (30, 30, 40, 100), confidence=.2),
                 box(2, (55, 30, 65, 100), text=" ")]
        self.assertEqual(self.regions(boxes)[0], {})

    def test_missing_image_keeps_all_boxes_via_fallback(self):
        boxes = [box(0, (10, 10, 20, 90)), box(1, (28, 10, 38, 90))]
        with patch.object(self.cv2, "imread", return_value=None):
            groups = self.ocr._group_boxes_manga(boxes)
        self.assertEqual(len(groups), 1)
        self.assertCountEqual([b.box_id for b in groups[0].group], [0, 1])
        self.assertEqual(groups[0].metadata["region_status"], "image_unavailable")


if __name__ == "__main__":
    unittest.main()
