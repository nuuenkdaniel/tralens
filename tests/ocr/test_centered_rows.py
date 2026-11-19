import unittest

import numpy as np

from tests.ocr.test_grouping import box
from tralens.ocr.grouping import group_manga_boxes


class CenteredDialogueTests(unittest.TestCase):
    def test_centered_indent_needs_clear_seam_and_continuing_text(self):
        image = np.full((100, 180, 3), 255, dtype=np.uint8)
        rows = [box(0, (58, 20, 122, 32), text='まだ'),
                box(1, (24, 37, 156, 49), text='終わっていません')]

        def groups(pixels):
            return [[b.box_id for b in g.group] for g in group_manga_boxes(
                list(reversed(rows)), original_pixels=pixels)]

        self.assertEqual(groups(image), [[0, 1]])
        self.assertEqual(groups(None), [[0], [1]])
        divided = image.copy()
        divided[34] = 0
        self.assertEqual(groups(divided), [[0], [1]])
        rows[0] = box(0, (58, 20, 122, 32), text='ねえ！')
        self.assertEqual(groups(image), [[0], [1]])

    def test_sloping_form_rule_separates_overlapping_ocr_rows(self):
        from tralens.ocr.automatic_units import horizontal_rule_between
        rows = [box(0,(20,12,216,45)),box(1,(23,40,170,70))]
        pixels = np.full((85,240,3),255,dtype=np.uint8)
        for x in range(240):
            y = 42+x//60
            pixels[y:y+2,x] = 0
        self.assertTrue(horizontal_rule_between(*rows,pixels))
        pixels[:,85:175] = 255
        self.assertFalse(horizontal_rule_between(*rows,pixels))
