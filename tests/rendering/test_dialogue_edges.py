import unittest

import numpy as np

from tralens.rendering.aggressive import adjacent_ellipsis, light_dialogue_frame


class DialogueEdgeTests(unittest.TestCase):
    def test_ellipsis_must_be_complete_separate_and_unclaimed(self):
        crop = np.full((80, 160, 3), 255, dtype=np.uint8)
        own = np.zeros(crop.shape[:2], bool)
        own[25:49, 40:120] = True
        protected = np.zeros_like(own)
        for x in (17, 24, 31):
            crop[40:43,x:x+3] = 0
        mask = adjacent_ellipsis(crop,own,protected)
        self.assertTrue(mask[41,18] and mask[41,25] and mask[41,32])
        self.assertFalse((mask & own).any())
        protected[41,32] = True
        self.assertFalse(adjacent_ellipsis(crop,own,protected).any())
        crop[40:43,24:27] = 255
        self.assertFalse(adjacent_ellipsis(crop,own,np.zeros_like(own)).any())

    def test_long_panel_rule_disqualifies_white_dialogue_frame(self):
        crop = np.full((80, 160, 3), 255, dtype=np.uint8)
        own = np.zeros(crop.shape[:2], bool)
        own[28:50,35:125] = True
        allowed = np.ones_like(own)
        self.assertIsNotNone(light_dialogue_frame(crop,own,allowed))
        crop[12:14] = 0
        self.assertIsNone(light_dialogue_frame(crop,own,allowed))

    def test_math_sign_requires_real_font_coverage(self):
        from tralens.rendering.natural import validate_text
        target = '8.10 × 2'
        validate_text(target,set(map(ord,target)))
        with self.assertRaisesRegex(ValueError,'missing glyph coverage'):
            validate_text(target,set(range(128)))
        with self.assertRaises(ValueError):
            validate_text('a\u202eb',{ord('a'),ord('b'),0x202e})
