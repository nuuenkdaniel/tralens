import unittest

import numpy as np

from tralens.rendering.printed_contours import vertical_continuations


class HollowEffectTests(unittest.TestCase):
    def test_interrupted_rule_needs_matching_ink_on_both_sides(self):
        crop = np.full((80,60,3),255,np.uint8)
        crop[:27,25] = 50
        crop[51:,25] = 55
        hidden = np.zeros(crop.shape[:2],bool)
        hidden[30:48,18:34] = True
        protected = np.zeros_like(hidden)
        protected[40,25] = True
        strokes,tones = vertical_continuations(crop,hidden,protected)
        self.assertTrue(strokes[35,25])
        self.assertFalse(strokes[40,25])
        self.assertTrue(np.all((tones[strokes]>=50)&(tones[strokes]<=55)))
        crop[51:,25] = 255
        self.assertFalse(vertical_continuations(crop,hidden,protected)[0].any())
