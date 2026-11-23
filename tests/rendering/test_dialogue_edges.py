import unittest

import cv2
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

    def test_divider_next_to_round_balloon_does_not_claim_the_panel(self):
        crop = np.full((120, 160, 3), 255, np.uint8)
        cv2.ellipse(crop, (80, 62), (43, 44), 0, 0, 360, (0, 0, 0), 2)
        crop[108:110] = 0
        own = np.zeros(crop.shape[:2], bool)
        own[79:94, 68:92] = True
        allowed = np.ones_like(own)
        frame = light_dialogue_frame(crop, own, allowed, enclosed=True)
        self.assertTrue(frame[62, 80])
        self.assertFalse(frame[108, 80] or frame[62, 12])
        crop.fill(255)
        cv2.rectangle(crop, (30, 17), (130, 106), (0, 0, 0), 2)
        self.assertIsNone(light_dialogue_frame(crop, own, allowed, enclosed=True))

    def test_enclosed_retry_uses_paper_around_source_and_respects_obstacles(self):
        crop = np.zeros((90, 150, 3), np.uint8)
        crop[10:80, 10:140] = 255
        own = np.zeros(crop.shape[:2], bool)
        own[35:55, 55:95] = True
        allowed = np.ones_like(own)
        crop[24, 35:37] = 238
        allowed[24, 36] = False
        crop[30, 35] = 0
        frame = light_dialogue_frame(crop, own, allowed, enclosed=True)
        self.assertTrue(frame[24, 35])
        self.assertFalse(frame[24, 36] or frame[30, 35] or frame[8, 60])
        crop[:10, 60] = 255
        self.assertIsNone(light_dialogue_frame(crop, own, allowed, enclosed=True))

    def test_math_sign_requires_real_font_coverage(self):
        from tralens.rendering.natural import validate_text
        target = '8.10 × 2'
        validate_text(target,set(map(ord,target)))
        with self.assertRaisesRegex(ValueError,'missing glyph coverage'):
            validate_text(target,set(range(128)))
        with self.assertRaises(ValueError):
            validate_text('a\u202eb',{ord('a'),ord('b'),0x202e})
