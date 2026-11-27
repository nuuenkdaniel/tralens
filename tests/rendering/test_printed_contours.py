import unittest

import cv2
import numpy as np

from tralens.rendering.printed_contours import colored_title, vertical_continuations, vertical_ellipsis, pale_speedline


class HollowEffectTests(unittest.TestCase):
    def test_faint_ellipsis_requires_three_aligned_supported_dots(self):
        crop=np.full((100,40,3),240,np.uint8)
        for y in (68,78,88):
            cv2.circle(crop,(15,y),3,(120,119,118),1)
        cv2.line(crop,(32,41),(28,96),(72,61,78),2)
        probability=np.zeros(crop.shape[:2],np.uint8)
        probability[60:,8:22]=250
        ink=vertical_ellipsis(crop,probability)
        self.assertEqual(int(ink.sum()),np.count_nonzero(crop[:,8:22,0]<150))
        probability[83:]=0
        self.assertFalse(vertical_ellipsis(crop,probability).any())

    def test_long_white_art_line_survives_a_crossing_letter(self):
        crop=np.full((170,90,3),(50,65,60),np.uint8)
        own=np.zeros(crop.shape[:2],bool)
        own[5:165,20:63]=True
        cv2.line(crop,(40,0),(40,169),(255,255,255),1)
        for y in (50,80,115):
            cv2.line(crop,(27,y),(53,y),(255,255,255),3)
        commit,full=pale_speedline(crop,own,np.zeros_like(own))
        self.assertTrue(full[80,40])
        self.assertFalse(commit[80,40])
        self.assertTrue(commit[80,32])

    def test_colored_title_keeps_unrelated_ink_and_protected_print(self):
        paper = np.full((90,200,3),(254,225,192),np.uint8)
        ink=(255,145,10)
        for x in (18,58,98,138):
            cv2.rectangle(paper,(x-3,16),(x+22,68),(48,38,25),-1)
            cv2.rectangle(paper,(x-1,14),(x+20,65),(255,245,206),-1)
            cv2.rectangle(paper,(x,16),(x+16,61),ink,-1)
        paper[70:81,179:184]=(43,37,73)
        paper[34:45,19:24]=(27,35,39)
        protected=np.zeros(paper.shape[:2],bool)
        protected[34:45,19:24]=True
        hue=int(cv2.cvtColor(np.uint8([[ink]]),cv2.COLOR_RGB2HSV)[0,0,0])
        mask,plate,color = colored_title(paper,np.ones_like(protected),protected,hue)
        self.assertEqual(color,list(ink))
        self.assertTrue(mask[30,61])
        self.assertTrue(np.array_equal(plate[30,61],paper[1,1]))
        self.assertFalse(mask[protected].any())
        self.assertTrue(np.array_equal(plate[70:81,179:184],paper[70:81,179:184]))
        self.assertIsNone(colored_title(paper,np.ones_like(protected),protected,(hue+90)%180))

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
