import unittest

import cv2
import numpy as np

from tralens.rendering import repeated_strokes


class RepeatedStrokeTests(unittest.TestCase):
    def test_two_pale_beats_only_match_repeated_strokes_on_the_painted_surface(self):
        crop = np.full((105,150,3),[56,87,108],np.uint8)
        lines = []
        for y in (22,53):
            poly = [[24,y],[105,y],[105,y+23],[24,y+23]]
            lines.append({'text':'さら','polygon':poly})
            for x in [31,54,78,93]:
                cv2.line(crop,(x,y+5),(x+4,y+14),(232,240,246),2)
        cv2.line(crop,(57,73),(68,70),(224,235,244),1)
        guard = np.zeros(crop.shape[:2],bool)
        guard[24:43,28:39] = True
        result = repeated_strokes.pale_pair(crop,lines,np.array([0,0]),guard)
        self.assertIsNotNone(result)
        _,mask,style,_ = result
        self.assertTrue(mask[61,55])
        self.assertFalse(mask[72,61] or mask[guard].any())
        self.assertGreater(min(style['fill']),220)
        self.assertIsNone(repeated_strokes.pale_pair(crop,[lines[0],dict(lines[1],text='かさ')],np.array([0,0]),guard))

    def test_shared_ink_preserves_unrelated_marks_and_a_protected_neighbor(self):
        tiles = []
        for i in range(5):
            art = np.full((96,96,3),250,np.uint8)
            cv2.line(art,(18,71),(74,25),(0,0,0),8)
            cv2.line(art,(25,29),(72,29),(0,0,0),7)
            cv2.rectangle(art,(4,5+11*i),(12,9+11*i),(0,0,0),-1)
            prob = np.zeros((96,96),np.uint8)
            prob[20:77,15:80] = 190
            tiles.append((art,prob))
        masks = list(repeated_strokes.shared_ink(tiles))
        self.assertEqual(len(masks),5)
        for i,mask in enumerate(masks):
            self.assertTrue(mask[47,47])
            self.assertFalse(mask[7+11*i,8])
        guard = np.zeros((96,96),bool)
        guard[:,50:55] = True
        args = ({3:((0,0,96,96),masks[0])},[3],(0,0,96,96),tiles[0][0],guard)
        commit,inference = repeated_strokes.local(*args)
        self.assertTrue(commit[47,47])
        self.assertFalse(commit[guard].any() or (commit&~inference).any())
        self.assertIsNone(repeated_strokes.local(args[0],[3,4],*args[2:]))
