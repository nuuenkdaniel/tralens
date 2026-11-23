import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image

from tralens.ocr import handwritten
from tralens.ocr.types import Text_Box


def region(i, text, confidence, x, y, r, b):
    return Text_Box(i,confidence,[(x,y),(r,y),(r,b),(x,b)],text)


class HandwrittenTests(unittest.TestCase):
    def test_neighboring_confident_reading_prevents_cluster_retry(self):
        blank = region(1,'',0,110,330,166,415)
        boxes = [blank,region(2,'ああ',.9,300,100,365,199),
                 region(3,'別の台詞',.95,180,275,210,335)]
        reader = Mock()
        self.assertIsNone(handwritten.read_cluster(Image.new('RGB',(500,500),'white'),boxes,blank,reader))
        reader.assert_not_called()

    def test_two_whole_reads_also_need_independent_remainder(self):
        rgb = np.full((500,500,3),255,np.uint8)
        blank = region(1,'',0,110,330,166,415)
        leading = region(2,'ああ',.89,305,100,365,205)
        fragment = region(3,'き',.3,100,255,166,325)
        for x,y in [(325,140),(340,190),(220,235),(232,280),(125,280),(144,348)]:
            cv2.line(rgb,(x-8,y+5),(x+9,y-5),(0,0,0),6)
        image = Image.fromarray(rgb)
        parts = [blank,leading,fragment]
        full = 'ああごめんなさい'
        self.assertIsNone(handwritten.read_cluster(image,parts,blank,Mock(side_effect=[full,full,'違う'])))
        result = handwritten.read_cluster(image,parts,blank,Mock(side_effect=[full,full,'ごめんなさい']))
        self.assertEqual(result[0],full)
        self.assertEqual(result[2],parts)

    def test_stroke_repair_preserves_panel_rule_and_reserved_ink(self):
        rgb = np.full((500,500,3),255,np.uint8)
        cv2.line(rgb,(100,50),(100,450),(0,0,0),3)
        cv2.line(rgb,(110,310),(148,290),(0,0,0),7)
        blank = region(1,'',0,110,330,166,415)
        detection = {'reason':'confirmed_handwritten_cluster','ocr_metadata':{'original_fragments':[
            {'text':'','polygon':blank.poly.tolist()}]}}
        own = np.ones(rgb.shape[:2],bool)
        protected = np.zeros_like(own)
        protected[:,140:] = True
        core,halo = handwritten.stroke_masks(rgb,(0,0,500,500),detection,own,protected)
        self.assertTrue(core[301,127])
        self.assertFalse(halo[:,100].any())
        self.assertFalse((halo & protected).any())
        self.assertFalse((core & ~halo).any())
