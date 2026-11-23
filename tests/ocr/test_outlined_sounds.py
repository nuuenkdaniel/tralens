import unittest

import cv2
import numpy as np

from tralens.ocr import outlined_sounds, sideways_sound, automatic_units
from tralens.ocr.types import Text_Box, Text_Group
from tralens.rendering import outlined_sound_cleanup, replacement


class OutlinedSoundTests(unittest.TestCase):
    def test_blank_effect_requires_matching_kana_across_all_views(self):
        from PIL import Image, ImageDraw
        page = Image.new('RGB',(150,150),'white')
        box = Text_Box(7,0,[(10,10),(95,10),(95,82),(10,82)],'')
        self.assertIsNone(outlined_sounds.blank_sound(page,box,lambda view:self.fail('blank crop')))
        ImageDraw.Draw(page).line((30,25,65,50),fill='black',width=4)
        for words,expected in [(['サッ']*3,'サッ'), (['サッ','ザッ','サッ'],None), (['11']*3,None)]:
            reads = iter(words)
            self.assertEqual(outlined_sounds.blank_sound(page,box,lambda view:next(reads)),expected)

    def test_framed_mask_leaves_a_white_symbol_and_rules_intact(self):
        from tralens.rendering.printed_contours import framed_outline_mask
        image = np.full((120,240,3),185,np.uint8)
        cv2.rectangle(image,(10,15),(225,100),(0,0,0),3)
        cv2.circle(image,(148,56),18,(255,255,255),-1)
        for x in (38,61,87):
            cv2.line(image,(x+16,24),(x,92),(255,255,255),13)
            cv2.line(image,(x+16,24),(x,92),(0,0,0),5)
        own = np.zeros(image.shape[:2],bool)
        own[16:101,12:161] = True
        mask = framed_outline_mask(image,own,np.zeros_like(own))
        self.assertIsNotNone(mask)
        self.assertTrue(mask[90,38])
        self.assertFalse(mask[:,128:].any())
        self.assertFalse(mask[15].any())
        self.assertFalse(mask[100].any())
        image[10:20] = 185
        self.assertIsNone(framed_outline_mask(image,own,np.zeros_like(own)))

    def test_repetition_must_come_from_whole_views(self):
        beat = 'パチ'
        readings = [beat*4,beat*3,beat*5,beat*2,beat*2]
        self.assertEqual(outlined_sounds.confirmed_text(readings),beat*4)
        self.assertIsNone(outlined_sounds.confirmed_text([*readings[:3],beat,'あ']))
        self.assertIsNone(outlined_sounds.confirmed_text([*readings[:2],'ドンドン',*readings[3:]]))

    def test_repeated_cores_avoid_unrelated_lettering(self):
        rgb = np.full((100,220,3),(130,165,199),np.uint8)
        for x in range(15,210,12):
            cv2.rectangle(rgb,(x,34),(x+7,52),(0,0,0),-1)
            cv2.rectangle(rgb,(x+2,36),(x+5,50),(246,246,246),-1)
        owned = np.zeros(rgb.shape[:2],bool)
        owned[:,:120] = True
        protected = np.zeros_like(owned)
        protected[:,180:] = True
        mask = outlined_sound_cleanup.joint_mask(rgb,owned,protected,(0,0,220,100))
        self.assertTrue(mask[42,161])
        self.assertFalse(mask[:,180:].any())
        self.assertFalse(mask[:20].any())
        self.assertFalse(outlined_sound_cleanup.joint_mask(rgb,owned & False,protected,(0,0,220,100)).any())

    def test_classical_joint_repair_preserves_gutter_and_unmasked_art(self):
        rgb = np.full((100,160,3), (202,146,72), np.uint8)
        rgb[60:75] = 255
        rgb[75:] = (45,90,145)
        mask = np.zeros(rgb.shape[:2], bool)
        mask[41:82,60:97] = True
        rgb[mask] = 0
        result = outlined_sound_cleanup.classical_repair(rgb, mask)
        self.assertTrue(np.array_equal(result[~mask], rgb[~mask]))
        self.assertTrue((result[60:75] == 255).all())
        self.assertTrue(np.allclose(result[54,78], (202,146,72), atol=1))
        self.assertTrue(np.allclose(result[79,78], (45,90,145), atol=1))

    def test_sideways_reading_requires_independent_visible_suffix(self):
        def agrees(*readings):
            return sideways_sound.matches({'readings':readings})
        self.assertTrue(agrees('グアオオオ','グアオオオ','グアオオオ','オオオ'))
        self.assertFalse(agrees('グアオオオ','クアオオオ','グアオオオ','オオオ'))
        self.assertFalse(agrees('グアオオオ','グアオオオ','グアオオオ','アオ'))

    def test_obligation_requires_clear_seam_and_matching_predicate(self):
        pixels = np.full((100,140,3),255,np.uint8)
        groups = []
        for i,(y,word) in enumerate([(25,'実演しないと'),(50,'いけないなんて')]):
            box = Text_Box(i,.99,[(20,y),(120,y),(120,y+20),(20,y+20)],word)
            groups.append(Text_Group([box],direction='horizontal'))
        def join():
            return automatic_units.join_horizontal_continuations(groups,pixels,[],{},
                {i:(None,None,i) for i in range(2)},set(),{})
        self.assertEqual(len(join()),1)
        pixels[47] = 0
        self.assertEqual(join(),groups)
        pixels[47] = 255
        groups[1].group[0].text = 'どうして？'
        self.assertEqual(join(),groups)
        groups[0].group[0].text = 'お茶を'
        groups[1].group[0].text = 'いただきます'
        self.assertEqual(len(join()),1)
        pixels[47] = 0
        self.assertEqual(join(),groups)

    def test_gutter_removes_only_connected_and_unprotected_ink(self):
        src=np.full((90,160,3),255,np.uint8)
        src[:35]=[163,128,67]
        src[24:55,47:51]=0
        src[40:52,118:121]=0
        old=np.zeros(src.shape[:2],bool)
        old[24:38,45:53]=True
        protected=np.zeros_like(old)
        protected[51:,50]=True
        plate=src.copy()
        plate[old]=255
        before=plate.copy()
        tails=outlined_sound_cleanup.repair_gutter(src,plate,old,protected)
        self.assertTrue(tails[49,48])
        self.assertFalse(tails[protected].any())
        self.assertTrue(np.array_equal(plate[~tails],before[~tails]))
        self.assertEqual(int(plate[46,120,0]),0)

    def test_painted_gutter_bounds_require_continuous_bands(self):
        rgb = np.full((110,150,3),(185,138,95),np.uint8)
        rgb[:,20:28] = 255
        rgb[85:93,:] = 255
        rgb[56:58,:] = 255
        self.assertEqual(replacement.painted_panel_limits(rgb,(48,37,96,50)),[28,0,150,85])
