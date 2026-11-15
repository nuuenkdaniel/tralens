import unittest

import cv2
import numpy as np

from tralens.rendering import source_support


class SourceSupport(unittest.TestCase):
    def fixture(self):
        image=np.full((240,320,3),245,np.uint8)
        # A deliberately oversized detector footprint includes a panel divider
        # and two hollow letters. The low probability field is not itself truth.
        image[205:210]=0
        cv2.rectangle(image,(90,55),(116,145),(0,0,0),3)
        cv2.ellipse(image,(165,100),(20,45),0,0,360,(0,0,0),3)
        own=np.zeros((240,320),bool);own[25:220,55:220]=True
        probability=np.zeros(own.shape,np.uint8)
        probability[57:144,92:115]=220
        cv2.ellipse(probability,(165,100),(19,44),0,0,360,220,-1)
        return image,own,probability

    def test_hollow_lettering_removed_without_rectangle_or_panel_erasure(self):
        image,own,probability=self.fixture()
        mask,_=source_support.derive(image,own,np.zeros_like(own),probability)
        letters=(image[:,:,0]<20);letters[205:210]=False
        self.assertTrue(mask[letters].all())
        self.assertTrue(mask[80:130,98:110].all())
        self.assertFalse(mask[205:210].any())
        self.assertFalse(mask[25:40].any())
        self.assertLess(mask.sum(),own.sum()*.5)

    def test_prior_edits_are_immutable_even_inside_semantic_support(self):
        image,own,probability=self.fixture()
        protected=np.zeros_like(own);protected[95:105,80:190]=True
        mask,_=source_support.derive(image,own,protected,probability)
        self.assertFalse((mask&protected).any())

    def test_partial_protection_does_not_discard_remaining_hollow_letter(self):
        image,own,probability=self.fixture()
        # Protection crosses an outline. Its far side and hollow interior must
        # still be completed even where the semantic prediction misses the edge.
        protected=np.zeros_like(own);protected[95:105,80:130]=True
        probability[:,90:100]=0
        mask,_=source_support.derive(image,own,protected,probability)
        letter=np.zeros_like(own);letter[53:148,88:119]=image[53:148,88:119,0]<20
        self.assertTrue(mask[letter&~protected].all())
        self.assertTrue(mask[80:90,98:110].all())
        self.assertFalse((mask&protected).any())

    def test_narrow_line_completes_missed_trailing_characters_in_both_polarities(self):
        image=np.full((220,180,3),245,np.uint8)
        cv2.rectangle(image,(76,35),(92,60),(0,0,0),3)
        cv2.ellipse(image,(84,130),(10,16),0,0,360,(0,0,0),3)
        image[:,130:133]=0  # Unrelated artwork outside the source line.
        own=np.zeros(image.shape[:2],bool);own[25:155,68:102]=True
        probability=np.zeros(own.shape,np.uint8);probability[35:61,76:93]=220
        for pixels in (image,255-image):
            with self.subTest(light_ink=pixels is not image):
                mask,details=source_support.derive(pixels,own,np.zeros_like(own),probability)
                letters=(image[:,:,0]<20)&own
                self.assertTrue(mask[letters].all())
                self.assertTrue(mask[125:135,80:89].all())
                self.assertFalse(mask[:,130:].any())
                self.assertFalse(mask[80:100].any())
                self.assertGreater(details['line_completed_components'],0)

    def test_broad_box_does_not_promote_disconnected_unpredicted_art(self):
        image,own,probability=self.fixture()
        image[175:185,70:80]=0
        mask,details=source_support.derive(image,own,np.zeros_like(own),probability)
        self.assertFalse(details['narrow_source_line'])
        self.assertFalse(mask[175:185,70:80].any())

    def test_slanted_narrow_line_uses_physical_thickness_not_axis_aligned_width(self):
        image=np.full((240,240,3),245,np.uint8)
        image[36:47,57:68]=0;image[171:182,151:162]=0
        own=np.zeros((240,240),np.uint8)
        cv2.fillPoly(own,[np.array([[40,20],[70,10],[190,190],[160,210]])],1)
        own=own!=0;probability=np.zeros(own.shape,np.uint8);probability[36:47,57:68]=220
        mask,details=source_support.derive(image,own,np.zeros_like(own),probability)
        self.assertGreater(min(cv2.boundingRect(own.astype(np.uint8))[2:]),96)
        self.assertTrue(mask[171:182,151:162].all())
        self.assertLess(details['source_line_thickness'],96)
        self.assertFalse(mask[100:120,100:120].any())

    def test_attached_long_art_stroke_cannot_expand_removal_across_crop(self):
        image,own,probability=self.fixture()
        image[100:103,110:315]=0
        mask,_=source_support.derive(image,own,np.zeros_like(own),probability)
        self.assertFalse(mask[:,240:].any())
        self.assertFalse(mask[100:103,210:].any())

    def test_unsupported_source_must_use_removal_fallback(self):
        image,own,probability=self.fixture()
        probability[:]=0
        with self.assertRaisesRegex(ValueError,'insufficient'):
            source_support.derive(image,own,np.zeros_like(own),probability)

    def test_light_letters_on_dark_surface(self):
        image,own,probability=self.fixture()
        mask,_=source_support.derive(255-image,own,np.zeros_like(own),probability)
        letters=(image[:,:,0]<20);letters[205:210]=False
        self.assertTrue(mask[letters].all())
        self.assertFalse(mask[205:210].any())


if __name__=='__main__':unittest.main()
