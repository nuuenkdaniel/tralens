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
