import unittest

import cv2
import numpy as np

from tralens.rendering import source_support


class SourceSupport(unittest.TestCase):
    def test_painted_stroke_fill_continues_gradient_without_changing_art(self):
        y,x = np.indices((80,100))
        original = np.stack((150+x//2,170+x//2,190+x//2), axis=2).astype(np.uint8)
        mask = (y>25) & (y<32) & (x>10) & (x<85)
        image = original.copy()
        image[mask] = 0
        obstacle = np.zeros_like(mask)
        obstacle[23:25,45:63] = True
        image[obstacle] = 20
        result = source_support.painted_stroke_fill(image,mask,obstacle)
        self.assertTrue(np.array_equal(result[~mask],image[~mask]))
        self.assertLess(np.abs(result[mask].astype(float)-original[mask]).mean(), 1.)
        self.assertIsNone(source_support.painted_stroke_fill(image, (x>10)&(x<70)&(y>10)&(y<60), obstacle))

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

    def test_wide_source_completes_disconnected_weak_stroke_in_both_polarities(self):
        image,own,probability=self.fixture()
        # A separate hollow character has only weak prediction; an unrelated
        # original shape and a disconnected probability-only island are retained.
        cv2.rectangle(image,(75,165),(102,190),(0,0,0),2)
        probability[164:168,74:103]=45
        image[175:185,145:155]=0;probability[30:40,70:80]=45
        protected=np.zeros_like(own);protected[174:179,72:105]=True
        for pixels in (image,255-image):
            with self.subTest(light_ink=pixels is not image):
                mask,details=source_support.derive(pixels,own,protected,probability)
                letter=np.zeros_like(own);letter[163:193,73:105]=image[163:193,73:105,0]<20
                self.assertTrue(mask[letter&~protected].all())
                self.assertTrue(mask[181:187,81:96].all())
                self.assertFalse((mask&protected).any())
                self.assertFalse(mask[175:185,145:155].any())
                self.assertFalse(mask[30:40,70:80].any())
                self.assertFalse(mask[205:210].any())
                self.assertEqual(len(details['weak_completed_components']),1)

    def test_weak_component_outside_source_cannot_pull_in_neighbor(self):
        image,own,probability=self.fixture()
        cv2.rectangle(image,(210,165),(245,190),(0,0,0),2)
        probability[164:168,209:246]=45
        mask,details=source_support.derive(image,own,np.zeros_like(own),probability)
        self.assertFalse(mask[163:193,209:247].any())
        self.assertEqual(details['weak_completed_components'],[])

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

    def contact_fixture(self):
        image=np.full((240,320,3),245,np.uint8)
        image[45:48,:]=0  # A thin panel line joins both thick glyph pieces.
        image[45:145,110:150]=0;image[45:72,180:204]=0
        image[90:93,145:315]=0  # A long thin art branch from the first piece.
        image[110:130,210:225]=0  # Unconnected, unpredicted artwork.
        image[45:80,260:290]=0  # Another thick piece outside source ownership.
        own=np.zeros(image.shape[:2],bool);own[30:165,80:240]=True
        probability=np.zeros(own.shape,np.uint8);probability[80:145,110:150]=220
        protected=np.zeros_like(own);protected[55:60,175:210]=True
        return image,own,protected,probability

    def test_art_connected_thick_source_parts_are_removed_in_both_polarities(self):
        image,own,protected,probability=self.contact_fixture()
        for pixels in (image,255-image):
            with self.subTest(light_ink=pixels is not image):
                inputs=[v.copy() for v in (pixels,own,protected,probability)]
                mask,details=source_support.derive(pixels,own,protected,probability)
                # The second head has NO learned prediction of its own, but its
                # parent contains substantial source-local lettering evidence.
                self.assertTrue(mask[48:72,180:204][~protected[48:72,180:204]].all())
                self.assertTrue(mask[48:145,110:150].all())
                self.assertFalse((mask&protected).any())
                self.assertFalse(mask[:,250:].any())
                self.assertFalse(mask[90:93,210:].any())
                self.assertFalse(mask[110:130,210:225].any())
                self.assertGreater(details['contact_added_pixels'],0)
                for before,after in zip(inputs,(pixels,own,protected,probability)):
                    self.assertTrue(np.array_equal(before,after))

    def test_sparse_parent_evidence_cannot_promote_unpredicted_attached_piece(self):
        image,own,protected,probability=self.contact_fixture()
        probability[:]=0;probability[120:130,110:150]=220
        mask,details=source_support.derive(image,own,protected,probability)
        self.assertFalse(mask[48:72,180:204].any())
        self.assertEqual(details['contact_completed_components'],[])
        self.assertEqual(details['contact_added_pixels'],0)

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
