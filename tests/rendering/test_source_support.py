import unittest

import cv2
import numpy as np

from tralens.rendering import source_support


class SourceSupport(unittest.TestCase):
    def test_white_keyline_on_striped_paint_uses_only_matching_sides(self):
        ground=np.full((125,90,3),[174,191,213],np.uint8)
        ground[::9]=[192,209,230]
        ink=np.zeros(ground.shape[:2],np.uint8)
        for y in (24,54,84):
            cv2.ellipse(ink,(43,y),(8,10),0,0,360,1,-1)
        ink=ink.astype(bool)
        protected=np.zeros_like(ink)
        protected[:,60:62]=True
        ground[protected]=10
        printed=ground.copy()
        printed[ink]=250
        printed[cv2.erode(np.uint8(ink),np.ones((5,5),np.uint8))!=0]=55
        mask,repair=source_support.painted_keyline_fill(printed,ink,protected)
        self.assertFalse((mask&protected).any())
        np.testing.assert_array_equal(repair[mask],ground[mask])
        printed[ink]=55
        self.assertIsNone(source_support.painted_keyline_fill(printed,ink,protected))

    def test_touching_glyph_releases_its_stem_and_preserves_the_crossing_contour(self):
        from tralens.rendering.aggressive import crossing_ink,split_touching_glyph
        crop = np.full((70,92,3),255,np.uint8)
        cv2.line(crop,(5,48),(88,48),(0,0,0),2)
        cv2.line(crop,(45,27),(45,47),(0,0,0),3)
        own = np.zeros(crop.shape[:2],bool)
        own[23:49,39:50] = True
        guard = crossing_ink(crop,own)
        kept = split_touching_glyph(crop,guard,own,np.zeros_like(own))
        self.assertTrue(kept[48,45])
        self.assertFalse(kept[34,45])
        self.assertTrue(kept[48,14])

    def test_pale_effect_style_requires_strokes_across_a_colored_baseline(self):
        from tralens.rendering.aggressive import light_effect_style
        polygon = np.array([[15,30],[145,30],[145,62],[15,62]])
        own = np.zeros((90,160),np.uint8)
        cv2.fillPoly(own,[polygon],1)
        art = np.full((90,160,3),[90,134,152],np.uint8)
        for x in [27,54,91,119]:
            cv2.line(art,(x,37),(x+5,56),(28,28,28),8)
            cv2.line(art,(x,37),(x+5,56),(252,252,252),4)
        style = light_effect_style(art,own!=0,polygon,'ガタン')
        self.assertEqual(style['outline'],2)
        self.assertEqual(style['rotation'],0)
        self.assertIsNone(light_effect_style(art,own!=0,polygon,'今日は晴れ'))
        art[:]=255
        self.assertIsNone(light_effect_style(art,own!=0,polygon,'ガタン'))

    def test_light_painted_letters_need_a_common_surface_and_local_text_support(self):
        art=np.full((100,160,3),[149,118,85],np.uint8)
        own=np.zeros((100,160),bool)
        own[32:66,23:135]=True
        probability=np.zeros(own.shape,np.uint8)
        probability[39:61,33:39]=160
        for x in (33,65,104):
            art[38:61,x:x+6]=[240,235,231]
        blocked=np.zeros_like(own)
        blocked[40:42,34]=True
        mask,_=source_support.pale_sign_strokes(art,own,blocked,probability)
        self.assertTrue(mask[47,106])
        self.assertFalse(mask[40,34])
        self.assertFalse(mask[28,32])
        self.assertIsNone(source_support.pale_sign_strokes(art,own,blocked,probability*0))
        plate=source_support.painted_stroke_fill(art,mask,blocked,pale_ink=True)
        self.assertLess(np.abs(plate[47,106].astype(int)-art[34,78]).max(),3)

    def test_gray_band_needs_matching_print_on_both_sides_and_respects_ink(self):
        paper = np.full((100,105,3),255,np.uint8)
        paper[39:49] = 194
        own = np.zeros(paper.shape[:2],bool)
        own[16:84,40:65] = True
        mask = np.zeros_like(own)
        mask[20:77,48:55] = True
        paper[39:49,44:60] = 255
        paper[39:49,47] = 23
        protected = np.zeros_like(own)
        protected[41:44,45] = True
        plate = paper.copy()
        plate[mask] = 255
        support = source_support.restore_paper_band(paper,plate,own,mask,protected)
        self.assertEqual(plate[43,50,0],194)
        self.assertEqual(plate[43,44,0],194)
        self.assertFalse(support[43,45])
        self.assertEqual(plate[43,47,0],23)
        self.assertFalse(support[43,47])
        paper[39:49,65:] = 255
        np.testing.assert_array_equal(source_support.restore_paper_band(paper,plate,own,mask,protected),mask)

    def test_hollow_effect_preserves_rim_art_and_requires_multiple_supported_strokes(self):
        crop = np.full((110,180,3),255,np.uint8)
        own = np.ones(crop.shape[:2],bool)
        prob = np.zeros(own.shape,np.uint8)
        for x in (35,74,112):
            cv2.line(crop,(x,26),(x,81),(0,0,0),15)
            cv2.line(crop,(x,26),(x,81),(255,255,255),7)
            cv2.line(prob,(x,26),(x,81),225,8)
        cv2.line(crop,(125,50),(160,50),(0,0,0),1)
        protected = np.zeros_like(own)
        protected[46:52,25:39] = True
        commit,inference = source_support.hollow_effect_masks(crop,own,protected,prob)
        self.assertTrue(commit[60,74])
        self.assertFalse(commit[protected].any())
        self.assertFalse(commit[50,125])
        self.assertFalse((commit & ~inference).any())
        prob[:,95:] = 0
        self.assertIsNone(source_support.hollow_effect_masks(crop,own,protected,prob))

    def test_solid_external_ink_is_protected_but_separate_type_is_erasable(self):
        crop=np.full((150,160,3),255,np.uint8)
        cv2.fillPoly(crop,[np.array([(0,12),(60,25),(84,125),(0,120)])],(0,0,0))
        cv2.line(crop,(93,35),(105,105),(0,0,0),4)
        own=np.zeros(crop.shape[:2],bool)
        own[22:127,54:130]=True
        probability=np.zeros(own.shape,np.uint8)
        probability[:,90:]=230
        guard=source_support.adjoining_solid_ink(crop,own,own,probability)
        self.assertIsNotNone(guard)
        self.assertTrue(guard[91,64])
        self.assertFalse(guard[77,100])
        self.assertFalse((guard&~own).any())
        self.assertIsNone(source_support.adjoining_solid_ink(crop,own,own,np.full_like(probability,230)))

    def test_white_stroke_tip_needs_dark_rim_and_model_support(self):
        crop = np.full((100,140,3),180,np.uint8)
        cv2.line(crop,(20,40),(104,55),(0,0,0),15)
        cv2.line(crop,(20,40),(104,55),(255,255,255),7)
        own = np.ones(crop.shape[:2],bool)
        prob = np.zeros(own.shape,np.uint8)
        prob[:,:91] = 200
        guard = np.zeros_like(own)
        guard[:,107:] = True
        mask = source_support.outlined_interiors(crop,own,guard,prob)
        self.assertTrue(mask[54,101])
        self.assertFalse(mask[guard].any())
        crop[np.all(crop==0,axis=2)] = 160
        self.assertFalse(source_support.outlined_interiors(crop,own,guard,prob).any())

    def test_wide_credit_strip_respects_visible_paper_grain(self):
        y,x = np.indices((64,700))
        shade = (193 + x//45 + (x+y)%3).astype(np.uint8)
        paper = np.stack([shade+9,shade+5,shade],axis=-1)
        ink = (x%32>9)&(x%32<17)&(y>22)&(y<41)
        page = paper.copy()
        page[ink] = 20
        mask = (x>18)&(x<680)&(y>14)&(y<50)
        restored = source_support.painted_stroke_fill(page,mask,np.zeros_like(mask))
        self.assertIsNotNone(restored)
        np.testing.assert_array_equal(restored[mask & ~ink],paper[mask & ~ink])
        self.assertLess(np.abs(restored[ink & mask].astype(float)-paper[ink & mask]).max(),5)

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
