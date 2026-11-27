import unittest

import cv2
import numpy as np

from tralens.rendering.aggressive import Geometry, adjacent_ellipsis, light_dialogue_frame, painted_balloon_paper


class DialogueEdgeTests(unittest.TestCase):
    def test_caption_band_stays_on_the_printed_side_of_a_rule_and_preserves_other_edits(self):
        from tralens.rendering.aggressive import ruled_caption_band
        own=np.zeros((70,210),bool)
        own[15:50,15:180]=True
        rules=np.zeros_like(own)
        rules[47:50,15:180]=True
        blocked=np.zeros_like(own)
        blocked[35,77]=True
        frame=np.asarray(ruled_caption_band(own,rules,blocked))!=0
        self.assertTrue(frame[35,78])
        self.assertFalse(frame[35,77] or frame[48,90] or frame[17,90])
        rules[30,15:180]=True
        self.assertIsNone(ruled_caption_band(own,rules,blocked))

    def test_original_star_keeps_its_whole_box_where_the_name_overlaps(self):
        raw = [dict(source_id=0,polygon=[(10,10),(74,10),(74,52),(10,52)]),
               dict(source_id=1,polygon=[(56,40),(80,40),(80,68),(56,68)],
                    reason='corroborated_native_line',text='☆')]
        bounds,own,protected=Geometry(raw,(90,80)).local([0],np.zeros((80,90),bool),source_priority=True)
        x,y,*_=bounds
        self.assertTrue(own[45-y,60-x] and protected[45-y,60-x])
        self.assertFalse(protected[45-y,35-x])

    def test_painted_balloon_excludes_its_curved_edges_and_other_white_panels(self):
        plate = np.full((110,180,3),(32,76,120),np.uint8)
        cv2.ellipse(plate,(70,55),(56,37),0,0,360,(255,255,255),-1)
        plate[:,150:] = 255
        own = np.zeros(plate.shape[:2],bool)
        own[41:65,45:90] = True
        blocked = np.zeros_like(own)
        blocked[50,70] = True
        frame = painted_balloon_paper(plate,own,blocked,24)
        self.assertTrue(frame[45,65])
        self.assertFalse(frame[19,70] or frame[50,70] or frame[50,160])
        self.assertIsNone(painted_balloon_paper(np.full_like(plate,255),own,blocked,24))

    def test_small_monochrome_balloon_requires_closed_paper(self):
        plate = np.full((105,135,3),255,np.uint8)
        cv2.ellipse(plate,(67,52),(32,43),0,0,360,(0,0,0),3)
        own = np.zeros(plate.shape[:2],bool)
        own[35:70,56:76] = True
        blocked = np.zeros_like(own)
        mask = painted_balloon_paper(plate,own,blocked,20,closed=True)
        self.assertTrue(mask[52,67])
        self.assertFalse(mask[52,35] or mask[10,10])
        plate[49:54,:] = 255
        self.assertIsNone(painted_balloon_paper(plate,own,blocked,20,closed=True))

    def test_label_extending_above_a_panel_corner_stays_inside_its_gutter(self):
        pixels = np.full((260,220,3),225,np.uint8)
        pixels[45:240,152:156] = 0
        pixels[:,156:] = 255
        raw = [{'source_id':0,'polygon':[(105,40),(158,40),(158,145),(105,145)]}]
        def reaches_gutter():
            bounds,_,mask = Geometry(raw,(220,260),pixels).local([0],np.zeros((260,220),bool))
            return mask[92-bounds[1],162-bounds[0]]
        self.assertTrue(reaches_gutter())
        pixels[:,156:] = 225
        self.assertFalse(reaches_gutter())

    def test_single_pixel_rule_needs_exterior_paper_to_bound_layout(self):
        crop = np.full((180, 160, 3), 224, np.uint8)
        crop[:,40] = 0
        raw = [{'source_id':0, 'polygon':[(48,65),(62,65),(62,110),(48,110)]}]
        def blocked():
            bounds, _, mask = Geometry(raw,(160,180),crop).local([0], np.zeros((180,160),bool))
            return mask[80-bounds[1],36-bounds[0]]
        self.assertFalse(blocked())
        crop[:,:39] = 255
        self.assertTrue(blocked())

    def test_grouped_dialogue_preserves_overlapping_emotion_mark_ink(self):
        pixels = np.full((140,160,3),255,np.uint8)
        pixels[72:76,50:55] = 0
        raw = [dict(source_id=i,polygon=[(x,y),(r,y),(r,b),(x,b)])
               for i,(x,y,r,b) in enumerate([(34,30,69,78),(73,30,106,78),(46,65,61,85)])]
        bounds,own,protected = Geometry(raw,(160,140),pixels).local([0,1],np.zeros((140,160),bool),source_priority=True)
        x,y,*_ = bounds
        self.assertTrue(own[73-y,51-x])
        self.assertTrue(protected[73-y,51-x])
        self.assertFalse(protected[43-y,55-x])

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
