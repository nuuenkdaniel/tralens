import unittest

import cv2
import numpy as np

from tralens.ocr import outlined_sounds, sideways_sound, automatic_units
from tralens.ocr.types import Text_Box, Text_Group
from tralens.rendering import outlined_sound_cleanup, replacement


class OutlinedSoundTests(unittest.TestCase):
    def test_split_reaction_requires_an_empty_gutter_between_aligned_panels(self):
        from PIL import Image,ImageDraw
        from tralens.ocr.display_readings import split_interjections
        from tralens.translation.rendition import reaction_beat
        image=Image.new('RGB',(330,180),'white')
        def glyph(i,x,word):
            return Text_Box(i,.96,[(x,82),(x+18,82),(x+18,102),(x,102)],word)
        right,left=glyph(1,249,'お'),glyph(2,48,'や')
        self.assertEqual(split_interjections(image,[left,right]),[left,right])
        draw=ImageDraw.Draw(image)
        for x in (148,161):draw.line((x,25,x,147),fill='black',width=3)
        annotated=split_interjections(image,[left,right])
        self.assertEqual([b.ocr_metadata['split_interjection']['beat'] for b in annotated],[1,0])
        self.assertNotIn('split_interjection',right.ocr_metadata)
        third=glyph(3,185,'え')
        self.assertEqual(split_interjections(image,[right,left,third]),[right,left,third])
        self.assertEqual(reaction_beat('{"first":"Well","second":"now!"}',1),'now!')
        with self.assertRaises(ValueError):reaction_beat('{"first":"The situation","second":"now!"}',1)

    def test_large_unvoiced_kana_needs_a_voiced_read_and_separately_isolated_strokes(self):
        from PIL import Image
        from unittest.mock import Mock
        from tralens.ocr.display_readings import voiced_motion
        img=np.full((225,240,3),255,np.uint8)
        for a,b in [((65,54),(98,153)),((119,55),(144,116)),((153,144),(164,160))]:
            cv2.line(img,a,b,(0,0,0),10)
        page=Image.fromarray(img)
        box=Text_Box(4,.46,[(48,40),(180,40),(180,178),(48,178)],'く')
        self.assertIsNone(voiced_motion(page,box,Mock(side_effect=['すっ'])))
        self.assertIsNone(voiced_motion(page,box,Mock(side_effect=['ぐっ']*3+['ぐつ'])))
        found=voiced_motion(page,box,Mock(side_effect=['ぐっ']*5))
        self.assertEqual(found.text,'ぐっ')
        self.assertGreaterEqual(found.ocr_metadata['heavy_print']['motion_radius'],2)

    def test_clipped_display_word_requires_optical_support_for_its_punctuation(self):
        from PIL import Image
        from unittest.mock import Mock
        from tralens.ocr.display_readings import clipped_exclamation
        img=np.full((290,220,3),255,np.uint8)
        for x,y in [(90,54),(71,94),(78,134)]:
            cv2.line(img,(x,y),(x+38,y+12),(0,0,0),7)
        for x in (83,104):
            cv2.line(img,(x+3,176),(x,214),(0,0,0),6)
            cv2.circle(img,(x,227),4,(0,0,0),-1)
        box=Text_Box(7,.49,[(35,30),(185,30),(185,250),(35,250)],'わかっ')
        page=Image.fromarray(img)
        self.assertIsNone(clipped_exclamation(page,box,Mock(side_effect=['わかった!!','?'])))
        recovered=clipped_exclamation(page,box,Mock(side_effect=['わかった!!','!!']+['わかった!!']*6))
        self.assertEqual(recovered.text,'わかった!!')
        self.assertGreater(recovered.bbox[0],box.bbox[0])
        self.assertLess(recovered.bbox[2],box.bbox[2])

    def test_blank_repeat_needs_two_independent_beats_and_preserves_edge_connected_art(self):
        from PIL import Image
        from unittest.mock import Mock
        from tralens.rendering.printed_contours import outlined_word_mask
        gray=np.full((270,210),224,np.uint8)
        for x,y in [(93,38),(108,83),(75,143),(83,180)]:
            cv2.ellipse(gray,(x,y),(19,14),0,0,360,0,3)
            cv2.ellipse(gray,(x,y),(16,11),0,0,360,255,-1)
        cv2.line(gray,(14,0),(22,269),0,3)
        page=Image.fromarray(gray).convert('RGB')
        original=Text_Box(2,0,[(10,10),(183,10),(183,249),(10,249)],'')
        original.reason='blank_paddle'
        part=Text_Box(3,.9,[(80,31),(106,31),(106,54),(80,54)],'はら')
        part.reason='frontier_discovery'
        def recover(*answers):
            return outlined_sounds.read_outlined_words(page,[original,part],Mock(side_effect=answers))
        self.assertEqual(recover('はらはら','はらはら','はろ','はろ','はろ'),[original,part])
        member,=recover('はらはら','はらはら','はら','はら',*['はらはら']*3)
        self.assertEqual(member.text,'はらはら')
        self.assertEqual(member.ocr_metadata['contained_discoveries'][0]['source_id'],3)
        guard=np.zeros_like(gray,dtype=bool)
        guard[35:40,91:97]=True
        mask=outlined_word_mask(np.asarray(page),[0,0,210,270],member.ocr_metadata,guard)
        self.assertTrue(mask[43,91])
        self.assertFalse(mask[:,:25].any())
        self.assertFalse((mask & guard).any())

    def test_thick_outline_reading_at_page_edge_excludes_thin_keys(self):
        from PIL import Image
        from unittest.mock import Mock
        from tralens.ocr.masked_lines import source_views
        from tralens.rendering.printed_contours import outlined_word_mask
        paper=np.full((150,340,3),255,np.uint8)
        for x in (38,90,142,194,246):
            cv2.line(paper,(x,55),(x+25,112),(0,0,0),20)
            cv2.line(paper,(x,55),(x+25,112),(255,255,255),9)
        cv2.rectangle(paper,(46,129),(78,144),(0,0,0),2)
        page=Image.fromarray(paper)
        box=Text_Box(5,.76,[(20,20),(300,20),(300,150),(20,150)],'まさか！')
        self.assertEqual(source_views(page,[box.poly])[0].height,130)
        self.assertIsNone(outlined_sounds.outlined_display(page,box,Mock(side_effect=['まさき!'])))
        member=outlined_sounds.outlined_display(page,box,Mock(return_value='まさか!'))
        self.assertEqual(member.text,'まさか！')
        mask=outlined_word_mask(paper,[0,0,340,150],member.ocr_metadata,np.zeros(paper.shape[:2],bool))
        self.assertTrue(mask[55,38])
        self.assertFalse(mask[128:145].any())

    def test_only_distinct_surviving_pen_lines_trigger_structural_repair(self):
        from tralens.rendering.printed_contours import interrupted_ink
        page=np.full((100,140,3),255,np.uint8)
        mask=np.zeros(page.shape[:2],bool)
        mask[15:85,50:90]=True
        guard=np.zeros_like(mask)
        for y in (30,50):page[y,22:65]=0
        self.assertFalse(interrupted_ink(page,mask,guard))
        page[70,29:65]=0
        self.assertTrue(interrupted_ink(page,mask,guard))
        guard[70]=True
        self.assertFalse(interrupted_ink(page,mask,guard))

    def test_blank_outlined_effect_requires_both_halves_as_well_as_the_whole(self):
        from PIL import Image
        from unittest.mock import Mock
        page=np.full((155,225,3),255,np.uint8)
        for x,y in ((40,45),(79,65),(122,35),(154,64)):
            cv2.line(page,(x,y),(x+18,y+45),(0,0,0),17)
            cv2.line(page,(x,y),(x+18,y+45),(255,255,255),8)
        page=Image.fromarray(page)
        box=Text_Box(1,0,[(18,18),(199,18),(199,132),(18,132)],'')
        box.reason='blank_paddle'
        def read(*words):
            return outlined_sounds.outlined_display(page,box,Mock(side_effect=words))
        self.assertIsNone(read('ギュ','ギュ','ギ','二','ギ','二','ギ','二'))
        recovered=read('ギュ','ギュ','ギ','ュ','ギュ','ギュ','ギュ')
        self.assertEqual(recovered.text,'ギュ')

    def test_apparent_initials_need_matching_native_and_isolated_kana(self):
        from PIL import Image
        from unittest.mock import Mock
        page=np.full((145,185,3),255,np.uint8)
        for x,y in [(40,48),(75,64),(111,37)]:
            cv2.line(page,(x,y),(x+16,y+41),(0,0,0),13)
            cv2.line(page,(x,y),(x+16,y+41),(255,255,255),6)
        box=Text_Box(1,.43,[[29,24],[144,24],[144,123],[29,123]],'IT')
        recover=lambda *reads: outlined_sounds.hollow_print(Image.fromarray(page),box,Mock(side_effect=reads))
        self.assertIsNone(recover('ピカ','ピカ','ピア'))
        self.assertEqual(recover(*['ピカ']*4).selected_text,'ピカ')
        self.assertIsNone(recover('IT'))

    def test_outlined_word_keeps_crossing_art_and_needs_three_dots_and_matching_readings(self):
        from PIL import Image
        from unittest.mock import Mock
        from tralens.rendering.printed_contours import outlined_word_mask

        gray=np.full((260,380),255,np.uint8)
        cv2.line(gray,(0,143),(379,96),0,1)
        for start,end in [((45,85),(92,162)),((102,90),(159,109)),
                          ((110,149),(186,144)),((202,142),(227,100))]:
            cv2.line(gray,start,end,0,19)
            cv2.line(gray,start,end,255,8)
        for x in (242,265,288):
            cv2.circle(gray,(x,148),10,0,-1)
            cv2.circle(gray,(x,148),5,255,-1)
        box=Text_Box(3,.41,[[37,76],[295,76],[295,175],[37,175]],'父')
        def read(*words):
            return outlined_sounds.read_outlined_words(Image.fromarray(gray).convert('RGB'),
                [box],Mock(side_effect=words))[0]
        self.assertIs(read('まて...','まだ...'),box)
        revised=read('まて...','まて...')
        self.assertEqual(revised.text,'まて...')
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        protected=np.zeros(gray.shape,bool)
        protected[145:150,261:268]=True
        mask=outlined_word_mask(rgb,[0,0,380,260],revised.ocr_metadata,protected)
        self.assertTrue(mask[148,242])
        self.assertFalse(mask[148,265])
        self.assertFalse(mask[125,145])
        cv2.circle(gray,(288,148),11,255,-1)
        self.assertIs(read(),box)

    def test_large_digit_requires_both_ends_of_a_descending_effect(self):
        from unittest.mock import Mock
        from PIL import Image
        from tralens.ocr.display_readings import diagonal_effects
        page=Image.new('RGB',(240,220),'white')
        box=Text_Box(0,.57,[[21,30],[209,30],[209,182],[21,182]],'7')
        def reread(end):
            return diagonal_effects(page,[box],Mock(side_effect=['キィィッ']*3+['キ',end]))[0]
        self.assertEqual(reread('ッ').text,'キィィッ')
        self.assertIs(reread('フ'),box)
    def test_split_color_title_needs_three_matching_readings_and_preserves_neighbors(self):
        from PIL import Image, ImageDraw
        from tralens.ocr.display_readings import colored_word
        image = Image.new('RGB', (480, 320), 'white')
        draw = ImageDraw.Draw(image)
        for x in [35, 64, 93, 122, 151]:
            draw.rectangle((x, 40, x+20, 85), fill='orange')
        a = Text_Box(1, .5, [(29,35),(100,35),(100,91),(29,91)], 'う')
        b = Text_Box(2, .5, [(85,35),(177,35),(177,91),(85,91)], 'ら')
        subtitle = Text_Box(3, .96, [(53,12),(153,12),(153,26),(53,26)], '新作')
        boxes = [a, b, subtitle]
        result = colored_word(image, boxes, lambda _: 'うみのいろ')
        self.assertEqual([v.text for v in result], ['うみのいろ', '新作'])
        self.assertIs(result[1], subtitle)
        self.assertEqual(a.text, 'う')
        divergent = iter(['うみのいろ', 'うみいろ', 'うみのいろ'])
        self.assertIsNone(colored_word(image, boxes, lambda _: next(divergent)))

    def test_nameplate_reads_both_names_and_their_separate_pronunciations(self):
        from PIL import Image
        from tralens.ocr.display_readings import nameplates
        def box(i, rect, text):
            l,t,r,b=rect
            return Text_Box(i,.98,[(l,t),(r,t),(r,b),(l,b)],text)
        role=box(0,(106,25,131,151),'研究会会長')
        boxes=[role,box(1,(75,25,101,74),'山田'),box(2,(76,85,101,129),'春菜'),
               box(3,(96,25,109,73),'やまだ'),box(4,(96,86,108,130),'はるな')]
        page=Image.new('RGB',(180,210),'white')
        answers=['研究会会長山田春菜']*2+['やまだ']*2+['はるな']*2
        observed=iter(answers)
        result=nameplates(page,boxes,lambda _:next(observed))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].ocr_metadata['nameplate']['kana'],['やまだ','はるな'])
        observed=iter(answers[:-1]+['はるね'])
        self.assertEqual(nameplates(page,boxes,lambda _:next(observed)),boxes)
        self.assertEqual(nameplates(page,boxes[:-1],lambda _:self.fail('no second ruby')),boxes[:-1])

    def test_alternate_reading_absorbs_only_weak_fragments_in_same_printed_strip(self):
        from PIL import Image
        from types import SimpleNamespace
        from tralens.ocr.display_readings import annotated_passages
        def box(i, bounds, word, confidence):
            l,t,r,b=bounds
            return Text_Box(i,confidence,[(l,t),(r,t),(r,b),(l,b)],word)
        first=box(0,(70,25,95,145),'空飛ぶ',.99)
        base=box(1,(39,25,63,145),'機械',.99)
        side=box(2,(62,42,69,61),'ロ',.45)
        frag=box(3,(62,68,69,84),'久',.32)
        prop=SimpleNamespace(direction='vertical',main_ids=[0,1],member_ids=[0,1,2],
                             annotation_parents=[(2,1)],annotation_ids=[2],crop_bounds=[39,25,95,145])
        def check(other,readings):
            engine=SimpleNamespace(raw_detections=[[first,base,side,other]],image_path='test.png',
                                   automatic_unit_proposals=[SimpleNamespace(proposals=[prop])])
            answers=iter(readings)
            from unittest.mock import patch
            with patch('tralens.ocr.automatic_units.propose_page_from_boxes'):
                annotated_passages(engine,Image.new('RGB',(180,180),'white'),lambda _:next(answers))
            return engine.raw_detections[0]
        words=['空飛ぶ機械']*2+['ロボット']*2
        merged=check(frag,words)
        self.assertEqual([v.text for v in merged],['空飛ぶ機械'])
        self.assertEqual(merged[0].ocr_metadata['printed_annotation']['reading'],'ロボット')
        self.assertEqual(len(check(frag,words[:-1]+['ロケット'])),4)
        self.assertEqual(len(check(box(4,(75,53,87,72),'人',.98),[])),4)

    def test_small_terminal_kana_requires_threshold_agreement_and_bound_pixels(self):
        from PIL import Image, ImageDraw
        from tralens.ocr.handwritten_lines import terminal_kana
        from tralens.rendering.aggressive import terminal_kana_mask
        page = Image.new('RGB', (150, 190), 'white')
        draw = ImageDraw.Draw(page)
        draw.arc((44,98,57,111), -150, 110, fill='black', width=3)
        draw.line((45,126,49,130), fill='black', width=2)
        column = Text_Box(1, .4, [(40,20),(90,20),(90,150),(40,150)], 'よっ')
        self.assertIsNone(terminal_kana(page, column, lambda _: 'っ'))
        proof = terminal_kana(page, column, lambda crop: 'っ' if crop.width>12 else 'し')
        self.assertIsNotNone(proof)
        raw = {'text': column.text, 'ocr_metadata': {'terminal_kana': proof}}
        mask = terminal_kana_mask(np.asarray(page), (0,0,*page.size), raw)
        self.assertGreater(mask.sum(), 10)
        self.assertFalse(mask[125:].any())
        reads = iter(['っ', 'し', 'し'])
        self.assertIsNone(terminal_kana(page, column, lambda _: next(reads)))
        draw.point((50,106), fill='red')
        self.assertIsNone(terminal_kana_mask(np.asarray(page), (0,0,*page.size), raw))

    def test_repeated_large_type_requires_matching_beats_in_separate_footprints(self):
        import json
        from tralens.ocr import repeated_effects
        from PIL import Image,ImageDraw
        page = Image.new('RGB',(320,320),'white')
        draw = ImageDraw.Draw(page)
        boxes=[]
        for i,x in enumerate((30,125)):
            box=Text_Box(i,.22,[(x,40),(x+90,40),(x+90,230),(x,230)],'')
            boxes.append(box)
            draw.rectangle((x+20,70,x+31,210),fill='black')
        reads=iter(['バタバタッ']*3+['バタッ']*3)
        pairs=repeated_effects.read_pairs(page,boxes,lambda chip:next(reads))
        self.assertEqual([pairs[i].text for i in (0,1)],['バタバタッ','バタッ'])
        lines=[dict(source_id=i,reason=b.reason,polygon=b.poly.tolist(),text=b.text,
                    metadata_json=json.dumps(b.ocr_metadata)) for i,b in pairs.items()]
        self.assertTrue(repeated_effects.bound_pair(lines,page))
        raw = [dict(r,ocr_metadata=pairs[r['source_id']].ocr_metadata) for r in lines]
        self.assertTrue(repeated_effects.bound_pair(lines[:1],page,raw))
        self.assertFalse(repeated_effects.bound_pair(lines[:1],page,raw[:1]))
        draw.point((90,80),fill='black')
        self.assertFalse(repeated_effects.bound_pair(lines,page))
        reads=iter(['バタバタッ']*3+['ドキッ']*3)
        self.assertFalse(repeated_effects.read_pairs(page,boxes,lambda chip:next(reads)))

    def test_curving_sound_requires_a_repeated_suffix_in_each_bound_view(self):
        from PIL import Image
        from tralens.ocr.repeated_effects import curved_repeats
        image = Image.new('RGB',(250,190),'white')
        head = Text_Box(5,.75,[(15,18),(155,28),(149,79),(10,69)],'ヒュルル')
        tail = Text_Box(7,.46,[(130,56),(167,49),(189,149),(152,156)],'22')
        readings = iter(['ルル','ヒュルルッ']+['ヒュルルルル']*3)
        result = curved_repeats(image,[head,tail],lambda _: next(readings))
        self.assertEqual(result[5].ocr_metadata['curved_repeat']['ids'],[5,7])
        self.assertNotIn('curved_repeat',head.ocr_metadata)
        readings = iter(['ルル','ヒュルルッ','ヒュルルルル','ヒュルルッ'])
        self.assertFalse(curved_repeats(image,[head,tail],lambda _:next(readings)))
        tail.text = '12'
        self.assertFalse(curved_repeats(image,[head,tail],lambda _:self.fail('unrelated number')))

    def test_twin_requires_separate_matching_print_and_full_crop_agreement(self):
        from PIL import Image,ImageDraw
        from tralens.ocr.repeated_effects import printed_twins
        im = Image.new('RGB',(270,180),'white')
        draw = ImageDraw.Draw(im)
        for x in (30,150):
            for y in (26,34,43):
                draw.line((x,y,x+14,y+4),fill='black',width=3)
            draw.line((x+8,53,x+11,78),fill='black',width=4)
            draw.line((x+8,83,x+26,91),fill='black',width=4)
        seed=Text_Box(3,.58,[(27,23),(48,23),(48,51),(27,51)],'ジ')
        paired=printed_twins(im,[seed],lambda _: 'ジーン')
        self.assertEqual({i:b.text for i,b in paired.items()},{3:'ジーン',4:'ジーン'})
        known = paired[4]
        self.assertEqual(set(printed_twins(im,[seed,known],lambda _: 'ジーン')),{3})
        self.assertFalse(printed_twins(im,[seed],lambda _: 'キーン'))
        draw.rectangle((120,0,265,170),fill='white')
        self.assertFalse(printed_twins(im,[seed],lambda _:self.fail('no second glyph')))

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

    def test_large_blank_detection_uses_reader_pixel_budget(self):
        from PIL import Image, ImageDraw
        page = Image.new('RGB',(850,620),'white')
        ImageDraw.Draw(page).line((70,80,425,210),fill='black',width=10)
        box = Text_Box(7,0,[(60,65),(470,65),(470,240),(60,240)],'')
        self.assertEqual(outlined_sounds.blank_sound(page,box,lambda view:'ザッ'),'ザッ')
        oversized = Text_Box(8,0,[(0,0),(849,0),(849,619),(0,619)],'')
        self.assertIsNone(outlined_sounds.blank_sound(page,oversized,lambda view:self.fail('oversized')))

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

    def test_separate_initial_kana_needs_a_matching_word_and_clear_column_gap(self):
        pixels = np.full((90,120,3),255,np.uint8)
        first = Text_Box(0,.96,[(57,12),(79,12),(79,34),(57,34)],'な')
        next_word = Text_Box(1,.98,[(30,13),(52,13),(52,72),(30,72)],'なに？')
        groups = [Text_Group([first],direction='unknown'),Text_Group([next_word],direction='vertical')]
        def run():
            return automatic_units.join_stuttered_onsets(groups,pixels,set())
        self.assertEqual([b.box_id for b in run()[0].group],[0,1])
        pixels[:50,54] = 0
        self.assertEqual(run(),groups)
        pixels[:] = 255
        next_word.text = 'だれ？'
        self.assertEqual(run(),groups)

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

    def test_adjective_and_noun_share_aligned_rows_and_clear_paper(self):
        words = ['ここには','大きくて立派な','建物が','あります']
        lines = [Text_Box(i,.99,[(12,y),(r,y),(r,y+18),(12,y+18)],word)
                 for i,(word,y,r) in enumerate(zip(words,[5,26,47,68],[122,102,72,115]))]
        groups = [Text_Group(lines[:2],direction='horizontal'),Text_Group(lines[2:],direction='horizontal')]
        pixels = np.full((105,150,3),255,np.uint8)
        def join():
            return automatic_units.join_horizontal_continuations(groups,pixels,[],{},
                {i:(None,None,i) for i in range(4)},set(),{})
        self.assertEqual([b.box_id for b in join()[0].group],[0,1,2,3])
        pixels[45] = 0
        self.assertEqual(join(),groups)

    def test_causal_quote_requires_a_matching_close_and_clear_seam(self):
        words = ['「今日は','休みなので','家に','います」']
        lines = [Text_Box(i,.96,[(16,y),(100,y),(100,y+18),(16,y+18)],word)
                 for i,(word,y) in enumerate(zip(words,[4,26,48,70]))]
        groups = [Text_Group(lines[:2],direction='horizontal'),Text_Group(lines[2:],direction='horizontal')]
        pixels = np.full((100,130,3),255,np.uint8)
        def join():
            return automatic_units.join_horizontal_continuations(groups,pixels,[],{},
                {i:(None,None,i) for i in range(4)},set(),{})
        self.assertEqual(len(join()),1)
        lines[-1].text = 'います』'
        self.assertEqual(join(),groups)

    def test_topic_and_predicate_require_one_balloon(self):
        pixels = np.full((140,180,3),255,np.uint8)
        # Centered short topic over two wider lines in the same balloon.
        lines = [Text_Box(i,.99,[(x,y),(140-x,y),(140-x,y+20),(x,y+20)],word)
                 for i,(x,y,word) in enumerate(((50,10,'明日は'),(20,32,'雨が降ると'),(27,55,'聞きました')))]
        groups = [Text_Group(lines[:1],direction='horizontal'),Text_Group(lines[1:],direction='horizontal')]
        owners = {i:(1,.99) for i in range(3)}
        def join():
            return automatic_units.join_horizontal_continuations(groups,pixels,[],owners,
                {i:(None,None,i) for i in range(3)},set(),{})
        self.assertEqual([box.box_id for box in join()[0].group],[0,1,2])
        owners[2] = (2,.99)
        self.assertEqual(join(),groups)
        owners[2] = (1,.99)
        pixels[31] = 0
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

    def test_overlapping_sounds_commit_only_after_all_new_targets_fit(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from PIL import Image
        from tralens.ocr import crop_readings

        image = Image.new('RGB',(220,120),(183,142,110))
        reading = SimpleNamespace(revision='checked',text='ドンドン',origin=crop_readings.ENGINE)
        unit = SimpleNamespace(selected_revision=reading.revision,readings=[reading])
        raw = [{'source_id':i,'polygon':[(x,55),(x+47,38),(x+54,55),(x+7,72)]}
               for i,x in enumerate((25,90))]
        def trial(fits):
            rows = [dict(status='skipped',reason_code='coverage_uncertain_geometry',
                         binding={'member_ids':[i],'text':'Boom, boom'}) for i in range(2)]
            clean = image.copy()
            occupied = np.zeros((120,220),bool)
            erase = occupied.copy()
            def render(text,font,coverage,frame,obstacle,style,**kwargs):
                if not fits:
                    raise ValueError('no fit')
                self.assertGreater(style['rotation'],0)
                return Image.new('RGBA',(12,12),'black'),frame[:2],{}
            with patch.object(outlined_sounds,'zones',return_value=[(0,0,220,120)]), \
                 patch.object(outlined_sound_cleanup,'source_centered_lettering',return_value=None), \
                 patch.object(replacement,'patch_layout',side_effect=render), \
                 patch.object(outlined_sound_cleanup,'joint_mask',side_effect=lambda rgb,own,*a,**k:own.copy()):
                result,layer,_ = outlined_sound_cleanup.finish(image,clean,Image.new('RGBA',image.size),occupied,
                    erase,raw,[unit,unit],rows,{},None,{},lambda row:(None,None),{'min_size':12,'max_size':20})
            return rows,erase,layer
        rows,erase,layer = trial(False)
        self.assertFalse(erase.any())
        self.assertIsNone(layer.getbbox())
        rows,erase,layer = trial(True)
        self.assertTrue(erase.any())
        self.assertIsNotNone(layer.getbbox())
        self.assertTrue(all(row.get('cleanup_method') == 'outlined_sound_group' for row in rows))

    def test_fully_detected_long_refrain_requires_corroboration_elsewhere_on_page(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from PIL import Image
        from tralens.ocr import crop_readings
        page=Image.new('RGB',(510,360),(175,130,86))
        polys=[[(40,85),(380,85),(380,130),(40,130)],[(180,265),(248,265),(248,298),(180,298)]]
        raw=[dict(source_id=i,polygon=p) for i,p in enumerate(polys)]
        def run(other, reason='coverage_uncertain_geometry'):
            units=[]
            rows=[]
            for i,word in enumerate(('ザワ'*5,other)):
                reading=SimpleNamespace(text=word,revision=str(i),origin=crop_readings.ENGINE)
                units.append(SimpleNamespace(selected_revision=str(i),readings=[reading]))
                rows.append(dict(status='skipped',reason_code=reason if i==0 else 'coverage_uncertain_geometry',
                            binding=dict(member_ids=[i],text='Murmur')))
            mask=np.zeros((page.height,page.width),bool)
            with patch.object(outlined_sounds,'zones',return_value=[]), \
                 patch.object(outlined_sound_cleanup,'source_centered_lettering',
                    return_value=(Image.new('RGBA',(16,12),'black'),(42,27),{})), \
                 patch.object(outlined_sound_cleanup,'joint_mask',
                    side_effect=lambda rgb,own,protected,zone,**kw:own[zone[1]:zone[3],zone[0]:zone[2]].copy()):
                _,_,reports=outlined_sound_cleanup.finish(page,page.copy(),Image.new('RGBA',page.size),
                    mask.copy(),mask,raw,units,rows,{},None,{},lambda row:(None,None),dict(min_size=12,max_size=18))
            return reports
        self.assertEqual(run('ドンドン'),[])
        self.assertEqual([r['source_ids'] for r in run('ザワザワ')],[[0]])
        self.assertEqual(run('ドンドン','uncorroborated_source'),[])
        self.assertEqual([r['source_ids'] for r in run('ザワザワ','uncorroborated_source')],[[0]])

    def test_painted_gutter_bounds_require_continuous_bands(self):
        rgb = np.full((110,150,3),(185,138,95),np.uint8)
        rgb[:,20:28] = 255
        rgb[85:93,:] = 255
        rgb[56:58,:] = 255
        self.assertEqual(replacement.painted_panel_limits(rgb,(48,37,96,50)),[28,0,150,85])

    def test_open_rims_beside_paper_preserve_colored_lines_and_neighbor_ink(self):
        pixels = np.full((100,180,3),(181,123,94),np.uint8)
        pixels[58:] = 255
        for x,color in [(52,(3,3,3)),(73,(60,31,18)),(88,(4,4,4))]:
            pixels[47:72,x:x+6] = color
        mask = np.zeros((100,180),bool)
        mask[42:67,20:30] = True
        own = np.ones_like(mask)
        neighbor = np.zeros_like(mask)
        neighbor[:,88:94] = True
        result = outlined_sound_cleanup.clipped_rims(pixels,mask,own,neighbor)
        self.assertTrue(result[56,54])
        self.assertFalse(result[56,75])
        self.assertFalse(result[56,90])
        mono = np.repeat(pixels[:,:,:1],3,axis=2)
        self.assertTrue(np.array_equal(outlined_sound_cleanup.clipped_rims(mono,mask,own,neighbor),mask))
