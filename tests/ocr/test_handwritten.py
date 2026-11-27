import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from tralens.ocr import handwritten
from tralens.ocr.types import Text_Box


def region(i, text, confidence, x, y, r, b):
    return Text_Box(i,confidence,[(x,y),(r,y),(r,b),(x,b)],text)


class HandwrittenTests(unittest.TestCase):
    def test_forum_nickname_requires_a_second_printed_occurrence(self):
        from types import SimpleNamespace
        from tralens.ocr.handwritten_lines import forum_post_readings
        page=Image.new('RGB',(640,310),'white')
        boxes=[region(0,'Fedora日本語フォーラム',.9,20,10,325,34),
               region(1,'ふぇ、とら終了',.7,70,63,288,97),
               region(2,'オケwwww',.75,70,95,370,139),
               region(3,'朝からふぇど動か',.7,70,141,388,181),
               region(4,'どうしよwww',.7,70,184,356,229)]
        groups=[SimpleNamespace(direction='horizontal',member_ids=ids,main_ids=ids,annotation_ids=())
                for ids in ((0,),(1,),(2,3,4))]
        def read(other):
            words=['■ふぇどら終る','□ふぇどら終る','オケwwww','オケwwww',
                   other,other,'どうしよwww','どうしよwww']
            return forum_post_readings(page,boxes,groups,Mock(side_effect=words))
        self.assertFalse(read('朝から動かない'))
        result=read('朝からふぇどら動かない')
        self.assertEqual(result[1].text,'ふぇどら終了')
        self.assertEqual(result[2].ocr_metadata['forum_post']['product'],'Fedora')

    def test_repeated_technical_passage_requires_a_second_copy_and_matching_suffix(self):
        from types import SimpleNamespace
        from tralens.ocr.handwritten_lines import repeated_printed_passages
        large=['同じ環境で確認','原因はDebian12.3が設','修正の予定があります']
        small=['同じ環境で確認しました','原因はDebian12.3が七','修正の予定があります。']
        boxes=[region(i,text,.8,10,10+48*i,390,47+48*i) for i,text in enumerate(large)]
        boxes.extend(region(3+i,text,.8,420,32+24*i,630,48+24*i) for i,text in enumerate(small))
        groups=[SimpleNamespace(direction='horizontal',member_ids=ids,annotation_ids=()) for ids in ((0,1,2),(3,4,5))]
        page=Image.new('RGB',(660,230),'white')
        full=large[1]+'定ファイルを開けない'
        def read(second):
            return repeated_printed_passages(page,boxes,groups,Mock(side_effect=[large[1],full,second,large[1]]))
        self.assertFalse(read('別の理由'))
        result=read(full)
        self.assertEqual(result[4].text,full)
        self.assertEqual(result[1].text,large[1])
        self.assertEqual(result[0].ocr_metadata['repeated_printed_line']['full'],full)
        self.assertFalse(repeated_printed_passages(page,boxes,groups[:1],Mock(side_effect=AssertionError)))

    def test_decimal_version_needs_the_same_digits_from_native_and_rectified_views(self):
        from types import SimpleNamespace
        from tralens.ocr.handwritten_lines import printed_versions
        page=Image.new('RGB',(350,160),'white')
        rows=[region(4,'環境はDebian12、少',.92,24,28,212,51),region(5,'更新です',.9,24,61,143,83)]
        group=SimpleNamespace(member_ids=(4,5),direction='horizontal',annotation_ids=())
        def trial(last):
            return printed_versions(page,rows,[group],Mock(side_effect=['環境はDebian12.10','環境はDeban12.10',last]))
        self.assertEqual(trial('環境はDebian12.10')[4].text,'環境はDebian12.10')
        self.assertFalse(trial('環境はDebian12.1'))

    def test_wrapped_polite_auxiliary_binds_only_to_matching_adjacent_verb(self):
        from types import SimpleNamespace
        from tralens.ocr.handwritten_lines import polite_continuations
        page=Image.new('RGB',(350,160),'white')
        rows=[region(4,'資料を送付さ',.85,25,28,254,53),region(5,'せて頂',.93,26,58,85,84)]
        groups=[SimpleNamespace(member_ids=(r.box_id,),direction='horizontal',annotation_ids=(),crop_bounds=r.bbox) for r in rows]
        def trial(word):
            return polite_continuations(page,rows,groups,Mock(side_effect=[word]*4+['せて頂']*3))
        self.assertEqual(trial('資料を送付さ')[4].ocr_metadata['masked_passage']['ids'],[4,5])
        self.assertFalse(trial('きれいな青さ'))

    def test_navigation_requires_an_aligned_account_and_complete_menu_labels(self):
        from tralens.ocr.handwritten_lines import forum_header
        page=Image.new('RGB',(440,140),'white')
        menu=region(1,'ホーム検索設廷',.7,12,12,322,34)
        field=region(2,'ユ-サ名:atelier',.76,14,36,198,57)
        text='ホーム検索設定'
        label='ユーザー名:atelier'
        def read(menu_text,account):
            return forum_header(page,[menu,field],Mock(side_effect=[account]*3+[menu_text]*3))
        rows=read(text,label)
        self.assertEqual(rows[1].ocr_metadata['navigation_items'],['ホーム','検索','設定'])
        self.assertEqual(rows[2].text,label)
        self.assertFalse(read('ホーム検索の設定',label))
        self.assertFalse(read(text,'ユーザー名:another'))

    def test_katakana_heading_keeps_separate_membership_only_when_crops_confirm(self):
        from types import SimpleNamespace
        from tralens.ocr import handwritten_lines as lines
        page=Image.new('RGB',(240,160),'white')
        top=region(1,'力夕口',.72,20,12,190,55)
        lower=region(2,'説明です',.89,24,61,148,82)
        group=SimpleNamespace(member_ids=(1,2),direction='horizontal',annotation_ids=(),annotation_parents=())
        def check(views):
            engine=SimpleNamespace(raw_detections=[[top,lower]],automatic_unit_proposals=[SimpleNamespace(proposals=[group])],image_path='page')
            with patch.object(lines,'propose_page_from_boxes',return_value=None):
                lines.printed_fields(engine,page,Mock(side_effect=views))
            return engine.raw_detections[0][0].text
        self.assertEqual(check(['カタロ','カタロ','カタロ','カタ']), 'カタロ')
        self.assertEqual(check(['カタロ','カタ','カタ','カタロ']), '力夕口')

    def test_grammatically_linked_columns_must_share_paper_and_agree_on_the_whole_line(self):
        from types import SimpleNamespace
        from tralens.ocr import handwritten_lines
        def attempt(divider,readings):
            img=np.full((550,550,3),255,np.uint8)
            cv2.rectangle(img,(90,90),(165,181),(0,0,0),3)
            if divider:
                cv2.line(img,(129,91),(129,180),(0,0,0),2)
            cols=[region(2,'早く',.96,133,110,151,157),region(3,'来て',.99,106,109,126,157)]
            proposals=[SimpleNamespace(main_ids=(v.box_id,),member_ids=(v.box_id,),proposal_id=v.box_id) for v in cols]
            engine=SimpleNamespace(raw_detections=[cols],automatic_unit_proposals=[SimpleNamespace(proposals=proposals)],image_path='test')
            with patch.object(handwritten_lines,'propose_page_from_boxes'):
                handwritten_lines.printed_balloon_columns(engine,Image.fromarray(img),Mock(side_effect=readings))
            return cols[0].ocr_metadata.get('corroborated_columns')
        self.assertEqual(attempt(False,['早く来て']*2),{'ids':[2,3],'reading':'早く来て'})
        self.assertIsNone(attempt(True,[]))
        self.assertIsNone(attempt(False,['早く来て','早く見て']))

    def test_wrapped_auxiliary_needs_its_matching_columns_and_an_empty_gap(self):
        from types import SimpleNamespace
        from tralens.ocr.handwritten_lines import wrapped_auxiliaries
        lead=region(1,'なぜ来て',.94,120,30,145,176)
        second=region(2,'くれま',.68,97,116,120,177)
        last=region(3,'せんか',.75,72,116,95,175)
        props=[SimpleNamespace(member_ids=(1,),main_ids=(1,),direction='vertical'),
               SimpleNamespace(member_ids=(2,3),main_ids=(2,3),direction='vertical',crop_bounds=(72,116,120,177))]
        rows=[lead,second,last]
        whole='なぜ来てくれませんか'
        def read(words,extra=()):
            page=Image.new('RGB',(210,240),'white')
            return wrapped_auxiliaries(page,rows+list(extra),props,Mock(side_effect=words))
        self.assertEqual([v.text for v in read([whole]*3+['なぜ来て']*3+['くれま']*3+['せんか']*3).values()],
                         ['なぜ来て','くれま','せんか'])
        self.assertFalse(read([whole,whole,'なぜ来ているの']))
        self.assertFalse(read([], [region(4,'違う',.95,114,96,127,119)]))

    def test_a_digit_beside_a_question_must_read_as_punctuation_by_itself(self):
        from types import SimpleNamespace
        from tralens.ocr.handwritten_lines import printed_balloon_columns
        img=np.full((500,500,3),255,np.uint8)
        cv2.rectangle(img,(90,85),(200,202),(0,0,0),3)
        def attempt(mark):
            cols=[region(1,'本当で',.98,155,110,178,173),region(2,'すか',.92,129,110,151,170),
                  region(3,'2',.83,106,129,125,161)]
            props=[SimpleNamespace(main_ids=(v.box_id,),member_ids=(v.box_id,),proposal_id=v.box_id) for v in cols]
            engine=SimpleNamespace(raw_detections=[cols],automatic_unit_proposals=[SimpleNamespace(proposals=props)],image_path='page')
            with patch('tralens.ocr.handwritten_lines.propose_page_from_boxes'):
                printed_balloon_columns(engine,Image.fromarray(img),Mock(side_effect=['本当ですか?']*3+[mark]*3))
            return cols[0].ocr_metadata.get('corroborated_columns')
        self.assertIsNone(attempt('2'))
        self.assertEqual(attempt('?')['ids'],[1,2,3])

    def test_long_vowel_expansion_requires_a_hollow_stroke_crossing_the_title_edge(self):
        from tralens.ocr.handwritten_lines import outlined_vowel
        box = region(2,'タイトル',.96,15,25,105,55)
        pixels = np.full((80,160,3),255,np.uint8)
        cv2.rectangle(pixels,(100,39),(115,46),(0,0,0),1)
        result = outlined_vowel(Image.fromarray(pixels),box)
        self.assertIsNotNone(result)
        self.assertTrue(115 <= result[1,0] <= 119)
        cv2.rectangle(pixels,(100,39),(115,46),(0,0,0),-1)
        self.assertIsNone(outlined_vowel(Image.fromarray(pixels),box))

    def test_small_separate_kana_require_independent_transcriptions(self):
        pixels = np.full((230,160,3),255,np.uint8)
        parent = region(0,'発表',.9,25,30,105,173)
        for x,y in ((40,160),(55,164),(70,168)):
            cv2.polylines(pixels,[np.int32([[x+9,y],[x,y+7],[x+7,y+14]])],False,(0,0,0),2)
        detector = Mock()
        detector.predict.return_value = [{'rec_texts':['くくく'],'rec_scores':[.91]}]
        image = Image.fromarray(pixels)
        self.assertIsNotNone(handwritten.trailing_kana(image,parent,Mock(return_value='くくく'),detector))
        self.assertIsNone(handwritten.trailing_kana(image,parent,Mock(return_value='へへへ'),detector))

    def test_location_suffix_needs_two_group_views_and_a_separate_source_read(self):
        from types import SimpleNamespace
        from tralens.ocr import handwritten_lines as lines
        img = Image.new('RGB',(200,180),'white')
        role = region(3,'メン八',.65,40,30,114,50)
        location = region(4,'From：地',.76,40,54,150,80)
        group = SimpleNamespace(member_ids=(3,4),annotation_ids=(),annotation_parents=(),direction='horizontal')
        def attempt(last):
            engine = SimpleNamespace(raw_detections=[[role,location]],image_path='page',
                                     automatic_unit_proposals=[SimpleNamespace(proposals=[group])])
            reader = Mock(side_effect=['メンバーFrom:大阪','メンバーFrom:大阪','メンバFrom:',
                                       *['メンバ']*3,last])
            with patch.object(lines,'propose_page_from_boxes',return_value=None):
                lines.printed_fields(engine,img,reader)
            return [v.text for v in engine.raw_detections[0]]
        self.assertEqual(attempt('メンバFrom:大阪'),['メンバ','From:大阪'])
        self.assertEqual(attempt('メンバFrom:大分'),['メン八','From：地'])

    def test_bilingual_gloss_requires_an_actual_printed_annotation_and_quote(self):
        from types import SimpleNamespace
        from tralens.ocr import handwritten_lines as lines
        img = Image.new('RGB',(240,170),'white')
        small = region(3,'Stay curious',.99,48,29,147,44)
        large = region(4,'「好奇心を持て」',.99,30,46,174,87)
        p = SimpleNamespace(member_ids=(4,3),direction='horizontal',annotation_ids=(3,),annotation_parents=((3,4),))
        def check(full):
            engine = SimpleNamespace(raw_detections=[[small,large]],automatic_unit_proposals=[SimpleNamespace(proposals=[p])],image_path='page')
            with patch.object(lines,'propose_page_from_boxes',return_value=None):
                lines.printed_fields(engine,img,Mock(side_effect=[full]*3+['Stay curious']*3))
            return engine.raw_detections[0][1].ocr_metadata.get('printed_gloss')
        self.assertEqual(check('Staycurious「好奇心を持て」')['text'],'Stay curious')
        self.assertIsNone(check('Staycurious「誰ですか」'))

    def test_zoom_recovery_needs_a_matching_head_and_grammatical_continuation(self):
        from tralens.ocr import handwritten_lines
        image = Image.new('RGB',(350,250),'white')
        anchor = region(1,'歩きながら',.81,30,40,190,90)
        def line(y,text):
            return np.float32([(13,y),(173,y),(173,y+40),(13,y+40)]),text,.91
        head,body,end = line(17,'歩きながら'),line(53,'話さない'),line(89,'でーっ')
        with patch.object(handwritten_lines,'lines',side_effect=[[head,body,end],[head,body,end]]):
            self.assertEqual(len(handwritten_lines.continuation(image,anchor,Mock(),Mock())),3)
        unmatched = line(89,'掲示板')
        with patch.object(handwritten_lines,'lines',side_effect=[[head,body,unmatched],[head,body,unmatched]]):
            self.assertIsNone(handwritten_lines.continuation(image,anchor,Mock(),Mock()))
    def test_tier_columns_need_agreeing_full_reads_in_right_to_left_order(self):
        from tralens.ocr.handwritten_lines import corroborate_columns
        image = Image.new('RGB',(260,300),'white')
        boxes = [region(i,t,.9,x,42,x+35,135) for i,(x,t) in enumerate(
            [(150,'昨夜'),(120,'家に'),(88,'戻った'),(20,'明日')])]
        first = boxes[0]
        first.ocr_metadata = {'tier_split':{}}
        self.assertIsNone(corroborate_columns(image,first,boxes[1:],Mock(side_effect=['昨夜家に戻った','昨夜学校に戻った'])))
        self.assertEqual(corroborate_columns(image,first,boxes[1:],Mock(return_value='昨夜家に戻った')),
                         {'ids':[0,1,2],'reading':'昨夜家に戻った'})

    def test_punctuated_column_needs_visible_dots_and_a_separate_read(self):
        canvas = np.full((150,180,3),255,np.uint8)
        blank = region(1,'',0,77,32,111,112)
        text = region(2,'いた',.71,54,30,84,116)
        for y in range(43,104,12):
            cv2.circle(canvas,(96,y),2,(0,0,0),-1)
        for y in (51,75,98):
            cv2.line(canvas,(59,y),(76,y-4),(0,0,0),2)
        def attempt(words):
            return handwritten.punctuated_column(Image.fromarray(canvas),[blank,text],blank,Mock(side_effect=words))
        self.assertEqual(attempt(['…いたよ','いたよ'])[0],'…いたよ')
        self.assertIsNone(attempt(['…いたよ','いた']))
        canvas[:,93:100] = 255
        self.assertIsNone(attempt([]))

    def test_clipped_ending_must_match_full_ink_and_avoid_other_print(self):
        parent=region(1,'見てL',.5,50,40,170,120)
        pixels=np.full((260,260,3),255,np.uint8)
        for x in (84,109):
            cv2.polylines(pixels,[np.int32([[x,133],[x+13,141],[x+2,155]])],False,(0,0,0),3)
        image=Image.fromarray(pixels)
        def run(neighbors,full):
            return handwritten.clipped_handwriting(image,parent,[parent,*neighbors],
                Mock(side_effect=['見てい','ると',full,full,full]))
        self.assertEqual(run([],'見ていると').text,'見ていると')
        self.assertIsNone(run([],'見ているよ'))
        self.assertIsNone(run([region(2,'別',.99,80,129,140,159)],'見ていると'))

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
