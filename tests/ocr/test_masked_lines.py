import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from tralens.ocr import masked_lines
from tralens.ocr.types import Text_Box


def rectangle(x,y,w,h):
    return np.float32([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])


class MaskedLinesTests(unittest.TestCase):
    def test_handwritten_repeat_needs_the_same_opening_beat_when_read_separately(self):
        from PIL import ImageDraw
        page=Image.new('RGB',(240,230),'white')
        draw=ImageDraw.Draw(page)
        for i in range(4):
            draw.rectangle((84+i*19,65+i*26,94+i*19,78+i*26),fill='black')
        probability=np.uint8(np.asarray(page.convert('L'))<128)*255
        head=Text_Box(0,.96,rectangle(80,60,40,42),'ま')
        def read(beat):
            return masked_lines.handwritten_aside(page,probability,head,[head],
                Mock(side_effect=['まごまごっ','まごまごっ',beat,'まごまごっ']))
        self.assertIsNone(read('まよ'))
        result=read('まご')
        self.assertEqual(result.text,'まごまごっ')
        self.assertTrue(result.ocr_metadata['handwritten_aside']['touching'])
        from tralens.rendering.printed_contours import diagonal_aside_mask
        detection={'text':result.text,'ocr_metadata':result.ocr_metadata}
        protected=np.zeros(probability.shape,bool)
        def removal():
            return diagonal_aside_mask(np.asarray(page),(0,0,*page.size),probability,detection,protected)
        original=removal()
        edge=next(zip(*np.nonzero(original & (probability==0))))
        protected[edge]=True
        clipped=removal()
        self.assertEqual(clipped.sum(),original.sum()-1)
        self.assertFalse(clipped[edge])
        protected[next(zip(*np.nonzero(original & (probability>0))))]=True
        self.assertIsNone(removal())

    def test_alias_keeps_the_original_ornament_only_if_each_side_of_the_gap_reads_back(self):
        from PIL import ImageDraw
        from tralens.ocr.display_readings import printed_alias
        page=Image.new('RGB',(180,150),'white')
        draw=ImageDraw.Draw(page)
        draw.rectangle((61,87,78,100),fill='black')
        draw.rectangle((90,87,100,100),fill='black')
        rows=[Text_Box(2,.9,rectangle(68,25,33,15),'AKA'),
              Text_Box(3,.7,rectangle(35,45,95,29),'ルメル'),
              Text_Box(4,.9,rectangle(56,80,53,24),'さま☆')]
        readings=['AKA']*3+['ルミエル']*3+['さま☆']*3+['AKAルミエルさま☆']*3
        result=printed_alias(page,rows,Mock(side_effect=readings+['☆']*3+['さま']*3),9)
        self.assertEqual([(r.box_id,r.text) for r in result],[(2,'AKA'),(3,'ルミエル'),(4,'さま'),(9,'☆')])
        self.assertEqual(result[0].ocr_metadata['masked_passage']['ids'],[2,3,4])
        self.assertLess(result[-2].bbox[2],result[-1].bbox[0])
        kept=printed_alias(page,rows,Mock(side_effect=readings+['メ']),9)
        self.assertEqual(kept[-1].text,'さま☆')

    def test_missing_column_needs_agreement_in_isolation_and_in_the_whole_phrase(self):
        from tralens.ocr.handwritten_lines import missing_middle_column
        page = Image.new('RGB',(150,140),'white')
        right = Text_Box(2,.93,rectangle(75,20,30,86),'私の')
        left = Text_Box(1,.71,rectangle(30,23,30,75),'だ…')
        extra = Text_Box(9,0,rectangle(65,26,8,18),'友')
        extra.reason = 'frontier_discovery'
        for boxes in ([right,left],[right,left,extra]):
            self.assertEqual(missing_middle_column(page,boxes,Mock(return_value='友達')),boxes)
            reads = Mock(side_effect=['友達']*3+['私の友達だ...']*3)
            result = missing_middle_column(page,boxes,reads)
            center = next(v for v in result if v.text=='友達')
            self.assertEqual(result[0].ocr_metadata['masked_passage']['ids'],[2,center.box_id,1])
            if extra in boxes:
                self.assertEqual(center.box_id,extra.box_id)
        extra.reason = 'paddle_confidence_accepted'
        boxes = [right,left,extra]
        self.assertEqual(missing_middle_column(page,boxes,Mock(side_effect=AssertionError)),boxes)

    def test_small_label_reads_each_row_and_the_title_together(self):
        from tralens.ocr.handwritten_lines import small_stacked_labels
        page = Image.new('RGB',(250,150),'white')
        head = Text_Box(0,.7,rectangle(30,35,80,26),'かぜの')
        cells = [Text_Box(i+1,.7,rectangle(30+27*i,61,26,20),word) for i,word in enumerate('資料集')]
        boxes = [head,*cells]
        def read(final):
            return small_stacked_labels(page,boxes,Mock(side_effect=['かぜの']*4+['資料集']*4+[final]*3))
        self.assertEqual(read('無関係'),boxes)
        rows = read('かぜの資料集')
        self.assertEqual([r.text for r in rows],['かぜの','資料集'])
        self.assertEqual([c['id'] for c in rows[1].ocr_metadata['label_fragments']],[1,2,3])

    def test_short_column_must_be_a_possessive_read_independently_in_the_same_order(self):
        from tralens.ocr.handwritten_lines import possessive_column
        parent = Text_Box(3,.7,rectangle(70,20,28,75),'友達に')
        child = Text_Box(7,.6,rectangle(100,35,17,39),'ぽく')
        page = Image.new('RGB',(180,120),'white')
        def run(readings):
            return possessive_column(page,parent,child,Mock(side_effect=readings))
        self.assertIsNone(run(['とも']))
        self.assertIsNone(run(['ぼくの','ぼくの','ぼくの','友達に','友達にぼくの','ぼくの友達に','ぼくの友達に']))
        corrected = run(['ぼくの']*3 + ['友達に'] + ['ぼくの友達に']*3)
        self.assertEqual(corrected[0].ocr_metadata['masked_passage']['reading'],'ぼくの友達に')
        self.assertEqual([row.reason for row in corrected],['corroborated_native_line']*2)

    def test_split_repeat_requires_two_complete_peers_and_matches_the_new_ink(self):
        from tralens.ocr.display_readings import clipped_repetitions
        page = Image.new('RGB',(300,170),'white')
        first = Text_Box(0,.7,rectangle(10,60,20,25),'大')
        second = Text_Box(1,.8,rectangle(32,61,21,24),'変')
        peers = [Text_Box(i,.9,rectangle(150,25*i,80,20),'大変だ') for i in (2,3)]
        boxes = [first,second,*peers]
        self.assertEqual(clipped_repetitions(page,boxes[:-1],Mock(side_effect=AssertionError)),boxes[:-1])
        self.assertEqual(clipped_repetitions(page,boxes,Mock(return_value='別の語')),boxes)
        readings = Mock(side_effect=['大変だ']*3+['変だ']*3+['大'])
        result = clipped_repetitions(page,boxes,readings)
        self.assertEqual(result[0].ocr_metadata['masked_passage']['ids'],[0,1])
        self.assertEqual(result[1].text,'変だ')

    def test_repeated_small_kana_need_a_second_native_inscription(self):
        from PIL import ImageDraw
        page=Image.new('RGB',(160,90),'white')
        draw=ImageDraw.Draw(page)
        for x in (14,88):
            draw.arc((x,20,x+17,34),10,290,fill='black',width=3)
            draw.arc((x+6,38,x+24,51),20,300,fill='black',width=3)
        prob=np.uint8(np.asarray(page.convert('L'))<180)*255
        seed=Text_Box(4,.51,rectangle(12,18,28,36),'ぽう')
        out=masked_lines.nearby_repeated_kana(page,prob,[seed],Mock(return_value='ぽろ'))
        self.assertEqual([(v.box_id,v.text) for v in out],[(4,'ぽろ'),(5,'ぽろ')])
        self.assertEqual(out[0].ocr_metadata['repeated_kana']['peer_digest'],out[1].ocr_metadata['repeated_kana']['source_digest'])
        prob[:,70:]=0
        self.assertEqual(masked_lines.nearby_repeated_kana(page,prob,[seed],Mock(return_value='ぽろ')),[seed])

    def setUp(self):
        self.image = Image.new('RGB',(240,220),'white')
        self.poly = rectangle(30,25,48,101)

    def test_disconnected_kana_row_requires_native_and_isolated_readings(self):
        from PIL import ImageDraw
        page = Image.new('RGB', (240,150), 'white')
        draw = ImageDraw.Draw(page)
        draw.arc((35,47,61,85), -145,100, fill='black', width=5)
        draw.line((77,64,119,61), fill='black', width=5)
        draw.line([(137,80),(153,45),(157,76),(177,64)],fill='black',width=5)
        probability = np.uint8(np.asarray(page.convert('L'))<150)*255
        result = masked_lines.drawn_kana_rows(page,probability,[],Mock(return_value='カーン'),5)
        self.assertEqual([(v.box_id,v.text) for v in result],[(5,'カーン')])
        self.assertFalse(masked_lines.drawn_kana_rows(page,probability,[],Mock(side_effect=['カーン','かーん']),5))
        self.assertFalse(masked_lines.drawn_kana_rows(page,probability,result,Mock(),6))

    def test_repetition_needs_both_original_halves_or_a_second_recognizer(self):
        def read(answers, native='11'):
            return masked_lines.read_line(self.image,self.image,self.poly,native,.82,Mock(side_effect=answers))
        self.assertEqual(read(['ドキドキ','ドキドキ','ドキ','ドキ']),'ドキドキ')
        self.assertIsNone(read(['ドキドキ','ドキドキ','ド','ドキ']))
        self.assertIsNone(read(['くく','くく']))
        self.assertIsNone(read(['わん','がん'],native='わん'))
        self.assertEqual(read(['わん']*3,native='わん'),'わん')

    def test_crossed_columns_need_independent_row_and_whole_source_agreement(self):
        original = [Text_Box(40,.4,rectangle(35,16,40,90),''),
                    Text_Box(52,.7,rectangle(60,16,40,90),'文字')]
        rows = [('元字',.8,rectangle(35,22,70,37)),('名簿',.9,rectangle(33,55,72,38))]
        def attempt(final):
            readings = ['名字']*5+['名簿']*5+['名字名簿']*5+[final]
            return masked_lines.crossed_rows(self.image,self.image,Mock(side_effect=readings),original,rows)
        self.assertIsNone(attempt('名字日記'))
        recovered = attempt('名字名簿')
        self.assertEqual([r.text for r in recovered],['名字','名簿'])
        self.assertEqual(recovered[0].ocr_metadata['masked_passage']['ids'],[40,52])
        self.assertEqual(recovered[1].ocr_metadata['candidates']['paddle'],'文字')

    def test_empty_page_can_recover_its_first_independently_read_word(self):
        engine = SimpleNamespace(raw_detections=[[]],automatic_unit_proposals=[None],ocr=Mock(),image_path='page')
        engine.ocr.predict.return_value = [{'rec_texts':['わん'],'rec_scores':[.88],'rec_polys':[self.poly]}]
        with patch.object(masked_lines,'propose_page_from_boxes',return_value='new'):
            masked_lines.discover(engine,self.image,Mock(return_value='わん'),np.full((220,240),255,np.uint8))
        self.assertEqual([(b.box_id,b.text) for b in engine.raw_detections[0]],[(0,'わん')])
        self.assertEqual(engine.automatic_unit_proposals,['new'])

    def test_slanted_suffix_needs_an_unclaimed_polygon_and_whole_inscription(self):
        poly = np.float32([(109,31),(133,31),(169,154),(144,154)])
        head = Text_Box(7,.70,poly,'まって')
        side = poly - [28,0]
        x,y,r,b = head.bbox
        l,t = max(0,x-round(.32*(r-x+1))),max(0,y-round(.25*(b-y+1)))
        detector = Mock()
        detector.predict.return_value = [{'rec_polys':[side-[l,t]],'rec_texts':['ちよ'],'rec_scores':[.76]}]
        def read(last,neighbors):
            reader = Mock(side_effect=['まってよね','まって',*['よね']*3,'まってよね','まってよね',last])
            return masked_lines.slanted_continuation(self.image,reader,detector,head,neighbors,18)
        self.assertIsNone(read('まってたよ',[head]))
        self.assertIsNone(read('まってよね',[head,Text_Box(9,.98,side,'誰')]))
        parts = read('まってよね',[head])
        self.assertEqual([(p.box_id,p.text) for p in parts],[(7,'まって'),(18,'よね')])
