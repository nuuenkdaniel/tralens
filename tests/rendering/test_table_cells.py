import unittest

import numpy as np
from PIL import Image

from tralens.rendering import table_cells
from .font_fixture import synthetic_font


class TableCellTests(unittest.TestCase):
    def test_section_title_must_read_back_above_two_checked_names(self):
        from unittest.mock import Mock
        from PIL import ImageDraw
        from tralens.ocr.table_rows import checkbox_headings
        from tralens.ocr.types import Text_Box

        page = Image.new('RGB',(220,170),'white')
        draw = ImageDraw.Draw(page)
        def row(i,x,y,w,h,label):
            return Text_Box(i,.95,[(x,y),(x+w,y),(x+w,y+h),(x,y+h)],label)
        boxes = [row(1,46,21,113,28,'推易'),row(2,63,50,90,34,'editor'),
                 row(3,63,100,88,34,'viewer')]
        for y in (54,104):
            draw.rectangle((40,y,63,y+23),outline='black',width=2)
            draw.line([(44,y+11),(48,y+16),(58,y+5)],fill='black',width=2)
        read = Mock(return_value='推奨')
        result = checkbox_headings(page,boxes,read)
        self.assertEqual(result[0].text,'推奨')
        self.assertTrue(all(v.reason=='confirmed_checkbox_item' for v in result[1:]))
        unstable = Mock(side_effect=['推奨','別物'])
        self.assertEqual(checkbox_headings(page,boxes,unstable),boxes)
        draw.rectangle((38,102,65,129),fill='white')
        self.assertEqual(checkbox_headings(page,boxes,read),boxes)

    def test_tabs_require_matching_enclosures_and_independent_readings(self):
        from unittest.mock import Mock
        from PIL import ImageDraw
        from tralens.ocr.table_rows import split_tab_labels
        from tralens.ocr.types import Text_Box

        page = Image.new('RGB',(200,90),'white')
        draw = ImageDraw.Draw(page)
        for left in (26,76,126):
            draw.rectangle((left,26,left+50,58),outline='black',width=2)
            draw.rectangle((left+14,35,left+33,47),fill='black')
        merged = Text_Box(7,.94,[(27,26),(175,26),(175,59),(27,59)],'設定 USB 表示')
        def split(last):
            reads = ['設定','設定','USB','USB','表示',last]
            return split_tab_labels(page,[merged],Mock(side_effect=reads))
        labels = split('表示')
        self.assertEqual([(b.box_id,b.text) for b in labels],[(7,'設定'),(8,'USB'),(9,'表示')])
        self.assertEqual(split('別物'),[merged])
        draw.rectangle((73,39,79,45),fill='white')
        unread = Mock()
        self.assertEqual(split_tab_labels(page,[merged],unread),[merged])
        unread.assert_not_called()

    def test_ink_and_english_stay_between_the_original_rules(self):
        pixels = np.full((140,210,3),250,np.uint8)
        for x in range(210):
            for y in (29,49,69,89,109):
                pixels[y+round(.07*(x-130)),x] = 180
        pixels[:,119] = 170
        pixels[:,188:193] = 0
        pixels[55:63,125:149] = 0
        pixels[22:120,158] = 125
        source = Image.fromarray(pixels)
        row = dict(unit_id='one',status='skipped',binding=dict(member_ids=[1],text='ABC'),reason_code='ruled_table_cell')
        clean = source.copy()
        text = Image.new('RGBA',source.size)
        erase = np.zeros(pixels.shape[:2],bool)
        layout = erase.copy()
        data = synthetic_font()
        proposals = {'one':{'grouping_method':'separate_printed_table_rows'}}
        detection = dict(source_id=1,polygon=[(119,51),(164,54),(164,70),(119,66)])
        args = source,clean,text,erase,layout,[row],proposals,[detection],lambda r:(data,set(range(32,127)))
        self.assertTrue(table_cells.place(*args))
        self.assertEqual(row['ruled_cell']['text'],'ABC')
        np.testing.assert_array_equal(np.asarray(clean)[:,158],pixels[:,158])
        self.assertFalse(erase[:,:121].any() or erase[:,188:].any())
        self.assertTrue((np.asarray(text.getchannel('A'))[~layout]==0).all())
        self.assertTrue((np.asarray(clean)[~erase]==pixels[~erase]).all())
        row['status'] = 'skipped'
        self.assertFalse(table_cells.place(*args))
        self.assertIsNone(table_cells.interior(np.full((140,210),250,np.uint8),detection['polygon']))

    def test_full_binding_must_fit_at_native_print_size(self):
        data = synthetic_font()
        self.assertIsNone(table_cells.lettering('ABC '*50,data,38,12))
        fit = table_cells.lettering('A B',data,60,18)
        self.assertEqual(' '.join(fit[2]),'A B')

    def test_window_title_stays_between_printed_rules(self):
        from PIL import ImageDraw
        from tralens.ocr.table_rows import tab_interiors
        page=Image.new('RGB',(210,110),'white')
        draw=ImageDraw.Draw(page)
        draw.rectangle((32,34,163,65),fill='#eeeeee',outline='black',width=2)
        for x in (63,79,94,110,127):
            draw.rectangle((x,42,x+6,58),fill='black')
        poly=[(53,35),(146,35),(146,66),(53,66)]
        cells=tab_interiors(np.asarray(page.convert('L')),poly,1)
        self.assertEqual(len(cells),1)
        xx,top,bottom,_=cells[0][-1]
        self.assertGreaterEqual(top.min(),36)
        self.assertLess(bottom.max(),64)
        self.assertGreaterEqual(xx[0],53)
        pixels=np.asarray(page)
        gray=pixels.max(2)
        probability=np.zeros_like(gray)
        probability[40:61,60:138]=220
        bounds,strokes,tone=table_cells.title_repair(pixels,gray,probability,cells[0][-1],False)
        protected=np.ones_like(gray,bool)
        protected[40:62,58:139]=False
        x,y,r,b=bounds
        self.assertFalse((strokes & protected[y:b,x:r]).any())
        self.assertTrue(strokes.any())
        self.assertEqual(tone,238)
        draw.rectangle((85,33,97,37),fill='white')
        self.assertFalse(tab_interiors(np.asarray(page.convert('L')),poly,1))

    def test_overlapping_checklist_label_preserves_rule_and_neighbor(self):
        from PIL import ImageDraw
        paper = Image.new('RGB',(180,100),'white')
        draw = ImageDraw.Draw(paper)
        poly = [(34,31),(144,31),(144,61),(34,61)]
        draw.line((23,60,149,60),fill='black',width=2)
        for x in (40,60,82,103,124):
            draw.rectangle((x,39,x+8,52),fill='black')
        alpha = np.zeros((100,180),bool)
        alpha[29:35,60:75] = True
        probability = np.zeros(alpha.shape,np.uint8)
        probability[38:55,38:138] = 161
        result = table_cells.checklist_line(paper,poly,alpha,probability)
        (x,y,r,b),strokes,forbidden,_ = result
        self.assertTrue(strokes.any())
        self.assertFalse(strokes[60-y].any())
        self.assertTrue(forbidden[60-y,50-x])
        self.assertFalse((strokes & alpha[y:b,x:r]).any())
        self.assertIsNone(table_cells.checklist_line(paper,poly,alpha,probability*0))
        draw.line((75,24,75,68),fill='black',width=3)
        self.assertIsNone(table_cells.checklist_line(paper,poly,alpha,probability))
