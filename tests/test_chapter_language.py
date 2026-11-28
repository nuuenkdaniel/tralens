import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np

from tralens.ocr.automatic_units import group_slanted_caption_rows, join_chinese_inline
from tralens.ocr.chapter_language import language_votes
from tralens.ocr.mixed_script import recover_caption_ends, recover_vertical_hanja
from tralens.ocr.types import Text_Box, Text_Group


class ChapterLanguageTests(unittest.TestCase):
    def test_script_votes_ignore_short_shared_signs_and_low_confidence(self):
        for shared, korean, expected in [
            (("学校", .99), ("학교", .30), {}),
            (("世界你好", .99), ("", .10), {"zh": 4}),
            (("今日は", .97), ("세", .60), {"ja": 6}),
            (("三立", .56), ("반갑다", .97), {"ko": 3}),
            (("ﾏﾝｶﾞ", .98), ("", .1), {"ja": 12}),
            (("𠀀𠀁𠀂", .98), ("", .1), {"zh": 3}),
            (("", .1), ("한글", .97), {"ko": 2}),
            (("ｰｰｰ", .99), ("", .1), {}),
            (("2026!", .99), ("2026!", .99), {}),
        ]:
            with self.subTest(shared=shared):
                votes = language_votes([shared], [korean])
                self.assertEqual({lang: n for lang, n in votes.items() if n}, expected)

    def test_caption_initials_keep_panel_order_and_respect_borders(self):
        def group(identity, text, x, y, width):
            return Text_Group([SimpleNamespace(box_id=identity, text=text, bbox=(x, y, x+width, y+12))])

        paper = np.full((70, 160, 3), 255, dtype=np.uint8)
        paper[:, 86] = 0
        groups = [group(0, "右方文字", 87, 12, 60), group(1, "日", 5, 40, 12),
                  group(2, "照大地", 17, 40, 55), group(3, "守", 72, 12, 12)]
        joined = join_chinese_inline(groups, paper)
        self.assertEqual([[box.box_id for box in g.group] for g in joined], [[0], [1, 2], [3]])

    def test_parallel_slanted_captions_keep_paragraph_breaks(self):
        theta = np.deg2rad(-17)
        rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        boxes = []
        for i, y in enumerate((0, 17, 34, 90, 107)):
            polygon = np.array([(0,y), (72,y), (72,y+16), (0,y+16)]) @ rotation + 80
            boxes.append(Text_Box(i, .99, polygon, "文字"))
        groups = [Text_Group(boxes[:2]), Text_Group([boxes[2],boxes[4]]), Text_Group([boxes[3]])]
        result = group_slanted_caption_rows(groups)
        self.assertEqual([[b.box_id for b in g.group] for g in result], [[0,1,2],[3,4]])

    def test_caption_extension_needs_independent_character_and_context_agreement(self):
        image = np.full((75, 150, 3), 255, np.uint8)
        image[22:42,99:120] = 0
        line = Text_Box(0, .99, [(10,20),(98,20),(98,45),(10,45)], "如此回答")
        rec = Mock()
        def result(s): return {"rec_text":s, "rec_score":.98}
        for context, count in [("如此回答",1), ("如此回答好",2)]:
            boxes = [line]
            rec.predict.return_value = [result("好"),result(context)]
            recover_caption_ends(image, boxes, lambda _:rec)
            self.assertEqual(len(boxes), count)
            if count == 2:
                self.assertEqual(boxes[-1].bbox, (99,22,119,41))
                recover_caption_ends(image, boxes, lambda _:rec)
                self.assertEqual(len(boxes), 2)  # Already owned by the first pass.

    def test_vertical_korean_suffix_requires_consistent_script_readings(self):
        image = np.full((125,25,3), 255, np.uint8)
        for y in (5,25,45,65,85):
            image[y:y+17,4:17] = 0
        box = Text_Box(1, .8, [(0,0),(21,0),(21,106),(0,106)], "새로운")
        def result(s): return {"rec_text":s,"rec_score":.95}
        han, korean = Mock(), Mock()
        korean.predict.return_value = [result("새로운")]*3
        for readings, expected in [(("文化","文","花"),"새로운"), (("文化","文","化"),"새로운 文化")]:
            boxes = [box]
            han.predict.return_value = [result(s) for s in readings]
            recover_vertical_hanja(image, boxes, lambda n: korean if n.startswith("korean") else han)
            self.assertEqual(boxes[0].text, expected)
