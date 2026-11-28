import unittest
from types import SimpleNamespace

import numpy as np

from tralens.ocr.automatic_units import join_chinese_inline
from tralens.ocr.chapter_language import language_votes
from tralens.ocr.types import Text_Group


class ChapterLanguageTests(unittest.TestCase):
    def test_script_votes_ignore_short_shared_signs_and_low_confidence(self):
        for shared, korean, expected in [
            (("学校", .99), ("학교", .30), {}),
            (("世界你好", .99), ("", .10), {"zh": 4}),
            (("今日は", .97), ("세", .60), {"ja": 6}),
            (("三立", .56), ("반갑다", .97), {"ko": 3}),
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
