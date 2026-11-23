import unittest

import numpy as np
from PIL import Image, ImageDraw
from tralens.rendering.banner import flat_plan


class BannerTests(unittest.TestCase):
    def test_outline_and_face_crossing_the_title_box_survive(self):
        image = Image.new('RGB',(210,100),'white')
        pen = ImageDraw.Draw(image)
        pen.rectangle((8,12,200,92),outline='black',width=3)
        pen.ellipse((140,26,168,55),outline='black',width=3)
        pen.rectangle((151,45,155,48),fill='black')
        rows = [dict(source_id=0,polygon=[[18,21],[72,21],[72,44],[18,44]]),
                dict(source_id=1,polygon=[[17,41],[184,41],[184,81],[17,81]])]
        for x,y in [(25,26),(44,28),(39,53),(82,52),(125,56)]:
            pen.rectangle((x,y,x+5,y+11),fill='black')
        plan=flat_plan(image,rows,[0,1])
        self.assertIsNotNone(plan)
        self.assertTrue(plan.erase[30,27])
        self.assertTrue(plan.erase[57,84])
        self.assertFalse(plan.erase[26:56,140:169].any())
        self.assertFalse(plan.layout[46,152])
        self.assertFalse(plan.erase[:,8:11].any())
