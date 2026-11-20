import unittest
import numpy as np
from PIL import Image,ImageDraw

from tools.experimental import recovery_layout as layout


class Presentation(unittest.TestCase):
    def test_closed_presentation_expands_without_owning_foreign_ink(self):
        image=Image.new('RGB',(200,200),'white');draw=ImageDraw.Draw(image)
        draw.ellipse((15,15,185,185),outline='black',width=3)
        draw.rectangle((95,65,105,135),fill='black')
        draw.line((40,75,65,115),fill='black',width=5)
        own=np.zeros((200,200),bool);own[50:150,90:110]=True
        protected=np.zeros_like(own);protected[45:60,70:130]=True
        mask=layout.presentation(np.asarray(image),own,protected)
        self.assertGreater(int(mask.sum()),int(own.sum()))
        self.assertFalse((mask&protected).any());self.assertFalse(mask[90,50])
        self.assertFalse(mask[0].any())

    def test_page_connected_or_cropped_region_is_not_presentation_space(self):
        own=np.zeros((100,100),bool);own[30:70,40:60]=True
        with self.assertRaises(ValueError):layout.presentation(np.full((100,100,3),255,np.uint8),own,np.zeros_like(own))
