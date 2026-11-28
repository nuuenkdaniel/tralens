import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import balloon_expansion, natural
from .font_fixture import synthetic_font


class BalloonExpansion(unittest.TestCase):
    def setUp(self):
        self.page = Image.new('RGB', (200, 140), (190, 170, 145))
        draw = ImageDraw.Draw(self.page)
        draw.polygon([(78, 95), (88, 96), (65, 118)], fill='white')
        draw.ellipse((38, 32, 157, 101), fill='white')
        self.source = np.zeros((140, 200), bool)
        self.source[55:78, 90:105] = True
        self.blocked = np.zeros_like(self.source)
        self.layer = Image.new('RGBA', self.page.size)
        for x in range(100, 164, 8):
            ImageDraw.Draw(self.layer).rectangle((x, 58, x+6, 73), fill='black')
        self.proof = {'size': 24, 'original_text': 'Wait'}
        self.font = synthetic_font()

    def finish(self):
        return balloon_expansion.finish(np.asarray(self.page), self.source, self.blocked,
                                        self.layer, self.proof, self.font, natural.font_coverage(self.font))

    def grow(self):
        with patch.object(natural, 'layout', side_effect=ValueError('no fit')):
            return self.finish()

    def test_rewrap_uses_original_paper_and_type_size_first(self):
        layer, proof = self.finish()
        self.assertEqual(proof['balloon_backing'], 'rewrapped_inside')
        self.assertEqual(proof['size'], self.proof['size'])
        self.assertEqual(proof['original_text'], 'Wait')
        ink = np.asarray(layer.getchannel('A')) != 0
        self.assertTrue((np.asarray(self.page)[ink] == 255).all())

    def test_shallow_growth_preserves_letters_and_tail(self):
        layer, proof = self.grow()
        self.assertEqual(proof['balloon_backing'], 'expanded')
        self.assertEqual(proof['size'], self.proof['size'])
        old = np.asarray(self.layer)
        new = np.asarray(layer)
        np.testing.assert_array_equal(new[old[:, :, 3] == 255], old[old[:, :, 3] == 255])
        np.testing.assert_array_equal(new[65,163], [255,255,255,255])
        self.assertFalse(new[105:, :, 3].any())

    def test_avoids_nearby_text_inked_border_and_artwork(self):
        untouched = self.page.copy()
        for condition in ('neighbor', 'border', 'black line', 'color line', 'second balloon'):
            with self.subTest(condition=condition):
                self.page = untouched.copy()
                self.blocked.fill(False)
                draw = ImageDraw.Draw(self.page)
                if condition == 'neighbor':
                    self.blocked[64, 159] = True
                elif condition == 'border':
                    draw.ellipse((38, 32, 157, 101), outline='black', width=2)
                elif condition == 'second balloon':
                    draw.ellipse((160, 62, 170, 75), fill='white')
                else:
                    draw.line((161, 50, 161, 85), fill='black' if condition == 'black line' else (220, 20, 160), width=2)
                layer, proof = self.grow()
                self.assertIs(layer, self.layer)
                self.assertIs(proof, self.proof)

    def test_no_change_for_ink_that_fits_or_excessive_spill(self):
        for end in (136, 189):
            with self.subTest(end=end):
                self.layer = Image.new('RGBA', self.page.size)
                ImageDraw.Draw(self.layer).rectangle((90, 60, end, 72), fill='black')
                layer, proof = self.grow()
                self.assertIs(layer, self.layer)
                self.assertIs(proof, self.proof)
