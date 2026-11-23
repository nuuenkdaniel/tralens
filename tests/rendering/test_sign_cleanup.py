import unittest

import numpy as np
from tralens.rendering.sign_cleanup import attached_ink


class SignCleanupTests(unittest.TestCase):
    def test_only_terminal_strokes_can_be_removed_at_the_rule(self):
        paper = (235, 225, 198)
        pixels = np.full((70, 110, 3), paper, np.uint8)
        foreground = np.zeros(pixels.shape[:2], bool)
        foreground[20:46, 12:31] = True
        foreground[20:46, 35:54] = True
        pixels[foreground] = 20
        pixels[:, 76:79] = 20
        pixels[:, 75] = 95  # Antialiased interior edge of the sign.
        extra = np.zeros_like(foreground)
        extra[23:45, 61:64] = True
        extra[25:28, 62:75] = extra[38:41, 62:75] = True
        pixels[extra] = 20
        member = np.zeros_like(extra)
        member[15:51, 9:83] = True
        protected = np.zeros_like(extra)

        def recover(text):
            return attached_ink(pixels, foreground, member, protected, np.array(paper), text)
        np.testing.assert_array_equal(recover('職員室'), extra)
        self.assertIsNone(recover('職員'))
        protected[26, 65] = True
        self.assertIsNone(recover('職員室'))
        protected[:] = False
        pixels[38:41, 64:75] = paper
        self.assertIsNone(recover('職員室'))

    def test_touching_prefix_requires_terminal_box_and_keeps_rule(self):
        paper = np.array([239, 228, 182])
        pixels = np.full((70, 112, 3), paper, np.uint8)
        foreground = np.zeros(pixels.shape[:2], bool)
        foreground[20:46, 12:36] = foreground[20:46, 38:62] = True
        foreground[31, 36:38] = True
        pixels[foreground] = 20
        member = np.zeros_like(foreground)
        member[16:50, 9:93] = True
        protected = np.zeros_like(member)
        for rail in (75, 85):
            with self.subTest(rail=rail):
                image = pixels.copy()
                image[:, rail:rail+3] = 20
                image[:, rail] = 100
                glyph = np.zeros_like(member)
                glyph[21:45, 65:68] = glyph[21:45, 82:86] = True
                glyph[21:24, 65:86] = glyph[40:43, 65:86] = True
                glyph[:, rail:] = False
                image[glyph] = 20

                def recover(text='改札口'):
                    return attached_ink(image, foreground, member, protected, paper, text)

                np.testing.assert_array_equal(recover(), glyph)
                self.assertIsNone(recover('改札室'))
                protected[22, 70] = True
                self.assertIsNone(recover())
                protected[22, 70] = False
                # Mostly antialiased strokes can still have the same solid ink.
                ink_y, ink_x = np.where(glyph)
                softened = np.arange(len(ink_y)) % 3 != 0
                image[ink_y[softened], ink_x[softened]] = (paper + 20) // 2
                np.testing.assert_array_equal(recover(), glyph)
                image[ink_y[softened], ink_x[softened]] = [220, 60, 100]
                self.assertIsNone(recover())
                image[glyph] = 20
                image[26:38, 65:68] = paper  # Two border marks without a left stroke.
                self.assertIsNone(recover())
