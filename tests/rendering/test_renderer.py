"""Headless checks of the unchanged production image replacement functions."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageFont
from tralens.rendering import cover_text, draw_text


class RendererTests(unittest.TestCase):
    def test_cover_text_copies_image_and_clips_padding(self):
        image = Image.new("RGB", (40, 40), "black")
        group = SimpleNamespace(group=[SimpleNamespace(bbox=(0, 0, 10, 10))])
        covered = cover_text(image, [group], padding=5)
        self.assertIsNot(covered, image)
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(covered.getpixel((14, 14)), (255, 255, 255))
        self.assertEqual(covered.getpixel((20, 20)), (0, 0, 0))

    def test_draw_text_returns_same_image_and_keeps_box_data(self):
        image = Image.new("RGB", (80, 50), "white")
        box = SimpleNamespace(bbox=(0, 0, 50, 40), text="A<br>B")
        group = SimpleNamespace(group=[box])
        with patch("tralens.rendering.renderer.ImageFont.truetype",
                   return_value=ImageFont.load_default()):
            result = draw_text(image, [group], font_path="test-only-font")
        self.assertIs(result, image)
        self.assertEqual(box.text, "A<br>B")
        self.assertTrue(any(pixel != (255, 255, 255) for pixel in image.getdata()))
