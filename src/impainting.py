from PIL import Image, ImageDraw, ImageFont
from ocr import Text_Group

def cover_text(image: Image.Image, groups: list[Text_Group], padding=5) -> Image.Image:
    new_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(new_image)

    for group in groups:
        for box in group.group:
            x1, y1, x2, y2 = map(int, box.bbox)
            x1, y1 = max(0, x1-padding), max(0, y1-padding)
            x2, y2 = min(image.width, x2+padding), min(image.height, y2+padding)
            draw.rectangle((x1, y1, x2, y2), fill="white")
    return new_image

def draw_text(image: Image.Image, groups: list[Text_Group], font_path="/usr/share/fonts/liberation/LiberationMono-Regular.ttf"):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 12)

    for group in groups:
        for box in group.group:
            text = box.text.replace("<br>", "\n")
            draw.text((box.bbox[0], box.bbox[1]), text, font=font, fill="black")

    return image

from ocr import OCR
from translate import Translate

if __name__ == "__main__":
    # image_path = "/home/Danuu/Downloads/japsigns.jpg"
    image_path = "/home/Danuu/Downloads/lmgJZ.jpg"
    predictor = OCR(image_path)
    groups = predictor.process_image()

    translator = Translate("100.121.133.88")
    translated_groups = translator.translate_groups(groups, image_path)

    image = cover_text(predictor.image, groups, 5)
    image = draw_text(image, translated_groups)
    image.show()
