from paddleocr import PaddleOCR

class Text_Box:
    def __init__(self, confidence, bbox, text):
        self.confidence = confidence
        self.bbox = bbox
        self.text = text

class Text_Group:
    def __init__(self, group: list[Text_Box]):
        self.group = group

    def get(self, index: int):
        return self.group[index]

class OCR:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict(self) -> list[list[Text_Box]]:
        results = self.ocr.predict(self.image_path)
        images_text_box = []
        for page in results:
            res = getattr(page, "res", page)
            dt_polys = res.get("dt_polys", [])
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])

            text_boxes = []
            for i, score in enumerate(rec_scores):
                if score > 0.5:
                    polys = dt_polys[i] if i < len(dt_polys) else None
                    text = rec_texts[i] if i < len(rec_texts) else ""
                    text_boxes.append(Text_Box(score, polys, text))
                    print(f"confidence: {score} | {text}")

            images_text_box.append(text_boxes)
        return images_text_box

if __name__ == "__main__":
    image = "images/japsigns.jpg"
    ocr = OCR(image)
    results = ocr.predict()
