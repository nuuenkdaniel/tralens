from paddleocr import PaddleOCR
import numpy as np
from paddlex.inference.models.ts_forecasting.result import visualize
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw, ImageFont

class Text_Box:
    def __init__(self, confidence, poly, text):
        self.confidence = confidence
        self.poly = np.array(poly, dtype=np.float32)
        self.text = text

        x1 = int(np.min(self.poly[:, 0]))
        y1 = int(np.min(self.poly[:, 1]))
        x2 = int(np.max(self.poly[:, 0]))
        y2 = int(np.max(self.poly[:, 1]))
        self.bbox = (x1, y1, x2, y2)

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

    def _calc_edge_dist(self, box1: Text_Box, box2: Text_Box):
        b1 = box1.bbox
        b2 = box2.bbox
        
        dx = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
        dy = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
        edge_dist = np.sqrt(dx**2 + dy**2)
        h1, w1 = b1[3] - b1[1], b1[2] - b1[0]
        h2, w2 = b2[3] - b2[1], b2[2] - b2[0]
        size1 = min(h1, w1) 
        size2 = min(h2, w2)
        avg_size = (size1+size2)/2
        normalized_dist = edge_dist/max(avg_size, 1e-5)
        
        if h1 > w1 and h2 > w2:
            normalized_dist = np.sqrt((dx*0.5)**2 + dy**2)/avg_size

        return normalized_dist

    def _group_boxes(self, text_boxes: list[Text_Box]):
        if not text_boxes:
            return []

        n = len(text_boxes)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = self._calc_edge_dist(text_boxes[i], text_boxes[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        db = DBSCAN(eps=1.1, min_samples=1, metric='precomputed').fit(dist_matrix)
        labels = db.labels_
        next_unique_id = max(labels) + 1 if len(labels) > 0 else 0

        grouped_results = {}
        for idx, label in enumerate(labels):
            if label == -1:
                target_id = next_unique_id
                next_unique_id += 1
            else:
                target_id = label
                
            if target_id not in grouped_results:
                grouped_results[target_id] = []
            grouped_results[target_id].append(text_boxes[idx])

        return [Text_Group(group) for group in grouped_results.values()]

    def process_images(self):
        images_text_boxes = self.predict()
        images_text_groups = []
        for text_boxes in images_text_boxes:
            images_text_groups.append(self._group_boxes(text_boxes))
        return images_text_groups


    def visualize_groups(self, groups: list[Text_Group], output_path="visualized_groups.png"):
        image = Image.open(self.image_path)
        canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        
        # Try to load a readable font (Linux usually has DejaVuSans or LiberationSans)
        try:
            font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 14)
        except IOError:
            font = ImageFont.load_default()

        print(f"\n--- Visualizing {len(groups)} Groups ---")

        for i, tg in enumerate(groups):
            # 1. Calculate and Draw Group Boundary (Blue)
            all_x1 = [box.bbox[0] for box in tg.group]
            all_y1 = [box.bbox[1] for box in tg.group]
            all_x2 = [box.bbox[2] for box in tg.group]
            all_y2 = [box.bbox[3] for box in tg.group]
            
            group_bbox = (min(all_x1), min(all_y1), max(all_x2), max(all_y2))
            draw.rectangle(group_bbox, outline="blue", width=3)
            
            # Label the Group ID
            draw.text((group_bbox[0], group_bbox[1] - 20), f"Group {i}", fill="blue", font=font)
            
            print(f"Group {i}:")

            for box in tg.group:
                # 2. Draw Individual Box (Red)
                draw.rectangle(box.bbox, outline="red", width=1)
                
                # 3. Draw Text & Confidence Label on Image
                # Format: "Text (0.95)"
                label = f"{box.text} ({box.confidence:.2f})"
                
                # Draw a small red background behind text so it's readable
                text_x, text_y = box.bbox[0], box.bbox[1] - 15
                if text_y < 0: text_y = box.bbox[3] + 5 # Handle top edge cases
                
                # Get text size to make background box
                left, top, right, bottom = draw.textbbox((text_x, text_y), label, font=font)
                draw.rectangle((left-2, top-2, right+2, bottom+2), fill="red")
                draw.text((text_x, text_y), label, fill="white", font=font)
                print(f"  - Conf: {box.confidence:.4f} | Text: {box.text}")

        canvas.save(output_path)
        print(f"Visualization saved to {output_path}")
        canvas.show()

if __name__ == "__main__":
    image = "images/japsigns.jpg"
    ocr = OCR(image)
    results = ocr.process_images()
    ocr.visualize_groups(results[0])
