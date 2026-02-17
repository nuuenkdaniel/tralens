from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from surya.foundation import FoundationPredictor
from surya.recognition import OCRResult, RecognitionPredictor
from surya.detection import DetectionPredictor
import numpy as np
from sklearn.cluster import DBSCAN

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
        self.image = Image.open(image_path)

    @staticmethod
    def surya_predict(image) -> list[OCRResult]:
        foundation_predictor = FoundationPredictor()
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        detection_predictor = DetectionPredictor()
        return recognition_predictor([image], det_predictor=detection_predictor, math_mode=False)

    def _crop_bbox(self, bbox):
        image_w, image_h = self.image.size
        x1, y1, x2, y2 = map(int, bbox)
        bbox_w = x2-x1
        bbox_h = y2-y1
        pad = int(bbox_w*.08) if bbox_w < bbox_h else int(bbox_h*.08)

        padded_bbox = (
            max(0, x1-pad),
            max(0, y1-pad),
            min(image_w, x2+pad),
            min(image_h, y2+pad)
        )
        return (self.image.crop(padded_bbox), padded_bbox)

    def predict_test_with_retry(self) -> list[Text_Box]:
        init_prediction = self.surya_predict(self.image)
        text_boxes = []

        for text_box in init_prediction[0].text_lines:
            confidence = text_box.confidence
            bbox = text_box.bbox
            
            if confidence and 0.8 <= confidence < 1.0:
                text_boxes.append(Text_Box(confidence, bbox, text_box.text))
            else:
                cropped, padded_bbox = self._crop_bbox(bbox)
                cropped_enhanced = ImageEnhance.Contrast(cropped).enhance(1.1)
                prediction_retry = self.surya_predict(cropped_enhanced)
                old_x1, old_y1, _, _ = padded_bbox
                
                for new_text_box in prediction_retry[0].text_lines:
                    new_confidence = new_text_box.confidence
                    x1, y1, x2, y2 = new_text_box.bbox
                    if new_confidence and 0.7 <= new_confidence < 1.0:
                        new_bbox = (x1+old_x1, y1+old_y1, x2+old_x1, y2+old_y1)
                        text_boxes.append(Text_Box(new_confidence, new_bbox, new_text_box.text))

        return text_boxes

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

    def process_image(self):
        text_boxes = self.predict_test_with_retry()
        return self._group_boxes(text_boxes)

    # def visualize_groups(self, groups: list[Text_Group], output_path="visualized_groups.png"):
    #     canvas = self.image.convert("RGB")
    #     draw = ImageDraw.Draw(canvas)
    #     for i, tg in enumerate(groups):
    #         all_x1 = [box.bbox[0] for box in tg.group]
    #         all_y1 = [box.bbox[1] for box in tg.group]
    #         all_x2 = [box.bbox[2] for box in tg.group]
    #         all_y2 = [box.bbox[3] for box in tg.group]
    #         group_bbox = (min(all_x1), min(all_y1), max(all_x2), max(all_y2))
    #         for box in tg.group:
    #             draw.rectangle(box.bbox, outline="red", width=1)
    #         draw.rectangle(group_bbox, outline="blue", width=3)
    #         draw.text((group_bbox[0], group_bbox[1] - 10), f"Group {i}", fill="blue")
    #     canvas.save(output_path)
    #     canvas.show()

    def visualize_groups(self, groups: list[Text_Group], output_path="visualized_groups.png"):
        canvas = self.image.convert("RGB")
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
