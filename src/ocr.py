import math
import statistics

import cv2
import numpy as np
from paddleocr import PaddleOCR
from sklearn.cluster import DBSCAN

class Text_Box:
    def __init__(self, box_id, confidence, poly, text, review=0):
        self.box_id = box_id
        self.confidence = confidence
        self.poly = np.array(poly, dtype=np.float32)
        self.text = text
        self.review = review

        x1 = int(np.min(self.poly[:, 0]))
        y1 = int(np.min(self.poly[:, 1]))
        x2 = int(np.max(self.poly[:, 0]))
        y2 = int(np.max(self.poly[:, 1]))
        self.bbox = (x1, y1, x2, y2)

class Text_Group:
    def __init__(self, group: list[Text_Box], *, method="legacy", status="grouped",
                 direction=None, metadata=None):
        self.group = group
        self.method = method
        self.status = status
        self.direction = direction
        self.metadata = metadata or {}

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
                polys = dt_polys[i] if i < len(dt_polys) else None
                text = rec_texts[i] if i < len(rec_texts) else ""
                text_boxes.append(Text_Box(i, score, polys, text))
                print(f"confidence: {score} | {text}")

            images_text_box.append(text_boxes)
        return images_text_box

    def _calc_edge_dist(self, box1: Text_Box, box2: Text_Box):
        b1 = box1.bbox
        b2 = box2.bbox
        
        dx = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
        dy = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
        edge_dist = math.sqrt(dx**2 + dy**2)
        h1, w1 = b1[3] - b1[1], b1[2] - b1[0]
        h2, w2 = b2[3] - b2[1], b2[2] - b2[0]
        size1 = min(h1, w1) 
        size2 = min(h2, w2)
        avg_size = (size1+size2)/2
        normalized_dist = edge_dist/max(avg_size, 1e-5)
        
        if h1 > w1 and h2 > w2:
            normalized_dist = math.sqrt((dx*0.5)**2 + dy**2)/avg_size

        return normalized_dist

    def _group_boxes_manga(self, text_boxes: list[Text_Box]):
        if not text_boxes: return []

        regions, region_status = self._manga_regions(text_boxes)
        clusters = [[i] for i in range(len(text_boxes))]
        # Complete-link safety prevents a chain of locally close boxes from
        # bridging distant blocks. A strict local edge is still required.
        while True:
            candidates = []
            for a, first in enumerate(clusters):
                for b in range(a + 1, len(clusters)):
                    second = clusters[b]
                    pairs = [(i, j) for i in first for j in second]
                    if not all(self._manga_compatible(text_boxes[i], text_boxes[j],
                                                     relaxed=True) for i, j in pairs):
                        continue
                    known = {regions[i][0] for i in first + second if i in regions}
                    if len(known) > 1:
                        continue
                    local = [(i, j) for i, j in pairs
                             if self._manga_compatible(text_boxes[i], text_boxes[j])]
                    if local:
                        distance = min(self._calc_edge_dist(text_boxes[i], text_boxes[j])
                                       for i, j in local)
                        candidates.append((distance, a, b))
            if not candidates:
                break
            _, a, b = min(candidates)
            clusters[a].extend(clusters.pop(b))

        groups = []
        for indices in clusters:
            boxes = [text_boxes[i] for i in indices]
            direction = self._manga_direction(boxes[0])
            # Vertical manga columns read right-to-left; horizontal rows read
            # top-to-bottom, then left-to-right. Bucket jittered row/column starts.
            scale = statistics.median(min(b.bbox[2] - b.bbox[0],
                                          b.bbox[3] - b.bbox[1]) for b in boxes)
            axis = 0 if direction == "vertical" else 1
            bands = []
            for box in sorted(boxes, key=lambda b: (b.bbox[axis], b.bbox[1-axis],
                                                     str(b.box_id))):
                if not bands or box.bbox[axis] - bands[-1][0] > max(1, scale * .5):
                    bands.append((box.bbox[axis], []))
                bands[-1][1].append(box)
            if direction == "vertical":
                bands.reverse()
            ordered = [box for _, band in bands for box in
                       sorted(band, key=lambda b: (b.bbox[1-axis], str(b.box_id)))]
            supported = [i for i in indices if i in regions]
            method = ("region_geometry" if len(supported) == len(indices)
                      else "mixed" if supported else "geometry_fallback")
            groups.append(Text_Group(
                ordered, method=method,
                status="singleton" if len(indices) == 1 else "grouped",
                direction=direction,
                metadata={"source_indices": tuple(indices),
                          "region_status": region_status,
                          "regions": {i: regions[i] for i in supported},
                          "policy": "complete_link_geometry_v1"}))
        # Page order: top-to-bottom with right-to-left ties (manga convention).
        groups.sort(key=lambda g: (min(b.bbox[1] for b in g.group),
                                   -max(b.bbox[2] for b in g.group),
                                   min(g.metadata["source_indices"])))
        return groups

    @staticmethod
    def _manga_direction(box):
        x1, y1, x2, y2 = box.bbox
        if x2 <= x1 or y2 <= y1:
            return "unknown"
        if y2 - y1 >= 1.3 * (x2 - x1):
            return "vertical"
        if x2 - x1 >= 1.3 * (y2 - y1):
            return "horizontal"
        return "unknown"

    def _manga_compatible(self, first, second, relaxed=False):
        direction = self._manga_direction(first)
        if direction == "unknown" or direction != self._manga_direction(second):
            return False
        a, b = first.bbox, second.bbox
        aw, ah = a[2] - a[0], a[3] - a[1]
        bw, bh = b[2] - b[0], b[3] - b[1]
        sa, sb = min(aw, ah), min(bw, bh)
        if max(sa, sb) > 1.6 * min(sa, sb):
            return False
        scale = min(sa, sb)
        dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        ox = max(0, min(a[2], b[2]) - max(a[0], b[0])) / min(aw, bw)
        oy = max(0, min(a[3], b[3]) - max(a[1], b[1])) / min(ah, bh)
        if direction == "vertical":
            columns = (oy >= .6 and abs(a[1] - b[1]) <= 1.5 * scale
                       and dx <= (4 if relaxed else 1.8) * scale)
            fragments = ox >= .7 and dy <= .8 * scale
            return columns or fragments
        rows = (ox >= .55 and abs(a[0] - b[0]) <= 2 * scale
                and dy <= (2.5 if relaxed else 1.3) * scale)
        fragments = oy >= .7 and dx <= .8 * scale
        return rows or fragments

    def _manga_regions(self, text_boxes):
        """Accept only enclosed white regions strongly containing OCR polygons.

        Removing OCR ink from the Otsu mask exposes bubble interiors without
        treating open page background or large panel interiors as bubbles.
        Failure to obtain trustworthy evidence never drops an input box.
        """
        image = cv2.imread(self.image_path)
        if image is None:
            return {}, "image_unavailable"
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, ink = cv2.threshold(
                gray_scale,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                )

        text_mask = np.zeros_like(gray_scale, dtype=np.uint8)

        for box in text_boxes:
            if not box.text.strip(): continue

            poly = box.poly.astype(np.int32)
            cv2.fillPoly(text_mask, [poly], 255)

        free = cv2.bitwise_not(cv2.bitwise_and(ink, cv2.bitwise_not(text_mask)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
        height, width = gray_scale.shape
        regions = {}
        for i, box in enumerate(text_boxes):
            if not box.text.strip() or box.confidence < .5:
                continue
            x1, y1, x2, y2 = box.bbox
            if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height or x2 <= x1 or y2 <= y1:
                continue
            # Local raster avoids a full-page allocation for every OCR polygon.
            polygon = box.poly.astype(np.int32) - np.array([x1, y1])
            mask = np.zeros((y2-y1+1, x2-x1+1), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            values = labels[y1:y2+1, x1:x2+1][mask != 0]
            if not len(values):
                continue
            votes = np.bincount(values, minlength=count)
            label = int(np.argmax(votes[1:]) + 1) if count > 1 else 0
            if label == 0:
                continue
            confidence = float(votes[label] / len(values))
            x, y, w, h, area = stats[label]
            box_area = (x2-x1) * (y2-y1)
            if (confidence >= .9 and x > 0 and y > 0
                    and x+w < width and y+h < height
                    and area >= box_area and area <= 30 * box_area
                    and area / (w*h) >= .55):
                regions[i] = (label, confidence)
        return regions, "region_analysis_complete"

    def _group_boxes_proximity(self, text_boxes: list[Text_Box]):
        if not text_boxes: return []

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
            images_text_groups.append(self._group_boxes_manga(text_boxes))
        return images_text_groups

    def visualize_groups(self, groups: list[Text_Group], output_path="visualized_groups.png"):
        from PIL import Image, ImageDraw, ImageFont

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

    def visualize_groups_polygon(self, groups: list[Text_Group], output_path="visualized_groups_polygon.png"):
        """Visualize text groups with PaddleOCR polygons instead of rectangles."""
        from PIL import Image, ImageDraw, ImageFont

        image = Image.open(self.image_path)
        canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 14)
        except IOError:
            font = ImageFont.load_default()

        print(f"\n--- Visualizing {len(groups)} Groups as Polygons ---")

        for i, tg in enumerate(groups):
            if not tg.group:
                continue

            group_points = np.vstack([box.poly for box in tg.group]).astype(np.float32)
            points = sorted({tuple(map(float, point)) for point in group_points})

            def cross(origin, first, second):
                return (
                        (first[0] - origin[0]) * (second[1] - origin[1])
                        - (first[1] - origin[1]) * (second[0] - origin[0])
                        )

            lower = []
            for point in points:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                    lower.pop()
                lower.append(point)

            upper = []
            for point in reversed(points):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                    upper.pop()
                upper.append(point)

            group_hull = lower[:-1] + upper[:-1]
            group_polygon = [tuple(map(int, point)) for point in group_hull]
            draw.polygon(group_polygon, outline="blue", width=3)

            label_x = int(np.min(group_points[:, 0]))
            label_y = int(np.min(group_points[:, 1])) - 20
            draw.text((label_x, max(0, label_y)), f"Group {i}", fill="blue", font=font)

            print(f"Group {i}:")
            for box in tg.group:
                polygon = [tuple(map(int, point)) for point in box.poly]
                draw.polygon(polygon, outline="red", width=1)

                label = f"{box.text} ({box.confidence:.2f})"
                text_x = int(np.min(box.poly[:, 0]))
                text_y = int(np.min(box.poly[:, 1])) - 15
                if text_y < 0:
                    text_y = int(np.max(box.poly[:, 1])) + 5

                left, top, right, bottom = draw.textbbox((text_x, text_y), label, font=font)
                draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill="red")
                draw.text((text_x, text_y), label, fill="white", font=font)
                print(f"  - Conf: {box.confidence:.4f} | Text: {box.text}")

        canvas.save(output_path)
        print(f"Polygon visualization saved to {output_path}")
        canvas.show()

if __name__ == "__main__":
    image = "images/lmgJZ.jpg"
    ocr = OCR(image)
    results = ocr.process_images()
    ocr.visualize_groups_polygon(results[0])
