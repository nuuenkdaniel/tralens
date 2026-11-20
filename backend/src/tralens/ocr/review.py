import math
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image

from .types import MediaType, OCREngine, OCRReviewPolicy, OCRStatus, normalize_ocr_text


class ReviewMixin:
    def _review_crop(self, box, boxes, image, padding):
        """Original-pixel crop with polygon-aware, confidence-independent isolation."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            return None, None, "image_unavailable"
        # Convex OCR polygons only: do not repair a malformed polygon by taking
        # its hull, which could silently enlarge the inferred text footprint.
        epsilon = 1e-7  # Numerical area/edge tolerance, not a glyph-size cutoff.

        def signed_area(poly):
            if len(poly) < 3:
                return 0.
            return float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                         - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))) / 2

        def cross(edge, points):
            return edge[0] * points[..., 1] - edge[1] * points[..., 0]

        def validate(poly):
            try:
                poly = np.asarray(poly, dtype=np.float64)
            except (TypeError, ValueError):
                return None, "invalid_polygon"
            if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
                return None, "invalid_polygon"
            if np.array_equal(poly[0], poly[-1]):
                poly = poly[:-1]
            area = signed_area(poly)
            if not math.isfinite(area):
                return None, "invalid_polygon"
            if abs(area) <= epsilon:
                if any(np.any(abs(cross(end-start, poly-start)) > epsilon)
                       for start, end in zip(poly, np.roll(poly, -1, axis=0))):
                    return None, "invalid_polygon"
                return None, "invalid_crop"
            orientation = 1 if area > 0 else -1
            for start, end in zip(poly, np.roll(poly, -1, axis=0)):
                if np.array_equal(start, end) or np.any(
                        orientation * cross(end-start, poly-start) < -epsilon):
                    return None, "invalid_polygon"
            return poly, None

        def intersection(subject, clip):
            # Sutherland-Hodgman in float64 preserves fractional Paddle geometry.
            orientation = 1 if signed_area(clip) > 0 else -1
            output = subject
            for start, end in zip(clip, np.roll(clip, -1, axis=0)):
                if not len(output):
                    break
                vertices = []
                previous = output[-1]
                previous_side = orientation * cross(end-start, previous-start)
                for current in output:
                    current_side = orientation * cross(end-start, current-start)
                    if (current_side >= 0) != (previous_side >= 0):
                        vertices.append(previous + (current-previous)
                                        * previous_side / (previous_side-current_side))
                    if current_side >= 0:
                        vertices.append(current)
                    previous, previous_side = current, current_side
                output = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
            return output

        def rectangle(bounds):
            a, b, c, d = bounds
            return np.array([(a, b), (c, b), (c, d), (a, d)], dtype=np.float64)

        def overlap(poly, bounds):
            return abs(signed_area(intersection(poly, rectangle(bounds))))

        def extent(poly):
            return (math.floor(float(np.min(poly[:, 0]))),
                    math.floor(float(np.min(poly[:, 1]))),
                    math.ceil(float(np.max(poly[:, 0]))),
                    math.ceil(float(np.max(poly[:, 1]))))

        height, width = image.shape[:2]
        page = rectangle((0, 0, width, height))
        source, error = validate(box.paddle_poly)
        if error:
            return None, None, error
        source = intersection(source, page)
        source_area = abs(signed_area(source))
        if source_area <= epsilon:
            return None, None, "invalid_crop"
        a, b, c, d = extent(source)
        core = (max(0, a), max(0, b), min(width, c), min(height, d))
        bounds = (max(0, core[0]-padding), max(0, core[1]-padding),
                  min(width, core[2]+padding), min(height, core[3]+padding))
        audit = {"policy": "polygon_isolation_v1", "neighbors": []}
        box.ocr_metadata["crop_isolation"] = audit
        source_text = normalize_ocr_text(box.paddle_text)
        neighbors = []
        tiny_area_total = 0.
        for neighbor in sorted((n for n in boxes if n is not box),
                               key=lambda n: str(n.paddle_box_id)):
            polygon, error = validate(neighbor.paddle_poly)
            entry = {"source_box_id": neighbor.paddle_box_id}
            audit["neighbors"].append(entry)
            if error:
                try:
                    raw = np.asarray(neighbor.paddle_poly, dtype=np.float64)
                except (TypeError, ValueError):
                    raw = None
                if (raw is not None and raw.ndim == 2 and raw.shape[1] == 2 and len(raw)
                        and np.isfinite(raw).all()):
                    a, b, c, d = extent(raw)
                    if max(a, bounds[0]) >= min(c, bounds[2]) or max(b, bounds[1]) >= min(d, bounds[3]):
                        entry["decision"] = "invalid_outside_crop"
                        continue
                entry["decision"] = "invalid_ambiguous_neighbor"
                return None, bounds, "invalid_neighbor_polygon"
            polygon = intersection(polygon, page)
            area = abs(signed_area(polygon))
            if area <= epsilon or overlap(polygon, bounds) <= epsilon:
                entry["decision"] = "outside_crop"
                continue
            shared = abs(signed_area(intersection(polygon, source)))
            entry.update(source_overlap_area=shared, neighbor_area=area)
            neighbor_text = normalize_ocr_text(neighbor.paddle_text)
            # Geometry AND corroborating text, never confidence. Tiny contained
            # fragments have a total 10% area budget; near-duplicates must cover
            # at least 95% of BOTH polygons and agree on their full text.
            duplicate = (source_text and neighbor_text == source_text
                         and shared >= .95 * max(area, source_area))
            fragment = (source_text and neighbor_text and neighbor_text in source_text
                        and abs(area-shared) <= epsilon
                        and tiny_area_total + area <= .1 * source_area)
            if duplicate or fragment:
                entry["decision"] = "near_duplicate" if duplicate else "contained_text_fragment"
                if not duplicate:
                    tiny_area_total += area
                continue
            neighbors.append(polygon)
            # Disjoint source polygons do NOT guarantee a clean rectangular crop.
            if overlap(polygon, core) > epsilon:
                entry["decision"] = "source_conflict" if shared > epsilon else "rectangle_contamination"
                return None, core, "neighbor_overlap" if shared > epsilon else "crop_contamination"
            entry["decision"] = "padding_only"
            # Only trim padding; never cut the source polygon to manufacture
            # isolation. The unpadded core is always a safe candidate here.
            a, b, c, d = extent(polygon)
            left = {bounds[0], core[0]}
            top = {bounds[1], core[1]}
            right = {bounds[2], core[2]}
            bottom = {bounds[3], core[3]}
            if bounds[0] <= c <= core[0]:
                left.add(c)
            if bounds[1] <= d <= core[1]:
                top.add(d)
            if core[2] <= a <= bounds[2]:
                right.add(a)
            if core[3] <= b <= bounds[3]:
                bottom.add(b)
            candidates = [(l, t, r, s) for l in left for t in top for r in right for s in bottom
                          if overlap(polygon, (l, t, r, s)) <= epsilon]
            bounds = min(candidates, key=lambda v: (-(v[2]-v[0])*(v[3]-v[1]), v))
        # Final check against the actual supplied rectangle, not just the source.
        if any(overlap(polygon, bounds) > epsilon for polygon in neighbors):
            return None, bounds, "crop_contamination"
        x1, y1, x2, y2 = bounds
        rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), bounds, None

    def review_boxes(self, text_boxes, image, *, media_type=None, regions=None):
        """Assess raw detections without dropping them or comparing engine scores."""
        media = MediaType(media_type if media_type is not None else
                          getattr(self, "media_type", MediaType.MANGA))
        policy = getattr(self, "review_policy", OCRReviewPolicy())
        recognizer = getattr(self, "manga_recognizer", None)
        if regions is None:
            regions = (self._manga_regions(text_boxes, image=image, include_uncertain=True)[0]
                       if media == MediaType.MANGA and image is not None else {})
        for index, box in enumerate(text_boxes):
            # Reprocessing the same objects cannot spend another attempt or
            # reset a previously selected candidate/status.
            if box.ocr_metadata.get("assessed"):
                continue
            box.ocr_metadata.update(assessed=True, media_type=media.value,
                                    source_box_id=box.paddle_box_id,
                                    region_evidence=regions.get(index))
            nonblank = bool(normalize_ocr_text(box.paddle_text))
            score = box.paddle_confidence
            if nonblank and math.isfinite(score) and score >= policy.accept_confidence:
                box.set_status(OCRStatus.ACCEPTED, "paddle_confidence_accepted")
                continue
            eligible = nonblank or (policy.blank_region_review and index in regions)
            status = (OCRStatus.REVIEW if eligible and not policy.reject_low_confidence
                      else OCRStatus.REJECTED)
            box.set_status(status, "uncertain_paddle" if nonblank else "blank_paddle")
            if media != MediaType.MANGA or not eligible or status == OCRStatus.REJECTED:
                continue
            crop, bounds, error = self._review_crop(box, text_boxes, image, policy.crop_padding)
            box.ocr_metadata["crop_bounds"] = bounds
            if error:
                box.set_status(OCRStatus.REJECTED if error == "invalid_crop" else OCRStatus.REVIEW,
                               error)
                continue
            if recognizer is None:
                box.set_status(OCRStatus.REVIEW, "manga_recognizer_disabled")
                continue
            box.ocr_metadata["fallback_attempts"] = 1
            box.set_status(OCRStatus.REVIEW, "manga_recognition_in_progress")
            try:
                candidate = recognizer(crop, source_box_id=box.paddle_box_id, crop_bounds=bounds)
            except Exception:
                box.set_status(OCRStatus.REVIEW, "manga_recognition_failed")
                raise
            if not isinstance(candidate, str):
                box.set_status(OCRStatus.REVIEW, "invalid_manga_recognizer_output")
                raise TypeError("Manga OCR recognizer must return text only (str)")
            box.ocr_metadata["candidates"][OCREngine.MANGA.value] = candidate
            paddle_normal = normalize_ocr_text(box.paddle_text)
            manga_normal = normalize_ocr_text(candidate)
            if paddle_normal and paddle_normal == manga_normal:
                box.set_status(OCRStatus.ACCEPTED, "normalized_engine_agreement")
            elif not manga_normal:
                box.set_status(OCRStatus.REVIEW, "blank_manga_candidate")
            elif not paddle_normal:
                box.selected_text = candidate
                box.selected_engine = OCREngine.MANGA
                box.set_status(OCRStatus.REVIEW, "blank_paddle_recovery_requires_review")
            else:
                box.ocr_metadata["differing_spans"] = [
                    {"operation": op, "paddle_span": (a, b), "manga_span": (c, d),
                     "paddle_text": paddle_normal[a:b], "manga_text": manga_normal[c:d]}
                    for op, a, b, c, d in SequenceMatcher(
                        None, paddle_normal, manga_normal, autojunk=False).get_opcodes()
                    if op != "equal"]
                box.set_status(OCRStatus.REVIEW, "engine_disagreement")
        return text_boxes
