import copy
import math
import unicodedata
from dataclasses import dataclass
from enum import Enum

import numpy as np


class MediaType(str, Enum):
    MANGA = "manga"
    OTHER = "other"


class OCRStatus(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"


class OCREngine(str, Enum):
    PADDLE = "paddle"
    MANGA = "manga"


@dataclass(frozen=True)
class OCRReviewPolicy:
    accept_confidence: float = .85
    reject_low_confidence: bool = False
    blank_region_review: bool = True
    crop_padding: int = 2

    def __post_init__(self):
        if not 0 <= self.accept_confidence <= 1:
            raise ValueError("accept_confidence must be between 0 and 1")
        if not isinstance(self.crop_padding, int) or not 0 <= self.crop_padding <= 4:
            raise ValueError("crop_padding must be an integer between 0 and 4")


def normalize_ocr_text(text):
    # Keep punctuation, ASCII and digits: all can be meaningful manga text.
    return "".join(unicodedata.normalize("NFKC", text).split())


class Text_Box:
    def __init__(self, box_id, confidence, poly, text, review=0):
        self.box_id = box_id
        self.confidence = confidence
        self.poly = np.array(poly, dtype=np.float32)
        self.text = text
        self.review = review
        self.paddle_text = text
        self.paddle_confidence = confidence
        self.paddle_poly = np.array(poly, copy=True)
        self.paddle_box_id = box_id
        self.selected_engine = OCREngine.PADDLE
        self.status = (OCRStatus.REVIEW if review else OCRStatus.REJECTED
                       if not normalize_ocr_text(text) else OCRStatus.ACCEPTED
                       if math.isfinite(confidence) and confidence >= OCRReviewPolicy().accept_confidence
                       else OCRStatus.REVIEW)
        self.review = int(self.status == OCRStatus.REVIEW)
        self.reason = "unassessed_paddle"
        self.ocr_metadata = {"candidates": {OCREngine.PADDLE.value: text},
                             "fallback_attempts": 0}

        x1 = int(np.min(self.poly[:, 0]))
        y1 = int(np.min(self.poly[:, 1]))
        x2 = int(np.max(self.poly[:, 0]))
        y2 = int(np.max(self.poly[:, 1]))
        self.bbox = (x1, y1, x2, y2)

    @property
    def selected_text(self):
        return self.text

    @selected_text.setter
    def selected_text(self, value):
        self.text = value

    def set_status(self, status, reason):
        self.status = OCRStatus(status)
        self.reason = reason
        self.review = int(self.status == OCRStatus.REVIEW)

    def transformed(self, text):
        """Preserve OCR evidence when a downstream caller changes display text."""
        result = copy.deepcopy(self)
        result.ocr_metadata["ocr_selected_text"] = self.selected_text
        result.text = text
        return result


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
