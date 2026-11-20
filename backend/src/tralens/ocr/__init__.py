"""Public OCR facade and data types; engines initialize only on explicit use."""
from .types import MediaType, OCREngine, OCRReviewPolicy, OCRStatus, Text_Box, Text_Group

__all__ = ["OCR", "MediaType", "OCREngine", "OCRReviewPolicy", "OCRStatus", "Text_Box", "Text_Group"]


def __getattr__(name):
    if name == "OCR":
        from .detection import OCR
        return OCR
    raise AttributeError(name)
