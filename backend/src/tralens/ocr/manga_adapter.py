"""Explicitly enabled, local-only Manga OCR adapter (optional dependency)."""
import os
from pathlib import Path

# Prevent Hugging Face from fetching files even when a supplied local model is
# incomplete. No environment values or credentials are inspected here.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from manga_ocr import MangaOcr


class MangaOCRAdapter:
    def __init__(self, model_path):
        if model_path is None or not Path(model_path).is_dir():
            raise ValueError("Manga OCR requires an existing local model directory; downloads are disabled")
        self.recognizer = MangaOcr(pretrained_model_name_or_path=str(Path(model_path).resolve()))

    def __call__(self, crop, *, source_box_id, crop_bounds):
        return self.recognizer(crop)
