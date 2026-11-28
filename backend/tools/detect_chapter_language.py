"""Disposable worker for chapter OCR language detection."""
import json
import sys
from tralens.ocr.chapter_language import detect

if __name__ == "__main__":
    print(json.dumps(detect(sys.argv[1], sys.argv[2], sys.argv[3:])))
