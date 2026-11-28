"""Identify a chapter locally from the same lettering read by both recognizers."""
from pathlib import Path
import re
import time
import unicodedata

LANGUAGES = {"ja": "Japanese", "zh": "Chinese", "ko": "Korean"}
KANA = re.compile(r"[ぁ-ゖァ-ヺｦ-ｯｱ-ﾝ]")
HAN = re.compile(r"[㐀-䶿一-鿿\U00020000-\U0002a6df\U0002a700-\U0002ebe0\U00030000-\U0003134a]")
HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣᄀ-하-ᅵᆨ-ᇂ]")


def language_votes(shared, korean):
    counts = dict.fromkeys(LANGUAGES, 0)
    for (text, score), (other, prob) in zip(shared, korean, strict=True):
        text = unicodedata.normalize("NFC", text)
        syllables = len(HANGUL.findall(unicodedata.normalize("NFC", other)))
        kana, han = len(KANA.findall(text)), len(HAN.findall(text))
        if syllables and prob >= .68 and (not (kana or han) or prob + .04 >= score):
            counts["ko"] += min(12, syllables)
        elif score >= .65:
            if kana:
                counts["ja"] += min(16, 4 * kana + min(4, han))
            elif han >= 3:
                counts["zh"] += min(8, han)
    return counts


def detect(model_root, device, pages):
    from PIL import Image
    start = time.monotonic()
    import numpy as np

    def load(cls, name):
        return cls(model_name=name, model_dir=str(Path(model_root)/name), device=device,
                   cpu_threads=2, enable_mkldnn=False)

    detector = None
    votes = dict.fromkeys(LANGUAGES, 0)
    sampled = 0
    for path in pages:
        with Image.open(path) as page:
            rgb = page.convert("RGB")
        if all(low == high for low, high in rgb.getextrema()):
            continue
        if detector is None:
            from paddleocr import TextDetection, TextRecognition
            from .paddle_profiles import PROFILES
            det, shared = PROFILES["v6_medium"]
            detector = load(TextDetection, det)
            readers = [load(TextRecognition, name) for name in (shared, PROFILES["v6_korean"][1])]
        bgr = np.asarray(rgb)[:, :, ::-1].copy()
        polys = next(iter(detector.predict(bgr, limit_side_len=800, limit_type="max")))["dt_polys"]
        bounds = []
        for polygon in polys:
            x, y = np.maximum(polygon.min(axis=0), 0).astype(int)
            r, b = np.minimum(polygon.max(axis=0)+1, rgb.size).astype(int)
            if min(r-x, b-y) >= 7:
                bounds.append((x, y, r, b))
        sampled += 1
        if not bounds:
            continue
        # Sample across the page, even on dense pages with hundreds of regions.
        crops = []
        for index in np.linspace(0, len(bounds) - 1, min(18, len(bounds)), dtype=int):
            x, y, r, b = bounds[index]
            crop = bgr[y:b, x:r]
            crops.append(np.ascontiguousarray(np.rot90(crop) if b-y >= 1.5*(r-x) else crop))
        pairs = [[(r["rec_text"], float(r["rec_score"])) for r in reader.predict(crops, batch_size=8)]
                 for reader in readers]
        for lang, count in language_votes(*pairs).items():
            votes[lang] += count
        first, second = sorted(votes.values(), reverse=True)[:2]
        if first >= 12 and first >= 2 * second + 4:
            break
    ranked = sorted(votes, key=votes.get, reverse=True)
    confident = votes[ranked[0]] >= 4 and votes[ranked[0]] > votes[ranked[1]] * 1.3
    return dict(language=ranked[0] if confident else "ja", fallback=not confident,
                pages=sampled or len(pages), seconds=round(time.monotonic()-start, 3), votes=votes)
