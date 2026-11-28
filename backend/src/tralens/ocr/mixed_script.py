"""Local crop rereads for horizontal Chinese captions and vertical Korean mixed script."""
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .chapter_language import HAN, HANGUL
from .paddle_profiles import PROFILES
from .types import OCREngine, Text_Box


def reread_lettering(image, boxes, language, settings):
    @lru_cache(maxsize=2)
    def reader(name):
        from paddleocr import TextRecognition
        directory = settings.get("text_recognition_model_dir")
        if directory and name != settings["text_recognition_model_name"]:
            directory = str(Path(directory).parent / name)
            if not Path(directory).is_dir():
                return None
        return TextRecognition(model_name=name, model_dir=directory, device=settings.get("device"),
                               cpu_threads=settings["cpu_threads"], enable_mkldnn=False)

    if language == "zh":
        recover_caption_initials(image, boxes, reader)
        recover_caption_ends(image, boxes, reader)
    elif language == "ko":
        recover_vertical_hanja(image, boxes, reader)


def recover_caption_initials(image, boxes, reader):
    """Confirm a weak enlarged initial both alone and in the printed caption."""
    attempts = 0
    for index, box in enumerate(boxes):
        x,y,r,b = box.bbox
        height = b-y
        if (attempts >= 6 or box.confidence >= .75 or not 12 <= height <= 250
                or not .65*height <= r-x <= 1.5*height):
            continue
        neighbors = [n for n in boxes if n.confidence >= .85 and len(HAN.findall(n.text)) >= 2
                     and x < n.bbox[0] and -.45*height <= n.bbox[0]-r <= .20*height
                     and n.bbox[2]-n.bbox[0] >= 1.5*height
                     and y <= (n.bbox[1]+n.bbox[3])/2 <= b]
        if len(neighbors) != 1:
            continue
        neighbor = neighbors[0]
        attempts += 1
        _,top,right,bottom = neighbor.bbox
        # A detector rectangle around a drop cap can extend into the artwork below the line.
        top = max(0, min(y, top))
        x = max(0, x)
        base = min(b + 1, bottom + max(2, height // 6))
        if x >= r or top >= base:
            continue
        cap = image[top:base, x:r]
        pad = max(2, round(height * .08))
        framed = cv2.copyMakeBorder(cap, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
        initial, framed_initial, line = reader(PROFILES["v6_medium"][1]).predict([cap, framed, image[top:base, x:right+1]])
        initial = max((initial, framed_initial), key=lambda res: res['rec_score'] if HAN.fullmatch(res['rec_text']) else 0)
        glyph = initial['rec_text']
        confidence = min(float(initial['rec_score']), float(line['rec_score']))
        if HAN.fullmatch(glyph) and confidence >= .90 and line['rec_text'].startswith(glyph + neighbor.text[:2]):
            confirmed = Text_Box(box.box_id, confidence, box.paddle_poly, glyph)
            confirmed.ocr_metadata.update(deepcopy(box.ocr_metadata))
            confirmed.ocr_metadata["candidates"][OCREngine.PADDLE.value] = glyph
            confirmed.ocr_metadata['caption_reread'] = dict(previous_text=box.text, previous_confidence=box.confidence,
                crop=glyph, caption=line['rec_text'], neighbor=neighbor.box_id)
            boxes[index] = confirmed

def recover_caption_ends(image, boxes, reader):
    """A printed caption can end with a larger character outside Paddle's polygon."""
    attempts = 0
    for line in tuple(boxes):
        x, y, r, b = line.bbox
        h = b - y
        if (not 12 <= h <= 160 or r-x < 2*h or line.confidence < .85
                or not re.fullmatch(r"[一-鿿]{3,}", line.text) or attempts >= 12 or len(boxes) >= 512):
            continue
        left, top = max(0, r-2), max(0, y-h//3)
        strip = image[top:b+h//2, left:r+round(1.5*h)]
        if not strip.size:
            continue
        mask = (strip.max(axis=2) < 162).astype(np.uint8)
        stats = cv2.connectedComponentsWithStats(mask)[2][1:]
        for dx, dy, width, height, area in stats:
            if not (dx <= .30*h and .55*h <= width <= 1.45*h and .65*h <= height <= 1.5*h
                    and area >= .12*width*height):
                continue
            l, t = int(left+dx), int(top+dy)
            rr, bb = l+int(width), t+int(height)
            if abs((t+bb)-(y+b)) > h or any(other is not line and other.bbox[0] < rr and
                    other.bbox[2] > l and other.bbox[1] < bb and other.bbox[3] > t for other in boxes):
                continue
            attempts += 1
            pad = max(2, h//12)
            glyph = image[max(0,t-pad):bb+pad, max(0,l-pad):rr+pad]
            full = image[max(0,min(t,y)-pad):max(bb,b)+pad, max(0,x):rr+pad]
            isolated, context = reader(PROFILES["v6_medium"][1]).predict([glyph, full])
            suffix = isolated["rec_text"]
            confidence = min(float(isolated["rec_score"]), float(context["rec_score"]))
            if confidence >= .90 and HAN.fullmatch(suffix) and context["rec_text"].strip() == line.text + suffix:
                box = Text_Box(max(box.box_id for box in boxes)+1, confidence,
                               [(l,t),(rr-1,t),(rr-1,bb-1),(l,bb-1)], suffix)
                box.ocr_metadata["caption_extension"] = dict(neighbor=line.box_id, caption=context["rec_text"])
                boxes.append(box)
            break


def trailing_glyphs(column):
    """Find the last two separated, roughly square ink bands in a vertical column."""
    width = column.shape[1]
    ink = (cv2.cvtColor(column, cv2.COLOR_BGR2GRAY) < 155).sum(axis=1) >= max(1, width*.06)
    bands = []
    for start, end in zip(np.flatnonzero(ink & ~np.r_[False, ink[:-1]]),
                          np.flatnonzero(ink & ~np.r_[ink[1:], False])+1):
        if bands and end - bands[-1][0] < .85*width:
            bands[-1][1] = end
        else:
            bands.append([start,end])
    if len(bands) >= 4 and all(.48*width <= b-a <= 1.4*width for a,b in bands[-2:]):
        return (bands[-3][1]+bands[-2][0])//2, bands[-2:]


def recover_vertical_hanja(image, boxes, reader):
    """Korean OCR can silently omit a Hanja compound alongside correctly read Hangul."""
    attempts = 0
    for index, box in enumerate(boxes):
        x,y,r,b = box.bbox
        if (attempts >= 12 or not 12 <= r-x <= 150 or b-y < 4*(r-x) or x < 0 or y < 0
                or not HANGUL.search(box.text) or box.confidence < .55):
            continue
        column = image[y:b+1, x:r+1]
        if not column.size:
            continue
        tail = trailing_glyphs(column)
        if not tail:
            continue
        han = reader(PROFILES["v6_medium"][1])
        if han is None:
            return
        cut, bands = tail
        attempts += 1
        separate = [cv2.copyMakeBorder(column[a:b], 2,2,3,3, cv2.BORDER_CONSTANT, value=(255,255,255))
                    for a,b in bands]
        pair, first, second = han.predict([np.rot90(column[cut:]).copy(), *separate])
        suffix = pair["rec_text"]
        confidence = min(float(v["rec_score"]) for v in (pair, first, second))
        if (confidence < .80 or len(suffix) != 2 or not all(HAN.fullmatch(c) for c in suffix)
                or suffix != first["rec_text"] + second["rec_text"]):
            continue
        korean = reader(PROFILES["v6_korean"][1])
        pad = max(2, (r-x)//16)
        prefixes = [np.rot90(column[:cut+offset]).copy() for offset in (-pad, 0, pad)]
        accepted = {}
        original = "".join(HANGUL.findall(box.text))
        for result in korean.predict(prefixes):
            prefix = result["rec_text"].strip()
            letters = "".join(HANGUL.findall(prefix))
            # Preserve the original Hangul in order, allowing the tighter reread to recover missing letters.
            sequence = iter(letters)
            if (float(result["rec_score"]) >= .75 and len(original) >= 2 and
                    len(original) <= len(letters) <= len(original)+2 and
                    all(letter in sequence for letter in original) and re.fullmatch(r"[가-힣 ]+", prefix)):
                accepted[prefix] = max(accepted.get(prefix, 0), float(result["rec_score"]))
        if not accepted:
            continue
        prefix = max(accepted, key=lambda text: (len(text.replace(" ", "")), accepted[text]))
        recovered = Text_Box(box.box_id, min(confidence, accepted[prefix]), box.paddle_poly, prefix + " " + suffix)
        recovered.ocr_metadata.update({key:value for key,value in box.ocr_metadata.items() if key != "candidates"})
        recovered.ocr_metadata["mixed_script_reread"] = dict(previous_text=box.text, previous_confidence=box.confidence,
                                                             hangul=prefix, hanja=suffix, cut=int(cut))
        boxes[index] = recovered
