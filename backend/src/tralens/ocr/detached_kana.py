"""Recover short handwritten readings from text-mask islands without a detector box."""
import re

import cv2
import numpy as np
from PIL import Image, ImageOps


def islands(probability, claimed):
    seed = (probability > 90) & (claimed == 0)
    joined = cv2.dilate(np.uint8(seed), np.ones((11, 11), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    for index, (x, y, w, h, _) in enumerate(stats[1:], 1):
        if min(w, h) < 20 or max(w, h) > 380:
            continue
        tile = np.s_[y:y+h, x:x+w]
        support = (labels[tile] == index) & seed[tile]
        dx, dy, width, height = cv2.boundingRect(np.uint8(support))
        if min(width, height) < 11 or support.sum() < 70 or probability[tile][support].max() < 165:
            continue
        left, top = int(x + dx), int(y + dy)
        if not claimed[top:top+height, left:left+width].any():
            yield left, top, left + width, top + height


def ink_views(image, probability, bounds):
    left, top, right, bottom = bounds
    margin = max(8, round(.48 * min(right-left, bottom-top)))
    x, y = max(0, left-margin), max(0, top-margin)
    region = (x, y, min(image.width, right+margin), min(image.height, bottom+margin))
    native = image.crop(region)
    gray = np.asarray(native.convert('L'))
    tile = probability[y:region[3], x:region[2]]
    center = np.s_[top-y:bottom-y, left-x:right-x]
    background = np.median(gray[center])
    # In dense flat lettering, ink can occupy a majority of its tight rectangle.
    solid = gray[center][tile[center] >= 165]
    if solid.size and np.mean(np.abs(solid.astype(float) - background)) < 12:
        unmarked = gray[tile <= 28]
        if unmarked.size:
            background = np.median(unmarked)
    ink = gray < background-44 if background >= 128 else gray > background+44
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink))
    votes = np.bincount(labels[tile > 90], minlength=len(stats))
    votes[0] = 0
    cx, cy, w, h, area = stats.T
    retained = ((votes >= 3) & (votes >= .30 * area) & (cx > 0) & (cy > 0)
                & (cx+w < native.width) & (cy+h < native.height))
    # A clipped stroke often belongs to a much larger drawing outside the island.
    # A pair of sweat drops or motion lines can read identically as two kana.
    if retained.sum() < 3 or votes[~retained].sum() > .16 * votes[retained].sum():
        return
    mask = np.uint8(retained[labels])
    dx, dy, w, h = cv2.boundingRect(mask)
    rect = (x+dx, y+dy, x+dx+w, y+dy+h)
    halo = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(bool)
    pixels = np.asarray(native)
    foreground = pixels if background >= 128 else 255-pixels
    isolated = Image.fromarray(np.where(halo[:, :, None], foreground, 255).astype(np.uint8))
    isolated = ImageOps.expand(isolated.crop((dx, dy, dx+w, dy+h)), 4, 'white')
    pad = 4
    original = image.crop((max(0, rect[0]-pad), max(0, rect[1]-pad),
                           min(image.width, rect[2]+pad), min(image.height, rect[3]+pad)))
    return rect, isolated, original, mask[dy:dy+h, dx:dx+w]


def terminal_dot(image, bounds, word, reader):
    """A detached dot can make a curved final stroke a question mark."""
    left, top, right, bottom = bounds
    width = right - left
    if not word.endswith(('っ', 'ッ')) or bottom-top < width:
        return
    tail = image.crop((left, bottom, right, min(image.height, bottom+max(8, round(.5*width)))))
    if not tail.height:
        return
    gray = np.asarray(tail.convert('L'))
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray < np.median(gray)-50))
    limit = max(3, round(width*.15))
    dots = [(x, y, w, h) for x,y,w,h,area in stats[1:]
            if 0 < x and x+w < width and 0 < y and y+h < tail.height
            and 2 <= w <= limit and 2 <= h <= limit and area >= .5*w*h and .5 <= w/h <= 2]
    if len(dots) != 1:
        return
    x, y, w, h = map(int, dots[0])
    dot = (left+x, bottom+y, left+x+w, bottom+y+h)
    # The dot should sit under the hook, next to the terminal ink.
    tip = image.crop((dot[0]-limit, bottom-limit, dot[2]+limit, bottom)).convert('L')
    if np.min(tip) >= np.median(gray)-50:
        return
    extended = (left, top, right, dot[3])
    punctuated = word[:-1] + '?'
    for pad in (2, 4):
        crop = image.crop((max(0,left-pad), max(0,top-pad), min(image.width,right+pad),
                           min(image.height,dot[3]+pad)))
        if (reader(crop) or '').replace('？', '?') != punctuated:
            return
    return punctuated, extended, dot


def readings(image, probability, polygons, reader):
    claimed = np.zeros(probability.shape, np.uint8)
    for poly in polygons:
        cv2.fillPoly(claimed, [np.int32(poly)], 1)
    claimed = cv2.dilate(claimed, np.ones((7, 7), np.uint8))
    attempts = 0
    for bounds in islands(probability, claimed):
        crops = ink_views(image, probability, bounds)
        if crops is None:
            continue
        (left, top, right, bottom), isolated, original, _ = crops
        if claimed[top:bottom, left:right].any():
            continue
        attempts += 1
        try:
            word = reader(isolated)
            if (isinstance(word, str) and re.fullmatch(r'[ぁ-ゖァ-ヺ][ぁ-ゖァ-ヺー]{1,7}', word)
                    and reader(original) == word):
                # Tilted English lettering can also look like kana when read down the page.
                if any(re.fullmatch(r'[A-Za-z]{3,}', reader(isolated.transpose(turn)) or '')
                       for turn in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270)):
                    continue
                dot = None
                punctuated = terminal_dot(image, (left, top, right, bottom), word, reader)
                if punctuated:
                    candidate, extended, dot = punctuated
                    if claimed[top:extended[3], left:right].any():
                        dot = None
                    else:
                        word, bottom = candidate, extended[3]
                score = float(probability[top:bottom, left:right].max()) / 255
                yield word, (left, top, right, bottom), score, bounds, dot
                claimed[top:bottom, left:right] = 1
        except (OSError, ValueError, RuntimeError, cv2.error):
            pass
        if attempts >= 12:
            break
