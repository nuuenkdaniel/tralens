"""Preserve surviving paint inside a broad lettering box on a light background."""
import cv2
import numpy as np


def refine(crop, commit, unavailable, candidate):
    x, y, width, height = cv2.boundingRect(commit.astype(np.uint8))
    if not (16 <= min(width, height) and width * height <= 8192
            and commit.sum() == width * height):
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    ring = cv2.dilate(commit.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ring &= ~(commit | unavailable)
    if ring.sum() < 32:
        return None
    values = gray[commit]
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ink = commit & (gray <= threshold)
    if not ink.any() or ink.sum() == values.size:
        return None
    dark = gray[ink]
    surround = np.percentile(gray[ring], 1)
    kernel = np.ones((3, 3), np.uint8)
    # On antialiased printing the tones between ink and paint must sit next
    # to a solid core. Otherwise the dark shape may be part of the artwork.
    if np.ptp(dark) > 24 or int(gray[commit & ~ink].min()) - int(dark.max()) < 28:
        peak = int(np.bincount(dark).argmax())
        if surround < peak + 64:
            return None
        core = commit & (gray <= peak + 8)
        middle = commit & (gray < (peak + surround) / 2)
        if np.any(middle & (cv2.dilate(core.astype(np.uint8), kernel) == 0)):
            return None
        ink = cv2.dilate(middle.astype(np.uint8), kernel).astype(bool)
    elif surround < int(dark.max()) + 35:
        return None
    if not .08 <= ink.sum() / values.size <= .75:
        return None
    collar = cv2.dilate(ink.astype(np.uint8), kernel).astype(bool) & ~ink
    if np.any(collar & ~commit) or np.median(gray[collar]) > np.percentile(gray[ring], 95) + 8:
        return None
    return interpolate(crop, ink, unavailable & ~commit, candidate)


def interpolate(crop, ink, unavailable, candidate):
    """Blend short chords between similar pairs of known colors across each stroke."""
    y, x = np.nonzero(ink)
    height, width = ink.shape
    blocked = ink | unavailable
    weighted = np.zeros((len(x), 3))
    total = np.zeros(len(x))
    for angle in np.arange(16) * np.pi / 16:
        ends = []
        for sign in (-1, 1):
            distance = np.zeros(len(x), dtype=int)
            colors = np.zeros((len(x), 3))
            for step in range(1, 35):
                yy = y + round(sign * step * np.sin(angle))
                xx = x + round(sign * step * np.cos(angle))
                inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
                yy, xx = yy.clip(0, height - 1), xx.clip(0, width - 1)
                hit = inside & ~blocked[yy, xx] & (distance == 0)
                colors[hit] = crop[yy[hit], xx[hit]]
                distance[hit] = step
                if distance.all():
                    break
            ends.append((distance, colors))
        (a, first), (b, second) = ends
        span = a + b
        cost = np.abs(first - second).mean(axis=1) + .65 * span
        weight = np.where((a > 0) & (b > 0), np.exp(-cost / 4), 0)
        blend = (first * b[:, None] + second * a[:, None]) / span.clip(1)[:, None]
        weighted += weight[:, None] * blend
        total += weight
    output = crop.copy()
    # Where no chord agrees on the color, keep the learned repair.
    reliable = total > np.exp(-12)
    output[y, x] = candidate[y, x]
    output[y[reliable], x[reliable]] = (weighted[reliable] / total[reliable, None]).round()
    return output
