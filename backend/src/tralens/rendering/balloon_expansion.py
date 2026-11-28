"""Finish overflowing dialogue on a closed, borderless white balloon."""
import cv2
import numpy as np
from PIL import Image

from . import natural


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def finish(clean, source, blocked, layer, proof, font, coverage):
    """Try the existing paper first. Any growth is a backing below the English."""
    if source.size > 1_048_576 or proof.get('contrast_backing'):
        return layer, proof
    ink = np.asarray(layer.getchannel('A')) > 0
    size = proof['size']
    white = np.uint8(clean.min(axis=2) >= 247)
    if not (ink & (white == 0)).any():
        return layer, proof

    # Opening removes thin tails and narrow connections to surrounding paper.
    bodies = cv2.morphologyEx(white, cv2.MORPH_OPEN, disk(max(2, size // 2)))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bodies, connectivity=4)
    hits = np.bincount(labels[source], minlength=len(stats))
    hits[0] = 0
    owner = hits.argmax()
    if not hits[owner] or hits[owner] < .65 * source.sum():
        return layer, proof
    x, y, w, h, area = stats[owner]
    body = np.uint8(labels == owner)
    if (min(x, y, clean.shape[1] - x - w, clean.shape[0] - y - h) < 2
            or not .52 < area / (w * h) < .93 or (ink & (body != 0)).sum() < .65 * ink.sum()):
        return layer, proof

    # A dark printed rim belongs to the drawing. Never paint over it.
    rim = (cv2.dilate(body, disk(3)) != 0) & (white == 0)
    if not rim.any() or (clean[rim].max(axis=1) < 120).mean() > .10:
        return layer, proof

    pad = max(2, round(size * .1))
    inset = cv2.erode(body, disk(pad)) != 0
    inside = inset & ~blocked
    if inside.any():
        try:
            new_layer, fit = natural.layout(proof['original_text'], font, coverage,
                Image.fromarray(inside.astype(np.uint8) * 255),
                {'min_size': size, 'max_size': size, 'outline': 0, 'leading': .08})
        except ValueError:
            pass
        else:
            fit['balloon_backing'] = 'rewrapped_inside'
            return new_layer, fit

    # Round only the protruding corners. Erode before forming the hull so
    # unused sides keep their contour, and exclude the tail from the hull.
    anchors = inset | ink
    hull = cv2.convexHull(cv2.findNonZero(anchors.astype(np.uint8)))
    shape = cv2.fillConvexPoly(np.zeros_like(body), hull, 255)
    backing = cv2.GaussianBlur(cv2.dilate(shape, disk(pad)), (3, 3), .7)
    # Protect disconnected white shapes (including neighboring balloons).
    _, sheets, regions, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    sheet = np.bincount(sheets[body != 0]).argmax()
    spreading = backing != 0
    other_paper = regions[:, cv2.CC_STAT_AREA] >= pad * pad
    other_paper[[0, sheet]] = False
    if np.any(spreading & other_paper[sheets]):
        return layer, proof
    backing[white != 0] = 0
    growth = backing != 0
    if (not growth.any() or growth.sum() > .4 * area or (growth & blocked).any()
            or growth[0].any() or growth[-1].any() or growth[:, 0].any() or growth[:, -1].any()):
        return layer, proof
    distance = cv2.distanceTransform(1 - body, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    if distance[growth].max() > min(.8 * size, .38 * min(w, h)):
        return layer, proof

    # Reject ink, including thin strokes that take up little of the total area.
    gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    detail = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, disk(3))
    dark = clean.max(axis=2) < 105
    # The white body itself makes its immediate colored edge look like a
    # dark stroke to a black-hat filter. Test texture outside that collar.
    collar = cv2.dilate(white, disk(3)) != 0
    color_detail = cv2.morphologyEx(clean, cv2.MORPH_BLACKHAT, disk(3)).max(axis=2)
    danger = (white == 0) & (dark | (((detail > 55) | (color_detail > 105)) & ~collar))
    if np.any(growth & danger):
        # Let the extreme rim taper away from a nearby stroke. If linework
        # enters the body of the proposed extension, abandon the whole edit.
        interior = cv2.erode(np.uint8(spreading), disk(pad + 1)) != 0
        if (growth & danger & interior).any():
            return layer, proof
        separation = cv2.distanceTransform(np.uint8(~danger), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        backing = np.minimum(backing, np.uint8(np.clip(separation - 1, 0, 1) * 255))
        glyph = (np.asarray(layer)[:, :, :3].max(axis=2) < 120) & ink
        if np.any(glyph & (white == 0) & (backing < 245)):
            return layer, proof
        growth = backing != 0

    result = Image.new('RGBA', layer.size, 'white')
    result.putalpha(Image.fromarray(backing))
    result.alpha_composite(layer)
    return result, {**proof, 'balloon_backing': 'expanded', 'balloon_added_pixels': int(growth.sum())}
