"""Finish overflowing dialogue on a closed, borderless white balloon."""
import cv2
import numpy as np
from PIL import Image

from . import natural


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def finish(clean, source, blocked, layer, proof, font, coverage):
    """Try the existing paper first. Any growth is a backing below the English."""
    if source.size > 1_048_576:
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

    pad = max(2, round(size * .1))
    inset = cv2.erode(body, disk(pad)) != 0
    fallback = layer, proof
    work = natural.LayoutWork()
    rim = (cv2.dilate(body, disk(3)) != 0) & (white == 0)
    for expand in (False, True):
        if expand:
            # Outlined balloons can be relettered inside their own paper.
            # Extending the paper would erase their printed border.
            if (proof.get('contrast_backing') or not rim.any()
                    or (clean[rim].max(axis=1) < 120).mean() > .10):
                return fallback
            room = cv2.dilate(body, disk(max(pad, size // 2))) != 0
        else:
            room = inset & (white != 0)
        style = {'min_size': size, 'max_size': size, 'outline': 0, 'leading': .04 if expand else .08}
        try:
            alternative, metrics = natural.layout(proof['original_text'], font, coverage,
                Image.fromarray(np.uint8(room & ~blocked) * 255), style, work=work)
        except natural.ResourceLimit:
            return fallback
        except ValueError:
            continue
        if not expand:
            metrics['balloon_backing'] = 'rewrapped_inside'
            return alternative, metrics
        # On white backing we can drop the contrasting stroke. Use this
        # setting only if it reduces the ink outside the original inset.
        alternative_ink = np.asarray(alternative.getchannel('A')) != 0
        if (alternative_ink & ~inset).sum() < (ink & ~inset).sum():
            layer, proof, ink = alternative, metrics, alternative_ink

    # Round the protruding corners. Inset the anchors so the unused sides and
    # tail stay put, then leave an opaque margin around each visible glyph.
    margin = max(pad, round(size * .16))
    anchors = (cv2.erode(body, disk(margin)) != 0) | ink
    hull = cv2.convexHull(cv2.findNonZero(anchors.astype(np.uint8)))
    shape = cv2.fillConvexPoly(np.zeros_like(body), hull, 255)
    outline = cv2.dilate(shape, disk(margin))
    rounded = cv2.GaussianBlur(outline, (0, 0), max(1., size / 6))
    backing = np.uint8(np.clip((rounded.astype(np.float32) - 110) * 5, 0, 255))
    glyph_margin = cv2.dilate(np.uint8(ink) * 255, disk(2))
    backing = np.maximum(backing, cv2.GaussianBlur(glyph_margin, (3, 3), .65))
    # Protect disconnected white shapes (including neighboring balloons).
    _, sheets, regions, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    sheet = np.bincount(sheets[body != 0]).argmax()
    spreading = backing != 0
    other_paper = regions[:, cv2.CC_STAT_AREA] >= pad * pad
    other_paper[[0, sheet]] = False
    if np.any(spreading & other_paper[sheets]):
        return fallback
    backing[white != 0] = 0
    growth = backing != 0
    if (not growth.any() or growth.sum() > .4 * area or (growth & blocked).any()
            or growth[0].any() or growth[-1].any() or growth[:, 0].any() or growth[:, -1].any()):
        return fallback
    distance = cv2.distanceTransform(1 - body, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    if distance[growth].max() > min(.8 * size, .38 * min(w, h)):
        return fallback

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
            return fallback
        separation = cv2.distanceTransform(np.uint8(~danger), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        backing = np.minimum(backing, np.uint8(np.clip(separation - 1, 0, 1) * 255))
        glyph = (np.asarray(layer)[:, :, :3].max(axis=2) < 120) & ink
        if np.any(glyph & (white == 0) & (backing < 245)):
            return fallback
        growth = backing != 0

    result = Image.new('RGBA', layer.size, 'white')
    result.putalpha(Image.fromarray(backing))
    result.alpha_composite(layer)
    return result, {**proof, 'balloon_backing': 'expanded', 'balloon_added_pixels': int(growth.sum())}
