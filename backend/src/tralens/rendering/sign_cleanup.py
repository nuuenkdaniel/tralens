"""Recover a terminal printed character clipped by a straight panel rule."""
import cv2
import numpy as np


def attached_ink(pixels, foreground, member, protected, background, text):
    """Return only residual components on the lettering side of a verified rule."""
    if not (2 <= len(text) <= 12 and all('\u3400' <= c <= '\u9fff' for c in text)):
        return None
    columns = np.flatnonzero(foreground.any(axis=0))
    if not len(columns):
        return None
    _, y, _, height = cv2.boundingRect(foreground.astype(np.uint8))
    # Require one missing character in an otherwise resolved horizontal run.
    groups = np.split(columns, np.where(np.diff(columns) > max(3, height // 16))[0] + 1)
    if len(groups) < len(text) - 1:
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
    if height < 12:
        return None
    infer_cells = (len(groups) != len(text) - 1
                   or any(group[-1] - group[0] + 1 < .35 * height for group in groups))
    if infer_cells:
        # Antialiasing can fill the last clear column between adjacent kanji.
        # Look for thin ink at each expected boundary of roughly square cells.
        cells = len(text) - 1
        pitch = (columns[-1] - columns[0] + 1) / cells
        if not text.endswith('口') or not .8 * height <= pitch <= 1.2 * height:
            return None
        profile = foreground.sum(axis=0)
        radius = max(1, round(.12 * pitch))
        for index in range(1, cells):
            center = round(columns[0] + index * pitch)
            valley = profile[center-radius:center+radius+1]
            if valley.min() > max(1, height // 12):
                return None
    last = columns[-1]
    residual = np.max(np.abs(pixels.astype(np.int16) - background), axis=2)
    band = residual[max(0, y-height//3):min(len(pixels), y+height+height//3)]
    rails = np.flatnonzero((band > 12).all(axis=0))
    rails = rails[(rails > last + 3) & (rails <= last + height)]
    if not len(rails):
        return None
    rail = rails[0]
    rule_width = np.sum(np.cumprod((band[:, rail:] > 12).all(axis=0)))
    if not 2 <= rule_width <= max(8, height // 8):
        return None
    # Preserve even the antialiased edge of the rule. Split the original
    # component there and require each attached piece to fit inside this OCR unit.
    contrast = residual[:, :rail] > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(contrast.astype(np.uint8))
    if count > 8192:
        return None
    chosen = []
    for label in np.unique(labels[:, -1]):
        if not label:
            continue
        x, top, w, h, area = stats[label]
        if (x <= last or top < y - height // 3 or top+h > y + height*3//2
                or h > height or w > height or area < 3):
            continue
        piece = labels == label
        if np.any(piece & (~member[:, :rail] | protected[:, :rail])):
            continue
        chosen.append(label)
    if not chosen:
        return None
    mask = np.zeros_like(member)
    mask[:, :rail] = np.isin(labels, chosen)
    x, top, w, h = cv2.boundingRect(mask.astype(np.uint8))
    # A rule intersection or single decorative stroke is insufficient evidence.
    contacts = mask[:, rail-1]
    runs = np.count_nonzero(np.diff(np.r_[False, contacts, False])) // 2
    if infer_cells and runs >= 2:
        # A clipped 口 has two horizontal strokes joined on the left. Exclude
        # isolated border decorations even if they also touch the rule twice.
        glyph = mask[top:top+h, x:rail]
        contacts_y = np.flatnonzero(glyph[:, -1])
        left_side = glyph[contacts_y[0]:contacts_y[-1]+1, :max(1, w//3)]
        if (runs != 2 or not left_side.any(axis=1).all()
                or not np.all(glyph[contacts_y].mean(axis=1) >= .75)
                or contacts_y[-1] - contacts_y[0] < .65 * h):
            return None
    if runs < 2:
        # A box-shaped kanji can meet the rule in one continuous stroke.
        # Require multiple enclosed counters as independent shape evidence.
        inverse = (~mask[:, x:rail]).astype(np.uint8)
        _, holes, sizes, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
        exterior = np.unique(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1], 0])
        counters = np.setdiff1d(np.flatnonzero(sizes[:, cv2.CC_STAT_AREA] >= 4), exterior)
        if infer_cells or len(counters) < 2:
            if len(counters) != 1 or not text.endswith('口'):
                return None
            # 口 has one rectangular opening. When its right stroke overlaps
            # the rule, the visible glyph touches the rule in a single run.
            _, _, inner_w, inner_h, area = sizes[counters[0]]
            if not (.7 <= w / h <= 1.2 and .35 <= inner_w / w <= .85
                    and .35 <= inner_h / h <= .85 and area >= .85 * inner_w * inner_h):
                return None
    if w < height*.2 or h < height*.45 or x-last > height*.45:
        return None
    if np.max(np.abs(np.median(pixels[mask], axis=0) - np.median(pixels[foreground], axis=0))) > 16:
        if not text.endswith('口'):
            return None
        # On thin printing the median may be mostly antialiasing. Check the
        # solid ink and require all other samples to blend into the same paper.
        prefix_core = foreground & (residual >= np.percentile(residual[foreground], 75))
        ink = np.median(pixels[prefix_core], axis=0) - background
        if np.max(np.abs(ink)) < 48:
            return None
        colors = pixels[mask].astype(float) - background
        alpha = np.clip((colors @ ink) / (ink @ ink), 0, 1)
        if (np.mean(alpha >= .9) < .15
                or np.max(np.abs(colors - alpha[:, None] * ink)) > 16):
            return None
    # Include detached antialias flecks only when the entire component is
    # adjacent to the recovered ink, faint, and clear of protected pixels.
    fringe = cv2.dilate(mask[:, :rail].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    small = np.flatnonzero((stats[:, cv2.CC_STAT_AREA] <= 4))
    for label in np.intersect1d(labels[fringe], small):
        piece = labels == label
        if (label and residual[:, :rail][piece].max() <= 48
                and np.all((fringe & member[:, :rail] & ~protected[:, :rail])[piece])):
            mask[:, :rail] |= piece
    return mask
