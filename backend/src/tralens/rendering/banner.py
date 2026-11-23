"""Clean dark lettering on a flat paper banner while preserving enclosed illustration."""
import cv2
import numpy as np

from . import segmentation


def flat_plan(rgb, detections, member_ids):
    from .automatic import AutomaticPlan
    own, other = segmentation.raster(detections, member_ids, rgb.size)
    x, y, width, height = cv2.boundingRect(own.astype(np.uint8))
    if width < 2 * height or (width+32)*(height+32) > segmentation.MAX_PIXELS:
        return None
    box = max(0,x-16), max(0,y-16), min(rgb.width,x+width+16), min(rgb.height,y+height+16)
    l,t,r,b = box
    tile = np.s_[t:b,l:r]
    paper = np.asarray(rgb.crop(box))
    inside = own[tile]
    colors, counts = np.unique(paper[inside], axis=0, return_counts=True)
    fill = colors[counts.argmax()]
    if fill.min() < 210 or counts.max() < inside.sum()*.45:
        return None
    delta = np.abs(paper.astype(np.int16)-fill).max(axis=2)
    core = (delta > 100).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core)
    if count > 4096:
        return None
    near = segmentation.dilate(inside, 4)
    protected = segmentation.dilate(other[tile], 2)
    touching = np.unique(labels[inside & (core > 0)])
    if not 2 <= len(touching) <= 128:
        return None
    chosen = []
    for label in touching:
        component = labels == label
        if np.any(component & ~near) or np.any(component & protected):
            # A face or emblem may extend into the OCR box. Protect its
            # interior as well as the contour that crosses the box boundary.
            _, _, w, h, _ = stats[label]
            if w > height or h > height or w > 3*h or h > 3*w:
                protected |= component
                continue
            points = cv2.findNonZero(component.astype(np.uint8))
            hull = np.zeros_like(core)
            cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
            protected |= hull.astype(bool) & inside
        else:
            chosen.append(label)
    ink = np.isin(labels, chosen) & ~protected
    if not ink.any():
        return None
    halo = segmentation.dilate(ink, 2) & ~protected
    collar = segmentation.dilate(halo, 1) & ~halo & ~protected
    if not collar.any() or (delta[collar] <= 4).mean() < .92:
        return None
    # Preserve solid secondary paper colors along the edge of the lettering.
    erase = halo & (delta > 0)
    for color in colors[(counts > inside.sum()*.03) & (colors.min(axis=1) > 150)]:
        if np.array_equal(color, fill):
            continue
        _, patches = cv2.connectedComponents(np.all(paper == color, axis=2).astype(np.uint8))
        surrounding = np.unique(patches[~halo])
        erase &= ~np.isin(patches, surrounding[surrounding != 0])
    # Lay English only on original paper and the strokes just cleared.
    layout = inside & ((delta <= 4) | erase) & ~segmentation.dilate(protected, 2)
    full_erase, full_layout = np.zeros_like(own), np.zeros_like(own)
    full_erase[tile], full_layout[tile] = erase, layout
    obstacles = ~(full_erase | full_layout)
    return AutomaticPlan(full_erase.copy(), full_erase, full_layout, obstacles, fill.tolist(),
                         dict(cleanup_method='banner_flat_ink', background_rgb_at_center=fill.tolist(),
                              local_bounds=list(box), erase_pixels=int(erase.sum())))
