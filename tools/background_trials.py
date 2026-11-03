"""Experimental source-bound masks and phase-locked repair. No model acquisition.

CTD probabilities are evidence, never an erasure rectangle. All selected native
components are retained whole; unselected source/art remains a layout obstacle.
This module deliberately does not change the default renderer.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import automatic as a, natural as n


POLICY = 'native_components_semantic_association_v2'
MAX_PIXELS = 262144
MAX_PERIOD = 32


def dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)) != 0


def raster(detections, member_ids, size):
    n.require(type(member_ids) is list and 0 < len(member_ids) <= 32
        and all(type(i) in (int, str) for i in member_ids), 'typed source members required')
    keys = [(type(d['source_id']), d['source_id']) for d in detections]
    selected = {(type(i), i) for i in member_ids}
    n.require(len(keys) == len(set(keys)) and selected <= set(keys), 'unique bound source IDs required')
    own, other = Image.new('L', size), Image.new('L', size)
    for detection, key in zip(detections, keys):
        draw = ImageDraw.Draw(own if key in selected else other)
        draw.polygon([tuple(point) for point in detection['polygon']], fill=255)
    return np.array(own) != 0, np.array(other) != 0


def periodic_surface(pixels, known):
    """Infer an axis-aligned tile using ALL known samples, without outlier removal.

    No target/reference pixels participate. A crossing structure in known context
    invalidates the tile instead of being averaged out. This supports genuinely
    repeating local patterns, not arbitrary scanned/rotated screentones.
    """
    n.require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3
              and known.dtype == bool and known.shape == pixels.shape[:2], 'periodic input shape/type')
    h, w = known.shape
    n.budget_check(h*w <= MAX_PIXELS, 'periodic context budget')
    a.check(int(known.sum()) >= 2048, 'periodic_support', 'insufficient original known context')
    periods, comparisons = [], []
    for axis in (1, 0):
        accepted = None
        for period in range(1, min(MAX_PERIOD, (w if axis == 1 else h)//3)+1):
            if axis == 1:
                valid = known[:, period:] & known[:, :-period]
                delta = np.abs(pixels[:, period:].astype(np.int16)-pixels[:, :-period].astype(np.int16))
            else:
                valid = known[period:] & known[:-period]
                delta = np.abs(pixels[period:].astype(np.int16)-pixels[:-period].astype(np.int16))
            if int(valid.sum()) < 1024:
                continue
            error = delta[valid]
            if int(error.max()) <= 2 and float(error.mean()) <= 1:
                accepted = period
                comparisons.append(dict(axis=axis, period=period, samples=int(valid.sum()),
                                        maximum=int(error.max()), mae=float(error.mean())))
                break
        a.check(accepted is not None, 'periodic_unstable', 'no fully supported local translation period')
        periods.append(accepted)
    px, py = periods
    yy, xx = np.mgrid[:h, :w]
    phase = (yy % py)*px+(xx % px)
    counts = np.bincount(phase[known], minlength=px*py)
    a.check(int(counts.min()) >= 8, 'periodic_phase_support', 'some tile phases lack original samples')
    tile = np.stack([np.bincount(phase[known], weights=pixels[..., channel][known], minlength=px*py)/counts
                     for channel in range(3)], axis=1)
    tile = np.rint(tile).clip(0, 255).astype(np.uint8)
    predicted = tile[phase]
    error = np.abs(pixels.astype(np.int16)-predicted.astype(np.int16))[known]
    a.check(int(error.max()) <= 2 and float(error.mean()) <= 1,
            'periodic_model_mismatch', 'tile does not explain every known context sample')
    a.check(float(tile.std(axis=0).max()) >= 8, 'periodic_flat', 'no nontrivial repeating texture')
    return predicted, dict(method='phase_locked_tile', period_xy=periods,
        known_pixels=int(known.sum()), minimum_samples_per_phase=int(counts.min()),
        known_max_error=int(error.max()), known_mae=float(error.mean()),
        periodicity_checks=comparisons, outliers_discarded=0)


def derive(rgb, detections, member_ids, probability, *, surface='flat', source_fraction=.75,
           semantic_fraction=.5):
    """Build an experimental automatic plan from whole original residual strokes."""
    n.require(rgb.mode == 'RGB' and isinstance(probability, np.ndarray)
        and probability.shape == (rgb.height, rgb.width) and probability.dtype == np.uint8,
        'native uint8 CTD grayscale required')
    n.require(surface in ('flat', 'periodic'), 'unknown surface')
    n.budget_check(rgb.width*rgb.height <= 16_000_000 and len(detections) <= 512,
                   'trial source geometry budget')
    own, other = raster(detections, member_ids, rgb.size)
    x, y, w, h = cv2.boundingRect(own.astype(np.uint8))
    margin = 64 if surface == 'periodic' else 32
    left, top, right, bottom = max(0, x-margin), max(0, y-margin), min(rgb.width, x+w+margin), min(rgb.height, y+h+margin)
    n.budget_check((right-left)*(bottom-top) <= MAX_PIXELS, 'trial local context budget')
    pixels = np.array(rgb)[top:bottom, left:right]
    member = own[top:bottom, left:right]
    nonmember = other[top:bottom, left:right]
    semantic = probability[top:bottom, left:right]/255.
    near = dilate(member, 2)
    if surface == 'periodic':
        known = ~dilate(member | nonmember, 4)
        predicted, background = periodic_surface(pixels, known)
        floor = background['known_max_error']
    else:
        samples = member & (semantic < .1) & ~nonmember
        a.check(int(samples.sum()) >= 32, 'background_samples', 'insufficient low-text original background evidence')
        colors, counts = np.unique(pixels[samples], axis=0, return_counts=True)
        color = colors[np.argmax(counts)]
        predicted = np.full_like(pixels, color)
        floor = 0
        background = dict(method='measured_flat_mode', background_rgb=color.tolist(),
                          background_samples=int(samples.sum()))
    residual = np.max(np.abs(pixels.astype(np.int16)-predicted.astype(np.int16)), axis=2)
    contrast = residual > floor
    if surface == 'periodic':
        # The independently fitted, fully checked repeating surface separates
        # native strokes from dots even where CTD misses their endpoints. Require
        # unit-level semantic support, then associate WHOLE source-local residual
        # strokes rather than clipping them to the learned silhouette.
        a.check(int((member & (semantic >= .3) & (residual > 32)).sum()) >= 16,
                'periodic_semantic_support', 'periodic residual lacks learned unit-level text support')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(contrast.astype(np.uint8), connectivity=8)
    n.budget_check(count <= 8192, 'trial native component budget')
    chosen, rejected, ambiguous = [], [], []
    # Pixel membership, semantic support and component extent stay separate.
    for label in np.unique(labels[member]):
        if label == 0:
            continue
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        ownership = float((component & member).sum()/area)
        supported = float((semantic[component] >= .3).mean())
        semantic_required = semantic_fraction if surface == 'flat' else 0
        if ownership < source_fraction or supported < semantic_required:
            rejected.append(dict(label=int(label), source_fraction=ownership, semantic_fraction=supported))
            if ownership >= .4 and supported >= .25 and residual[component].max() > 32:
                ambiguous.append(int(label))
            continue
        fx, fy, fw, fh, _ = [int(value) for value in stats[label]]
        a.check(fx > 0 and fy > 0 and fx+fw < pixels.shape[1] and fy+fh < pixels.shape[0]
            and max(fw, fh) <= 96 and area <= 20000 and not (component & ~near).any(),
            'trial_component_extent', 'whole selected stroke leaves source neighborhood or may be artwork')
        chosen.append(int(label))
    a.check(0 < len(chosen) <= 128, 'trial_no_strokes', 'no bounded text-supported native components')
    a.check(not ambiguous, 'trial_ambiguous_stroke', 'partially supported original foreground remains ambiguous',
            {'ambiguous_labels': ambiguous})
    foreground = np.isin(labels, chosen)
    # Small disconnected antialias/ringing pixels are not strong independent art.
    # Include them whole only in a 3px ring of an already associated native stroke.
    ring = dilate(foreground, 3)
    faint = []
    for label in np.unique(labels[ring & near & ~foreground]):
        if label == 0:
            continue
        component = labels == label
        if int(stats[label, cv2.CC_STAT_AREA]) <= 64 and residual[component].max() <= 16 \
                and not (component & ~(ring & near)).any():
            faint.append(int(label))
    foreground |= np.isin(labels, faint)
    # A second ink/shadow/art color with a native interior is not antialiasing.
    # Keep one measured ink core and at most a 2px transition to it. This rejects
    # the development contained-gray-art failure instead of calling its lower
    # text-remnant score a success. Multi-fill/outlined lettering stays unsupported.
    core_samples = foreground & (residual > 32)
    a.check(int(core_samples.sum()) >= 8, 'trial_contrast_core', 'insufficient native high-contrast ink core')
    colors, counts = np.unique(pixels[core_samples], axis=0, return_counts=True)
    ink = colors[np.argmax(counts)]
    ink_core = foreground & (np.max(np.abs(pixels.astype(np.int16)-ink.astype(np.int16)), axis=2) <= 32)
    unsupported_ink = foreground & (residual > 16) & ~dilate(ink_core, 2)
    ink_count, ink_labels, ink_stats, _ = cv2.connectedComponentsWithStats(unsupported_ink.astype(np.uint8), connectivity=8)
    antialias_interiors = 0
    for label in range(1, ink_count):
        _, _, iw, ih, area = [int(value) for value in ink_stats[label]]
        selected_interior = ink_labels == label
        values = pixels[selected_interior].astype(np.int16)
        varying_transition = int(np.ptp(values, axis=0).max()) > 2
        if area <= 4 and max(iw, ih) <= 3 and (area < 3 or varying_transition):
            unsupported_ink[selected_interior] = False
            antialias_interiors += area
    a.check(not unsupported_ink.any(), 'trial_multiple_ink_interiors',
            'different-color interior may be artwork, shadow or unsupported multicolor lettering',
            {'ink_rgb': ink.tolist(), 'unsupported_pixels': int(unsupported_ink.sum()),
             'unsupported_semantic_min': float(semantic[unsupported_ink].min()) if unsupported_ink.any() else None,
             'unsupported_semantic_mean': float(semantic[unsupported_ink].mean()) if unsupported_ink.any() else None,
             'unsupported_residual_range': [int(residual[unsupported_ink].min()), int(residual[unsupported_ink].max())] if unsupported_ink.any() else None,
             'unsupported_components': ink_stats[1:33].tolist()})
    a.check(not (foreground & dilate(nonmember, 2)).any(), 'trial_neighbor',
            'selected whole foreground touches neighboring source geometry')
    unexplained = member & (semantic >= .5) & (residual > 32) & ~foreground
    # A tiny intersection with an externally owned component is NOT a reason to
    # cut that component. Keep it whole and protected as an obstacle instead.
    for rejected_component in rejected:
        if rejected_component['source_fraction'] < .4:
            unexplained[labels == rejected_component['label']] = False
    a.check(not unexplained.any(), 'trial_text_remnant', 'strong target text evidence remains unassociated',
        {'unexplained_pixels': int(unexplained.sum()), 'rejected': rejected,
         'selected_components': len(chosen)})
    erase = dilate(foreground, 1)
    a.check(not (erase & nonmember).any(), 'trial_neighbor', 'erase mask overlaps a neighboring source')
    a.check(not (erase[0].any() or erase[-1].any() or erase[:, 0].any() or erase[:, -1].any()),
            'trial_frame', 'complete original clearance unavailable')
    collar = dilate(erase, 2) & ~erase
    a.check(int(collar.sum()) >= 32 and int(residual[collar].max()) <= max(2, floor),
            'trial_clearance', 'selected lettering does not have an independently explained collar',
            {'collar_residual_max': int(residual[collar].max()),
             'collar_unexplained_pixels': int((residual[collar] > max(2, floor)).sum())})
    local_obstacles = (contrast & ~foreground) | nonmember
    obstacles = np.ones_like(own)
    obstacles[top:bottom, left:right] = local_obstacles
    full_fg, full_erase = np.zeros_like(own), np.zeros_like(own)
    full_fg[top:bottom, left:right] = foreground
    full_erase[top:bottom, left:right] = erase
    safe = ~dilate(obstacles, 2)
    safe[:2] = safe[-2:] = False
    safe[:, :2] = safe[:, -2:] = False
    mx, my = max(4, min(32, (w+1)//2)), max(4, min(24, (h+9)//10))
    window = [max(2, x-mx), max(2, y-my), min(rgb.width-2, x+w+mx), min(rgb.height-2, y+h+my)]
    # Periodic surface contrast is background, not layout ink: residual-based
    # obstacles preserve deviations from the supported pattern, not its dots.
    layout, bounds, steps, derivation = a.source_local_layout(safe, dilate(full_fg, 2), window)
    # A successful rectangle can still waste genuine original whitespace around
    # a tapered balloon. Keep the same window, obstacles and complete source
    # clearance, but expose its one 4-connected safe component to exact-glyph
    # layout. No rectangular corner is treated as new erasure permission.
    wl, wt, wr, wb = window
    local_seed = dilate(full_fg, 2)[wt:wb, wl:wr]
    component_count, free_labels = cv2.connectedComponents(safe[wt:wb, wl:wr].astype(np.uint8), connectivity=4)
    n.budget_check(component_count <= 8192, 'trial free-space component budget')
    seed_labels = np.unique(free_labels[local_seed])
    if len(seed_labels) == 1 and seed_labels[0] != 0:
        free = np.zeros_like(own)
        free[wt:wb, wl:wr] = free_labels == seed_labels[0]
        n.require(not (dilate(full_fg, 2) & ~free).any() and not (free & ~safe).any(),
                  'trial free-space lost original clearance')
        if int(free.sum()) > int(layout.sum()):
            old_pixels = int(layout.sum())
            layout = free
            lx, ly, lw, lh = cv2.boundingRect(layout.astype(np.uint8))
            bounds = [lx, ly, lx+lw, ly+lh]
            derivation = dict(method='original_connected_free_space', rectangle_pixels=old_pixels,
                              connected_pixels=int(free.sum()), original_derivation=derivation,
                              window_unchanged=True, source_clearance_discarded=0)
    evidence = dict(policy=POLICY, surface=background, source_fraction=source_fraction,
        semantic_fraction=semantic_fraction, selected_components=len(chosen), faint_components=len(faint),
        ink_rgb=ink.tolist(), ink_interior_policy='one measured fill core within 32 channel levels and <=2px antialias transition',
        small_varying_antialias_pixels=antialias_interiors,
        rejected_components=rejected, foreground_pixels=int(foreground.sum()), erase_pixels=int(erase.sum()),
        layout_bounds=bounds, layout_derivation=derivation, layout_expansion_tests=steps,
        local_bounds=[left, top, right, bottom], native_mask='whole residual components, no rectangle erasure',
        uncertainty='machine source-local hypothesis; not semantic certification or recovery of unique hidden art')
    return a.AutomaticPlan(full_fg, full_erase, layout, obstacles,
        background.get('background_rgb'), evidence,
        restored_pixels=predicted[erase].copy() if surface == 'periodic' else None)
