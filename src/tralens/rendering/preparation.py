"""Original-pixel flat-enclosure proposals; explicit confirmation is separate.

No OCR engine/facade, text-masked region labels, hull masks or source writes.
Geometry is a measured hypothesis, never ownership/full-utterance proof.
"""
from io import BytesIO
import html
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import natural as n
from .styles import face_metadata, resolve_style
from .review import CSP


PURPOSE = 'flat_geometry_proposals'
ALGORITHM = 'original_closed_flat_enclosure_boundary_inset_v3'
MAX_PIXELS = 4_000_000
MAX_SIDE = 4096
MAX_UNITS = 16
MAX_DETECTIONS = 512
MAX_PREFLIGHT_PIXELS = 512_000_000
BOUNDARY_RADIUS = 8
MAX_ADDITIONAL_INSET = 3
DEFAULTS = {'clearance': 3, 'member_padding': 2, 'glyph_padding': 2, 'layout_inset': 1}


def settings(values=None):
    values = {} if values is None else values
    n.require(type(values) is dict and set(values) <= set(DEFAULTS), 'unknown preparation option')
    result = {**DEFAULTS, **values}
    for key, lo, hi in (('clearance', 2, 8), ('member_padding', 0, 4), ('glyph_padding', 2, 4), ('layout_inset', 0, 4)):
        n.integer(result[key], lo, hi)
    return result


def implementation_hash():
    root = Path(__file__).parent
    return n.revision({name: n.digest(n.read_explicit(root/name, 256*1024))
                       for name in ('preparation.py', 'natural.py', 'styles.py', 'review.py')})


class Unsupported(ValueError):
    pass


def supported(condition, message):
    if not condition:
        raise Unsupported(message)


def dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)) != 0


def erode(mask, radius):
    return cv2.erode(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0


def filled_outer(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(result, contours, -1, 1, cv2.FILLED)
    return result != 0


def split_pixel_contour(points, work):
    """Decompose integer vertex/edge contacts into simple polygon cycles.

    This repairs raster contour representation only, not balloon connectivity.
    Every result is still checked against original allowed/required pixels.
    """
    pending, result = [points], []
    while pending:
        polygon = pending.pop()
        polygon = [p for i, p in enumerate(polygon) if i == 0 or p != polygon[i-1]]
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon.pop()
        if len(polygon) < 3:
            continue
        split = False
        count = len(polygon)
        for i, point in enumerate(polygon):
            for edge, start in enumerate(polygon):
                work[0] += 1
                n.budget_check(work[0] <= 4_000_000, 'pixel contour splitting budget')
                if edge == i or (edge+1) % count == i:
                    continue
                end = polygon[(edge+1) % count]
                cross = (end[0]-start[0])*(point[1]-start[1])-(end[1]-start[1])*(point[0]-start[0])
                if cross == 0 and min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and min(start[1], end[1]) <= point[1] <= max(start[1], end[1]):
                    cycle = polygon[i:]+polygon[:i]
                    position = (edge-i) % count
                    pending.extend((cycle[:position+1], [cycle[0]]+cycle[position+1:]))
                    split = True
                    break
            if split:
                break
        if not split:
            area = sum(p[0]*q[1]-q[0]*p[1] for p, q in zip(polygon, polygon[1:]+polygon[:1]))
            if abs(area) >= 2:
                result.append(polygon)
        supported(len(pending)+len(result) <= 256, 'pixel contour cycle budget')
    return result


def strip_polygons(mask, height):
    """Exact clipped contours in overlapping strips; no pixel expansion."""
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    polygons, work = [], [0]
    for top in range(y, y+h, height-1):
        strip = mask[top:min(top+height, y+h), x:x+w].astype(np.uint8)
        count, labels = cv2.connectedComponents(strip, connectivity=4)
        supported(count <= 129, 'strip component budget')
        for label in range(1, count):
            contours, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = contour[:, 0, :]
                if not 3 <= len(points) <= 128:
                    continue
                points = points+np.array([x, top])
                polygons.extend(split_pixel_contour(points.tolist(), work))
                supported(len(polygons) <= 128, 'strip polygon budget')
    return polygons


def polygons_inside(allowed, required=None):
    """Exact contours first; bounded inward approximations, checked pixel by pixel.

    No hull is used. Holes must already have been explicitly accounted for.
    """
    required = np.zeros_like(allowed) if required is None else required
    original_contours, original_hierarchy = cv2.findContours(allowed.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    supported(original_hierarchy is not None and len(original_contours) <= 128
              and all(h[3] == -1 and h[2] == -1 for h in original_hierarchy[0]),
              'polygonization has unsupported holes/component budget')
    for inset, epsilon in ((0, 0), (1, .5), (1, 1), (1, 1.5), (1, 2)):
        target = erode(allowed, inset) if inset else allowed
        contours, hierarchy = cv2.findContours(target.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) > 128 or any(h[3] != -1 or h[2] != -1 for h in hierarchy[0]):
            continue
        polygons = []
        valid = True
        for contour in contours:
            contour = cv2.approxPolyDP(contour, epsilon, True) if epsilon else contour
            points = contour[:, 0, :].tolist()
            if not 3 <= len(points) <= 128:
                valid = False
                break
            polygons.append(points)
        if not valid:
            continue
        try:
            actual = n.union_mask(polygons, (allowed.shape[1], allowed.shape[0]))
        except ValueError:
            continue
        if actual.any() and not (actual & ~allowed).any() and not (required & ~actual).any():
            return polygons, actual
    for transposed in (False, True):
        for height in (32, 16, 8):
            try:
                polygons = strip_polygons(allowed.T if transposed else allowed, height)
                if transposed:
                    polygons = [[[y, x] for x, y in polygon] for polygon in polygons]
                actual = n.union_mask(polygons, (allowed.shape[1], allowed.shape[0]))
            except (Unsupported, ValueError):
                continue
            if actual.any() and not (actual & ~allowed).any() and not (required & ~actual).any():
                return polygons, actual
    raise Unsupported('polygonization cannot preserve required pixels within safe bounds/holes budget')


class OriginalRegions:
    def __init__(self, rgb):
        n.budget_check(rgb.mode == 'RGB' and rgb.width*rgb.height <= MAX_PIXELS and max(rgb.size) <= MAX_SIDE,
                       'preparation source pixel/side budget')
        self.rgb = rgb
        light = (np.asarray(rgb).min(axis=2) >= 180).astype(np.uint8)
        self.white = []
        for connectivity in (4, 8):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=connectivity)
            n.budget_check(count <= 8192, 'original light-component budget')
            self.white.append((labels, stats))
        count, self.dark, self.dark_stats, _ = cv2.connectedComponentsWithStats(1-light, connectivity=8)
        n.budget_check(count <= 8192, 'original dark-component budget')


def outline_adjacency(context, envelope, chosen_label, detections, selected):
    """Separate local shared boundaries from a remote common black component.

    Containment or four-edge surrounding light is measured as ambient context.
    Peer status needs independent source geometry. Unassigned exterior pockets
    stay unclassified: this is never a semantic balloon/ownership verdict.
    """
    labels, stats = context.white[0]
    ring = dilate(envelope, 1) & ~envelope
    outline_ids = np.unique(context.dark[ring])
    outline_ids = outline_ids[outline_ids != 0]
    supported(0 < len(outline_ids) <= 64, 'original outline component support unavailable')
    lookup = np.zeros(len(context.dark_stats), bool)
    lookup[outline_ids] = True
    global_outline = lookup[context.dark]
    adjacent = dilate(global_outline, 1)
    global_votes = np.bincount(labels[adjacent], minlength=len(stats))
    # A closed white art pocket/counter is not evidence of another balloon.
    # Associate peer contacts with independent original source geometry, including
    # uncertain/blank detections; no confidence/text/translation filtering.
    peer_sources = {}
    for detection in detections:
        key = (type(detection['source_id']).__name__, detection['source_id'])
        if key in selected:
            continue
        mask = np.asarray(n.polygon_mask(detection['polygon'], context.rgb.size, evidence=True)) != 0
        votes = np.bincount(labels[mask], minlength=len(stats))
        positive = int(votes[1:].sum())
        if positive < .5*int(mask.sum()) or positive == 0:
            continue
        label = int(np.argmax(votes[1:])+1)
        if votes[label] >= .9*positive:
            peer_sources.setdefault(label, []).append(detection['source_id'])
    x, y, w, h = cv2.boundingRect(envelope.astype(np.uint8))
    pad = BOUNDARY_RADIUS+1
    left, top = max(0, x-pad), max(0, y-pad)
    right, bottom = min(labels.shape[1], x+w+pad), min(labels.shape[0], y+h+pad)
    local_envelope = envelope[top:bottom, left:right]
    local_band = dilate(local_envelope, BOUNDARY_RADIUS) & ~local_envelope
    local_ink = global_outline[top:bottom, left:right] & local_band
    count, local_labels = cv2.connectedComponents(local_ink.astype(np.uint8), connectivity=8)
    n.budget_check(count <= 8192, 'local outline-component budget')
    seed_ids = np.unique(local_labels[ring[top:bottom, left:right]])
    seed_ids = seed_ids[seed_ids != 0]
    supported(len(seed_ids) > 0, 'no original local outline support')
    local_lookup = np.zeros(count, bool)
    local_lookup[seed_ids] = True
    local_contact = dilate(local_lookup[local_labels], 1)
    local_votes = np.bincount(labels[top:bottom, left:right][local_contact], minlength=len(stats))
    distance = cv2.distanceTransform((~envelope).astype(np.uint8), cv2.DIST_C, 3)
    nearest = np.full(len(stats), np.inf, np.float32)
    np.minimum.at(nearest, labels.ravel(), distance.ravel())
    contacts = []
    containers = 0
    for label in np.flatnonzero(global_votes):
        if label == 0 or label == chosen_label:
            continue
        bx, by, bw, bh, area = (int(v) for v in stats[label])
        if not (area > 0 and bx > 0 and by > 0 and bx+bw < labels.shape[1] and by+bh < labels.shape[0]):
            continue
        container = False
        surrounds_window = False
        if bx <= x and by <= y and bx+bw >= x+w and by+bh >= y+h:
            containers += 1
            n.budget_check(containers <= 8, 'enclosing ambient-region comparison budget')
            # Raster containment, not bbox/hull containment. This never supplies
            # an erase mask or fills unknown pixels in the actual geometry.
            outer = filled_outer(labels[by:by+bh, bx:bx+bw] == label)
            candidate = envelope[by:by+bh, bx:bx+bw]
            container = not (candidate & ~outer).any()
            # A narrow original artwork connection can turn an ambient hole into
            # a notch, so full contour containment alone is insufficient. Require
            # actual same-label light pixels on all four local-window edges.
            local_region = labels[top:bottom, left:right] == label
            surrounds_window = bool(local_region[0].any() and local_region[-1].any()
                                    and local_region[:, 0].any() and local_region[:, -1].any())
            container = container or surrounds_window
        sources = peer_sources.get(int(label), [])
        kind = ('enclosing_ambient' if container else 'nonlocal_contact' if not local_votes[label]
                else 'local_peer_contact' if sources else 'unassigned_external_contact')
        contacts.append({'label': int(label), 'bounds': [bx, by, bx+bw, by+bh], 'area': area,
            'global_contact_pixels': int(global_votes[label]), 'local_contact_pixels': int(local_votes[label]),
            'minimum_distance_to_envelope': int(nearest[label]), 'relation': kind,
            'peer_source_ids': sources[:32], 'peer_sources_truncated': len(sources) > 32,
            'surrounds_local_window': surrounds_window})
    contacts.sort(key=lambda item: (item['relation'] != 'local_peer_contact', item['minimum_distance_to_envelope'], item['label']))
    peers = [item for item in contacts if item['relation'] == 'local_peer_contact']
    return {'method': 'original_local_outline_band_v2', 'band_radius': BOUNDARY_RADIUS,
        'source_window': [left, top, right, bottom], 'outline_components': int(len(outline_ids)),
        'global_closed_contact_count': len(contacts), 'local_closed_contact_count': len(peers),
        'ignored_container_count': sum(item['relation'] == 'enclosing_ambient' for item in contacts),
        'ignored_nonlocal_contact_count': sum(item['relation'] == 'nonlocal_contact' for item in contacts),
        'unassigned_external_contact_count': sum(item['relation'] == 'unassigned_external_contact' for item in contacts),
        'contacts_sample': contacts[:32], 'contacts_truncated': len(contacts) > 32,
        'scope': 'local source-supported peer boundary evidence only; unassigned exterior pockets are not classified as balloons; no semantic ownership/whole-page outline independence certification'}


def trim_empty_boundary_spans(safe, required, limit):
    """Remove only a few non-single-span edge scanlines outside required ink.

    Caller must first validate flatness of the ENTIRE original core. No internal
    gap is filled or cut, no permission grows, and no required pixel may be lost.
    """
    result = safe.copy()
    x, y, w, h = cv2.boundingRect(result.astype(np.uint8))
    rx, ry, rw, rh = cv2.boundingRect(required.astype(np.uint8))
    left, top, right, bottom = x, y, x+w, y+h
    trimmed = [0]*4
    def multiple(values):
        return int(np.count_nonzero(values & ~np.pad(values[:-1], (1, 0)))) > 1
    for _ in range(limit):
        changed = False
        if top < ry and multiple(result[top, left:right]):
            result[top, :] = False
            top += 1
            trimmed[0] += 1
            changed = True
        if bottom > ry+rh and multiple(result[bottom-1, left:right]):
            result[bottom-1, :] = False
            bottom -= 1
            trimmed[1] += 1
            changed = True
        if left < rx and multiple(result[top:bottom, left]):
            result[:, left] = False
            left += 1
            trimmed[2] += 1
            changed = True
        if right > rx+rw and multiple(result[top:bottom, right-1]):
            result[:, right-1] = False
            right -= 1
            trimmed[3] += 1
            changed = True
        if not changed:
            break
    horizontal = result & ~np.pad(result[:, :-1], ((0, 0), (1, 0)))
    vertical = result & ~np.pad(result[:-1], ((1, 0), (0, 0)))
    supported(int(horizontal.sum(axis=1).max()) <= 1 and int(vertical.sum(axis=0).max()) <= 1,
              'non-single-span enclosure; internal gaps or excessive boundary irregularity remain unsupported')
    supported(not (required & ~result).any(), 'boundary-span trimming would discard required ink/collar')
    return result, {'method': 'flat_empty_boundary_scanlines_only', 'limit_per_edge': limit,
        'trimmed_top_bottom_left_right': trimmed, 'excluded_flat_pixels': int(safe.sum())-int(result.sum()),
        'required_pixels_discarded': 0}


def span_evidence(mask):
    result = {}
    for name, values in (('rows', mask), ('columns', mask.T)):
        gaps = []
        maximum = total = lines = 0
        for line, pixels in enumerate(values):
            positions = np.flatnonzero(pixels)
            differences = np.diff(positions)-1
            missing = np.flatnonzero(differences > 0)
            if len(missing):
                lines += 1
            for index in missing:
                width = int(differences[index])
                total += width
                maximum = max(maximum, width)
                if len(gaps) < 8:
                    gaps.append({'line': line, 'start': int(positions[index]+1), 'end': int(positions[index+1])})
        result[name] = {'non_single_span_lines': lines, 'maximum_gap': maximum, 'total_gap_pixels': total, 'sample': gaps}
    return result


def conservative_boundary_inset(context, envelope, safe, permission, members, error, member_padding):
    """One measured global inset for original-outline-connected edge fringe only.

    No tolerance/permission change, no spot masking, no retry search. Isolated
    marks, interior defects and any lost source member/collar remain blocked.
    """
    diagnostic = error.diagnostics
    evidence = {'method': 'original_outline_connected_fringe_inset', 'maximum_inset': MAX_ADDITIONAL_INSET,
                'geometry_inset_applied': False, 'original_cleanup_diagnostics': diagnostic}
    if diagnostic['code'] != 'nonflat_sample':
        evidence['refusal'] = 'not a sample-variance failure'
        return None, evidence
    color = np.array(diagnostic['background_estimate_rgb'], np.int16)
    delta = np.max(np.abs(np.asarray(context.rgb).astype(np.int16)-color), axis=2)
    contrast = delta > 2
    offending = safe & ~permission & contrast
    evidence['offending_count'] = int(offending.sum())
    supported(evidence['offending_count'] == diagnostic['offending_count'], 'boundary diagnostic/source mismatch')
    distance = cv2.distanceTransform(safe.astype(np.uint8), cv2.DIST_C, 3)
    inset = int(distance[offending].max()) if offending.any() else 0
    evidence['inset_pixels'] = inset
    if not 1 <= inset <= MAX_ADDITIONAL_INSET:
        evidence['refusal'] = 'sample contamination is not confined to the bounded original edge band'
        return None, evidence
    band = dilate(envelope, BOUNDARY_RADIUS) & ~erode(envelope, BOUNDARY_RADIUS)
    # Trace original non-background antialias pixels (including 1/2-level
    # bridges), not just the >2-level pixels which FAILED flatness. This does
    # not alter cleanup tolerance; the inset must still pass that exact guard.
    evidence['boundary_connection_minimum_delta'] = 1
    count, labels = cv2.connectedComponents(((delta > 0) & band).astype(np.uint8), connectivity=8)
    n.budget_check(count <= 8192, 'boundary fringe component budget')
    outline = (context.dark != 0) & (dilate(envelope, 1) & ~envelope)
    seeds = np.unique(labels[outline])
    seeds = seeds[seeds != 0]
    linked = np.zeros(count, bool)
    linked[seeds] = True
    offending_ids = np.unique(labels[offending])
    connected = bool(len(offending_ids) > 0 and 0 not in offending_ids and linked[offending_ids].all())
    evidence['connected_to_original_outline'] = connected
    if not connected:
        evidence['refusal'] = 'isolated/unassigned edge contrast is not original-outline-connected'
        return None, evidence
    selected = np.zeros(count, bool)
    selected[offending_ids] = True
    source_contact = selected[labels] & safe & dilate(members, member_padding)
    if source_contact.any():
        evidence['refusal'] = 'fringe component intersects source member neighborhood'
        evidence['source_contact'] = n.offending_pixels(source_contact)
        return None, evidence
    candidate = erode(safe, inset)
    lost = int(np.count_nonzero(permission & ~candidate))
    evidence['required_pixels_discarded'] = lost
    if lost:
        evidence['refusal'] = 'additional inset would discard required whole ink/collar'
        return None, evidence
    evidence['excluded_pixels'] = int(safe.sum())-int(candidate.sum())
    return candidate, evidence


def separator_evidence(context, envelope, first, second, glyph_padding):
    """Read original clearance topology, never invent a separator/ownership mask.

    A finding is diagnostic only. The current shared-region guard still rejects;
    wide shared speech regions are not split using source-box distances.
    """
    required = []
    for footprint in (first, second):
        ids = np.unique(context.dark[footprint])
        ids = ids[ids != 0]
        if not 0 < len(ids) <= 64:
            return {'method': 'original_clearance_neck_probe', 'scope': 'diagnostic only; no ownership split',
                    'reason': 'insufficient isolated source component support', 'separating_clearance_found': False}
        lookup = np.zeros(len(context.dark_stats), bool)
        lookup[ids] = True
        ink = lookup[context.dark]
        if (ink & ~envelope).any():
            return {'method': 'original_clearance_neck_probe', 'scope': 'diagnostic only; no ownership split',
                    'reason': 'source ink connects to border/art', 'separating_clearance_found': False}
        required.append(dilate(ink, glyph_padding))
    trials = []
    for radius in range(1, BOUNDARY_RADIUS+1):
        core = erode(envelope, radius)
        count, labels = cv2.connectedComponents(core.astype(np.uint8), connectivity=8)
        n.budget_check(count <= 8192, 'separator component budget')
        lost = [int(np.count_nonzero(mask & ~core)) for mask in required]
        anchors = [np.unique(labels[mask]).tolist() for mask in required]
        single = all(len(values) == 1 and values[0] != 0 for values in anchors)
        separate = bool(single and not any(lost) and anchors[0][0] != anchors[1][0])
        trials.append({'radius': radius, 'components': count-1, 'required_pixels_lost': lost,
                       'distinct_complete_ink_cores': separate})
    found = any(row['distinct_complete_ink_cores'] for row in trials)
    return {'method': 'original_clearance_neck_probe', 'scope': 'diagnostic only; no separator selected, ownership/translation unchanged',
            'maximum_tested_radius': BOUNDARY_RADIUS, 'separating_clearance_found': found, 'trials': trials,
            'reason': 'thin-neck observation requires separate source review; shared region remains manual' if found
                      else 'no disjoint complete-ink cores at supported clearances; no image-supported narrow separator'}


def propose_geometry(rgb, detections, member_ids, options=None, *, context=None):
    """Useful for closed, single-span light flat balloons; honest rejects otherwise."""
    options = settings(options)
    context = OriginalRegions(rgb) if context is None else context
    n.require(type(detections) is list and 1 <= len(detections) <= MAX_DETECTIONS, 'detection budget')
    keys = [(type(d['source_id']).__name__, d['source_id']) for d in detections]
    n.require(all(type(d['source_id']) in (int, str) and d['source_id'] != '' for d in detections)
              and len(set(keys)) == len(keys), 'typed detection IDs required')
    n.require(type(member_ids) is list and member_ids and all(type(i) in (str, int) for i in member_ids)
              and len(set(member_ids)) == len(member_ids), 'typed ordered members required')
    selected = {(type(i).__name__, i) for i in member_ids}
    n.require(selected <= set(keys), 'foreign member ID')
    n.budget_check(len(member_ids) <= 32 and len(member_ids)*rgb.width*rgb.height <= n.MAX_UNIT_PIXELS,
                   'member mask raster budget')
    measured = {'algorithm': ALGORITHM, 'settings': options, 'semantic_ownership': 'unverified',
                'full_utterance_coverage': 'unverified; outside-OCR punctuation is not certified'}
    try:
        member_masks = [np.asarray(n.polygon_mask(d['polygon'], rgb.size, evidence=True)) != 0
                        for d, key in zip(detections, keys) if key in selected]
        members = np.logical_or.reduce(member_masks)
        chosen = []
        for labels, stats in context.white:
            votes = []
            for mask in member_masks:
                values = labels[mask]
                counts = np.bincount(values, minlength=len(stats))
                supported(int(counts[1:].sum()) >= .5*int(mask.sum()), 'insufficient original light support inside members')
                label = int(np.argmax(counts[1:])+1)
                supported(counts[label] >= .9*counts[1:].sum(), 'ambiguous original background components')
                x, y, w, h, area = (int(v) for v in stats[label])
                supported(x > 0 and y > 0 and x+w < rgb.width and y+h < rgb.height,
                          'open/frame-connected original background')
                votes.append(label)
            supported(len(set(votes)) == 1, 'members span distinct original enclosures')
            chosen.append(votes[0])
        light_labels, stats = context.white[0]
        background = light_labels == chosen[0]
        envelope = filled_outer(background)
        area = int(envelope.sum())
        ex, ey, ew, eh = cv2.boundingRect(envelope.astype(np.uint8))
        measured.update(original_background_labels=chosen, enclosure_pixels=area,
                        original_enclosure_bounds=[ex, ey, ex+ew, ey+eh])
        supported(64 <= area <= min(1_000_000, int(rgb.width*rgb.height*.6), int(members.sum())*30),
                  'enclosure area unsupported relative to page/member evidence')
        supported(not (members & ~envelope).any(), 'member geometry crosses original enclosure boundary')
        boundary = outline_adjacency(context, envelope, chosen[0], detections, selected)
        measured['outline_adjacency'] = boundary
        measured['outline_components'] = boundary['outline_components']
        supported(boundary['local_closed_contact_count'] == 0,
                  'locally outline-connected closed regions; touching/seam-connected enclosure unsupported')
        for detection, key in zip(detections, keys):
            if key not in selected:
                neighbor = np.asarray(n.polygon_mask(detection['polygon'], rgb.size, evidence=True)) != 0
                intersection = neighbor & envelope
                if intersection.any():
                    measured['nonmember_contact'] = {'source_id': detection['source_id'], **n.offending_pixels(intersection)}
                    measured['separator_evidence'] = separator_evidence(context, envelope, members, neighbor, options['glyph_padding'])
                    raise Unsupported('nonmember/annotation geometry intersects original enclosure')
        owned_ids = np.unique(context.dark[members])
        owned_ids = owned_ids[owned_ids != 0]
        supported(0 < len(owned_ids) <= 64, 'no supported ink or excessive owned-component count')
        lookup = np.zeros(len(context.dark_stats), bool)
        lookup[owned_ids] = True
        owned = lookup[context.dark]
        measured['component_evidence'] = {'owned_component_count': int(len(owned_ids)),
            'owned_components_sample': [{'label': int(label), 'stats': [int(v) for v in context.dark_stats[label]]} for label in owned_ids[:16]],
            'outside_enclosure': n.offending_pixels(owned & ~envelope),
            'outside_member_neighborhood': n.offending_pixels(owned & ~dilate(members, options['member_padding'])),
            'unexplained_inside': n.offending_pixels(envelope & (context.dark != 0) & ~owned)}
        supported(not (owned & ~envelope).any(), 'member ink connected to border/art or clipped enclosure')
        supported(not (owned & ~dilate(members, options['member_padding'])).any(),
                  'whole ink component exceeds bounded member neighborhood')
        unexplained = envelope & (context.dark != 0) & ~owned
        supported(not unexplained.any(), 'unexplained ink inside enclosure; punctuation/art not silently omitted')
        safe = erode(envelope, options['clearance'])
        # Assess the usable core, not tiny excluded outline irregularities.
        # Required whole ink + padding is checked below; erosion cannot discard it.
        horizontal = safe & ~np.pad(safe[:, :-1], ((0, 0), (1, 0)))
        vertical = safe & ~np.pad(safe[:-1], ((1, 0), (0, 0)))
        permission = filled_outer(dilate(owned, options['glyph_padding']))
        supported(not (permission & ~safe).any(), 'ink/punctuation lacks boundary clearance; no erosion of required ink')
        # Validate the complete original core, with a single evidence-qualified
        # additional inward geometry step for boundary-connected fringe only.
        try:
            _, full_glyph, full_color = n.cleanup(rgb, safe, permission, np.zeros_like(safe))
        except n.CleanupBlocked as error:
            candidate, repair = conservative_boundary_inset(context, envelope, safe, permission, members,
                                                            error, options['member_padding'])
            measured['additional_boundary_inset'] = repair
            if candidate is None:
                raise
            _, full_glyph, full_color = n.cleanup(rgb, candidate, permission, np.zeros_like(candidate))
            safe = candidate
            repair['geometry_inset_applied'] = True
            repair['retained_cleanup_pixels'] = int(full_glyph.sum())
            horizontal = safe & ~np.pad(safe[:, :-1], ((0, 0), (1, 0)))
            vertical = safe & ~np.pad(safe[:-1], ((1, 0), (0, 0)))
        if int(horizontal.sum(axis=1).max()) > 1 or int(vertical.sum(axis=0).max()) > 1:
            measured['span_evidence'] = span_evidence(safe)
            # Do not hide texture, gray punctuation or border contamination by
            # cropping: the ENTIRE current core has passed first (any preceding
            # boundary-only inset is separately evidenced above).
            safe, core = trim_empty_boundary_spans(safe, permission, options['clearance'])
            core.update(full_original_core_cleanup_pixels=int(full_glyph.sum()), full_original_core_background=full_color)
            measured['boundary_span_trim'] = core
        measured['polygonization_stage'] = 'interior'
        interior_polys, interior = polygons_inside(safe, permission)
        measured['polygonization_stage'] = 'layout'
        layout_polys, layout = polygons_inside(erode(interior, options['layout_inset']))
        measured['polygonization_stage'] = 'cleanup_permission'
        permission_polys, permission = polygons_inside(permission, dilate(owned, 1))
        measured['polygonization_stage'] = 'complete'
        measured['polygon_counts'] = [len(interior_polys), len(layout_polys), len(permission_polys)]
        supported(not (permission & ~interior).any(), 'permission escapes safe interior')
        supported(sum(len(poly) for group in (interior_polys, layout_polys, permission_polys) for poly in group) <= 4096,
                  'aggregate polygon vertex budget')
        # The unchanged renderer remains the independent flatness/collar guard.
        _, glyph, color = n.cleanup(rgb, interior, permission, np.zeros_like(interior))
        supported(not (owned & ~glyph).any(), 'required original ink absent from derived cleanup mask')
        measured.update(owned_components=int(len(owned_ids)), owned_ink_pixels=int(owned.sum()),
                        cleanup_pixels=int(glyph.sum()), background_rgb=color,
                        accounted_hole_pixels=int((envelope & ~background).sum()),
                        protection='clearance excludes boundary; exterior locality enforced by renderer; no art-hole permission')
        return {'status': 'proposed', 'geometry': {'interior': interior_polys, 'layout': layout_polys,
                'cleanup_permission': permission_polys, 'protected': []}, 'measured': measured}
    except (Unsupported, n.CleanupBlocked) as error:
        result = {'status': 'rejected', 'reason': str(error), 'measured': measured}
        if isinstance(error, n.CleanupBlocked):
            result['cleanup_diagnostics'] = error.diagnostics
        return result


def source_rgb(opened):
    """Expand only explicit opaque channels; never guess a transparency matte.

    The caller checks dimensions before decoding. Alpha-bearing originals are
    accepted only if EVERY alpha sample is 255. Original bytes remain evidence;
    canonical RGB values do not depend on a chosen display background.
    """
    n.budget_check(opened.width*opened.height <= MAX_PIXELS and max(opened.size) <= MAX_SIDE,
                   'preparation source pixel/side budget')
    n.require(opened.mode in ('RGB', 'L', 'RGBA', 'LA') and getattr(opened, 'n_frames', 1) == 1
              and 'transparency' not in opened.info and not opened.info.get('icc_profile')
              and opened.getexif().get(274, 1) == 1, 'unsupported source mode/frame/profile/orientation/transparency')
    if 'A' in opened.getbands():
        n.require(opened.getchannel('A').getextrema() == (255, 255),
                  'unsupported source transparency; an explicit opaque source is required')
    return opened.convert('RGB')


def _sources(preview, image, fonts):
    paths = [Path(value).absolute() for value in (preview, image, fonts)]
    contents = [n.read_explicit(path, limit) for path, limit in zip(paths, (16*1024*1024, 32*1024*1024, 256*1024))]
    p, f = n.document(contents[0]), n.document(contents[2])
    with Image.open(BytesIO(contents[1])) as opened:
        n.budget_check(opened.width*opened.height <= MAX_PIXELS and max(opened.size) <= MAX_SIDE,
                       'preparation source pixel/side budget')
        rgb = source_rgb(opened)
    source = {'path': str(paths[1].resolve()), 'sha256': n.digest(contents[1]),
              'pixels_sha256': n.digest(f'RGB:{rgb.width}:{rgb.height}:'.encode()+rgb.tobytes()), 'size': list(rgb.size)}
    n.require(type(p['schema_version']) is int and p['schema_version'] in (1, 2, 3, 4) and p['preview_only'] is True
              and n.canonical(p['input']) == n.canonical(source), 'saved preview/current source binding drift')
    n.require(type(p['units']) is list and len(p['units']) <= 512 and len(p['ocr']['raw_detections']) == 1,
              'single-page unit budget')
    ids = [r['unit']['unit_id'] for r in p['units']]
    n.require(all(type(i) is str and 0 < len(i) <= 128 for i in ids) and len(set(ids)) == len(ids), 'saved unit ID drift')
    return paths, contents, p, f, rgb, source


def _binding(record, p, rgb, source, contents, font_id, style):
    """Reuse the renderer's complete source/request/result validator, without I/O.

    This transient record is binding preflight only, never exported as a selection
    and never supplies geometry or operator confirmation.
    """
    supported(record['status'] == 'preview' and record.get('input_kind', 'strict_utterance') == 'strict_utterance'
              and record.get('binding_current') is True, 'strict candidate unavailable; blocked/fragments/alternates are retained, not promoted')
    unit = record['unit']
    result = record['result']
    supported(type(result) is dict and result['status'] == 'preview' and len(result['translations']) == 1,
              'strict translated result unavailable')
    uid, text = result['translations'][0]
    supported(uid == unit['unit_id'] and type(text) is str and 0 < len(text) <= 1000, 'strict translated text/budget mismatch')
    binding = {'unit_id': uid, 'member_ids': unit['member_ids'], 'order': unit['order'],
        'unit_revision': record['unit_revision'], 'request_revision': record['request_revision'],
        'result_sha256': n.revision(result), 'text': text, 'font_id': font_id, 'style': style}
    transient = {'schema_version': 1, 'purpose': 'candidate_only', 'source': source,
        'preview_sha256': n.digest(contents[0]), 'fonts_sha256': n.digest(contents[2]),
        'units': [{**binding, 'geometry_intent': 'Automated binding preflight only; no geometry confirmation or human review.'}]}
    n.bind(p, source['path'], rgb, contents[1], contents[0], transient, contents[2])
    return binding


def _request(unit_ids, font_id, min_size, max_size, options):
    n.require(type(unit_ids) in (list, tuple) and 1 <= len(unit_ids) <= MAX_UNITS
              and all(type(i) is str and 0 < len(i) <= 128 for i in unit_ids)
              and len(set(unit_ids)) == len(unit_ids), 'explicit unique bounded unit IDs required')
    n.require(type(font_id) is str and re.fullmatch('[A-Za-z0-9_-]{1,64}', font_id), 'explicit font ID required')
    n.integer(min_size, 8, 144)
    n.integer(max_size, min_size, 144)
    return {'unit_ids': list(unit_ids), 'font_id': font_id,
            'style': {'min_size': min_size, 'max_size': max_size, 'outline': 0}, 'settings': settings(options)}


def _html(report):
    escape = html.escape
    head = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="'
        +escape(CSP, quote=True)+'"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Unconfirmed flat geometry proposals</title><style>*{box-sizing:border-box}body{font:16px sans-serif;'
        'margin:clamp(.75rem,3vw,2rem);overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1em}'
        'figure{margin:0;min-width:0}img{display:block;max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        '@media(max-width:600px){.grid{grid-template-columns:minmax(0,1fr)}}</style>'
        '<h1>Unconfirmed geometry proposals</h1><p>No erasure, translated composite, human/linguistic approval or automatic selection. '
        'Inspect the original source and full proposed extent before explicit confirmation. Unassigned punctuation or art cannot be certified by OCR.</p>'
        '<p>Overlay: green interior, blue layout, orange cleanup permission; rejected footprints are red. '
        'Exterior pixels are excluded by geometry; no protected art-hole permission is inferred.</p><p>Revision: '
        +escape(report['proposal_revision'])+'</p><a href="proposals.json">All measured evidence</a>')
    rows = []
    for index, row in enumerate(report['units']):
        rows.append('<section><h2>'+escape(row['unit_id'])+' — '+escape(row['status'])+'</h2><p>'+escape(row.get('reason', 'Proposal only; explicit confirmation required'))
            +'</p><p>Proposal ID: '+escape(row['proposal_id'])+'</p><div class="grid">'
            +''.join('<figure><figcaption>'+label+'</figcaption><img alt="'+label+'" src="unit-'+f'{index:03d}'+'-'+label+'.png"></figure>'
                     for label in ('source', 'overlay'))+'</div><details><summary>Binding, geometry, settings and uncertainty</summary><pre>'
            +escape(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False))+'</pre></details></section>')
    return (head+''.join(rows)+'</html>').encode()


def _build(paths, contents, p, f, rgb, source, request):
    """Build deterministic bounded bytes; also used to verify confirmation."""
    n.budget_check(len(request['unit_ids'])*rgb.width*rgb.height <= n.MAX_UNIT_PIXELS, 'preparation aggregate unit pixel budget')
    by_id = {r['unit']['unit_id']: r for r in p['units']}
    n.require(set(request['unit_ids']) <= set(by_id), 'foreign requested unit ID')
    fonts = n.load_fonts(f)
    n.require(request['font_id'] in fonts and not isinstance(fonts[request['font_id']], n.ResourceLimit), 'unavailable explicit font')
    font_data, coverage = fonts[request['font_id']]
    resolve_style(request['style'], face_metadata(font_data))
    raw = p['ocr']['raw_detections'][0]
    n.require(type(raw) is list and len(raw) <= 2048, 'raw detection budget')
    n.require(type(raw) is list and 1 <= len(raw) <= MAX_DETECTIONS, 'raw detection budget')
    n.budget_check(len(raw)*len(request['unit_ids'])*rgb.width*rgb.height <= MAX_PREFLIGHT_PIXELS,
                   'source binding/neighbor raster work budget')
    context = OriginalRegions(rgb)
    report = {'schema_version': 1, 'purpose': PURPOSE, 'algorithm': ALGORITHM,
        'implementation_sha256': implementation_hash(), 'source': source,
        'preview_sha256': n.digest(contents[0]), 'fonts_sha256': n.digest(contents[2]),
        'request': request, 'engine': {'pillow': n.pillow_version, 'opencv': cv2.__version__},
        'operator_confirmation': None, 'linguistic_approval': 'not_assessed', 'units': [],
        'limits': {'page_pixels': MAX_PIXELS, 'page_side': MAX_SIDE, 'units': MAX_UNITS,
                   'aggregate_unit_pixels': n.MAX_UNIT_PIXELS, 'components_per_labeling': 8192,
                   'raw_detections': MAX_DETECTIONS, 'binding_neighbor_raster_work': MAX_PREFLIGHT_PIXELS,
                   'member_masks_per_unit': 32, 'member_mask_pixels': n.MAX_UNIT_PIXELS,
                   'owned_components': 64, 'polygons_per_mask': 128, 'vertices_per_polygon': 128,
                   'aggregate_vertices_per_unit': 4096}}
    report['limits'].update(local_boundary_radius=BOUNDARY_RADIUS, enclosing_region_comparisons=8,
        contact_samples=32, pixel_contour_cycle_limit=256, pixel_contour_split_work_per_strip_trial=4_000_000,
        strip_trial_heights=[32, 16, 8], strip_trial_orientations=2,
        maximum_additional_boundary_inset=MAX_ADDITIONAL_INSET, separator_probe_maximum_radius=BOUNDARY_RADIUS)
    for uid in request['unit_ids']:
        record = by_id[uid]
        row = {'unit_id': uid, 'source_record_sha256': n.revision(record), 'binding_validated': False,
               'source_status': record['status'], 'status': 'unsupported'}
        try:
            binding = _binding(record, p, rgb, source, contents, request['font_id'], request['style'])
            row.update(binding_validated=True, selection_binding=binding)
            supported(all(ord(c) in coverage for c in binding['text'] if c != '\n'), 'chosen font lacks translated glyph coverage')
            row.update(propose_geometry(rgb, raw, binding['member_ids'], request['settings'], context=context))
        except Unsupported as error:
            row['reason'] = str(error)
        except (ValueError, KeyError, TypeError) as error:
            row.update(status='rejected', reason='binding/geometry validation failed: '+str(error))
        report['units'].append(row)
    # No proposal sharing/collision may be selected by changing iteration order.
    masks = [(i, n.union_mask(row['geometry']['interior'], rgb.size)) for i, row in enumerate(report['units']) if row['status'] == 'proposed']
    collisions = set()
    for pos, (i, mask) in enumerate(masks):
        for j, other in masks[pos+1:]:
            if (mask & other).any():
                collisions.update((i, j))
    for index in collisions:
        report['units'][index].update(status='rejected', reason='multiunit proposed interior collision')
    for row in report['units']:
        row['proposal_id'] = 'geometry-'+n.revision({'source': source, 'preview': report['preview_sha256'],
            'fonts': report['fonts_sha256'], 'implementation': report['implementation_sha256'], 'request': request, 'unit': row})[:32]
    report['proposal_revision'] = n.revision(report)
    blobs = {}
    total = 0
    def add(name, data):
        nonlocal total
        total += len(data)
        n.budget_check(len(data) <= 64*1024*1024 and total <= 128*1024*1024, 'preparation artifact byte budget')
        blobs[name] = data
    def add_png(name, image):
        stream = BytesIO()
        image.save(stream, format='PNG')
        add(name, stream.getvalue())
    for name, data in zip(('evidence-preview.json', 'original-file.bin', 'evidence-fonts.json'), contents):
        add(name, data)
    add('evidence-request.json', (n.canonical(request)+'\n').encode())
    add_png('source.png', rgb)
    for index, row in enumerate(report['units']):
        overlay = rgb.copy()
        draw = ImageDraw.Draw(overlay)
        if 'geometry' in row:
            for key, color in (('interior', (0, 150, 0)), ('layout', (20, 90, 255)), ('cleanup_permission', (255, 140, 0))):
                for polygon in row['geometry'][key]:
                    draw.line([tuple(point) for point in polygon]+[tuple(polygon[0])],
                              fill=color if row['status'] == 'proposed' else (220, 20, 20), width=1)
            footprint = n.union_mask(row['geometry']['interior'], rgb.size)
        else:
            footprint = np.zeros((rgb.height, rgb.width), bool)
            members = by_id[row['unit_id']]['unit']['member_ids']
            for detection in raw:
                if detection['source_id'] in members:
                    footprint |= np.asarray(n.polygon_mask(detection['polygon'], rgb.size, evidence=True)) != 0
                    draw.line([tuple(point) for point in detection['polygon']]+[tuple(detection['polygon'][0])], fill=(220, 20, 20), width=1)
        bounds = Image.fromarray(footprint.astype(np.uint8)*255).getbbox()
        n.require(bounds is not None, 'proposal has no source footprint')
        if row.get('measured', {}).get('original_enclosure_bounds'):
            original_bounds = row['measured']['original_enclosure_bounds']
            bounds = (min(bounds[0], original_bounds[0]), min(bounds[1], original_bounds[1]),
                      max(bounds[2], original_bounds[2]), max(bounds[3], original_bounds[3]))
        bounds = (max(0, bounds[0]-8), max(0, bounds[1]-8), min(rgb.width, bounds[2]+8), min(rgb.height, bounds[3]+8))
        scale = min(1, 1000/(bounds[2]-bounds[0]), 1000/(bounds[3]-bounds[1]))
        size = (max(1, int((bounds[2]-bounds[0])*scale)), max(1, int((bounds[3]-bounds[1])*scale)))
        row['overlay'] = {'source_bounds': list(bounds), 'size': list(size), 'scale': scale}
        add_png(f'unit-{index:03d}-source.png', rgb.resize(size, box=bounds))
        add_png(f'unit-{index:03d}-overlay.png', overlay.resize(size, box=bounds))
    # Overlay metadata is measured output too; bind it into the final revision.
    report['proposal_revision'] = n.revision({k: v for k, v in report.items() if k != 'proposal_revision'})
    page = _html(report)
    n.budget_check(len(page) <= 4*1024*1024, 'preparation report byte budget')
    add('review.html', page)
    report['artifacts'] = {name: n.digest(data) for name, data in blobs.items()}
    manifest = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    n.budget_check(len(manifest) <= 4*1024*1024, 'preparation manifest byte budget')
    add('proposals.json', manifest)
    return report, blobs


def prepare(*, preview, image, fonts, unit_ids, font_id, min_size, max_size, output, options=None):
    request = _request(unit_ids, font_id, min_size, max_size, options)
    output = Path(output).absolute()
    n.require(not output.exists() and output.parent.is_dir() and
              not any(p.is_symlink() for p in (output, *output.parents)), 'proposal output must be fresh without symlinks')
    paths, contents, p, f, rgb, source = _sources(preview, image, fonts)
    report, blobs = _build(paths, contents, p, f, rgb, source, request)
    n.require(all(n.read_explicit(path) == data for path, data in zip(paths, contents)), 'preparation inputs changed')
    output.mkdir(mode=0o700)
    for name, data in blobs.items():
        with (output/name).open('xb') as stream:
            stream.write(data)
    return report


def confirm_preparation(*, preview, image, fonts, proposal_bundle, proposal_ids, operator, intent, output):
    """Explicit unsigned geometry choice, never a human/linguistic approval flag."""
    n.require(type(proposal_ids) in (list, tuple) and 1 <= len(proposal_ids) <= MAX_UNITS and
              all(type(i) is str and re.fullmatch('geometry-[0-9a-f]{32}', i) for i in proposal_ids)
              and len(set(proposal_ids)) == len(proposal_ids), 'explicit unique chosen proposal IDs required')
    n.require(type(operator) is str and 0 < len(operator.strip()) <= 120 and
              type(intent) is str and 0 < len(intent.strip()) <= 2000, 'explicit operator identity and intent required')
    root = Path(proposal_bundle).absolute()
    raw = n.read_explicit(root/'proposals.json', 4*1024*1024)
    supplied = n.document(raw)
    n.require(supplied['schema_version'] == 1 and supplied['purpose'] == PURPOSE and supplied['algorithm'] == ALGORITHM
              and supplied['implementation_sha256'] == implementation_hash(), 'stale/foreign preparation implementation or purpose')
    request = supplied['request']
    checked = _request(request['unit_ids'], request['font_id'], request['style']['min_size'], request['style']['max_size'], request['settings'])
    n.require(n.canonical(checked) == n.canonical(request), 'preparation request drift')
    paths, contents, p, f, rgb, source = _sources(preview, image, fonts)
    expected, blobs = _build(paths, contents, p, f, rgb, source, checked)
    n.require(raw == blobs['proposals.json'], 'preparation source/binding/geometry/manifest drift')
    # Names come exclusively from regenerated output, never supplied JSON paths.
    for name, data in blobs.items():
        n.require(n.read_explicit(root/name) == data, 'preparation artifact drift: '+name)
    by_id = {row['proposal_id']: row for row in expected['units']}
    n.require(set(proposal_ids) <= set(by_id), 'foreign proposal ID')
    n.require(all(by_id[i]['status'] == 'proposed' and by_id[i]['binding_validated'] for i in proposal_ids),
              'rejected/unsupported proposals cannot be confirmed')
    units = [{**by_id[i]['selection_binding'], **by_id[i]['geometry'],
              'geometry_intent': f'Explicit preparation confirmation by {operator}: {intent}. Unsigned assertion; no identity or linguistic approval.'}
             for i in proposal_ids]
    selection = {'schema_version': 1, 'purpose': 'candidate_only', 'source': source,
        'preview_sha256': n.digest(contents[0]), 'fonts_sha256': n.digest(contents[2]), 'units': units,
        'preparation_confirmation': {'scope': 'geometry_candidate_only', 'operator': operator, 'intent': intent,
            'identity_verified': False, 'linguistic_approval': 'not_assessed', 'proposal_ids': list(proposal_ids),
            'proposal_manifest_sha256': n.digest(raw), 'proposal_revision': expected['proposal_revision'],
            'implementation_sha256': implementation_hash()}}
    n.bind(p, source['path'], rgb, contents[1], contents[0], selection, contents[2])
    n.require(all(n.read_explicit(path) == data for path, data in zip(paths, contents)), 'confirmation inputs changed')
    n.require(not Path(output).absolute().resolve().is_relative_to(root.resolve()), 'confirmation output must not modify proposal bundle')
    from .review import fresh_file
    data = (json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    n.budget_check(len(data) <= 4*1024*1024, 'confirmed selection byte budget')
    fresh_file(output, data)
    return selection
