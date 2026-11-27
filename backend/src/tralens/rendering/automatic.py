"""Automatic source-bound rendering, separate from operator-confirmed tools.

Flat fill first; bounded classical restoration is a separate automatic fallback.
No semantic balloon partition, human approval, model/provider or font lookup.
"""
from dataclasses import dataclass, replace
from io import BytesIO
import html
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import natural as n
from . import preparation as p
from .review import CSP
from . import typography
from . import replacement_confidence as rc


POLICY = 'automatic_original_tight_flat_v3'
MAX_SLOTS = 32
MAX_RETAINED_PIXELS = 128_000_000
MAX_ARTIFACTS = 512
MAX_WORK = 512_000_000


class AutomaticSkip(ValueError):
    def __init__(self, code, message, evidence=None):
        self.code = code
        self.evidence = {} if evidence is None else evidence
        super().__init__(message)


def retained_artwork(record):
    outcomes = (record.get('result') or {}).get('outcomes') or []
    return (record['status'] == 'abstained' and not record.get('luna_review_error')
            and record.get('luna_first_decision', {}).get('category', 'artwork') == 'artwork'
            and len(outcomes) == 1
            and outcomes[0].get('reasons') in (['visual_non_text'], ('visual_non_text',)))


def retained_label(record):
    if record['status'] != 'abstained' or record.get('luna_review_error'):
        return False
    decisions = record.get('luna_decision') or {}
    if decisions.get('category') == 'already_english':
        return True
    outcomes = (record.get('result') or {}).get('outcomes') or []
    return len(outcomes) == 1 and any(reason in ('original_lettering', 'already_latin_lettering')
                                      for reason in outcomes[0].get('reasons', ()))


def check(condition, code, message, evidence=None):
    if not condition:
        raise AutomaticSkip(code, message, evidence)


class OriginalComponents:
    def __init__(self, rgb):
        n.budget_check(rgb.mode == 'RGB' and rgb.width*rgb.height <= p.MAX_PIXELS and max(rgb.size) <= p.MAX_SIDE,
                       'automatic source pixel/side budget')
        self.pixels = np.asarray(rgb).copy()
        gray = cv2.cvtColor(self.pixels, cv2.COLOR_RGB2GRAY)
        count, self.labels, self.stats, _ = cv2.connectedComponentsWithStats((gray < 180).astype(np.uint8), connectivity=8)
        n.budget_check(count <= 8192, 'automatic original component budget')
        self.geometry_revision = None
        self.source_counts = None
        self.source_polygons = None

    def bind_geometry(self, detections, *, remaining_work=MAX_WORK):
        revision = n.revision([(type(d['source_id']).__name__, d['source_id'], d['polygon']) for d in detections])
        if self.geometry_revision is not None:
            n.require(revision == self.geometry_revision, 'source geometry changed since raster snapshot')
            return 0
        height,width = self.pixels.shape[:2]
        work = height*width
        n.budget_check(work <= remaining_work, 'automatic source geometry raster budget')
        counts = np.zeros((height,width), np.uint16)
        polygons = {}
        for detection in detections:
            points = detection['polygon']
            n.validate_polygon(points,(width,height),evidence=True)
            # Pillow rounds scan-line intersections at absolute x coordinates.
            # Keeping the page width preserves its exact boundary pixels.
            ys = [py for _,py in points]
            y,b = max(0,int(min(ys))-1),min(height,int(max(ys))+2)
            work += width*(b-y)
            n.budget_check(work <= remaining_work, 'automatic source geometry raster budget')
            tile = Image.new('L',(width,b-y))
            ImageDraw.Draw(tile).polygon([(px,py-y) for px,py in points],fill=1)
            counts[y:b] += np.asarray(tile)
            key = (type(detection['source_id']).__name__, detection['source_id'])
            polygons[key] = points
        self.source_counts, self.source_polygons, self.geometry_revision = counts, polygons, revision
        return work


@dataclass
class AutomaticPlan:
    foreground: np.ndarray
    erase: np.ndarray
    layout: np.ndarray
    obstacles: np.ndarray
    background: list | None
    evidence: dict
    restored_pixels: np.ndarray | None = None


class PackedPlan:
    """Retain four exact boolean masks at one bit/pixel, not four bytes/pixel.

    Full masks are materialized only while computing collisions/composition or
    writing one unit's evidence. Pixel coordinates and output PNGs are unchanged.
    """
    def __init__(self, plan):
        self.shape = plan.erase.shape
        self.masks = {name: np.packbits(getattr(plan, name), axis=None)
                      for name in ('foreground', 'erase', 'layout', 'obstacles')}
        self.background, self.evidence, self.restored_pixels = plan.background, plan.evidence, plan.restored_pixels

    def __getattr__(self, name):
        if name in self.masks:
            return np.unpackbits(self.masks[name], count=self.shape[0]*self.shape[1]).reshape(self.shape).view(bool)
        raise AttributeError(name)


def apply_cleanup(pixels, plan):
    """Commit only source-derived mask pixels, from fill or validated restoration."""
    if plan.restored_pixels is None:
        n.require(plan.background is not None, 'missing flat cleanup color')
        pixels[plan.erase] = plan.background
    else:
        n.require(plan.restored_pixels.dtype == np.uint8
            and plan.restored_pixels.shape == (int(plan.erase.sum()), 3), 'restoration sample shape/type mismatch')
        pixels[plan.erase] = plan.restored_pixels


def occupied_reservation(plan, tile):
    """Protect committed ink/cleanup plus 2px, not unused placement whitespace.

    Layout support is not an ownership claim. This is only a reservation for
    later native operations; no source foreground or erase permission changes.
    """
    layer, (x, y) = tile
    height, width = plan.erase.shape
    n.require(layer.mode == 'RGBA' and type(x) is int and type(y) is int
        and 0 <= x < x+layer.width <= width and 0 <= y < y+layer.height <= height,
        'invalid committed text tile geometry')
    alpha = np.zeros_like(plan.erase)
    alpha[y:y+layer.height, x:x+layer.width] = np.asarray(layer.getchannel('A')) != 0
    check(not (alpha & (~plan.layout | plan.obstacles)).any(), 'alpha_containment',
          'reservation text leaves original native layout support')
    return p.dilate(plan.erase | alpha, 2)


def reserve_layout(plan, reserved):
    """Trim only unused layout space; never clip a stroke or cleanup permission.

    A neighboring accepted output can reserve whitespace shared by two local
    layout windows. Rejecting that overlap needlessly forces coarse cleanup.
    Keep the complete native erase/foreground and protect every reserved pixel.
    Actual exact-glyph fitting still decides whether the remaining space works.
    """
    check(not (plan.erase & reserved).any(), 'region_collision', 'native cleanup touches accepted output')
    layout = plan.layout & ~reserved
    check(layout.any(), 'region_collision', 'no native layout space remains outside accepted output')
    evidence = {**plan.evidence, 'layout_reservation': {
        'method': 'subtract_reserved_layout_only', 'original_layout_pixels': int(plan.layout.sum()),
        'retained_layout_pixels': int(layout.sum()), 'cleanup_pixels_discarded': 0,
        'source_foreground_discarded': 0, 'scope': 'layout support only, not new erase permission'}}
    return replace(plan, layout=layout, obstacles=plan.obstacles | reserved, evidence=evidence)


def grow_rectangle(safe, seed, window):
    x, y, w, h = cv2.boundingRect(seed.astype(np.uint8))
    wl, wt, wr, wb = window
    blocked = np.zeros_like(safe)
    blocked[y:y+h, x:x+w] = ~safe[y:y+h, x:x+w]
    check(w > 0 and h > 0 and not blocked.any(), 'layout_seed_obstructed',
          'complete source ink cannot fit a clear local rectangle',
          {'seed_bounds': [x, y, x+w, y+h], 'obstructed_seed': n.offending_pixels(blocked)})
    check(wl <= x and wt <= y and x+w <= wr and y+h <= wb, 'layout_window',
          'complete source ink/collar lies outside bounded layout window')
    left, top, right, bottom = x, y, x+w, y+h
    active = [True]*4
    steps = 0
    while any(active):
        for side in range(4):
            if not active[side]:
                continue
            if side == 0:
                good = left > wl and safe[top:bottom, left-1].all()
                if good:
                    left -= 1
            elif side == 1:
                good = top > wt and safe[top-1, left:right].all()
                if good:
                    top -= 1
            elif side == 2:
                good = right < wr and safe[top:bottom, right].all()
                if good:
                    right += 1
            else:
                good = bottom < wb and safe[bottom, left:right].all()
                if good:
                    bottom += 1
            active[side] = bool(good)
            steps += 1
        n.budget_check(steps <= 2*(safe.shape[0]+safe.shape[1])+4, 'automatic rectangle search budget')
    result = np.zeros_like(safe)
    result[top:bottom, left:right] = True
    return result, [left, top, right, bottom], steps


def source_local_layout(safe, seed, window):
    """Prefer the existing rectangle, then retain safe irregular whitespace.

    Erasure and placement are independent. A rectangle's empty corner touching an
    obstacle does not imply the actual source ink/collar touches it. The fallback
    keeps the same window, original obstacles and clearance, and must retain the
    entire seed in one 4-connected safe component. It never fills holes, bridges
    disconnected pockets or invents balloon ownership.
    """
    try:
        layout, bounds, steps = grow_rectangle(safe, seed, window)
        return layout, bounds, steps, {'method': 'source_local_rectangle'}
    except AutomaticSkip as error:
        if error.code != 'layout_seed_obstructed':
            raise
        rejection = {'code': error.code, 'reason': str(error), 'evidence': error.evidence}
        # Preserve genuine source-ink/obstacle contact as a rejection, not a
        # reason to cut the offending seed pixels out of a layout mask.
        if (seed & ~safe).any():
            raise
    left, top, right, bottom = window
    x, y, width, height = cv2.boundingRect(seed.astype(np.uint8))
    check(width > 0 and height > 0 and left <= x and top <= y
          and x+width <= right and y+height <= bottom,
          'layout_window', 'complete source ink/collar lies outside bounded layout window')
    local_seed = seed[top:bottom, left:right]
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        safe[top:bottom, left:right].astype(np.uint8), connectivity=4)
    n.budget_check(count <= 8192, 'automatic local layout component budget')
    chosen = np.unique(labels[local_seed])
    check(len(chosen) == 1 and chosen[0] != 0, 'layout_seed_disconnected',
          'complete source ink/collar does not share one bounded safe whitespace component',
          {'rectangular_seed_rejection': rejection, 'seed_component_count': len(chosen)})
    local = labels == chosen[0]
    layout = np.zeros_like(safe)
    layout[top:bottom, left:right] = local
    n.require(not (seed & ~layout).any() and not (layout & ~safe).any(),
              'irregular layout lost source clearance or added unsafe pixels')
    x, y, width, height = cv2.boundingRect(local.astype(np.uint8))
    bounds = [left+x, top+y, left+x+width, top+y+height]
    return layout, bounds, 0, {'method': 'source_local_connected_safe_mask', 'connectivity': 4,
        'rectangular_seed_rejection': rejection, 'local_components': count-1,
        'required_seed_pixels': int(seed.sum()), 'required_seed_pixels_discarded': 0,
        'layout_pixels': int(local.sum()), 'scope': 'bounded safe whitespace, not a balloon partition'}


def derive_plan(rgb, detections, member_ids, *, context=None):
    """Original whole components + measured flat collars; no enclosure ownership."""
    context = OriginalComponents(rgb) if context is None else context
    n.require(np.array_equal(np.asarray(rgb), context.pixels), 'source pixels changed since component snapshot')
    n.require(type(detections) is list and 1 <= len(detections) <= 512, 'automatic detection budget')
    n.require(type(member_ids) is list and 1 <= len(member_ids) <= 32
              and all(type(i) in (int, str) and i != '' for i in member_ids) and len(set(member_ids)) == len(member_ids),
              'typed unique members required')
    n.budget_check(len(member_ids)*rgb.width*rgb.height <= n.MAX_UNIT_PIXELS, 'automatic raster work budget')
    keys = [(type(d['source_id']).__name__, d['source_id']) for d in detections]
    n.require(all(type(d['source_id']) in (int, str) and d['source_id'] != '' for d in detections)
              and len(set(keys)) == len(keys), 'typed unique detection IDs required')
    selected = {(type(i).__name__, i) for i in member_ids}
    n.require(selected <= set(keys), 'foreign source member')
    context.bind_geometry(detections)
    members = np.zeros((rgb.height, rgb.width), bool)
    own_counts = np.zeros(members.shape, np.uint16)
    for key in selected:
        mask = np.asarray(n.polygon_mask(context.source_polygons[key], rgb.size, evidence=True)) != 0
        members |= mask
        own_counts += mask
    # Exact counting preserves even a nonmember polygon fully contained in or
    # coincident with a member; subtracting boolean unions would lose it.
    nonmembers = context.source_counts > own_counts
    neighborhood = p.dilate(members, 2)
    ids = np.unique(context.labels[members])
    ids = ids[ids != 0]
    check(0 < len(ids) <= 64, 'dark_component_support', 'no bounded original dark-component support')
    lookup = np.zeros(len(context.stats), bool)
    lookup[ids] = True
    dark = lookup[context.labels]
    spill = dark & ~neighborhood
    evidence = {'segmentation': 'original RGB-to-gray <180 seeds; whole 8-connected components',
        'dark_components': int(len(ids)), 'whole_component_spill_pixels': int(spill.sum()),
        'source_member_padding': 2, 'erase_dilation': 1, 'background_tolerance': 2,
        'component_samples': [{'label': int(i), 'stats': [int(v) for v in context.stats[i]]} for i in ids[:16]],
        'ownership': 'machine source-local hypothesis, not semantic balloon ownership',
        'full_utterance_coverage': 'unverified; outside-source-polygon punctuation is not certified'}
    check(not spill.any(), 'component_spill', 'whole original component spills beyond source neighborhood (border/art/clipping unsupported)', evidence)
    for i in ids:
        x, y, w, h, area = (int(v) for v in context.stats[i])
        check(x > 0 and y > 0 and x+w < rgb.width and y+h < rgb.height and max(w, h) <= 96
              and w*h <= max(64, int(members.sum())//2) and area <= 20000,
              'component_extent', 'original component extent is unsupported; possible border/art/frame', evidence)
    check(not (p.dilate(dark, 1) & nonmembers).any(), 'nonmember_ink', 'source component/erase neighborhood intersects a nonmember polygon', evidence)
    sample_mask = p.dilate(dark, 4) & ~p.dilate(dark, 2)
    sample = context.pixels[sample_mask]
    check(len(sample) >= 32, 'insufficient_background', 'fewer than 32 original local collar samples', evidence)
    color = np.median(sample, axis=0).astype(np.uint8)
    spread = int(color.max())-int(color.min())
    evidence.update(background_rgb=color.tolist(), background_samples=len(sample))
    check(int(color.min()) >= 180 and spread <= 16, 'unsupported_background', 'only light neutral white/off-white local backgrounds are supported', evidence)
    difference = np.max(np.abs(context.pixels.astype(np.int16)-color.astype(np.int16)), axis=2)
    check(int(difference[sample_mask].max()) <= 2, 'nonflat_background', 'original collar contains nonflat color, gradient or neighboring ink', evidence)
    # Audit antialias and color with complete original RGB-contrast components,
    # not a blind one-pixel enlargement of the 180-threshold seed silhouette.
    contrast = difference > 2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(contrast.astype(np.uint8), connectivity=8)
    n.budget_check(count <= 8192, 'automatic full-contrast component budget')
    full_ids = np.unique(labels[dark])
    full_ids = full_ids[full_ids != 0]
    full_lookup = np.zeros(count, bool)
    full_lookup[full_ids] = True
    foreground = full_lookup[labels]
    check(not (foreground & ~neighborhood).any(), 'antialias_spill', 'whole contrast/antialias component escapes source neighborhood', evidence)
    check(not (dark & ~foreground).any(), 'missing_dark_coverage', 'full contrast mask does not cover original seed ink', evidence)
    unexplained = neighborhood & contrast & ~foreground
    check(not unexplained.any(), 'unassigned_contrast', 'unassigned original contrast/punctuation in source neighborhood',
          {**evidence, 'unassigned': n.offending_pixels(unexplained)})
    ink_colors = context.pixels[foreground]
    check(len(ink_colors) > 0 and int((ink_colors.max(axis=1).astype(np.int16)-ink_colors.min(axis=1)).max()) <= max(8, spread+2),
          'colored_ink', 'colored foreground/art is unsupported by the automatic monochrome policy', evidence)
    erase = p.dilate(foreground, 1)
    check(not (erase & nonmembers).any(), 'nonmember_erasure', 'complete erasure mask intersects a nonmember polygon', evidence)
    collar = p.dilate(erase, 2) & ~erase
    check(int(collar.sum()) >= 32 and not (contrast & collar).any(), 'clearance_contamination',
          'two-pixel erasure collar contains original non-background/border/art', evidence)
    check(not erase[0].any() and not erase[-1].any() and not erase[:, 0].any() and not erase[:, -1].any(),
          'source_edge', 'erasure reaches source frame; complete clearance unavailable', evidence)
    x, y, w, h = cv2.boundingRect(members.astype(np.uint8))
    mx, my = max(4, min(32, (w+1)//2)), max(4, min(24, (h+9)//10))
    window = [max(2, x-mx), max(2, y-my), min(rgb.width-2, x+w+mx), min(rgb.height-2, y+h+my)]
    obstacles = (contrast & ~foreground) | nonmembers
    safe = ~p.dilate(obstacles, 2)
    safe[:2] = safe[-2:] = False
    safe[:, :2] = safe[:, -2:] = False
    seed = p.dilate(foreground, 2)
    try:
        layout, bounds, steps, layout_derivation = source_local_layout(safe, seed, window)
    except AutomaticSkip as error:
        error.evidence = {**evidence, **error.evidence}
        raise
    check(not (layout & obstacles).any(), 'layout_obstacle', 'source-local layout covers original obstacle')
    evidence.update(full_contrast_components=int(len(full_ids)), dark_pixels=int(dark.sum()),
        foreground_pixels=int(foreground.sum()), antialias_pixels=int((foreground & ~dark).sum()),
        erase_pixels=int(erase.sum()), clearance_samples=int(collar.sum()),
        source_local_window=window, layout_bounds=bounds, layout_expansion_tests=steps,
        layout_derivation=layout_derivation,
        layout_scope='bounded original safe whitespace; not a semantic boundary/partition',
        unrelated_contrast_obstacles=int((contrast & ~foreground).sum()))
    return AutomaticPlan(foreground, erase, layout, obstacles, color.tolist(), evidence)


def implementation_hash():
    return n.revision({name: n.digest(n.read_explicit(Path(__file__).parent.resolve()/name, 256*1024))
                       for name in ('automatic.py', 'natural.py', 'font_faces.py', 'styles.py', 'preparation.py', 'review.py', 'typography.py', 'restoration.py', 'segmentation.py', 'source_support.py', 'source_extents.py', 'best_effort.py', 'replacement_confidence.py', 'layout_enclosures.py', 'localized_strokes.py', 'localization_requests.py', 'visible_translations.py', 'aggressive.py', 'printed_contours.py', 'outlined_sound_cleanup.py', 'table_cells.py', 'repeated_strokes.py', '../ocr/table_rows.py', 'replacement.py', 'reconstruction.py', 'reconstruction_worker.py', '../translation/coverage.py', '../translation/context.py', '../translation/contracts.py', '../ocr/manga_dialogue.py', '../ocr/crop_readings.py', '../ocr/upright_latin.py', '../ocr/sound_readings.py', '../ocr/outlined_sounds.py', '../ocr/handwritten.py', '../ocr/handwritten_lines.py', '../ocr/repeated_effects.py', '../ocr/masked_lines.py', '../ocr/sideways_sound.py', '../ocr/display_readings.py', '../ocr/ink_groups.py', '../ocr/quality_reread.py', '../ocr/quality_lines.py', '../ocr/quality_groups.py', '../ocr/source_localization.py')})


def _report_html(report):
    escape = html.escape
    head = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="'
        +escape(CSP, quote=True)+'"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Automatic tight-mask candidates</title><style>*{box-sizing:border-box}body{font:16px sans-serif;'
        'margin:clamp(.75rem,3vw,2rem);overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1em}'
        'figure{margin:0;min-width:0}img{display:block;max-width:100%;height:auto;background:#eee}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        '@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}'
        '@media(max-width:420px){.grid{grid-template-columns:minmax(0,1fr)}}</style>'
        '<h1>Automatic machine-selected candidates</h1><p>No operator confirmation, human/linguistic approval or final export. '
        'Partial output: skipped units remain original. Source-local layout is not balloon ownership/full-utterance proof.</p><pre>'
        +escape(json.dumps(report['coverage'], indent=2))+'</pre><p>Job '+escape(report['job_revision'])+'</p>'
        '<a href="manifest.json">Exact source, policy, hashes and per-unit reasons</a>')
    rows = []
    if 'display_summary' in report:
        rows.append('<h2>Visible translations — deliberately degraded display</h2><p><strong>'
            +escape(report['display_summary'].get('warning','Fallback backing may cover artwork.'))
            +'</strong> Captions retain their original source text; these are not clean replacements or verified reconstruction.</p><pre>'
            +escape(json.dumps(report['display_summary'],indent=2))+'</pre>')
    if 'replacement_confidence' in report:
        rows.append('<p><strong>EXPERIMENTAL CONFIDENCE GATE: some translations remain text-only while their source lettering '
            'is preserved. OCR scores are not accuracy probabilities; cleanup support is assessed separately.</strong></p><pre>'
            +escape(json.dumps({'settings': report['replacement_policy'], 'summary': report['replacement_confidence']}, indent=2))+'</pre>')
    if report.get('quality_summary', {}).get('best_effort_rendered'):
        rows.append('<p><strong>BEST-EFFORT OUTPUT: some edits use uncertain OCR footprints, may damage artwork, '
            'or use smaller/emergency-wrapped text. Originals are retained; inspect the marked units below.</strong></p>')
    for index, row in enumerate(report['units']):
        images = ''
        typography_note = ''
        background_note = ''
        if row.get('render_quality') == 'best_effort':
            background_note = '<p><strong>BEST-EFFORT replacement — background/art preservation is not guaranteed.</strong></p>'
        if row.get('background_attempts'):
            background_note += '<ul>'+''.join('<li>'+escape(attempt['method'])+': '+escape(attempt['status'])
                +(' — '+escape(attempt['reason']) if attempt.get('reason') else '')+'</li>'
                for attempt in row['background_attempts'])+'</ul>'
        if row.get('cleanup_method', '').startswith('inpaint_'):
            background_note += '<p>Classical restoration candidate: hidden background is estimated, not verified original artwork.</p>'
        if row.get('cleanup_method') == 'phase_locked_tile':
            background_note += '<p>Phase-locked periodic repair: estimated from original context, not verified hidden artwork.</p>'
        if row.get('layout', {}).get('typography'):
            layout = row['layout']
            fitted = layout['typography']
            typography_note = ('<p>Text size: '+str(layout['size'])+'px; preferred: '+str(fitted['preferred_size'])
                +'px; inner margin: '+str(fitted['margin_pixels'])+'px; '
                +str(len(layout['draw_text'].split('\n')))+' lines. '
                +f'<a href="unit-{index:03d}-content-mask.png">Padded text mask</a>.</p>')
        if row.get('specimen'):
            images = '<div class="grid">'+''.join('<figure><figcaption>'+name+'</figcaption><img alt="'+name+'" src="'
                +f'unit-{index:03d}-specimen-{name}.png'+'"></figure>' for name in ('source', 'clean', 'text', 'composite'))+'</div>'
            images += '<div class="grid">'+''.join('<figure><figcaption>'+name+'</figcaption><img alt="'+name+'" src="'
                 +f'unit-{index:03d}-specimen-{name}.png'+'"></figure>' for name in ('foreground', 'applied-erase', 'layout', 'obstacles'))+'</div>'
        if row.get('display_fallback',{}).get('status') == 'displayed':
            images += '<p><strong>'+escape(row['display_status'])+'; artwork may be obscured; source text removal is not certified.</strong></p>'
            images += '<div class="grid">'+''.join('<figure><figcaption>'+suffix+'</figcaption><img alt="fallback '+suffix+'" src="'
                +f'unit-{index:03d}-fallback-{suffix}.png'+'"></figure>' for suffix in ('patch','text','mask'))+'</div>'
        rows.append('<section><h2>'+escape(row['unit_id'])+' — '+escape(row['status'])+'</h2><p>'
            +escape(row.get('reason') or 'Automatically composed candidate; unverified wording/ownership')+'</p>'+background_note+typography_note+images
            +'<details><summary>Source evidence, settings and checks</summary><pre>'
            +escape(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False))+'</pre></details></section>')
    return (head+''.join(rows)+'</html>').encode()


def _font(fonts_doc, font_id):
    available = n.load_fonts(fonts_doc)
    if font_id is None:
        font_id = fonts_doc.get('default_font_id')
    if font_id is None:
        n.require(len(available) == 1, 'multiple approved faces require a global --font-id; no implicit font matching')
        font_id = next(iter(available))
    n.require(type(font_id) is str and re.fullmatch('[A-Za-z0-9_-]{1,64}', font_id), 'bounded automatic font ID required')
    n.require(font_id in available and not isinstance(available[font_id], n.ResourceLimit), 'unavailable explicit automatic font')
    return font_id, available[font_id]


def layout_settings(policy, render_policy, confidence_settings):
    n.require(type(policy) is str and policy in ('source-local', 'enclosed-retry'), 'unknown automatic layout policy')
    n.require(policy == 'source-local' or (render_policy == 'best-effort' and confidence_settings is not None),
              'enclosed-retry requires best-effort rendering and confidence-gated replacement')


def cleanup_settings(policy, render_policy, confidence_settings):
    n.require(type(policy) is str and policy in ('native', 'localized'), 'unknown automatic cleanup policy')
    n.require(policy == 'native' or (render_policy == 'best-effort' and confidence_settings is not None),
              'localized cleanup requires best-effort rendering and confidence-gated replacement')


def _render_policies(min_size, max_size, max_units, background_method, render_policy, replacement_policy,
                     min_ocr_confidence, layout_policy, cleanup_policy, visibility_policy, removal_policy,
                     reconstruction_policy, reconstruction_config):
    n.integer(min_size, 8, 144)
    n.integer(max_size, min_size, 144)
    if max_units is not None:
        n.integer(max_units, 1, MAX_SLOTS)
    n.require(type(background_method) is str and background_method in ('auto', 'flat'), 'unknown background method')
    n.require(type(render_policy) is str and render_policy in ('best-effort', 'conservative'), 'unknown rendering policy')
    confidence_settings = rc.settings(replacement_policy, min_ocr_confidence)
    layout_settings(layout_policy, render_policy, confidence_settings)
    cleanup_settings(cleanup_policy, render_policy, confidence_settings)
    from .visible_translations import settings as visibility_settings
    visibility_settings(visibility_policy)
    from . import aggressive, reconstruction
    aggressive.settings(removal_policy,reconstruction_policy,reconstruction_config)
    reconstruction_spec = reconstruction.configuration(reconstruction_config,reconstruction_policy) if reconstruction_policy!='classical' else None
    return confidence_settings, reconstruction_spec


def preflight_automatic(*, image, fonts, font_id=None, min_size=16, max_size=28, max_units=None, background_method='auto', text_mask_model=None, render_policy='best-effort', replacement_policy='all', min_ocr_confidence=None, layout_policy='source-local', cleanup_policy='native', visibility_policy='replacement-only', removal_policy='legacy', reconstruction_policy='classical', reconstruction_config=None):
    """Check rendering prerequisites before upstream inference, without a preview.

    This is not erase permission: run_automatic still validates the complete saved
    preview, source bindings, geometry, typography and actual output pixels.
    """
    confidence_settings, reconstruction_spec = _render_policies(
        min_size, max_size, max_units, background_method, render_policy, replacement_policy,
        min_ocr_confidence, layout_policy, cleanup_policy, visibility_policy, removal_policy,
        reconstruction_policy, reconstruction_config)
    image, fonts = Path(image).absolute(), Path(fonts).absolute()
    content = n.read_explicit(image, 32*1024*1024)
    font_manifest = n.read_explicit(fonts, 256*1024)
    with Image.open(BytesIO(content)) as opened:
        n.budget_check(opened.width*opened.height <= p.MAX_PIXELS and max(opened.size) <= p.MAX_SIDE,
                       'automatic source pixel/side budget')
        with p.source_rgb(opened) as rgb:
            source = {'path': str(image.resolve()), 'sha256': n.digest(content),
                'pixels_sha256': n.digest(f'RGB:{rgb.width}:{rgb.height}:'.encode()+rgb.tobytes()),
                'size': list(rgb.size)}
    selected, _ = _font(n.document(font_manifest), font_id)
    from .font_faces import ApprovedFaces
    ApprovedFaces(n.document(font_manifest), selected)
    n.require(n.features.check_feature('raqm'), 'RAQM unavailable')
    checked = {'source': source, 'fonts_sha256': n.digest(font_manifest), 'font_id': selected, 'render_policy': render_policy}
    if removal_policy!='legacy':
        checked.update(removal_policy=removal_policy,reconstruction_policy=reconstruction_policy,reconstruction_config=reconstruction_spec)
    if confidence_settings is not None:
        checked['replacement_policy'] = confidence_settings
    if layout_policy != 'source-local':
        checked['layout_policy'] = layout_policy
    if cleanup_policy != 'native':
        checked['cleanup_policy'] = cleanup_policy
    if visibility_policy != 'replacement-only':
        checked['visibility_policy'] = visibility_policy
    if text_mask_model is not None:
        from . import segmentation
        checked['text_mask_model'] = segmentation.model_spec(text_mask_model)
    return checked


def run_automatic(*, preview, image, fonts, output, font_id=None, min_size=16, max_size=28, max_title_size=None, max_units=None, background_method='auto', text_mask_model=None, render_policy='best-effort', replacement_policy='all', min_ocr_confidence=None, layout_policy='source-local', cleanup_policy='native', plan_localization=False, visibility_policy='replacement-only', removal_policy='legacy', reconstruction_policy='classical', reconstruction_config=None, prepared_text_mask=None):
    """All current strict translated candidates, with explicit machine skip states."""
    from . import aggressive, reconstruction, visible_translations
    confidence_settings, reconstruction_spec = _render_policies(
        min_size, max_size, max_units, background_method, render_policy, replacement_policy,
        min_ocr_confidence, layout_policy, cleanup_policy, visibility_policy, removal_policy,
        reconstruction_policy, reconstruction_config)
    n.require(type(plan_localization) is bool, 'localization planning needs explicit boolean enablement')
    if plan_localization:
        cleanup_settings('localized', render_policy, confidence_settings)
    from . import restoration
    from ..translation import coverage as tc
    model_spec = None
    if text_mask_model is not None:
        from . import segmentation
        model_spec = segmentation.model_spec(text_mask_model)
    output = Path(output).absolute()
    n.require(not output.exists() and output.parent.is_dir() and
              not any(path.is_symlink() for path in (output, *output.parents)), 'automatic output must be fresh without symlinks')
    paths, contents, preview_doc, fonts_doc, rgb, source = p._sources(preview, image, fonts)
    if max_title_size is not None:
        n.integer(max_title_size,8,144)
    raw = preview_doc['ocr']['raw_detections'][0]
    n.require(type(raw) is list and len(raw) <= 512 and (len(raw)>0 or not preview_doc['units']), 'automatic raw detection budget')
    outlines = {d['source_id']:d.get('ocr_metadata',{}).get('outline_reading',{})
                for d in raw if d.get('reason')=='corroborated_outline_reading'}
    title_bars = {d['source_id'] for d in raw if d.get('ocr_metadata',{}).get('tab_strip',{}).get('count')==1}
    pixels = rgb.width*rgb.height
    # Explicit machine binding contract; there is no geometry_intent, reviewed
    # flag, manual selection file, proposal ID or call to confirmation/export.
    source_contract = {'schema_version': 1, 'purpose': 'automatic_candidate_only', 'source': source,
        'preview_sha256': n.digest(contents[0]), 'fonts_sha256': n.digest(contents[2]), 'units': []}
    n.bind(preview_doc, paths[1], rgb, contents[1], contents[0], source_contract, contents[2], selection_kind='automatic_candidate_only')
    font_id, (font_data, coverage) = _font(fonts_doc, font_id)
    from .font_faces import ApprovedFaces, handwritten_columns
    faces = ApprovedFaces(fonts_doc, font_id)
    style = {'min_size': min_size, 'max_size': max_size, 'outline': 0}
    candidates = preview_doc['ocr'].get('automatic_recognition_results', [])
    units = n.automatic_units(preview_doc['ocr']['batches'][0], candidates[0] if candidates else ()) if raw else ()
    from ..ocr.crop_readings import confirms_raw
    crop_records = {item['proposal_id']: item for item in preview_doc['ocr'].get('crop_readings', {}).get('records', [])}
    proposals = {p['proposal_id']: p for p in preview_doc['ocr']['batches'][0]['proposals']} if raw else {}
    from ..translation.context import saved_requests
    contextual = saved_requests(preview_doc, units) if units else {}
    localized = {};extent_records={};extents={}
    if raw and cleanup_policy != 'native':
        from . import localized_strokes
        localized = localized_strokes.saved_records(preview_doc, rgb, raw, units)
        extent_records={uid:row for uid,row in localized.items() if row.get('crop_kind')=='extent'}
        localized={uid:row for uid,row in localized.items() if row.get('crop_kind','strokes')=='strokes'}
    report = {'schema_version': 1, 'purpose': 'automatic_candidate_output',
        'policy': POLICY if background_method == 'flat' else 'automatic_flat_then_classical_v1',
        'background_method': background_method,
        'render_policy': render_policy,
        'selection_method': 'strict translations first; explicitly bound uncertain text regions only in best-effort; machine geometric/typographic checks',
        'operator_confirmation': None, 'linguistic_approval': 'not_assessed', 'approved_export': False,
        'source': source, 'preview_sha256': n.digest(contents[0]), 'fonts_sha256': n.digest(contents[2]),
        'implementation_sha256': implementation_hash(), 'font_id': font_id, 'style': style,
        'typography_policy': typography.POLICY,
        'evaluation_cap': max_units, 'units': [],
        'limits': {'page_pixels': p.MAX_PIXELS, 'page_side': p.MAX_SIDE, 'processed_slots': MAX_SLOTS,
            'aggregate_unit_pixels': MAX_RETAINED_PIXELS, 'retained_mask_encoding': 'bitpacked_boolean_v1',
            'binding_neighbor_raster_work': MAX_WORK,
            'components_per_labeling': 8192, 'owned_components': 64, 'member_masks': 32, 'member_padding': 2,
            'seed_gray_threshold_exclusive': 180, 'component_side': 96, 'component_ink_pixels': 20000,
            'component_bbox_member_area_fraction': .5, 'component_bbox_area_floor': 64,
            'erase_dilation': 1, 'clearance': 2, 'sample_tolerance': 2, 'minimum_background_samples': 32,
            'background_channel_minimum': 180, 'background_channel_spread': 16,
            'layout_horizontal_expansion': 32, 'layout_vertical_expansion': 24,
            'irregular_layout_connectivity': 4, 'local_layout_components': 8192,
            'typography_probes': typography.MAX_PROBES, 'typography_margin_min_max': [2, 8],
            'typography_shared_raster_pixels': 64_000_000,
            'restoration_local_pixels': restoration.MAX_LOCAL_PIXELS, 'restoration_local_side': restoration.MAX_LOCAL_SIDE,
            'restoration_samples': restoration.MAX_SAMPLES, 'restoration_methods_per_unit': 2,
            'artifact_count': MAX_ARTIFACTS, 'artifact_bytes': 128*1024*1024, 'report_bytes': 4*1024*1024,
            'manifest_bytes': 8*1024*1024},
        'engine': {'pillow': n.pillow_version, 'opencv': cv2.__version__, 'raqm': n.features.version_feature('raqm'),
                   'harfbuzz': n.features.version_feature('harfbuzz'), 'freetype': n.features.version_module('freetype2')}}
    plans, tiles, content_masks = {}, {}, {}
    if removal_policy!='legacy':
        report.update(removal_policy=removal_policy,reconstruction_policy=reconstruction_policy,reconstruction_config=reconstruction_spec)
    layout_retry_plans = {}
    if confidence_settings is not None:
        report['replacement_policy'] = confidence_settings
    if layout_policy != 'source-local':
        report['layout_policy'] = layout_policy
    if cleanup_policy != 'native':
        report['cleanup_policy'] = cleanup_policy
    if visibility_policy != 'replacement-only':
        report['visibility_policy'] = visibility_policy
    restoration_context = None
    text_mask, text_mask_error = prepared_text_mask, None
    if text_mask is not None:
        n.require(model_spec is not None and isinstance(text_mask,segmentation.TextMask)
                  and text_mask.evidence['checkpoint']==model_spec
                  and text_mask.evidence['source_pixels_sha256']==n.digest(rgb.tobytes())
                  and text_mask.raw.shape==(rgb.height,rgb.width)
                  and text_mask.evidence['mask_sha256']==n.digest(text_mask.raw.tobytes()),
                  'prepared OCR text mask differs from the rendering source or checkpoint')
        report['text_mask_inference'] = text_mask.evidence
    if model_spec is not None:
        report['policy'] = 'automatic_flat_classical_native_semantic_v1'
        report['text_mask_model'] = model_spec

    def semantic_plan(binding, row, original_error):
        nonlocal text_mask, text_mask_error
        if model_spec is None:
            raise original_error
        if text_mask is None and text_mask_error is None:
            try:
                text_mask = segmentation.TextMask(rgb, model_spec)
                report['text_mask_inference'] = text_mask.evidence
            except (ValueError, OSError) as error:
                text_mask_error = str(error)
        if text_mask_error is not None:
            row['background_attempts'].append({'method': 'semantic_mask', 'status': 'skipped',
                'reason_code': 'text_mask_backend', 'reason': text_mask_error})
            raise original_error
        for surface in (('flat',) if background_method == 'flat' else ('flat', 'periodic')):
            method = 'semantic_flat_fill' if surface == 'flat' else 'phase_locked_tile'
            try:
                candidate = segmentation.derive(rgb, raw, binding['member_ids'], text_mask.raw, surface=surface)
                row['background_attempts'].append({'method': method, 'status': 'planned'})
                row['cleanup_method'] = 'flat_fill_semantic' if surface == 'flat' else 'phase_locked_tile'
                row['text_mask_method'] = segmentation.POLICY
                return candidate
            except (ValueError, AutomaticSkip) as error:
                row['background_attempts'].append({'method': method, 'status': 'skipped',
                    'reason_code': getattr(error, 'code', 'validation'), 'reason': str(error),
                    'evidence': getattr(error, 'evidence', {})})
        raise original_error
    work, eligible, coverage_eligible, attempts = 0, 0, 0, 0
    context, context_error, context_initialized = None, None, False

    def initialize_context():
        nonlocal context, context_error, context_initialized, work
        if not context_initialized:
            context_initialized = True
            try:
                context = OriginalComponents(rgb)
                work += context.bind_geometry(raw, remaining_work=MAX_WORK-work)
            except n.ResourceLimit as error:
                context, context_error = None, str(error)

    pairs = list(enumerate(zip(preview_doc['units'], units)))
    # Keep report/artifact indices in saved OCR order, but reserve budgets for
    # existing strict translations first, independent of uncertain page position.
    report['units'] = [None]*len(pairs)
    pairs.sort(key=lambda pair: pair[1][0].get('input_kind') == tc.KIND)
    for index, (record, unit) in pairs:
        saved_result = record.get('result')
        review_needed = saved_result.get('review_needed') if type(saved_result) is dict else None
        row = {'unit_id': unit.unit_id, 'source_status': record['status'], 'source_record_sha256': n.revision(record),
               'machine_selected': False, 'status': 'skipped', 'reason_code': 'not_strict_translated_candidate',
               'reason': 'blocked, untranslated, fragment/alternate-only or unexecuted source is not promoted',
               'source_uncertainty': {'completeness': unit.completeness,
                   'structural_reasons': list(unit.structural_reasons), 'lexical_reasons': list(unit.lexical_reasons),
                   'blocked_reasons': list(unit.blocked_reasons),
                   'translation_review_needed': review_needed if type(review_needed) is bool else None}}
        if (record['status'] == 'abstained' and unit.selected.origin=='raw_main_reading'
                and unit.selected.text.isascii() and any(c.isalnum() for c in unit.selected.text)):
            row['original_label_retained'] = True
        if retained_label(record):
            row['original_label_retained'] = True
        if retained_artwork(record):
            row['original_artwork_retained'] = True
        if type(record.get('luna_page_order')) is int:
            row['reading_order'] = record['luna_page_order']
        row['input_kind'] = record.get('input_kind', 'strict_utterance')
        report['units'][index] = row
        is_coverage = row['input_kind'] == tc.KIND
        if record['status'] != 'preview' or row['input_kind'] not in ('strict_utterance', tc.KIND):
            continue
        if is_coverage and render_policy != 'best-effort' and visibility_policy != 'always-show':
            row.update(reason_code='coverage_requires_best_effort', reason='uncertain text-region replacements require best-effort rendering')
            continue
        try:
            if is_coverage:
                n.require(preview_doc['schema_version'] == 4 and
                    preview_doc['settings'].get('translation_policy') == 'coverage-first', 'coverage translation policy/schema required')
                binding = tc.bound_result(record, unit, request=contextual.get(unit.unit_id))
                coverage_eligible += 1
                row['coverage'] = record['coverage']
            else:
                binding = n.strict_result_binding(record, unit, request=contextual.get(unit.unit_id))
                eligible += 1
            row.update(machine_selected=True, binding=binding)
            from ..ocr import sound_readings
            from ..translation.rendition import impact_sound
            source_lines = [d for d in raw if d['source_id'] in unit.member_ids]
            source_word = next((v.text for v in unit.readings if v.revision==binding.get('reread_selected_revision')),
                               unit.selected.text)
            if (len(source_lines)==1 and (impact_sound(source_word) or sound_readings.repeat(source_word)
                    or source_lines[0].get('ocr_metadata',{}).get('drawn_kana')
                    or source_lines[0].get('reason') in ('supplemental_rotated_sound_region','corroborated_repeated_effect')
                    or re.fullmatch(r'[ァ-ヺー]{2,6}ッ[!！]*',source_word))):
                row['lettering_role'] = 'shout'
            elif source_lines and (all(d.get('reason')=='confirmed_handwritten_line' or d.get('ocr_metadata',{}).get('handwritten_aside') for d in source_lines)
                                   or handwritten_columns(source_lines)):
                row['lettering_role'] = 'handwriting'
            elif (len(source_lines)==1 and re.fullmatch(r'(?:おしまい|つづく|おわり)[。！!…]*',source_word)
                    and min(point[1] for point in source_lines[0]['polygon']) > .85*rgb.height):
                row['lettering_role'] = 'handwriting'
            if visibility_policy == 'always-show':
                if len(source_lines)==1 and rc.pencil_hatching(rgb,source_lines[0]):
                    raise AutomaticSkip('uncorroborated_source','The detected strokes continue as hatching outside the text region.')
                selected = binding.get('reread_selected_revision')
                reviewed = next((reading.text for reading in unit.readings if reading.revision == selected), None)
                decision = record.get('luna_decision') or {}
                if preview_doc.get('settings', {}).get('profile') in ('page','panel','atlas','reviewed') and decision.get('category') == 'translation':
                    reviewed = decision.get('source') or reviewed
                if not reviewed and all(d.get('reason')=='corroborated_repeated_effect' or any(k in d.get('ocr_metadata',{}) for k in ('colored_word','painted_effect','outlined_display'))
                                        for d in raw if d['source_id'] in unit.member_ids):
                    if confirms_raw(crop_records.get(unit.unit_id),unit,proposals[unit.unit_id],rgb,raw):
                        reviewed = unit.selected.text
                preserve = rc.original_lettering_reason(unit, binding['text'], raw,
                    proposals[unit.unit_id]['crop_bounds'], rgb.size, reviewed_source=reviewed)
                if preserve:
                    raise AutomaticSkip(*preserve)
            if (is_coverage and removal_policy == 'aggressive' and not binding.get('reread_selected_revision')
                    and not (preview_doc.get('settings', {}).get('profile') in ('page', 'panel', 'atlas', 'reviewed')
                        and (record.get('luna_decision') or {}).get('category') == 'translation'
                        and record['luna_decision'].get('source', '').strip())
                    and unit.selected.origin == 'raw_main_reading'):
                assessment = rc.assess_source(unit, binding['text'], rc.DEFAULT_MINIMUM)
                if (assessment['reason_code'] in ('replacement_low_ocr_confidence', 'replacement_confidence_unavailable')
                        and not confirms_raw(crop_records.get(unit.unit_id), unit, proposals[unit.unit_id], rgb, raw)):
                    row['replacement_confidence'] = assessment
                    raise AutomaticSkip('uncorroborated_source', 'Weak OCR without a corroborated reading. Preserve the artwork.')
            if len(unit.member_ids)==1 and unit.member_ids[0] in outlines:
                angle = float(outlines[unit.member_ids[0]].get('angle',0))
                if np.isfinite(angle) and abs(angle)<=15:
                    row['effect_style'] = {'rotation':angle,'fill':[255]*3,'outline_fill':[0]*3,'outline':2}
            if visibility_policy == 'always-show' and binding['text'] == unit.selected.text:
                raise AutomaticSkip('replacement_unchanged_text', 'Target exactly matches source; intentionally keep original text.')
            if ((confidence_settings is not None or visibility_policy == 'always-show')
                    and not binding.get('reread_selected_revision')
                    and rc.retains_printed_words(unit, binding['text'])):
                raise AutomaticSkip('replacement_unchanged_text', 'Target repeats the printed Latin words with editorial punctuation; preserve original lettering.')
            if max_units is not None and eligible+coverage_eligible > max_units:
                raise AutomaticSkip('evaluation_cap', 'explicit evaluation cap reached; candidate preserved, not rendered')
            if confidence_settings is not None:
                confidence_unit = tc.derive(unit, binding['reread_selected_revision']) if binding.get('reread_selected_revision') else unit
                assessment = rc.assess_source(confidence_unit, binding['text'], confidence_settings['minimum_main_line_score'])
                row['replacement_confidence'] = assessment
                if not assessment['allowed']:
                    raise AutomaticSkip(assessment['reason_code'], assessment['reason'])
            font_data, coverage = faces.resolve(row)
            n.validate_text(binding['text'], coverage)
            if (reconstruction_policy=='auto' and proposals[unit.unit_id].get('grouping_method')=='separate_printed_table_rows'
                    and len(binding['member_ids'])==1 and (model_spec is not None or binding['member_ids'][0] not in title_bars)):
                raise AutomaticSkip('ruled_table_cell','Use the printed cell boundaries for cleanup and lettering.')
            if is_coverage:
                # Delay not just geometry but also its slot/raster budget until
                # ALL strict placements and their fallback attempts finish.
                row['coverage_pending'] = True
                raise AutomaticSkip('coverage_uncertain_geometry', 'source-bound uncertain region; try local routes after strict placements')
            unit_work = pixels*len(binding['member_ids'])
            if attempts >= MAX_SLOTS or (attempts+1)*pixels > MAX_RETAINED_PIXELS or work+unit_work > MAX_WORK:
                raise AutomaticSkip('resource_budget', 'automatic per-job mask/raster-work budget reached')
            attempts += 1
            work += unit_work
            row['attempted'] = True
            row['background_attempts'] = []
            initialize_context()
            try:
                if context_error is not None:
                    raise n.ResourceLimit(context_error)
                plan = derive_plan(rgb, raw, binding['member_ids'], context=context)
                row['background_attempts'].append({'method': 'flat_fill', 'status': 'planned'})
                row['cleanup_method'] = 'flat_fill'
            except (AutomaticSkip, n.ResourceLimit) as flat_error:
                row['background_attempts'].append({'method': 'flat_fill', 'status': 'skipped',
                    'reason_code': getattr(flat_error, 'code', 'resource_budget'), 'reason': str(flat_error)})
                if background_method == 'flat':
                    plan = semantic_plan(binding, row, flat_error)
                else:
                    try:
                        if restoration_context is None:
                            extra_work = pixels*len(raw) if context is None else 0
                            n.budget_check(work+extra_work <= MAX_WORK, 'restoration geometry raster-work budget')
                            restoration_context = restoration.OriginalGeometry(rgb, raw, geometry=context)
                            work += extra_work
                        plan = restoration.derive_plan(rgb, raw, binding['member_ids'], context=restoration_context)
                        row['background_attempts'].append({'method': 'classical_inpaint', 'status': 'planned'})
                        row['cleanup_method'] = 'inpaint_'+plan.evidence['method']
                    except (restoration.RestorationSkip, AutomaticSkip, n.ResourceLimit) as restore_error:
                        row['background_attempts'].append({'method': 'classical_inpaint', 'status': 'skipped',
                            'reason_code': getattr(restore_error, 'code', 'resource_budget'), 'reason': str(restore_error),
                            'evidence': getattr(restore_error, 'evidence', {})})
                        plan = semantic_plan(binding, row, flat_error)
            if reconstruction_policy == 'auto' and proposals[unit.unit_id].get('grouping_method') == 'stacked_vertical_subject':
                from .layout_enclosures import stacked_paper
                plan = stacked_paper(rgb,raw,binding['member_ids'],plan)
            plans[index] = PackedPlan(plan)
            row['mask_evidence'] = plan.evidence
            check(not (plan.erase & plan.obstacles).any(), 'erasure_obstacle', 'automatic erasure intersects original obstacles')
            unit_style = style
            if plan.restored_pixels is not None or row.get('text_mask_method'):
                luminance = float(np.dot(plan.evidence['background_rgb_at_center'], [.2126, .7152, .0722]))
                fill = [255, 255, 255] if luminance < 128 else [0, 0, 0]
                outline = [255-value for value in fill]
                unit_style = {**style, 'fill': fill, 'outline': 1, 'outline_fill': outline}
                row['text_contrast'] = {'fill': fill, 'outline_fill': outline, 'background_luminance_estimate': luminance}
            layer, provenance, content = typography.layout_dialogue(binding['text'], font_data, coverage,
                Image.fromarray(plan.layout.astype(np.uint8)*255), {**unit_style,**row.get('effect_style',{})})
            if confidence_settings is not None:
                reason = rc.layout_rejection(provenance, min_size)
                check(reason is None, 'replacement_layout_confidence', reason)
            content_masks[index] = content.convert('1')
            provenance['typography']['content_mask_artifact'] = f'unit-{index:03d}-content-mask.png'
            box = layer.getchannel('A').getbbox()
            check(box is not None, 'empty_alpha', 'automatic text has no raster alpha')
            alpha = np.asarray(layer.getchannel('A')) != 0
            row['outside_content_alpha_pixels'] = int(np.count_nonzero(alpha & (np.asarray(content) == 0)))
            check(row['outside_content_alpha_pixels'] == 0, 'typography_containment',
                  'automatic alpha escapes comfortable text margins')
            check(not (alpha & (~plan.layout | plan.obstacles)).any(), 'alpha_containment', 'automatic alpha escapes local layout/obstacle checks')
            plate_pixels = np.asarray(rgb).copy()
            apply_cleanup(plate_pixels, plan)
            plate = Image.fromarray(plate_pixels, 'RGB')
            composed = Image.alpha_composite(plate.convert('RGBA'), layer).convert('RGB')
            row['candidate_pixel_audit'] = n.pixel_audit(rgb, plate, composed, plan.erase, alpha, plan.obstacles)
            try:
                n.verify_pixel_audit(row['candidate_pixel_audit'])
            except ValueError as error:
                raise AutomaticSkip('runtime_locality', str(error)) from error
            row['candidate_audit_scope'] = 'isolated proposed operation before symmetric collision rejection; final global audit/masks describe committed pixels'
            row.update(status='ready', reason_code=None, reason=None, layout=provenance)
            tiles[index] = (layer.crop(box), box[:2])
        except AutomaticSkip as error:
            row.update(reason_code=error.code, reason=str(error), skip_evidence=error.evidence)
        except n.ResourceLimit as error:
            row.update(reason_code='resource_budget', reason=str(error))
        except (ValueError, OSError, KeyError, TypeError) as error:
            row.update(reason_code='source_or_layout_validation', reason=str(error))
    # Symmetric rejection, never reading-order priority or geometric ownership
    # splitting, if automatically derived layout/erasure regions collide.
    ready = [i for i, row in enumerate(report['units']) if row['status'] == 'ready']
    collisions = set()
    for position, i in enumerate(ready):
        for j in ready[position+1:]:
            if ((plans[i].layout | plans[i].erase) & (plans[j].layout | plans[j].erase)).any():
                collisions.update((i, j))
    for i in collisions:
        report['units'][i].update(status='skipped', reason_code='region_collision', reason='automatic layout/erasure overlaps another ready unit; both remain original')
    if render_policy == 'best-effort':
        from . import best_effort
        report['conservative_policy'] = report['policy']
        report['policy'] = 'automatic_coverage_first_v1'
        reserved = np.zeros((rgb.height, rgb.width), bool)
        for i in ready:
            if i not in collisions:
                reserved |= (occupied_reservation(plans[i], tiles[i]) if confidence_settings is not None
                             else plans[i].layout | plans[i].erase)
        if confidence_settings is not None:
            report['layout_reservation_policy'] = 'committed_alpha_and_native_erase_plus_2px_not_unused_layout'
        for i, _ in pairs:
            row = report['units'][i]
            if row['status'] != 'skipped' or 'binding' not in row:
                continue
            if row.pop('coverage_pending', False):
                unit_work = pixels*len(row['binding']['member_ids'])
                if attempts >= MAX_SLOTS or (attempts+1)*pixels > MAX_RETAINED_PIXELS or work+unit_work > MAX_WORK:
                    row.update(reason_code='resource_budget', reason='automatic per-job mask/raster-work budget reached after strict placements')
                    continue
                attempts += 1
                work += unit_work
                row.update(attempted=True, background_attempts=[])
                initialize_context()
            if not row.get('attempted'):
                continue
            # These are runtime integrity failures, not artistic uncertainty.
            if row['reason_code'] in ('typography_containment', 'alpha_containment', 'erasure_obstacle', 'runtime_locality'):
                continue
            original_failure = {key: row.get(key) for key in ('reason_code', 'reason', 'cleanup_method')}
            row['best_effort_attempts'] = []
            existing = plans.get(i)
            fallback_work = n.LayoutWork()
            routes = ('source_local', 'native_hint', 'footprint') if row['input_kind'] == tc.KIND else ('reuse', 'native_hint', 'footprint')
            if row['input_kind'] == tc.KIND and model_spec is not None:
                # Explicit optional evidence must reach coverage regions too.
                # Preserve native attempts/strict reservations; use the same
                # single page inference and unchanged association/repair guards.
                routes = (*routes[:-1], 'semantic', routes[-1])
            if confidence_settings is not None or visibility_policy == 'always-show' or removal_policy != 'legacy':
                routes = tuple(route for route in routes if route != 'footprint')
            for route in routes:
                reuse = route == 'reuse'
                if reuse:
                    if existing is None or row['reason_code'] == 'region_collision':
                        continue
                    occupied = existing.erase if confidence_settings is not None else existing.layout | existing.erase
                    if (occupied & reserved).any():
                        continue
                try:
                    font_data, coverage = faces.resolve(row)
                    if route == 'source_local':
                        if context_error is not None:
                            raise n.ResourceLimit(context_error)
                        plan = derive_plan(rgb, raw, row['binding']['member_ids'], context=context)
                        method = 'flat_fill_uncertain_region'
                        if confidence_settings is not None:
                            plan = reserve_layout(plan, reserved)
                        check(not ((plan.layout | plan.erase) & reserved).any(), 'region_collision', 'uncertain region touches reserved strict output')
                    elif reuse:
                        plan = reserve_layout(existing, reserved) if confidence_settings is not None else existing
                        method = 'reuse_cleanup_relaxed_typesetting'
                    elif route == 'native_hint':
                        extra_work = pixels*len(row['binding']['member_ids'])
                        n.budget_check(work+extra_work <= MAX_WORK, 'best-effort native raster budget')
                        work += extra_work
                        plan = None
                        if (reconstruction_policy == 'auto' and
                                'punctuated_horizontal_heading' in row['source_uncertainty']['structural_reasons']):
                            from .banner import flat_plan
                            plan = flat_plan(rgb, raw, row['binding']['member_ids'])
                        if plan is None:
                            plan = best_effort.native_plan(rgb, raw, row['binding']['member_ids'],
                                reserved=None if confidence_settings is not None else reserved,
                                allow_tight_source_seed=confidence_settings is not None,
                                recover_sign=reconstruction_policy == 'auto')
                        if confidence_settings is not None:
                            plan = reserve_layout(plan, reserved)
                        method = plan.evidence['cleanup_method']
                    elif route == 'semantic':
                        extra_work = pixels*len(row['binding']['member_ids'])
                        n.budget_check(work+extra_work <= MAX_WORK, 'coverage semantic raster budget')
                        work += extra_work
                        plan = semantic_plan(row['binding'], row,
                            AutomaticSkip('semantic_cleanup_unavailable', 'explicit semantic cleanup supplied no supported plan'))
                        if confidence_settings is not None:
                            plan = reserve_layout(plan, reserved)
                        check(not ((plan.layout | plan.erase) & reserved).any(), 'region_collision',
                              'semantic coverage plan touches prior committed output')
                        method = row['cleanup_method']
                    else:
                        # Geometry/source checks are reused, not weakened. The
                        # artistic interpretation of the OCR footprint is weaker.
                        extra_work = pixels*len(row['binding']['member_ids'])
                        n.budget_check(work+extra_work <= MAX_WORK, 'best-effort aggregate raster budget')
                        work += extra_work
                        if restoration_context is None:
                            extra_work = pixels*len(raw) if context is None else 0
                            n.budget_check(work+extra_work <= MAX_WORK, 'best-effort geometry raster budget')
                            restoration_context = restoration.OriginalGeometry(rgb, raw, geometry=context)
                            work += extra_work
                        plan = best_effort.derive_plan(rgb, raw, row['binding']['member_ids'],
                            context=restoration_context, reserved=reserved, background_method=background_method)
                        method = plan.evidence['cleanup_method']
                    check(not (plan.erase & plan.obstacles).any(), 'erasure_obstacle', 'best-effort erasure touches protected output')
                    if reconstruction_policy == 'auto' and proposals[row['unit_id']].get('grouping_method') == 'stacked_vertical_subject':
                        from .layout_enclosures import stacked_paper
                        plan = stacked_paper(rgb,raw,row['binding']['member_ids'],plan,reserved)
                    background = plan.evidence.get('background_rgb_at_center', plan.background)
                    n.require(background is not None, 'best-effort contrast estimate unavailable')
                    light = float(np.dot(background, [.2126, .7152, .0722])) < 128
                    fill = [255, 255, 255] if light else [0, 0, 0]
                    fallback_style = {**style, 'fill': fill, 'outline_fill': [255-v for v in fill], 'outline': 1, **row.get('effect_style',{})}
                    if layout_policy == 'enclosed-retry' and plan.restored_pixels is None and plan.background is not None:
                        layout_retry_plans[i] = PackedPlan(plan)
                    try:
                        layer, provenance, content = best_effort.fit(row['binding']['text'], font_data, coverage,
                            Image.fromarray(plan.layout.astype(np.uint8)*255), fallback_style, work=fallback_work)
                    except ValueError as error:
                        if route != 'footprint' or isinstance(error, n.ResourceLimit) or not str(error).startswith('overflow:'):
                            raise
                        plan = best_effort.overlay_plan(plan)
                        row['best_effort_attempts'].append({'method': 'source_local_art_overlay', 'status': 'attempted'})
                        layer, provenance, content = best_effort.fit(row['binding']['text'], font_data, coverage,
                            Image.fromarray(plan.layout.astype(np.uint8)*255), fallback_style, work=fallback_work)
                    if confidence_settings is not None or visibility_policy == 'always-show' or removal_policy != 'legacy':
                        reason = rc.layout_rejection(provenance, min_size)
                        check(reason is None, 'replacement_layout_confidence', reason)
                    plate_pixels = np.asarray(rgb).copy()
                    apply_cleanup(plate_pixels, plan)
                    if (reconstruction_policy=='auto' and not light and not row.get('effect_style')
                            and len(row['binding']['member_ids'])>=2 and len(row['binding']['text'].split())>=2):
                        layer,provenance,content = best_effort.inset_retained_ink(layer,provenance,content,
                            plate_pixels,font_data,coverage,fallback_style,fallback_work)
                    box = layer.getchannel('A').getbbox()
                    check(box is not None, 'empty_alpha', 'best-effort text has no raster alpha')
                    alpha = np.asarray(layer.getchannel('A')) != 0
                    check(not (alpha & (~plan.layout | plan.obstacles | reserved)).any(),
                          'alpha_containment', 'best-effort alpha leaves declared local overlay support')
                    plate = Image.fromarray(plate_pixels)
                    composed = Image.alpha_composite(plate.convert('RGBA'), layer).convert('RGB')
                    audit = n.pixel_audit(rgb, plate, composed, plan.erase, alpha, plan.obstacles)
                    try:
                        n.verify_pixel_audit(audit)
                    except ValueError as error:
                        raise AutomaticSkip('runtime_locality', str(error)) from error
                    plans[i], content_masks[i], tiles[i] = PackedPlan(plan), content.convert('1'), (layer.crop(box), box[:2])
                    if not reuse and route != 'semantic':
                        row.pop('text_mask_method', None)
                    provenance['typography']['content_mask_artifact'] = f'unit-{i:03d}-content-mask.png'
                    row['best_effort_attempts'].append({'method': method, 'status': 'planned'})
                    row.update(status='ready', reason_code=None, reason=None, render_quality='best_effort',
                        cleanup_method=row.get('cleanup_method') if reuse else method,
                        text_contrast={'fill': fill, 'outline_fill': [255-v for v in fill],
                            'background_luminance_estimate': float(np.dot(background, [.2126, .7152, .0722]))},
                        layout=provenance, mask_evidence=plan.evidence, candidate_pixel_audit=audit,
                        candidate_audit_scope='isolated best-effort proposal; final global audit describes committed pixels',
                        outside_content_alpha_pixels=0,
                        best_effort={'original_failure': original_failure, 'reused_cleanup': reuse,
                            'warnings': (['Typography relaxed; exact text preserved.'] if reuse else
                                plan.evidence.get('warnings', ['OCR region ownership/order is uncertain; full utterance not certified.']))
                                +(['Font size is below the requested minimum.'] if provenance['typography']['below_requested_minimum'] else [])})
                    if i not in ready:
                        ready.append(i)
                    collisions.discard(i)
                    reserved |= (occupied_reservation(plan, tiles[i]) if confidence_settings is not None
                                 else plan.layout | plan.erase)
                    break
                except (ValueError, OSError, KeyError, TypeError) as error:
                    row['best_effort_attempts'].append({'method': {'source_local': 'flat_fill_uncertain_region', 'reuse': 'reuse_cleanup', 'native_hint': 'native_contrast_hint',
                        'semantic': 'explicit_semantic_mask',
                        'footprint': 'ocr_footprint_replacement'}[route],
                        'status': 'failed', 'reason_code': getattr(error, 'code', 'validation'), 'reason': str(error)})
                    if isinstance(error, n.ResourceLimit):
                        row.update(reason_code='resource_budget', reason=str(error))
                    if isinstance(error, AutomaticSkip) and error.code == 'replacement_layout_confidence':
                        row.update(reason_code=error.code, reason=str(error))
                    if isinstance(error, AutomaticSkip) and error.code in ('runtime_locality', 'alpha_containment', 'erasure_obstacle'):
                        row.update(reason_code=error.code, reason=str(error))
                        break
                    if isinstance(error, n.ResourceLimit) and (work+pixels*len(row['binding']['member_ids']) > MAX_WORK
                            or fallback_work.rasterized_pixels >= 64_000_000
                            or fallback_work.search['measures'] >= 65536
                             or fallback_work.search['transitions'] >= 1_000_000):
                        break
            if confidence_settings is not None and row['status'] == 'skipped':
                row['best_effort_attempts'].append({'method': 'ocr_footprint_replacement', 'status': 'not_attempted',
                    'reason_code': 'replacement_cleanup_confidence',
                    'reason': 'Confidence-gated replacement requires native stroke support, not coarse OCR-footprint erasure.'})
                if row['reason_code'] not in ('resource_budget', 'runtime_locality', 'alpha_containment',
                                             'erasure_obstacle', 'replacement_layout_confidence'):
                    row.update(reason_code='replacement_cleanup_confidence',
                        reason='Native-mask cleanup/layout routes failed; coarse footprint replacement is disabled. Translation retained.')
    if layout_policy == 'enclosed-retry':
        from . import layout_enclosures
        enclosure_context = None
        enclosure_error = None
        retry_indices = [i for i, _ in pairs if report['units'][i]['status'] == 'skipped'
            and report['units'][i]['reason_code'] == 'replacement_layout_confidence']
        for i in retry_indices:
            row = report['units'][i]
            previous = layout_retry_plans.get(i, plans.get(i))
            if previous is None or previous.restored_pixels is not None or previous.background is None:
                continue
            attempt = {'policy': layout_enclosures.POLICY, 'status': 'rejected',
                       'original_failure': row['reason_code']}
            row['enclosure_layout_attempt'] = attempt
            try:
                font_data, coverage = faces.resolve(row)
                if enclosure_context is None and enclosure_error is None:
                    try:
                        n.budget_check(work+3*pixels <= MAX_WORK, 'enclosure original geometry work budget')
                        n.require(context is not None, 'original geometry unavailable for enclosure retry')
                        enclosure_context = layout_enclosures.Enclosures(rgb, raw, context)
                        work += 3*pixels
                    except (ValueError, OSError) as error:
                        enclosure_error = str(error)
                n.require(enclosure_error is None, enclosure_error or 'enclosure geometry unavailable')
                unit_work = pixels*len(row['binding']['member_ids'])
                n.budget_check(work+unit_work <= MAX_WORK, 'enclosure member raster work budget')
                work += unit_work
                plan = enclosure_context.expand(rgb, raw, row['binding']['member_ids'], previous, reserved)
                mask = Image.fromarray(plan.layout.astype(np.uint8)*255)
                attempt['evidence'] = plan.evidence['enclosure_layout']
                try:
                    layer, provenance, content = typography.layout_dialogue(row['binding']['text'], font_data, coverage, mask, style)
                    attempt['typesetting'] = 'comfortable_margins'
                except n.ResourceLimit:
                    raise
                except ValueError as error:
                    if not str(error).startswith('overflow:'):
                        raise
                    # The measured enclosure already has 5px boundary clearance.
                    # Relax extra comfort/leading, never the requested font floor.
                    layer, provenance = n.layout(row['binding']['text'], font_data, coverage, mask,
                        {**style, 'leading': 0})
                    content = mask
                    provenance['typography'] = {'policy': layout_enclosures.POLICY,
                        'margin_pixels': 0, 'boundary_clearance_pixels': 5,
                        'preferred_size': max(min_size, round(max_size*.9)),
                        'below_requested_minimum': False, 'requested_minimum': min_size}
                    attempt['typesetting'] = 'measured_boundary_margin_zero_extra_leading'
                check(rc.layout_rejection(provenance, min_size) is None, 'replacement_layout_confidence',
                      'enclosure lettering must meet the requested minimum')
                alpha = np.asarray(layer.getchannel('A')) != 0
                check(alpha.any() and not (alpha & (~plan.layout | plan.obstacles | reserved)).any(),
                      'alpha_containment', 'enclosure text leaves supported original whitespace')
                plate_pixels = np.asarray(rgb).copy()
                apply_cleanup(plate_pixels, plan)
                plate = Image.fromarray(plate_pixels)
                composite = Image.alpha_composite(plate.convert('RGBA'), layer).convert('RGB')
                audit = n.pixel_audit(rgb, plate, composite, plan.erase, alpha, plan.obstacles)
                n.verify_pixel_audit(audit)
                box = layer.getchannel('A').getbbox()
                plans[i], tiles[i], content_masks[i] = PackedPlan(plan), (layer.crop(box), box[:2]), content.convert('1')
                provenance['typography']['content_mask_artifact'] = f'unit-{i:03d}-content-mask.png'
                row.update(status='ready', reason_code=None, reason=None, render_quality='best_effort',
                    layout=provenance, mask_evidence=plan.evidence, candidate_pixel_audit=audit,
                    candidate_audit_scope='isolated enclosure layout retry; unchanged validated native cleanup',
                    outside_content_alpha_pixels=0,
                    best_effort={'original_failure': attempt['original_failure'], 'reused_cleanup': True,
                                 'warnings': ['Original closed flat-space hypothesis; balloon ownership and translation unverified.']})
                row.setdefault('cleanup_method', plan.evidence.get('cleanup_method', 'flat_fill_uncertain_region'))
                attempt['status'] = 'planned'
                ready.append(i)
                collisions.discard(i)
                reserved |= occupied_reservation(plan, tiles[i])
            except (ValueError, OSError, KeyError, TypeError) as error:
                attempt.update(reason_code=getattr(error, 'code', 'validation'), reason=str(error),
                               rejection_evidence=getattr(error, 'evidence', {}))
    demand = preview_doc['ocr'].get('source_localization', {}).get('request_plan')
    if plan_localization or demand is not None:
        from . import localization_requests
        requested, planning_seconds = localization_requests.plan(preview_doc, rgb, raw, units, report,
            policy=demand['policy'] if demand is not None else localization_requests.POLICY)
        if demand is not None:
            localization_requests.validate(demand, requested)
        report['localization_request_plan'] = requested
        report['localization_planning_seconds'] = planning_seconds
    if cleanup_policy == 'localized':
        from . import source_extents
        extent_checks=[]
        for unit,row in zip(units,report['units']):
            if unit.unit_id not in extent_records:continue
            check_row={'unit_id':unit.unit_id,'status':'not_used'};extent_checks.append(check_row)
            if removal_policy!='aggressive' or row['status']!='skipped' or row.get('reason_code') in aggressive.HARD_REASONS:
                check_row['reason']='requires_pending_aggressive_cleanup';continue
            try:
                extent=source_extents.derive(rgb,raw,unit,extent_records[unit.unit_id])
                extents[unit.unit_id]=extent;check_row.update(status='accepted',evidence=extent)
            except (ValueError,KeyError,TypeError,cv2.error) as error:
                check_row.update(status='rejected',reason=str(error))
        report['source_extent_summary']={'policy':source_extents.POLICY,'records':extent_checks,
            'accepted':len(extents),'raw_geometry_unchanged':True,'translations_reselected':0}
        localized_work, localized_retries = 0, 0
        retry_indices = [i for i, _ in pairs if report['units'][i]['status'] == 'skipped'
            and report['units'][i]['reason_code'] == 'replacement_cleanup_confidence'
            and report['units'][i].get('attempted') and units[i].unit_id in localized]
        directions = {proposal['proposal_id']:proposal['direction'] for proposal in preview_doc['ocr']['batches'][0]['proposals']}
        retry_indices.sort(key=lambda i:(directions[units[i].unit_id]!='vertical',len(units[i].order)<2,
                                         -len(units[i].selected.text),units[i].unit_id))
        if demand is not None:
            by_id = {units[i].unit_id:i for i in retry_indices}
            retry_indices = [by_id[uid] for uid in demand['requests'] if uid in by_id]
        for i in retry_indices:
            row = report['units'][i]
            attempt = {'policy': localized_strokes.POLICY, 'status': 'rejected',
                'original_failure': row['reason_code']}
            row['localized_cleanup_attempt'] = attempt
            try:
                font_data, coverage = faces.resolve(row)
                extra_work = pixels*(len(raw)+128)
                n.budget_check(localized_retries < localized_strokes.MAX_RETRIES
                    and localized_work+extra_work <= localized_strokes.MAX_RETRY_WORK,
                    'localized cleanup retry/raster budget')
                localized_retries += 1; localized_work += extra_work
                plan = localized_strokes.derive(rgb, raw, row['binding']['member_ids'],
                    localized[units[i].unit_id], reserved=reserved)
                attempt['evidence'] = plan.evidence
                mask = Image.fromarray(plan.layout.astype(np.uint8)*255)
                layer, provenance, content = typography.layout_dialogue(row['binding']['text'], font_data, coverage, mask, style)
                check(rc.layout_rejection(provenance, min_size) is None, 'replacement_layout_confidence',
                    'localized lettering must meet the requested minimum')
                alpha = np.asarray(layer.getchannel('A')) != 0
                check(alpha.any() and not (alpha & (~plan.layout | plan.obstacles | reserved)).any(),
                    'alpha_containment', 'localized lettering leaves supported original space')
                plate_pixels = np.asarray(rgb).copy(); apply_cleanup(plate_pixels, plan)
                plate = Image.fromarray(plate_pixels)
                composite = Image.alpha_composite(plate.convert('RGBA'), layer).convert('RGB')
                audit = n.pixel_audit(rgb, plate, composite, plan.erase, alpha, plan.obstacles)
                n.verify_pixel_audit(audit)
                box = layer.getchannel('A').getbbox()
                plans[i], tiles[i], content_masks[i] = PackedPlan(plan), (layer.crop(box), box[:2]), content.convert('1')
                provenance['typography']['content_mask_artifact'] = f'unit-{i:03d}-content-mask.png'
                row.update(status='ready', reason_code=None, reason=None, render_quality='best_effort',
                    cleanup_method=plan.evidence['cleanup_method'], layout=provenance, mask_evidence=plan.evidence,
                    candidate_pixel_audit=audit, candidate_audit_scope='isolated crop-corroborated whole-stroke hypothesis',
                    outside_content_alpha_pixels=0, best_effort={'original_failure': attempt['original_failure'],
                        'warnings': plan.evidence['warnings']})
                attempt['status'] = 'planned'
                ready.append(i); collisions.discard(i)
                reserved |= occupied_reservation(plan, tiles[i])
            except (ValueError, OSError, KeyError, TypeError) as error:
                attempt.update(reason_code=getattr(error, 'code', 'validation'), reason=str(error),
                    rejection_evidence=getattr(error, 'evidence', {}))
        report['localized_cleanup_summary'] = {'retries': localized_retries+len(extent_checks),
            'native_stroke_retries':localized_retries,'extent_checks':len(extent_checks),'raster_work': localized_work,
            'maximum_retries': localized_strokes.MAX_RETRIES, 'maximum_raster_work': localized_strokes.MAX_RETRY_WORK,
            'semantic_mask_inference': False, 'semantic_ownership_verified': False}
    erase_union = np.zeros((rgb.height, rgb.width), bool)
    alpha_union = np.zeros_like(erase_union)
    layout_union = np.zeros_like(erase_union)
    plate_pixels = np.asarray(rgb).copy()
    for i in ready:
        if i in collisions:
            continue
        plan = plans[i]
        apply_cleanup(plate_pixels, plan)
        erase_union |= plan.erase
        layout_union |= plan.layout
        report['units'][i]['status'] = 'automatic_composed'
    clean = Image.fromarray(plate_pixels, 'RGB')
    composite = clean.convert('RGBA')
    text = Image.new('RGBA', rgb.size)
    for i, (tile, xy) in tiles.items():
        if report['units'][i]['status'] != 'automatic_composed':
            continue
        composite.alpha_composite(tile, dest=xy)
        text.alpha_composite(tile, dest=xy)
        x, y = xy
        alpha_union[y:y+tile.height, x:x+tile.width] |= np.asarray(tile.getchannel('A')) != 0
    actual_alpha = np.asarray(text.getchannel('A')) != 0
    report['alpha_integrity'] = {'validated_pixels': int(alpha_union.sum()), 'actual_pixels': int(actual_alpha.sum()),
                                 'disagreement_pixels': int(np.count_nonzero(actual_alpha != alpha_union))}
    composite = composite.convert('RGB')
    protected = ~(erase_union | layout_union)
    report['pixel_audit'] = n.pixel_audit(rgb, clean, composite, erase_union, alpha_union, protected)
    try:
        n.require(report['alpha_integrity']['disagreement_pixels'] == 0 and not (actual_alpha & ~layout_union).any(),
                  'combined automatic alpha differs from validated unit alpha/layout')
        n.verify_pixel_audit(report['pixel_audit'])
    except ValueError as error:
        report['rejected_pixel_audit'] = report['pixel_audit']
        report['rejected_alpha_integrity'] = report['alpha_integrity']
        for row in report['units']:
            if row['status'] == 'automatic_composed':
                row.update(status='skipped', reason_code='runtime_locality', reason=str(error))
        clean, composite, text = rgb.copy(), rgb.copy(), Image.new('RGBA', rgb.size)
        erase_union[:] = False
        alpha_union[:] = False
        layout_union[:] = False
        protected[:] = True
        report['pixel_audit'] = n.pixel_audit(rgb, clean, composite, erase_union, alpha_union, protected)
        report['alpha_integrity'] = {'validated_pixels': 0, 'actual_pixels': 0, 'disagreement_pixels': 0}
    recovery_artifacts = {}
    if removal_policy == 'aggressive':
        backend=reconstruction.LocalReconstruction(reconstruction_spec) if reconstruction_spec else None
        if model_spec is not None and text_mask is None and text_mask_error is None:
            # Coverage-only pages may never enter the strict semantic route.
            # The explicitly configured evidence is also needed by recovery.
            try:
                text_mask=segmentation.TextMask(rgb,model_spec)
                report['text_mask_inference']=text_mask.evidence
            except (ValueError,OSError) as error:
                text_mask_error=str(error)
                report['text_mask_error']=text_mask_error
        repetitions = {}
        if reconstruction_policy=='auto' and text_mask is not None:
            from . import repeated_strokes
            try:
                repetitions = repeated_strokes.samples(rgb,raw,text_mask.raw,model_spec)
            except (ValueError,OSError,cv2.error) as error:
                report['repeated_stroke_error'] = str(error)
        try:
            report['units'],clean,text,composite,erase_union,layout_union,recovery_artifacts,report['reconstruction_summary'] = aggressive.recover(
                rgb,source,raw,report['units'],clean,text,erase_union,layout_union,*faces.primary,style,
                backend=backend,font_resolver=faces.resolve,repetitions=repetitions,max_title_size=max_title_size,
                probability=text_mask.raw if text_mask is not None else None,extents=extents)
        except reconstruction.BindingError:
            raise
        except (ValueError,OSError,MemoryError,cv2.error) as error:
            report['reconstruction_summary']={'policy':aggressive.POLICY,'status':'failed_primary_preserved',
                'error':type(error).__name__+': '+str(error)[:2000],'model_config':reconstruction_spec}
        from . import replacement
        report['units'],clean,text,composite,erase_union,layout_union,final_artifacts,report['removal_summary'] = replacement.finish(
            rgb,raw,report['units'],clean,text,erase_union,layout_union,*faces.primary,style,font_resolver=faces.resolve,repetitions=repetitions,
            probability=text_mask.raw if text_mask is not None else None,source=source,backend=backend,
            max_title_size=max_title_size, budget=reconstruction.Budget(maximum_calls=4,maximum_seconds=240 if reconstruction_policy in ("auto", "sdxl") else 40),extents=extents)
        recovery_artifacts.update(final_artifacts)
        from . import outlined_sound_cleanup
        clean, text, joint_repairs = outlined_sound_cleanup.finish(rgb,clean,text,layout_union,erase_union,raw,units,
            report['units'],source,backend,recovery_artifacts,faces.resolve,style)
        if joint_repairs:
            report['outlined_sound_cleanup'] = joint_repairs
            composite = Image.alpha_composite(clean.convert('RGBA'),text).convert('RGB')
        if reconstruction_policy == 'auto':
            from . import table_cells
            if table_cells.place(rgb,clean,text,erase_union,layout_union,report['units'],proposals,raw,faces.resolve,
                                 probability=text_mask.raw if text_mask is not None else None):
                composite = Image.alpha_composite(clean.convert('RGBA'),text).convert('RGB')
        alpha_union = np.asarray(text.getchannel('A')) != 0
        protected = ~(erase_union | layout_union)
        report['pixel_audit'] = n.pixel_audit(rgb,clean,composite,erase_union,alpha_union,protected)
        report['alpha_integrity'] = {'validated_pixels':int(alpha_union.sum()),'actual_pixels':int(alpha_union.sum()),'disagreement_pixels':0}
    display_tiles = {}
    if visibility_policy == 'always-show':
        native_composite, native_text = composite, text
        try:
            composite, text, patch_layer, fallback_text, display_mask, display_tiles, display_summary = visible_translations.display(
                rgb, raw, units, report['units'], clean, native_text, native_composite, *faces.primary, style,
                font_resolver=faces.resolve,
                max_displays=max(0,min(visible_translations.MAX_ATTEMPTS,(MAX_ARTIFACTS-24-14*len(plans))//3)),native_erase=erase_union,
                protect_artwork=removal_policy=='aggressive',
                replace_all=removal_policy=='aggressive' and confidence_settings is None)
            report['display_summary'] = display_summary
            report['pixel_audit_scope'] = 'native clean/text before separately audited opaque display overlays'
        except (ValueError, OSError, KeyError, TypeError) as error:
            # Optional display failure cannot remove any successful native edit.
            composite, text = native_composite, native_text
            patch_layer, fallback_text = Image.new('RGBA',rgb.size), Image.new('RGBA',rgb.size)
            display_mask = np.zeros((rgb.height,rgb.width),bool)
            display_tiles = {}
            for row in report['units']:
                previous_display = row.get('display_status')
                row['display_status'] = (visible_translations.composed_status(row) if row['status']=='automatic_composed' else
                    'missing' if 'binding' not in row else 'unchanged' if row.get('reason_code')=='replacement_unchanged_text'
                    else 'undisplayed')
                if previous_display in ('patch','caption'):
                    row['display_fallback'].update(status='unresolved',reason='display_pass_rolled_back')
            report['display_summary'] = {**visible_translations.display_counts(report['units']),
                'policy':visible_translations.POLICY,'error':str(error),'rolled_back':True,'displayed_translations':
                sum(r['status']=='automatic_composed'for r in report['units'])}
    rendered = sum(row['status'] == 'automatic_composed' for row in report['units'])
    strict_rendered = sum(row['status'] == 'automatic_composed' and row['input_kind'] == 'strict_utterance' for row in report['units'])
    report['coverage'] = {'saved_units': len(report['units']), 'strict_eligible': eligible, 'attempted': attempts,
        'coverage_eligible': coverage_eligible, 'total_eligible': eligible+coverage_eligible,
        'strict_rendered': strict_rendered, 'coverage_rendered': rendered-strict_rendered,
        'rendered': rendered, 'skipped': len(report['units'])-rendered, 'partial_output': rendered < len(report['units']),
        'strict_coverage_fraction': strict_rendered/eligible if eligible else None,
        'total_coverage_fraction': rendered/(eligible+coverage_eligible) if eligible+coverage_eligible else None,
        'has_rendered_output': bool(rendered)}
    if visibility_policy == 'always-show':
        report['native_alpha_integrity'] = report['alpha_integrity']
        final_alpha = np.asarray(text.getchannel('A')) != 0
        expected_alpha = alpha_union | (np.asarray(fallback_text.getchannel('A')) != 0)
        report['alpha_integrity'] = {'validated_pixels':int(expected_alpha.sum()), 'actual_pixels':int(final_alpha.sum()),
                                    'disagreement_pixels':int(np.count_nonzero(final_alpha != expected_alpha))}
        report['coverage']['displayed_translations'] = report['display_summary'].get('displayed_translations',rendered)
        report['coverage']['has_rendered_output'] = bool(report['coverage']['displayed_translations'])
        if not report['display_summary'].get('rolled_back'):
            report['coverage']['partial_output'] = bool(report['display_summary']['available_not_displayed']
                                                       or report['display_summary']['missing_translations'])
        protected &= ~display_mask
    best_count = sum(row['status'] == 'automatic_composed' and row.get('render_quality') == 'best_effort'
                     for row in report['units'])
    report['quality_summary'] = {'conservative_rendered': rendered-best_count, 'best_effort_rendered': best_count,
        'has_degraded_output': bool(best_count), 'best_effort_artwork_preservation': 'not_guaranteed' if best_count else 'not_used'}
    if confidence_settings is not None:
        report['replacement_confidence'] = rc.summary(report['units'])
        if visibility_policy == 'always-show':
            report['replacement_confidence']['note'] = 'Clean-replacement gate only; separate display_summary reports patches/captions and retained source text.'
    report['retained_masks'] = {'encoding': 'bitpacked_boolean_v1', 'plans': len(plans),
        'boolean_mask_bytes': sum(mask.nbytes for plan in plans.values() for mask in plan.masks.values()),
        'equivalent_unpacked_mask_bytes': 4*pixels*len(plans),
        'content_mask_storage': 'Pillow mode 1; expanded to L for unchanged PNG artifacts'}
    report['background_summary'] = {'flat_rendered': sum(row['status'] == 'automatic_composed'
            and row.get('cleanup_method') in ('flat_fill', 'flat_fill_semantic') for row in report['units']),
        'restored_rendered': sum(row['status'] == 'automatic_composed'
            and row.get('cleanup_method', '').startswith('inpaint_') for row in report['units']),
        'best_effort_cleanup_rendered': sum(row['status'] == 'automatic_composed'
            and row.get('cleanup_method', '').startswith('best_effort_') for row in report['units']),
        'restoration_attempted': sum(any(attempt['method'] == 'classical_inpaint'
             for attempt in row.get('background_attempts', [])) for row in report['units'])}
    if model_spec is not None:
        report['background_summary'].update(semantic_mask_rendered=sum(row['status'] == 'automatic_composed'
            and bool(row.get('text_mask_method')) for row in report['units']),
            periodic_rendered=sum(row['status'] == 'automatic_composed' and row.get('cleanup_method') == 'phase_locked_tile'
                for row in report['units']))
    if cleanup_policy != 'native':
        report['background_summary']['localized_whole_stroke_rendered'] = sum(row['status'] == 'automatic_composed'
            and row.get('cleanup_method') == 'localized_whole_stroke_flat_fill' for row in report['units'])
    if removal_policy!='legacy':
        # Exact target availability survives overflow, budgets and model failures.
        report['transcript'] = [{'unit_id':row['unit_id'],'binding':row['binding'],
            'image_display':row.get('display_status',visible_translations.composed_status(row) if row['status']=='automatic_composed' else 'undisplayed')}
            for row in report['units'] if 'binding' in row
            and row.get('reason_code') not in ('uncorroborated_source', 'replacement_unchanged_text')]
        report['transcript_only'] = sum(item['image_display']=='undisplayed' for item in report['transcript'])
    report['job_revision'] = n.revision({k: v for k, v in report.items() if k not in ('job_revision', 'artifacts')})
    blobs, total, recovery_total = {}, 0, 0
    def add(name, value):
        nonlocal total, recovery_total
        if name.startswith('recovery-'):
            recovery_total += len(value)
            n.budget_check(recovery_total <= 64*1024*1024 and sum(k.startswith('recovery-') for k in blobs)<1024,
                'separate reconstruction artifact budget')
            blobs[name] = value
            return
        total += len(value)
        n.budget_check(len(value) <= 64*1024*1024 and total <= 128*1024*1024, 'automatic artifact byte budget')
        n.budget_check(sum(not k.startswith('recovery-') for k in blobs) < MAX_ARTIFACTS, 'automatic artifact count budget')
        blobs[name] = value
    def png(name, value):
        stream = BytesIO()
        value.save(stream, format='PNG')
        add(name, stream.getvalue())
    for name, value in zip(('evidence-preview.json', 'original-file.bin', 'evidence-fonts.json'), contents):
        add(name, value)
    for name, value in (('source', rgb), ('clean', clean), ('composite', composite), ('text', text)):
        png(name+'.png', value)
    for name,value in recovery_artifacts.items():
        if isinstance(value,bytes):add(name,value)
        else:png(name,value)
    if removal_policy!='legacy':
        add('transcript.json',(n.canonical(report['transcript'])+'\n').encode())
    if visibility_policy == 'always-show':
        for name, value in (('patch',patch_layer),('fallback-text',fallback_text),('native-text',native_text),('native-composite',native_composite)):
            png(name+'.png',value)
        png('fallback-mask.png',Image.fromarray(display_mask.astype(np.uint8)*255))
        for i,(patch,lettering,xy) in display_tiles.items():
            for suffix,tile in (('patch',patch),('text',lettering)):
                layer = Image.new('RGBA',rgb.size);layer.paste(tile,xy)
                png(f'unit-{i:03d}-fallback-{suffix}.png',layer)
            mask = Image.new('L',rgb.size);mask.paste(patch.getchannel('A'),xy)
            png(f'unit-{i:03d}-fallback-mask.png',mask)
    if text_mask is not None:
        png('text-probability.png', Image.fromarray(text_mask.raw))
    for name, mask in (('erase-mask', erase_union), ('alpha-mask', np.asarray(text.getchannel('A'))!=0 if visibility_policy=='always-show' else alpha_union), ('protected-mask', protected),
                       ('cleanup-diff', np.any(np.asarray(clean) != np.asarray(rgb), axis=2)),
                       ('composite-diff', np.any(np.asarray(composite) != np.asarray(rgb), axis=2))):
        png(name+'.png', Image.fromarray(mask.astype(np.uint8)*255))
    for i, plan in plans.items():
        row = report['units'][i]
        applied = plan.erase if row['status'] == 'automatic_composed' and not row.get('reconstruction',{}).get('status')=='composed_unverified' and not row.get('removal_fallback') else np.zeros_like(plan.erase)
        masks = {'interior': plan.layout, 'foreground': plan.foreground, 'applied-erase': applied,
                 'layout': plan.layout, 'obstacles': plan.obstacles}
        metadata, rasters = n.specimen_rasters(rgb, clean, composite, masks, tiles.get(i))
        row['specimen'] = metadata
        for name, raster in rasters.items():
            png(f'unit-{i:03d}-specimen-{name}.png', raster)
        for name, mask in (('proposed-erase', plan.erase), ('applied-erase', applied), ('foreground', plan.foreground), ('layout-mask', plan.layout)):
            png(f'unit-{i:03d}-{name}.png', Image.fromarray(mask.astype(np.uint8)*255))
        if i in content_masks:
            png(f'unit-{i:03d}-content-mask.png', content_masks[i].convert('L'))
        if i in tiles:
            layer = Image.new('RGBA', rgb.size)
            layer.paste(tiles[i][0], tiles[i][1])
            png(f'unit-{i:03d}-text.png', layer)
    # Specimen metadata is part of the exact deterministic output revision too.
    report['job_revision'] = n.revision({k: v for k, v in report.items() if k not in ('job_revision', 'artifacts')})
    page = _report_html(report)
    n.budget_check(len(page) <= 4*1024*1024, 'automatic report byte budget')
    add('review.html', page)
    report['artifacts'] = {name: n.digest(value) for name, value in blobs.items()}
    manifest = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    n.budget_check(len(manifest) <= 8*1024*1024, 'automatic manifest byte budget')
    add('manifest.json', manifest)
    n.require(all(n.read_explicit(path) == value for path, value in zip(paths, contents)), 'automatic source inputs changed before publication')
    n.load_fonts(fonts_doc)  # Current explicit font/license bytes must still match.
    output.mkdir(mode=0o700)
    from ..storage import atomic_bytes, sync_directory
    sync_directory(output.parent)
    # A complete manifest is the publication marker; partial runs retain only
    # complete earlier artifacts, never a half-written manifest/image.
    for name, value in blobs.items():
        atomic_bytes(output/name,value)
    return report
