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
from PIL import Image

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

    def bind_geometry(self, detections):
        revision = n.revision([(type(d['source_id']).__name__, d['source_id'], d['polygon']) for d in detections])
        if self.geometry_revision is not None:
            n.require(revision == self.geometry_revision, 'source geometry changed since raster snapshot')
            return
        n.budget_check(len(detections)*self.pixels.shape[0]*self.pixels.shape[1] <= MAX_WORK,
                       'automatic source geometry raster budget')
        counts = np.zeros(self.pixels.shape[:2], np.uint16)
        polygons = {}
        for detection in detections:
            key = (type(detection['source_id']).__name__, detection['source_id'])
            mask = np.asarray(n.polygon_mask(detection['polygon'], (counts.shape[1], counts.shape[0]), evidence=True)) != 0
            counts += mask
            polygons[key] = detection['polygon']
        self.source_counts, self.source_polygons, self.geometry_revision = counts, polygons, revision


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
    n.budget_check(len(member_ids)*rgb.width*rgb.height <= n.MAX_UNIT_PIXELS
                   and len(detections)*rgb.width*rgb.height <= MAX_WORK, 'automatic raster work budget')
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
    return n.revision({name: n.digest(n.read_explicit(Path(__file__).parent/name, 256*1024))
                       for name in ('automatic.py', 'natural.py', 'styles.py', 'preparation.py', 'review.py', 'typography.py', 'restoration.py', 'segmentation.py', 'best_effort.py', 'replacement_confidence.py', '../translation/coverage.py', '../translation/contracts.py', '../ocr/quality_reread.py', '../ocr/quality_lines.py')})


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
        rows.append('<section><h2>'+escape(row['unit_id'])+' — '+escape(row['status'])+'</h2><p>'
            +escape(row.get('reason') or 'Automatically composed candidate; unverified wording/ownership')+'</p>'+background_note+typography_note+images
            +'<details><summary>Source evidence, settings and checks</summary><pre>'
            +escape(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False))+'</pre></details></section>')
    return (head+''.join(rows)+'</html>').encode()


def _options(min_size, max_size, max_units, background_method, render_policy='best-effort',
             replacement_policy='all', min_ocr_confidence=None):
    n.integer(min_size, 8, 144)
    n.integer(max_size, min_size, 144)
    if max_units is not None:
        n.integer(max_units, 1, MAX_SLOTS)
    n.require(type(background_method) is str and background_method in ('auto', 'flat'), 'unknown background method')
    n.require(type(render_policy) is str and render_policy in ('best-effort', 'conservative'), 'unknown rendering policy')
    return rc.settings(replacement_policy, min_ocr_confidence)


def _font(fonts_doc, font_id):
    available = n.load_fonts(fonts_doc)
    if font_id is None:
        n.require(len(available) == 1, 'multiple approved faces require a global --font-id; no implicit font matching')
        font_id = next(iter(available))
    n.require(type(font_id) is str and re.fullmatch('[A-Za-z0-9_-]{1,64}', font_id), 'bounded automatic font ID required')
    n.require(font_id in available and not isinstance(available[font_id], n.ResourceLimit), 'unavailable explicit automatic font')
    return font_id, available[font_id]


def preflight_automatic(*, image, fonts, font_id=None, min_size=16, max_size=28, max_units=None, background_method='auto', text_mask_model=None, render_policy='best-effort', replacement_policy='all', min_ocr_confidence=None):
    """Check rendering prerequisites before upstream inference, without a preview.

    This is not erase permission: run_automatic still validates the complete saved
    preview, source bindings, geometry, typography and actual output pixels.
    """
    confidence_settings = _options(min_size, max_size, max_units, background_method, render_policy,
                                   replacement_policy, min_ocr_confidence)
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
    n.require(n.features.check_feature('raqm'), 'RAQM unavailable')
    checked = {'source': source, 'fonts_sha256': n.digest(font_manifest), 'font_id': selected, 'render_policy': render_policy}
    if confidence_settings is not None:
        checked['replacement_policy'] = confidence_settings
    if text_mask_model is not None:
        from . import segmentation
        checked['text_mask_model'] = segmentation.model_spec(text_mask_model)
    return checked


def run_automatic(*, preview, image, fonts, output, font_id=None, min_size=16, max_size=28, max_units=None, background_method='auto', text_mask_model=None, render_policy='best-effort', replacement_policy='all', min_ocr_confidence=None):
    """All current strict translated candidates, with explicit machine skip states."""
    confidence_settings = _options(min_size, max_size, max_units, background_method, render_policy,
                                   replacement_policy, min_ocr_confidence)
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
    raw = preview_doc['ocr']['raw_detections'][0]
    n.require(type(raw) is list and 1 <= len(raw) <= 512, 'automatic raw detection budget')
    pixels = rgb.width*rgb.height
    n.budget_check(pixels*len(raw) <= MAX_WORK, 'automatic binding raster work budget')
    # Explicit machine binding contract; there is no geometry_intent, reviewed
    # flag, manual selection file, proposal ID or call to confirmation/export.
    source_contract = {'schema_version': 1, 'purpose': 'automatic_candidate_only', 'source': source,
        'preview_sha256': n.digest(contents[0]), 'fonts_sha256': n.digest(contents[2]), 'units': []}
    n.bind(preview_doc, paths[1], rgb, contents[1], contents[0], source_contract, contents[2], selection_kind='automatic_candidate_only')
    font_id, (font_data, coverage) = _font(fonts_doc, font_id)
    style = {'min_size': min_size, 'max_size': max_size, 'outline': 0}
    candidates = preview_doc['ocr'].get('automatic_recognition_results', [])
    units = n.automatic_units(preview_doc['ocr']['batches'][0], candidates[0] if candidates else ())
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
    if confidence_settings is not None:
        report['replacement_policy'] = confidence_settings
    restoration_context = None
    text_mask, text_mask_error = None, None
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
    work, eligible, coverage_eligible, attempts = pixels*len(raw), 0, 0, 0
    context, context_error, context_initialized = None, None, False

    def initialize_context():
        nonlocal context, context_error, context_initialized, work
        if not context_initialized:
            context_initialized = True
            try:
                n.budget_check(work+pixels*len(raw) <= MAX_WORK, 'automatic source geometry raster work budget')
                context = OriginalComponents(rgb)
                context.bind_geometry(raw)
                work += pixels*len(raw)
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
        row['input_kind'] = record.get('input_kind', 'strict_utterance')
        report['units'][index] = row
        is_coverage = row['input_kind'] == tc.KIND
        if record['status'] != 'preview' or row['input_kind'] not in ('strict_utterance', tc.KIND):
            continue
        if is_coverage and render_policy != 'best-effort':
            row.update(reason_code='coverage_requires_best_effort', reason='uncertain text-region replacements require best-effort rendering')
            continue
        try:
            if is_coverage:
                n.require(preview_doc['schema_version'] == 4 and
                    preview_doc['settings'].get('translation_policy') == 'coverage-first', 'coverage translation policy/schema required')
                binding = tc.bound_result(record, unit)
                coverage_eligible += 1
                row['coverage'] = record['coverage']
            else:
                binding = n.strict_result_binding(record, unit)
                eligible += 1
            row.update(machine_selected=True, binding=binding)
            if max_units is not None and eligible+coverage_eligible > max_units:
                raise AutomaticSkip('evaluation_cap', 'explicit evaluation cap reached; candidate preserved, not rendered')
            if confidence_settings is not None:
                confidence_unit = tc.derive(unit, binding['reread_selected_revision']) if binding.get('reread_selected_revision') else unit
                assessment = rc.assess_source(confidence_unit, binding['text'], confidence_settings['minimum_main_line_score'])
                row['replacement_confidence'] = assessment
                if not assessment['allowed']:
                    raise AutomaticSkip(assessment['reason_code'], assessment['reason'])
            n.validate_text(binding['text'], coverage)
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
                Image.fromarray(plan.layout.astype(np.uint8)*255), unit_style)
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
                reserved |= plans[i].layout | plans[i].erase
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
            if confidence_settings is not None:
                routes = tuple(route for route in routes if route != 'footprint')
            for route in routes:
                reuse = route == 'reuse'
                if reuse and (existing is None or row['reason_code'] == 'region_collision'
                              or ((existing.layout | existing.erase) & reserved).any()):
                    continue
                try:
                    if route == 'source_local':
                        if context_error is not None:
                            raise n.ResourceLimit(context_error)
                        plan = derive_plan(rgb, raw, row['binding']['member_ids'], context=context)
                        method = 'flat_fill_uncertain_region'
                        if confidence_settings is not None:
                            plan = reserve_layout(plan, reserved)
                        check(not ((plan.layout | plan.erase) & reserved).any(), 'region_collision', 'uncertain region touches reserved strict output')
                    elif reuse:
                        plan = existing
                        method = 'reuse_cleanup_relaxed_typesetting'
                    elif route == 'native_hint':
                        extra_work = pixels*len(row['binding']['member_ids'])
                        n.budget_check(work+extra_work <= MAX_WORK, 'best-effort native raster budget')
                        work += extra_work
                        plan = best_effort.native_plan(rgb, raw, row['binding']['member_ids'],
                            reserved=None if confidence_settings is not None else reserved,
                            allow_tight_source_seed=confidence_settings is not None)
                        if confidence_settings is not None:
                            plan = reserve_layout(plan, reserved)
                        method = plan.evidence['cleanup_method']
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
                    background = plan.evidence.get('background_rgb_at_center', plan.background)
                    n.require(background is not None, 'best-effort contrast estimate unavailable')
                    light = float(np.dot(background, [.2126, .7152, .0722])) < 128
                    fill = [255, 255, 255] if light else [0, 0, 0]
                    fallback_style = {**style, 'fill': fill, 'outline_fill': [255-v for v in fill], 'outline': 1}
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
                    if confidence_settings is not None:
                        reason = rc.layout_rejection(provenance, min_size)
                        check(reason is None, 'replacement_layout_confidence', reason)
                    box = layer.getchannel('A').getbbox()
                    check(box is not None, 'empty_alpha', 'best-effort text has no raster alpha')
                    alpha = np.asarray(layer.getchannel('A')) != 0
                    check(not (alpha & (~plan.layout | plan.obstacles | reserved)).any(),
                          'alpha_containment', 'best-effort alpha leaves declared local overlay support')
                    plate_pixels = np.asarray(rgb).copy()
                    apply_cleanup(plate_pixels, plan)
                    plate = Image.fromarray(plate_pixels)
                    composed = Image.alpha_composite(plate.convert('RGBA'), layer).convert('RGB')
                    audit = n.pixel_audit(rgb, plate, composed, plan.erase, alpha, plan.obstacles)
                    try:
                        n.verify_pixel_audit(audit)
                    except ValueError as error:
                        raise AutomaticSkip('runtime_locality', str(error)) from error
                    plans[i], content_masks[i], tiles[i] = PackedPlan(plan), content.convert('1'), (layer.crop(box), box[:2])
                    if not reuse:
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
                    reserved |= plan.layout | plan.erase
                    break
                except (ValueError, OSError, KeyError, TypeError) as error:
                    row['best_effort_attempts'].append({'method': {'source_local': 'flat_fill_uncertain_region', 'reuse': 'reuse_cleanup', 'native_hint': 'native_contrast_hint',
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
    rendered = sum(row['status'] == 'automatic_composed' for row in report['units'])
    strict_rendered = sum(row['status'] == 'automatic_composed' and row['input_kind'] == 'strict_utterance' for row in report['units'])
    report['coverage'] = {'saved_units': len(report['units']), 'strict_eligible': eligible, 'attempted': attempts,
        'coverage_eligible': coverage_eligible, 'total_eligible': eligible+coverage_eligible,
        'strict_rendered': strict_rendered, 'coverage_rendered': rendered-strict_rendered,
        'rendered': rendered, 'skipped': len(report['units'])-rendered, 'partial_output': rendered < len(report['units']),
        'strict_coverage_fraction': strict_rendered/eligible if eligible else None,
        'total_coverage_fraction': rendered/(eligible+coverage_eligible) if eligible+coverage_eligible else None,
        'has_rendered_output': bool(rendered)}
    best_count = sum(row['status'] == 'automatic_composed' and row.get('render_quality') == 'best_effort'
                     for row in report['units'])
    report['quality_summary'] = {'conservative_rendered': rendered-best_count, 'best_effort_rendered': best_count,
        'has_degraded_output': bool(best_count), 'best_effort_artwork_preservation': 'not_guaranteed' if best_count else 'not_used'}
    if confidence_settings is not None:
        report['replacement_confidence'] = rc.summary(report['units'])
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
    report['job_revision'] = n.revision({k: v for k, v in report.items() if k not in ('job_revision', 'artifacts')})
    blobs, total = {}, 0
    def add(name, value):
        nonlocal total
        total += len(value)
        n.budget_check(len(value) <= 64*1024*1024 and total <= 128*1024*1024, 'automatic artifact byte budget')
        n.budget_check(len(blobs) < MAX_ARTIFACTS, 'automatic artifact count budget')
        blobs[name] = value
    def png(name, value):
        stream = BytesIO()
        value.save(stream, format='PNG')
        add(name, stream.getvalue())
    for name, value in zip(('evidence-preview.json', 'original-file.bin', 'evidence-fonts.json'), contents):
        add(name, value)
    for name, value in (('source', rgb), ('clean', clean), ('composite', composite), ('text', text)):
        png(name+'.png', value)
    if text_mask is not None:
        png('text-probability.png', Image.fromarray(text_mask.raw))
    for name, mask in (('erase-mask', erase_union), ('alpha-mask', alpha_union), ('protected-mask', protected),
                       ('cleanup-diff', np.any(np.asarray(clean) != np.asarray(rgb), axis=2)),
                       ('composite-diff', np.any(np.asarray(composite) != np.asarray(rgb), axis=2))):
        png(name+'.png', Image.fromarray(mask.astype(np.uint8)*255))
    for i, plan in plans.items():
        row = report['units'][i]
        applied = plan.erase if row['status'] == 'automatic_composed' else np.zeros_like(plan.erase)
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
    for name, value in blobs.items():
        with (output/name).open('xb') as stream:
            stream.write(value)
    return report
