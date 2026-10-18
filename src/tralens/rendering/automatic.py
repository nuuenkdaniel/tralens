"""Automatic tight-mask flat rendering, separate from operator-confirmed tools.

Machine-selected strict candidates only. No semantic balloon partition, human
approval, textured restoration, inpainting, model/provider or implicit font lookup.
"""
from dataclasses import dataclass
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


POLICY = 'automatic_original_tight_flat_v1'
MAX_SLOTS = 16
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
    background: list
    evidence: dict


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
        layout, bounds, steps = grow_rectangle(safe, seed, window)
    except AutomaticSkip as error:
        error.evidence = {**evidence, **error.evidence}
        raise
    check(not (layout & obstacles).any(), 'layout_obstacle', 'source-local layout covers original obstacle')
    evidence.update(full_contrast_components=int(len(full_ids)), dark_pixels=int(dark.sum()),
        foreground_pixels=int(foreground.sum()), antialias_pixels=int((foreground & ~dark).sum()),
        erase_pixels=int(erase.sum()), clearance_samples=int(collar.sum()),
        source_local_window=window, layout_bounds=bounds, layout_expansion_tests=steps,
        layout_scope='bounded original whitespace rectangle; not a semantic boundary/partition',
        unrelated_contrast_obstacles=int((contrast & ~foreground).sum()))
    return AutomaticPlan(foreground, erase, layout, obstacles, color.tolist(), evidence)


def implementation_hash():
    return n.revision({name: n.digest(n.read_explicit(Path(__file__).parent/name, 256*1024))
                       for name in ('automatic.py', 'natural.py', 'styles.py', 'preparation.py', 'review.py')})


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
    for index, row in enumerate(report['units']):
        images = ''
        if row.get('specimen'):
            images = '<div class="grid">'+''.join('<figure><figcaption>'+name+'</figcaption><img alt="'+name+'" src="'
                +f'unit-{index:03d}-specimen-{name}.png'+'"></figure>' for name in ('source', 'clean', 'text', 'composite'))+'</div>'
            images += '<div class="grid">'+''.join('<figure><figcaption>'+name+'</figcaption><img alt="'+name+'" src="'
                +f'unit-{index:03d}-specimen-{name}.png'+'"></figure>' for name in ('foreground', 'applied-erase', 'layout', 'obstacles'))+'</div>'
        rows.append('<section><h2>'+escape(row['unit_id'])+' — '+escape(row['status'])+'</h2><p>'
            +escape(row.get('reason') or 'Automatically composed candidate; unverified wording/ownership')+'</p>'+images
            +'<details><summary>Source evidence, settings and checks</summary><pre>'
            +escape(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False))+'</pre></details></section>')
    return (head+''.join(rows)+'</html>').encode()


def run_automatic(*, preview, image, fonts, output, font_id=None, min_size=16, max_size=28, max_units=None):
    """All current strict translated candidates, with explicit machine skip states."""
    n.integer(min_size, 8, 144)
    n.integer(max_size, min_size, 144)
    if max_units is not None:
        n.integer(max_units, 1, MAX_SLOTS)
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
    available = n.load_fonts(fonts_doc)
    if font_id is None:
        n.require(len(available) == 1, 'multiple approved faces require a global --font-id; no implicit font matching')
        font_id = next(iter(available))
    n.require(type(font_id) is str and re.fullmatch('[A-Za-z0-9_-]{1,64}', font_id), 'bounded automatic font ID required')
    n.require(font_id in available and not isinstance(available[font_id], n.ResourceLimit), 'unavailable explicit automatic font')
    font_data, coverage = available[font_id]
    style = {'min_size': min_size, 'max_size': max_size, 'outline': 0}
    candidates = preview_doc['ocr'].get('automatic_recognition_results', [])
    units = n.automatic_units(preview_doc['ocr']['batches'][0], candidates[0] if candidates else ())
    report = {'schema_version': 1, 'purpose': 'automatic_candidate_output', 'policy': POLICY,
        'selection_method': 'all current strict translated candidates in saved order; machine geometric/typographic checks',
        'operator_confirmation': None, 'linguistic_approval': 'not_assessed', 'approved_export': False,
        'source': source, 'preview_sha256': n.digest(contents[0]), 'fonts_sha256': n.digest(contents[2]),
        'implementation_sha256': implementation_hash(), 'font_id': font_id, 'style': style,
        'evaluation_cap': max_units, 'units': [],
        'limits': {'page_pixels': p.MAX_PIXELS, 'page_side': p.MAX_SIDE, 'processed_slots': MAX_SLOTS,
            'aggregate_unit_pixels': n.MAX_UNIT_PIXELS, 'binding_neighbor_raster_work': MAX_WORK,
            'components_per_labeling': 8192, 'owned_components': 64, 'member_masks': 32, 'member_padding': 2,
            'seed_gray_threshold_exclusive': 180, 'component_side': 96, 'component_ink_pixels': 20000,
            'component_bbox_member_area_fraction': .5, 'component_bbox_area_floor': 64,
            'erase_dilation': 1, 'clearance': 2, 'sample_tolerance': 2, 'minimum_background_samples': 32,
            'background_channel_minimum': 180, 'background_channel_spread': 16,
            'layout_horizontal_expansion': 32, 'layout_vertical_expansion': 24,
            'artifact_count': 256, 'artifact_bytes': 128*1024*1024, 'report_bytes': 4*1024*1024,
            'manifest_bytes': 8*1024*1024},
        'engine': {'pillow': n.pillow_version, 'opencv': cv2.__version__, 'raqm': n.features.version_feature('raqm'),
                   'harfbuzz': n.features.version_feature('harfbuzz'), 'freetype': n.features.version_module('freetype2')}}
    plans, tiles = {}, {}
    work, eligible, attempts = pixels*len(raw), 0, 0
    try:
        context = OriginalComponents(rgb)
        n.budget_check(work+pixels*len(raw) <= MAX_WORK, 'automatic source geometry raster work budget')
        context.bind_geometry(raw)
        work += pixels*len(raw)
        context_error = None
    except n.ResourceLimit as error:
        context, context_error = None, str(error)
    for index, (record, unit) in enumerate(zip(preview_doc['units'], units)):
        saved_result = record.get('result')
        review_needed = saved_result.get('review_needed') if type(saved_result) is dict else None
        row = {'unit_id': unit.unit_id, 'source_status': record['status'], 'source_record_sha256': n.revision(record),
               'machine_selected': False, 'status': 'skipped', 'reason_code': 'not_strict_translated_candidate',
               'reason': 'blocked, untranslated, fragment/alternate-only or unexecuted source is not promoted',
               'source_uncertainty': {'completeness': unit.completeness,
                   'structural_reasons': list(unit.structural_reasons), 'lexical_reasons': list(unit.lexical_reasons),
                   'blocked_reasons': list(unit.blocked_reasons),
                   'translation_review_needed': review_needed if type(review_needed) is bool else None}}
        report['units'].append(row)
        if record['status'] != 'preview' or record.get('input_kind', 'strict_utterance') != 'strict_utterance':
            continue
        try:
            binding = n.strict_result_binding(record, unit)
            row.update(machine_selected=True, binding=binding)
            eligible += 1
            if max_units is not None and eligible > max_units:
                raise AutomaticSkip('evaluation_cap', 'explicit evaluation cap reached; candidate preserved, not rendered')
            unit_work = pixels*len(binding['member_ids'])
            if attempts >= MAX_SLOTS or (attempts+1)*pixels > n.MAX_UNIT_PIXELS or work+unit_work > MAX_WORK:
                raise AutomaticSkip('resource_budget', 'automatic per-job mask/raster-work budget reached')
            attempts += 1
            work += unit_work
            if context_error is not None:
                raise AutomaticSkip('resource_budget', context_error)
            plan = derive_plan(rgb, raw, binding['member_ids'], context=context)
            plans[index] = plan
            row['mask_evidence'] = plan.evidence
            check(not (plan.erase & plan.obstacles).any(), 'erasure_obstacle', 'automatic erasure intersects original obstacles')
            layer, provenance = n.layout(binding['text'], font_data, coverage,
                Image.fromarray(plan.layout.astype(np.uint8)*255), style)
            box = layer.getchannel('A').getbbox()
            check(box is not None, 'empty_alpha', 'automatic text has no raster alpha')
            alpha = np.asarray(layer.getchannel('A')) != 0
            check(not (alpha & (~plan.layout | plan.obstacles)).any(), 'alpha_containment', 'automatic alpha escapes local layout/obstacle checks')
            plate_pixels = np.asarray(rgb).copy()
            plate_pixels[plan.erase] = plan.background
            plate = Image.fromarray(plate_pixels, 'RGB')
            composed = Image.alpha_composite(plate.convert('RGBA'), layer).convert('RGB')
            row['candidate_pixel_audit'] = n.pixel_audit(rgb, plate, composed, plan.erase, alpha, plan.obstacles)
            n.verify_pixel_audit(row['candidate_pixel_audit'])
            row['candidate_audit_scope'] = 'isolated proposed operation before symmetric collision rejection; final global audit/masks describe committed pixels'
            row.update(status='ready', reason_code=None, reason=None, layout=provenance)
            tiles[index] = (layer.crop(box), box[:2])
        except AutomaticSkip as error:
            row.update(reason_code=error.code, reason=str(error), skip_evidence=error.evidence)
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
    erase_union = np.zeros((rgb.height, rgb.width), bool)
    alpha_union = np.zeros_like(erase_union)
    layout_union = np.zeros_like(erase_union)
    plate_pixels = np.asarray(rgb).copy()
    for i in ready:
        if i in collisions:
            continue
        plan = plans[i]
        plate_pixels[plan.erase] = plan.background
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
    report['coverage'] = {'saved_units': len(report['units']), 'strict_eligible': eligible, 'attempted': attempts,
        'rendered': rendered, 'skipped': len(report['units'])-rendered, 'partial_output': rendered < len(report['units']),
        'strict_coverage_fraction': rendered/eligible if eligible else None, 'has_rendered_output': bool(rendered)}
    report['job_revision'] = n.revision({k: v for k, v in report.items() if k not in ('job_revision', 'artifacts')})
    blobs, total = {}, 0
    def add(name, value):
        nonlocal total
        total += len(value)
        n.budget_check(len(value) <= 64*1024*1024 and total <= 128*1024*1024, 'automatic artifact byte budget')
        n.budget_check(len(blobs) < 256, 'automatic artifact count budget')
        blobs[name] = value
    def png(name, value):
        stream = BytesIO()
        value.save(stream, format='PNG')
        add(name, stream.getvalue())
    for name, value in zip(('evidence-preview.json', 'original-file.bin', 'evidence-fonts.json'), contents):
        add(name, value)
    for name, value in (('source', rgb), ('clean', clean), ('composite', composite), ('text', text)):
        png(name+'.png', value)
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
