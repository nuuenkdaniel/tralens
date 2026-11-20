"""Coverage-first, explicitly degraded replacements after conservative failure.

The OCR footprint is an uncertain edit region, NOT a certified lettering mask.
This route can damage art inside that region. Originals, known neighboring text,
exact target characters, resource limits and outside-edit locality stay enforced.
No downloads, provider calls, full-page redraws or implicit fonts.
"""
from bisect import bisect_left
from dataclasses import replace
from io import BytesIO
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import natural as n, preparation as p, restoration as r


POLICY = 'coverage_first_ocr_footprint_v1'
MIN_SIZE = 8
MAX_LOCAL_PIXELS = 262144
MAX_LOCAL_SIDE = 1024


def native_plan(rgb, detections, member_ids, *, reserved=None, allow_tight_source_seed=False):
    """Try tighter native strokes before resorting to an OCR-footprint repair.

    The support channel is explicitly a dark-contrast heuristic, NOT a learned
    text probability. It can mistake same-color artwork for text; label the edit
    best-effort even when its flat-background clearance checks succeed.
    """
    from . import segmentation
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    support = np.where(gray < 180, 245, 0).astype(np.uint8)
    plan = segmentation.derive(rgb, detections, member_ids, support, surface='flat',
                               allow_tight_source_seed=allow_tight_source_seed)
    if reserved is not None:
        n.require(not ((plan.erase | plan.layout) & reserved).any(), 'native best-effort plan conflicts with accepted output')
    evidence = dict(plan.evidence)
    evidence['native_association_policy'] = evidence['policy']
    evidence.update(policy=POLICY+'_native_hint', cleanup_method='best_effort_native_fill',
        foreground_is_text_certified=False, artwork_preservation_inside_mask='not_guaranteed',
        support_kind='native grayscale <180; heuristic only, not CTD inference',
        contrast_support_fraction=evidence.pop('semantic_fraction'),
        warnings=['Native contrast/ownership is uncertain and can mistake artwork for lettering.',
                  'This support is not a learned text probability or semantic certification.'])
    evidence['rejected_components'] = [{('contrast_support_fraction' if key == 'semantic_fraction' else key): value
        for key, value in component.items()} for component in evidence['rejected_components']]
    return replace(plan, evidence=evidence)


def derive_plan(rgb, detections, member_ids, *, context=None, reserved=None, background_method='auto'):
    from .automatic import AutomaticPlan, AutomaticSkip

    context = r.OriginalGeometry(rgb, detections) if context is None else context
    n.require(np.array_equal(np.asarray(rgb), context.pixels), 'best-effort source pixels changed')
    n.require(context.revision == n.revision([((type(d['source_id']).__name__, d['source_id']), d['polygon'])
        for d in detections]), 'best-effort source geometry changed')
    n.require(type(member_ids) is list and 0 < len(member_ids) <= 32
        and all(type(i) in (int, str) and i != '' for i in member_ids)
        and len(set(member_ids)) == len(member_ids), 'typed unique best-effort source members required')
    keys = {(type(i).__name__, i) for i in member_ids}
    n.require(keys <= set(context.polygons), 'foreign best-effort source member')
    members = np.zeros(context.counts.shape, bool)
    own_counts = np.zeros_like(context.counts)
    for key in keys:
        mask = np.asarray(n.polygon_mask(context.polygons[key], rgb.size, evidence=True)) != 0
        members |= mask
        own_counts += mask
    nonmembers = context.counts > own_counts
    n.require(reserved is None or (reserved.dtype == bool and reserved.shape == members.shape),
              'best-effort reservation shape/type')
    protected = p.dilate(nonmembers, 1)
    if reserved is not None:
        protected |= reserved
    x, y, w, h = cv2.boundingRect(members.astype(np.uint8))
    # Expansion is an explicit overlay allowance, not inferred blank/art-free space.
    mx, my = max(24, min(64, w//2)), max(16, min(40, h//4))
    bounds = [max(0, x-mx), max(0, y-my), min(rgb.width, x+w+mx), min(rgb.height, y+h+my)]
    left, top, right, bottom = bounds
    cw, ch = right-left, bottom-top
    n.budget_check(0 < cw*ch <= MAX_LOCAL_PIXELS and max(cw, ch) <= MAX_LOCAL_SIDE,
                   'best-effort local context budget')
    crop = context.pixels[top:bottom, left:right]
    local_members = members[top:bottom, left:right]
    erase = p.dilate(local_members, 1) & ~protected[top:bottom, left:right]
    if not erase.any():
        raise AutomaticSkip('best_effort_neighbor_conflict', 'source footprint is reserved by neighboring text/output')
    collar = p.dilate(erase, 4) & ~erase & ~protected[top:bottom, left:right]
    samples = crop[collar] if collar.any() else crop[~erase]
    if not len(samples):
        samples = crop.reshape(-1, 3)
    color = np.median(samples, axis=0).astype(np.uint8)
    attempts, candidates = [], []
    # Compare at most two installed classical methods on exactly the same mask.
    # The score is a seam/color heuristic, NEVER a hidden-art quality certificate.
    if background_method == 'auto' and (~erase).any():
        for method, flag in (('navier_stokes', cv2.INPAINT_NS), ('telea', cv2.INPAINT_TELEA)):
            try:
                candidate = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    erase.astype(np.uint8)*255, 3, flag), cv2.COLOR_BGR2RGB)
                n.require(candidate.dtype == np.uint8 and candidate.shape == crop.shape,
                          'best-effort backend shape/type')
                n.require(np.array_equal(candidate[~erase], crop[~erase]), 'best-effort backend changed known pixels')
                boundary = erase & ~cv2.erode(erase.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                support = candidate[boundary] if boundary.any() else candidate[erase]
                seam = float(np.abs(support.astype(float)-color).mean())
                score = seam + .05*float(candidate[erase].astype(float).std(axis=0).mean())
                candidates.append((score, method, candidate[erase].copy()))
                attempts.append({'method': method, 'status': 'candidate', 'heuristic_score': score})
            except (cv2.error, ValueError) as error:
                attempts.append({'method': method, 'status': 'failed', 'reason': str(error)[:500]})
    if candidates:
        _, method, restored = min(candidates, key=lambda item: (item[0], item[1]))
        cleanup = 'best_effort_inpaint_'+method
    else:
        # A visible/guessed backing is preferable to silently leaving a known
        # translation unplaced. This is explicitly reported as a low-quality edit.
        restored = np.tile(color, (int(erase.sum()), 1)).astype(np.uint8)
        cleanup = 'best_effort_color_backing'
    full_erase = np.zeros_like(members)
    full_erase[top:bottom, left:right] = erase
    overlay = np.zeros_like(members)
    overlay[top:bottom, left:right] = ~protected[top:bottom, left:right]
    if not overlay.any():
        raise AutomaticSkip('best_effort_no_layout', 'no source-local overlay space remains outside neighboring text')
    layout = overlay.copy()
    layout_scope = 'source-local overlay; may cover original artwork, not certified whitespace'
    # Prefer locally measured background space before resorting to an art overlay.
    # This does not strengthen the uncertain erase mask or certify a balloon.
    residual = np.max(np.abs(crop.astype(np.int16)-color.astype(np.int16)), axis=2)
    quiet = ((residual <= 12) | erase) & ~protected[top:bottom, left:right]
    close = np.zeros_like(quiet)
    close[max(0, y-24-top):min(ch, y+h+24-top), max(0, x-32-left):min(cw, x+w+32-left)] = True
    quiet &= close
    count, labels = cv2.connectedComponents(quiet.astype(np.uint8), connectivity=4)
    frequencies = np.bincount(labels[local_members & erase], minlength=count) if count <= 8192 else np.array([0])
    frequencies[0] = 0
    if frequencies.max() > 0:
        candidate = labels == int(frequencies.argmax())
        if int(candidate.sum()) >= 64 and int((candidate & local_members).sum()) >= .6*int(local_members.sum()):
            layout[:] = False
            layout[top:bottom, left:right] = candidate
            layout_scope = 'measured source-local low-contrast space after uncertain cleanup; not certified art-free'
    obstacles = ~overlay
    evidence = {'policy': POLICY, 'cleanup_method': cleanup, 'backend_attempts': attempts,
        'edit_support': 'OCR source polygons +1px, minus known neighbors and already accepted output',
        'foreground_is_text_certified': False, 'artwork_preservation_inside_mask': 'not_guaranteed',
        'source_footprint_pixels': int(members.sum()), 'erase_pixels': int(erase.sum()),
        'protected_source_pixels': int((members & protected).sum()), 'local_bounds': bounds,
        'background_rgb_at_center': color.tolist(), 'restoration_quality': 'best_effort_unverified',
        'candidate_selection': 'bounded seam/color heuristic; not ground-truth reconstruction',
        'layout_scope': layout_scope,
        'warnings': ['Uncertain OCR-footprint erasure may remove artwork or outlines.',
                     'Background reconstruction and complete source-text removal are unverified.']
                    +(['Translated lettering may overlay original artwork.'] if np.array_equal(layout, overlay) else [])}
    return AutomaticPlan(full_erase.copy(), full_erase, layout, obstacles, None, evidence,
                         restored_pixels=restored)


def overlay_plan(plan):
    """Last local placement allowance, without rerunning cleanup or changing words."""
    evidence = {**plan.evidence, 'layout_scope': 'source-local art overlay; may cover original artwork',
        'warnings': plan.evidence['warnings']+['Translated lettering may overlay original artwork.']}
    return replace(plan, layout=~plan.obstacles, evidence=evidence)


def _insert(text, offsets):
    for offset in reversed(offsets):
        text = text[:offset]+'\n'+text[offset:]
    return text


def emergency_wrap(text, font_data, width, *, size=MIN_SIZE, outline=1):
    """Insert line breaks without dropping spaces/punctuation or splitting marks."""
    n.require(type(text) is str and 0 < len(text) <= 1000, 'best-effort text budget')
    font = ImageFont.truetype(BytesIO(font_data), size, layout_engine=ImageFont.Layout.RAQM)
    probe = ImageDraw.Draw(Image.new('L', (1, 1)))
    offsets, start, index = [], 0, 0
    while index < len(text):
        if text[index] == '\n':
            start, index = index+1, index+1
            continue
        end = index+1
        while end < len(text) and unicodedata.category(text[end])[0] == 'M':
            end += 1
        advance = probe.textlength(text[start:end].rstrip(' '), font=font, direction='ltr', language='en')
        if advance > width-2*outline:
            n.require(index > start, 'overflow: one target grapheme cannot fit the best-effort layout')
            offsets.append(index)
            start = index
            continue
        index = end
    n.budget_check(len(offsets)+text.count('\n') < 32, 'best-effort line count budget')
    shaped = _insert(text, offsets)
    n.budget_check(len(shaped) <= 1000, 'best-effort wrapped text budget')
    return shaped, offsets


def fit(text, font_data, coverage, mask, style, *, work=None):
    """Reduced margins, then smaller type, then character-preserving hard wraps."""
    bounds = mask.getbbox()
    n.require(bounds is not None, 'empty best-effort layout')
    work = n.LayoutWork() if work is None else work
    attempts = []
    modes = [('reduced_margins', style), ('smaller_type', {**style, 'min_size': MIN_SIZE,
        'max_size': min(28, style['max_size']), 'leading': 0})]
    for mode, current in modes:
        try:
            layer, provenance = n.layout(text, font_data, coverage, mask, current, work=work)
            attempts.append({'method': mode, 'status': 'fit', 'size': provenance['size']})
            break
        except n.ResourceLimit:
            raise
        except ValueError as error:
            if str(error) != 'overflow: no contained layout within supplied size bounds':
                raise
            attempts.append({'method': mode, 'status': 'overflow'})
    else:
        shaped, offsets = emergency_wrap(text, font_data, bounds[2]-bounds[0], outline=style.get('outline', 0))
        layer, provenance = n.layout(shaped, font_data, coverage, mask,
            {**style, 'min_size': MIN_SIZE, 'max_size': MIN_SIZE, 'leading': 0}, work=work)
        inserted_positions = [offset+i for i, offset in enumerate(offsets)]
        additional = [offset-bisect_left(inserted_positions, offset) for offset in provenance['inserted_break_offsets']]
        combined = sorted(offsets+additional)
        n.require(_insert(text, combined) == provenance['draw_text'], 'best-effort character preservation failed')
        provenance.update(original_text=text, inserted_break_offsets=combined)
        attempts.append({'method': 'emergency_character_wrap', 'status': 'fit', 'size': provenance['size']})
    provenance['line_break_policy'] = 'best_effort_exact_characters_v1'
    provenance['typography'] = {'policy': POLICY, 'margin_pixels': 0, 'preferred_size': style['min_size'],
        'content_bounds': list(bounds), 'content_pixels': int(np.count_nonzero(np.asarray(mask))),
        'probes': attempts, 'below_requested_minimum': provenance['size'] < style['min_size'],
        'requested_minimum': style['min_size'], 'quality': 'degraded_best_effort'}
    return layer, provenance, mask.copy()
