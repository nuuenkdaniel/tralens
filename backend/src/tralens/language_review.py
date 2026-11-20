"""Render-independent bounded language-review routing, never an approval gate.

Signals request investigation, not corrections. This module consumes immutable
OCR/translation evidence and original pixels; renderer status is not an input.
"""
from dataclasses import asdict
import math
import re
import unicodedata

import cv2
import numpy as np

from .ocr.recognition_units import image_fingerprint
from .translation.contracts import (automatic_units, TranslationRequest, TranslationResult,
    UnitOutcome, canonical, revision)
from .translation.context import saved_requests
from .translation import coverage


POLICY = 'independent_language_review_queue_v1'
MAX_REQUESTS = 4
MAX_CROP_PIXELS = 262144


def require(value,message):
    if not value:
        raise ValueError(message)


def result_from(data):
    data = dict(data)
    data['translations'] = tuple(tuple(pair) for pair in data['translations'])
    data['outcomes'] = tuple(UnitOutcome(**{**o,'reasons':tuple(o['reasons'])}) for o in data['outcomes'])
    return TranslationResult(**data)


def primary(record,unit,request=None):
    if record['status'] != 'preview':
        return None
    if record.get('input_kind') == coverage.KIND:
        binding = coverage.bound_result(record,unit,request=request)
        return {'text':binding['text'],'request_revision':binding['request_revision'],
            'result_revision':revision(record['result'])}
    if record.get('input_kind','strict_utterance') != 'strict_utterance':
        return None
    request = request or TranslationRequest((unit,),(unit.unit_id,))
    require(unit.disposition == 'preview' and record['binding_current'] is True
        and record['unit_revision'] == unit.revision and record['request_revision'] == request.revision
        and canonical(record['request']) == canonical(asdict(request)), 'stale primary translation binding')
    result = result_from(record['result']);result.assert_current(request)
    require(result.status == 'preview' and len(result.translations) == 1
        and result.translations[0][0] == unit.unit_id, 'primary target/result drift')
    return {'text':result.translations[0][1],'request_revision':request.revision,'result_revision':revision(record['result'])}


def bounds(detections,ids):
    points = [p for d in detections if d['source_id'] in ids for p in d['polygon']]
    require(points,'missing original source geometry')
    return [math.floor(min(x for x,y in points)),math.floor(min(y for x,y in points)),
        math.ceil(max(x for x,y in points)),math.ceil(max(y for x,y in points))]


def context_bounds(box,size):
    x,y,r,b = box
    # Larger along the reading axis to expose clipped continuation. This is a
    # review allowance, NOT replacement geometry or a claim of text ownership.
    px = max(24,min(192,math.ceil((r-x)*.75)))
    py = max(24,min(96,math.ceil((b-y)*.5)))
    return [max(0,x-px),max(0,y-py),min(size[0],r+px),min(size[1],b+py)]


def scripts(text):
    return (any('LATIN' in unicodedata.name(c,'') for c in text),
        any(unicodedata.name(c,'').startswith(('HIRAGANA','KATAKANA','CJK UNIFIED')) for c in text))


def possible_menu(text,source):
    """Routing evidence only. Never split/select text, even if this returns true."""
    return ('/' not in source and len(re.findall(r'\s+/\s+',text)) >= 1
        and len(re.findall(r'[A-Za-z]{2,}',text)) >= 3)


def crosses_edge(pixels,box):
    x,y,r,b = box;h,w = pixels.shape[:2]
    left,top,right,bottom = max(0,x-4),max(0,y-4),min(w,r+5),min(h,b+5)
    local = (pixels[top:bottom,left:right].min(axis=2) < 180).astype(np.uint8)
    count,labels,stats,_ = cv2.connectedComponentsWithStats(local,connectivity=8)
    require(count <= 8192,'review component budget')
    own = np.zeros(local.shape,bool);own[y-top:min(b+1,bottom)-top,x-left:min(r+1,right)-left] = True
    for label in np.unique(labels[own]):
        if label == 0:
            continue
        xx,yy,ww,hh,area = map(int,stats[label])
        if area >= 4 and (xx+left < x or yy+top < y or xx+ww+left > r+1 or yy+hh+top > b+1):
            return True
    return False


def plan(preview,rgb,*,max_requests=MAX_REQUESTS):
    require(type(max_requests) is int and 0 <= max_requests <= 8,'language-review request budget')
    require(rgb.mode == 'RGB' and rgb.width*rgb.height <= 4_000_000
        and image_fingerprint(rgb) == preview['input']['pixels_sha256'],'language-review original pixels/budget')
    batch = preview['ocr']['batches'][0]
    candidates = preview['ocr'].get('automatic_recognition_results') or [()]
    units = automatic_units(batch,candidates[0])
    require(1 <= len(units) <= 256 and len(units) == len(preview['units']),'language-review unit budget/count')
    raw = preview['ocr']['raw_detections'][0]
    by_id = {d['source_id']:d for d in raw}
    require(len(by_id) == len(raw) <= 512,'language-review typed unique raw IDs')
    for d in batch['detections']:
        current = by_id[d['source_id']]
        require(all(canonical(current[k]) == canonical(d[k]) for k in ('source_id','text','polygon','confidence')),
            'language-review raw detection drift')
    contextual = saved_requests(preview,units)
    boxes = [bounds(raw,u.member_ids) for u in units]
    pixels = np.asarray(rgb)
    rows = []
    for i,(unit,record,box) in enumerate(zip(units,preview['units'],boxes)):
        require(canonical(record['unit']) == canonical(asdict(unit)) and record['unit_revision'] == unit.revision,
            'language-review saved unit drift')
        saved = primary(record,unit,contextual.get(unit.unit_id))
        text = unit.selected.text
        if record.get('translation_input'):
            source = record['translation_input']
            text = next(r['text'] for r in source['readings'] if revision(r) == source['selected_revision'])
        reasons = []
        scores = [by_id[j]['confidence'] for j in unit.order]
        latin,japanese = scripts(text)
        if min(scores) < .9:reasons.append('low_main_line_score')
        if len({r.text for r in unit.readings}) > 1:reasons.append('retained_reading_conflict')
        if latin and japanese:reasons.append('mixed_script_source')
        if latin and re.search(r'[-‐‑–ー]$',text.strip()):reasons.append('possible_truncated_latin_run')
        x,y,r,b = box
        if len(unit.order) <= 2 and r-x >= 3*max(1,b-y):reasons.append('horizontal_label_audit')
        if crosses_edge(pixels,box):reasons.append('ink_crosses_detector_edge')
        if re.search(r'([\u3041-\u30fa])[\s・…ー～-]{0,2}\1',text):reasons.append('phonetic_hesitation')
        if not text.strip():reasons.append('blank_selected_source')
        neighbors = []
        for j,other in enumerate(boxes):
            if i == j or set(unit.member_ids)&set(units[j].member_ids):continue
            a,c,d,e = other
            dx,dy = max(0,x-d,a-r),max(0,y-e,c-b)
            scale = max(8,min(r-x,b-y,d-a,e-c))
            if dx*dx+dy*dy <= (2*scale)**2:
                neighbors.append(units[j].unit_id)
        if neighbors:reasons.append('nearby_separate_units')
        if saved and possible_menu(saved['text'],text):reasons.append('possible_multiple_renditions')
        window = context_bounds(box,rgb.size)
        crop_pixels = (window[2]-window[0])*(window[3]-window[1])
        row = {'index':i,'unit_id':unit.unit_id,'unit_revision':unit.revision,
            'member_ids':list(unit.member_ids),'main_order':list(unit.order),'source_text':text,
            'primary':saved,'main_line_scores':scores,'confidence_is_accuracy':False,
            'reasons':reasons or ['stratified_audit_only'],'neighbor_unit_ids':neighbors,
            'detector_bounds':box,'context_bounds':window,'crop_pixels':crop_pixels,
            'status':'not_requested','reason':'not_selected_under_separate_review_budget',
            'source_change_authorized':False,'rendering_status_consulted':False}
        if len(unit.member_ids) > 8 or crop_pixels > MAX_CROP_PIXELS or max(window[2]-window[0],window[3]-window[1]) > 1024:
            row.update(status='unsupported',reason='review_member_or_crop_budget')
        rows.append(row)
    available = [r for r in rows if r['status'] != 'unsupported']
    # Reserve one audit slot; high confidence and lack of risk signals cannot
    # make a unit permanently ineligible. Every exclusion remains in the plan.
    def rank(row):
        rs = set(row['reasons'])
        tier = (0 if 'possible_truncated_latin_run' in rs else 1 if 'horizontal_label_audit' in rs else
            2 if 'retained_reading_conflict' in rs else 3 if 'phonetic_hesitation' in rs else
            4 if 'mixed_script_source' in rs else 5 if 'possible_multiple_renditions' in rs else 6)
        return tier,-len(rs),row['index']
    ranked = sorted(available,key=rank)
    requested = ranked[:max(0,max_requests-1)] if max_requests else []
    others = [r for r in available if r not in requested]
    if max_requests and others:
        requested.append(min(others,key=lambda r:revision({'source':preview['input']['sha256'],'unit_id':r['unit_id']})))
        requested[-1]['reasons'] = list(dict.fromkeys(requested[-1]['reasons']+['stratified_audit_sample']))
    for row in available:
        if row in requested:row.update(status='requested',reason='bounded_language_review')
        else:row.update(status='budget_excluded',reason='independent_language_review_budget')
    plan = {'policy':POLICY,'source':preview['input'],'preview_revision':revision(preview),
        'raw_revision':revision(raw),'requests':[r['unit_id'] for r in requested], 'records':rows,
        'maximum_requests':max_requests,'counts':{'considered':len(rows),'requested':len(requested),
            'budget_excluded':sum(r['status']=='budget_excluded' for r in rows),
            'unsupported':sum(r['status']=='unsupported' for r in rows)},
        'source_selection':'none; routing only','renderer_dependency':False}
    plan['revision'] = revision(plan)
    return plan
