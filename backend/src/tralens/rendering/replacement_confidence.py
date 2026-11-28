"""Opt-in replacement triage, never a translation gate or accuracy probability.

The weakest bound raw main-line score limits OCR trust; annotations are not mixed
into that score. Alternative readings cannot inherit Paddle's raw score. Native
cleanup support and requested-size lettering are separate requirements. No text,
script, source ID, visual-review label or expected word is special-cased here.
"""
import math
import re
import unicodedata
from urllib.parse import urlsplit
from dataclasses import replace

from ..translation.contracts import canonical, strict_json


POLICY = 'weakest_raw_line_native_cleanup_v1'
DEFAULT_MINIMUM = .70


def pencil_hatching(image, line):
    """Tiny parallel strokes continuing outside a weak OCR box are drawn texture."""
    if line.get('confidence',1)>=.55 or not re.fullmatch('[A-Za-z0-9]{1,4}',line.get('text','')):
        return False
    import cv2
    import numpy as np
    points=np.asarray(line['polygon'])
    (l,t),(r,b)=points.min(0).astype(int),points.max(0).astype(int)
    w,h=r-l+1,b-t+1
    if not (8<=w<=36 and 14<=h<=64):
        return False
    pad=min(24,max(10,w))
    x,y=max(0,l-pad),max(0,t-pad)
    tile=np.asarray(image.crop((x,y,min(image.width,r+pad+1),min(image.height,b+pad+1))).convert('L'))
    ink=np.uint8(tile<155)
    _,labels,stats,centers=cv2.connectedComponentsWithStats(ink)
    inside_area=ink[t-y:b-y+1,l-x:r-x+1].sum()
    if inside_area<16:
        return False
    inside=outside=covered=0
    for j,(_,_,width,height,area) in enumerate(stats[1:],1):
        if not (3<=width<=12 and 1<=height<=width and 3<=area<=26):
            continue
        yy,xx=np.nonzero(labels==j)
        vx,vy,*_=cv2.fitLine(np.float32(np.column_stack((xx,yy))),cv2.DIST_L2,0,.01,.01).ravel()
        if abs(vx)<.65 or not .15<vy/vx<1.35 or np.std(xx*vy-yy*vx)>.65:
            continue
        cx,cy=centers[j]
        if l<=cx+x<=r and t<=cy+y<=b:
            inside+=1
            covered+=area
        else:
            outside+=1
    return inside>=6 and outside>=3 and covered>=.67*inside_area


def settings(policy='all', minimum=None):
    if type(policy) is not str or policy not in ('all', 'confidence-gated'):
        raise ValueError('unknown replacement policy')
    if policy == 'all':
        if minimum is not None:
            raise ValueError('minimum OCR confidence requires confidence-gated replacement policy')
        return None
    minimum = DEFAULT_MINIMUM if minimum is None else minimum
    if type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError('minimum OCR confidence must be a finite number in 0..1')
    return {'name': policy, 'policy': POLICY, 'minimum_main_line_score': float(minimum),
        'score_is_accuracy_probability': False, 'coarse_footprint_cleanup_allowed': False,
        'below_requested_font_size_allowed': False, 'unchanged_target_preserves_source': True,
        'translation_policy_unchanged': True,
        'calibration_scope': 'provisional single-page model visual assessment; not held-out accuracy validation'}


def retains_printed_words(unit, target):
    """Keep confident Latin labels when the model only punctuates their rows."""
    reading = unit.selected
    if reading.origin != 'raw_main_reading' or not re.fullmatch(r'[A-Za-z]{2,128}', reading.text):
        return False
    if reading.text.isascii() and reading.text.casefold() == target.casefold():
        return True
    if len(target) > 256 or not re.fullmatch(r'[A-Za-z]+(?:[ ,/]+[A-Za-z]+)*[.]?', target):
        return False
    lines = strict_json(reading.evidence_json)
    if type(lines) is not list or len(lines) != len(unit.order):
        return False
    words = re.findall(r'[A-Za-z]+', target)
    if len(words) != len(lines):
        return False
    exact_words = set()
    for line, source_id, word in zip(lines, unit.order, words):
        if (type(line) is not dict or line.get('source_id') != source_id
                or line.get('status') != 'accepted' or type(line.get('confidence')) not in (int, float)
                or not .97 <= line['confidence'] <= 1):
            return False
        source = line.get('text')
        if type(source) is not str or not re.fullmatch(r'[A-Za-z]{2,32}', source):
            return False
        if source == word:
            exact_words.add(source)
        elif not (source in exact_words and source.istitle() and word == source.lower()):
            return False
    return ''.join(line['text'] for line in lines) == reading.text


def circled_letter(image, polygon, reading):
    """Confirm the enclosing ring before treating a lone OCR letter as a printed mark."""
    if reading not in ('R','C','P','®','©','℗'):
        return False
    import cv2
    import numpy as np
    xx,yy = zip(*polygon)
    l,t,r,b = int(min(xx)),int(min(yy)),int(max(xx))+1,int(max(yy))+1
    w,h = r-l,b-t
    if not (8<=w<=64 and 8<=h<=64 and .75<w/h<1.34):
        return False
    pixels = np.asarray(image.crop((max(0,l-3),max(0,t-3),min(image.width,r+3),min(image.height,b+3))).convert('L'))
    _,binary = cv2.threshold(pixels,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    for ink in (binary,255-binary):
        contours,tree = cv2.findContours(ink,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i,contour in enumerate(contours):
            x,y,cw,ch = cv2.boundingRect(contour)
            if (tree[0,i,3]>=0 or tree[0,i,2]<0 or x==0 or y==0
                    or x+cw>=ink.shape[1] or y+ch>=ink.shape[0] or min(cw/w,ch/h)<.72):
                continue
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour,True)
            if area/(cw*ch)>.63 and 4*math.pi*area/perimeter**2>.78:
                return True
    return False


def printed_sequence(member, detections):
    """Three consecutive letters on a regularly spaced row or column are labels."""
    labels = []
    for d in detections:
        word = d['text'].strip()
        if not re.fullmatch('[A-Z]',word) or d['confidence']<.9:
            continue
        xs,ys = zip(*d['polygon'])
        labels.append((ord(word),(min(xs)+max(xs))/2,(min(ys)+max(ys))/2,
                       max(max(xs)-min(xs),max(ys)-min(ys),1),d['source_id']))
    center = next((item for item in labels if item[4]==member),None)
    if center is None:
        return False
    for other in labels:
        delta = other[0]-center[0]
        if abs(delta)!=1:
            continue
        dx,dy = (other[1]-center[1])/delta,(other[2]-center[2])/delta
        step = math.hypot(dx,dy)
        if not (center[3]<step<9*center[3] and max(dx,dy)>.9*step
                and .65*center[3]<other[3]<1.5*center[3]):
            continue
        for third in labels:
            k = third[0]-center[0]
            if k not in (-delta,2*delta) or not .65*center[3]<third[3]<1.5*center[3]:
                continue
            if math.hypot(third[1]-center[1]-k*dx,third[2]-center[2]-k*dy)<.35*step:
                return True
    return False


def original_lettering_reason(unit, target, detections, bounds, page_size, *, reviewed_source=None, verify_crop=None):
    """Retain literal/bilingual type, and quarantine readings with insufficient evidence."""
    from ..ocr.types import is_credit_list, is_printed_literal, is_printed_number, is_printed_punctuation, printed_latin_lines
    source = unit.selected.text.strip()
    if (len(unit.member_ids)==1 and re.fullmatch('[A-Z]',source) and reviewed_source in (None,source)
            and printed_sequence(unit.member_ids[0],detections)):
        return 'replacement_unchanged_text','Preserve the printed sequence of letter labels.'
    if len(unit.member_ids)==1 and source.isascii() and any(
            z['source_id']==unit.member_ids[0] and z.get('reason')=='confirmed_checkbox_item' for z in detections):
        return 'replacement_unchanged_text', 'Preserve the printed software identifier beside its checkbox.'
    if is_credit_list(source):
        return 'replacement_unchanged_text', 'Preserve the printed list of names.'
    if is_printed_literal(source):
        return 'replacement_unchanged_text', 'Preserve the original technical literal.'
    evidence = reviewed_source.strip() if reviewed_source and reviewed_source.strip() else source
    if evidence.isascii() and source.isascii() and unit.selected.origin=='raw_main_reading':
        ids = set(unit.member_ids)
        lines = [line for line in detections if line['source_id'] in ids]
        if len(lines)>=2 and printed_latin_lines(source,lines):
            return 'replacement_unchanged_text', 'The corroborated Latin lines are already printed on the page.'
    if all(re.fullmatch(r'[. …・]+', value) for value in (source,evidence,target)):
        return 'replacement_unchanged_text', 'Preserve the original typeset pause and its orientation.'
    if is_printed_number(evidence) and is_printed_number(source):
        return 'replacement_unchanged_text', 'Keep the printed numeric label.'
    if evidence in ('を','へ','に') and not reviewed_source and unit.selected.origin == 'raw_main_reading':
        return 'uncorroborated_source', 'A detached grammatical particle does not establish an independent caption.'
    if is_printed_punctuation(evidence) and any(c.isalnum() for c in target):
        return 'uncorroborated_source', 'Punctuation or an arrow alone does not establish dialogue.'
    url_pattern = r'https?://[^\s<>()「」『』、。]+'
    urls = [url.rstrip('.,') for url in re.findall(url_pattern, source)]
    if urls != [url.rstrip('.,') for url in re.findall(url_pattern, target)]:
        return 'uncorroborated_source', 'The proposed translation changes a printed URL.'
    for url in urls:
        try:
            hostname = urlsplit(url).hostname or ''
        except ValueError:
            hostname = ''
        if not hostname or any(len({unicodedata.name(c, '').split(' ')[0] for c in label if c.isalpha()}
                                  & {'LATIN', 'GREEK', 'CYRILLIC'}) > 1 for label in hostname.split('.')):
            return 'uncorroborated_source', 'The domain contains an ambiguous OCR reading. Preserve the printed address.'
    from ..translation.rendition import expanded_short_label
    if expanded_short_label(evidence,target):
        return 'uncorroborated_source', 'The short source label does not support a full sentence.'
    if evidence == '第' and re.search(r'\d', target):
        return 'uncorroborated_source', 'The isolated ordinal marker supplies no number.'
    left, top, right, bottom = bounds
    width, height = page_size
    compact_name = ' '.join(target.casefold().split())
    pair = [word.strip() for word in compact_name.split('/')]
    if len(pair)==2 and source.casefold() in pair:
        compact_name = pair[1-pair.index(source.casefold())]
    if re.fullmatch(r'[a-z]+(?: [a-z]+){1,3}', compact_name) and re.search(r'[一-鿿]', source):
        nearby_words = []
        for detection in detections:
            if detection['source_id'] in unit.member_ids or detection['confidence'] < .90:
                continue
            xs, ys = zip(*detection['polygon'])
            if (max(xs) >= left and min(xs) <= right and min(ys) <= bottom + (bottom-top)
                    and max(ys) >= top - (bottom-top)):
                word = ' '.join(detection['text'].casefold().split())
                if word == compact_name:
                    return 'replacement_unchanged_text', 'The adjacent bilingual label already prints this name.'
                if re.fullmatch('[a-z]+', word) and left <= (min(xs)+max(xs))/2 <= right:
                    nearby_words.append((min(ys), min(xs), max(ys), word))
        # A narrow card may stack the romanization below the native name.
        words = sorted(nearby_words)
        if (' '.join(w[3] for w in words) == compact_name
                and all(0 < b[0]-a[0] <= 1.5*(a[2]-a[0]) and abs(b[1]-a[1]) <= a[2]-a[0]
                        for a,b in zip(words,words[1:]))):
            return 'replacement_unchanged_text', 'The adjacent bilingual label already prints this name.'
    confirmed_display = reviewed_source and all(d.get('reason')=='corroborated_repeated_effect'
        or any(d.get('ocr_metadata',{}).get(k,{}).get('reading')==reviewed_source for k in ('colored_word','outlined_display'))
        for d in detections if d['source_id'] in unit.member_ids)
    if (not confirmed_display and (right-left) > .34*width and bottom-top > .11*height and len(source) <= 12
            and all(d['confidence'] < .82 for d in detections if d['source_id'] in unit.member_ids)
            and not (verify_crop and verify_crop())):
        return 'uncorroborated_source', 'Large display lettering has no reliable original reading.'
    return None


def assess_source(unit, target, minimum):
    """Caller first validates original/unit/request/result with existing bindings."""
    settings('confidence-gated', minimum)
    assessment = {'policy': POLICY, 'source_unit_revision': unit.revision,
        'source_reading_revision': unit.selected_revision, 'minimum_required': float(minimum),
        'score_kind': 'minimum uncalibrated raw OCR main-line score, not accuracy probability',
        'source_target_exact_match': target == unit.selected.text,
        'line_scores': [], 'minimum_main_line_score': None, 'allowed': False,
        'reason_code': 'replacement_confidence_unavailable',
        'reason': 'Selected reading lacks usable source-bound raw main-line scores.'}
    if unit.selected.origin == 'raw_main_reading':
        try:
            raw = strict_json(unit.selected.evidence_json)
            if type(raw) is not list or not 1 <= len(raw) <= 32 or len(raw) != len(unit.order):
                raise ValueError('main-line coverage')
            for item, source_id in zip(raw, unit.order):
                if (type(item) is not dict or type(item['source_id']) is not type(source_id)
                        or item['source_id'] != source_id or canonical(item['page']) != unit.page_json
                        or type(item['text']) is not str or item['status'] not in ('accepted', 'review')):
                    raise ValueError('main-line binding/status')
                score = item['confidence']
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('unusable confidence')
            if ''.join(item['text'] for item in raw) != unit.selected.text:
                raise ValueError('raw characters differ from selected reading')
            assessment['line_scores'] = [{'source_id': item['source_id'], 'score': float(item['confidence'])}
                                         for item in raw]
            score = min(item['score'] for item in assessment['line_scores'])
            assessment['minimum_main_line_score'] = score
            assessment.update(allowed=score >= minimum,
                reason_code=None if score >= minimum else 'replacement_low_ocr_confidence',
                reason=('Bound main-line scores meet the experimental threshold; correctness remains unverified.'
                        if score >= minimum else 'At least one bound main-line OCR score is below the replacement threshold.'))
        except (ValueError, TypeError, KeyError):
            pass
    elif unit.selected.origin == 'paired_visual_reread_unverified':
        from ..ocr.quality_reread import preferred_reading
        raw = [reading for reading in unit.readings if reading.origin == 'raw_main_reading']
        if len(raw) == 1:
            candidate = preferred_reading(replace(unit, selected_revision=raw[0].revision))
            if candidate is not None and candidate.revision == unit.selected_revision:
                assessment.update(allowed=True, reason_code=None,
                    score_kind='unscored paired visual/line reread; no raw Paddle score inherited',
                    reason='Explicitly enabled, source-bound paired-reader hypothesis; semantic correctness remains unverified.')
                from ..ocr import quality_lines
                proof = strict_json(candidate.evidence_json)['quality_reread']
                if proof['policy'] == quality_lines.POLICY:
                    scores = [{'source_id': line['source_id'], 'score': line['confidence']}
                        for line in proof['lines'] if line['kind'] == 'raw_unchanged']
                    assessment.update(line_scores=scores,
                        minimum_main_line_score=min((line['score'] for line in scores), default=None),
                        score_kind='retained raw lines keep their own scores; changed lines are unscored paired-reader hypotheses')
                    if any(line['score'] < minimum for line in scores):
                        assessment.update(allowed=False, reason_code='replacement_low_ocr_confidence',
                            reason='A retained raw line is below the requested replacement threshold; reread scores are not inherited.')
    # Exact string identity justifies preserving source pixels, not certifying OCR.
    if assessment['source_target_exact_match']:
        assessment.update(allowed=False, reason_code='replacement_unchanged_text',
            reason='Translation exactly matches the selected source string; preserve original lettering without erasure.')
    return assessment


def layout_rejection(layout, requested_minimum):
    size = layout.get('size')
    if type(size) is not int or size < requested_minimum:
        return 'Translated text would need lettering below the requested minimum size.'
    return None


def summary(rows):
    def count(code):
        return sum(row.get('reason_code') == code for row in rows)
    return {'bound_translation_results_retained': sum('binding' in row for row in rows),
        'actual_replacements': sum(row['status'] == 'automatic_composed' for row in rows),
        'unchanged_text_preserved': count('replacement_unchanged_text'),
        'low_ocr_confidence_deferred': count('replacement_low_ocr_confidence'),
        'missing_confidence_deferred': count('replacement_confidence_unavailable'),
        'cleanup_support_deferred': count('replacement_cleanup_confidence'),
        'readability_deferred': count('replacement_layout_confidence'),
        'bound_translations_without_replacement': sum('binding' in row and row['status'] == 'skipped' for row in rows),
        'semantic_quality_verified': False,
        'note': 'Unedited source text remains; translations are retained, not counted as image replacements.'}
