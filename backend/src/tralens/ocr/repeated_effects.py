"""Read oversized neighboring effects when Paddle found their footprints but missed the ink."""
import re
from copy import deepcopy

import cv2
import numpy as np

from .recognition_units import image_fingerprint
from .sound_readings import repeat
from .types import OCREngine, Text_Box


def views(image, polygon):
    points = np.asarray(polygon)
    (x,y),(r,b) = points.min(0).astype(int),points.max(0).astype(int)
    for margin in (0,.04,.10):
        dx,dy = round((r-x)*margin),round((b-y)*margin/2)
        yield image.crop((max(0,x-dx),max(0,y-dy),min(image.width,r+dx+1),min(image.height,b+dy+1)))


def neighboring(a,b):
    x,y,r,bottom = a.bbox
    l,t,right,end = b.bbox
    width = min(r-x,right-l)
    return (.45*width < abs(x+r-l-right)/2 < 2*width
            and min(bottom,end)-max(y,t)>.50*min(bottom-y,end-t))


def read_pairs(image, boxes, reader):
    candidates = [b for b in boxes if len(b.text)<=2 and b.confidence<.60
                  and 60<=b.bbox[2]-b.bbox[0]<320 and 110<=b.bbox[3]-b.bbox[1]<570]
    if len(candidates)>12:
        return {}
    readings = {}
    for b in candidates:
        if not any(other is not b and neighboring(b,other) for other in candidates):
            continue
        chips = list(views(image,b.poly))
        density = np.mean(np.asarray(chips[0]).max(2)<160)
        if not .025<density<.65:
            continue
        first = reader(chips[0])
        if not isinstance(first,str) or not re.fullmatch('[ァ-ヺー]{2,9}ッ',first):
            continue
        if all(reader(c)==first for c in chips[1:]):
            readings[b.box_id] = (first,[image_fingerprint(c) for c in chips])
    changed = {}
    for b in candidates:
        if b.box_id not in readings:
            continue
        word,hashes = readings[b.box_id]
        for other in candidates:
            if other is b or other.box_id not in readings or not neighboring(b,other):
                continue
            other_word = readings[other.box_id][0]
            motif = repeat(word[:-1]) or repeat(other_word[:-1])
            if motif and all(s==motif or repeat(s)==motif for s in (word[:-1],other_word[:-1])):
                replacement = Text_Box(b.box_id,b.confidence,b.poly,word,review=1)
                replacement.reason = 'corroborated_repeated_effect'
                replacement.selected_engine = OCREngine.HAYAI
                replacement.ocr_metadata = {**b.ocr_metadata,'candidates':{**b.ocr_metadata.get('candidates',{}),'hayai':word}}
                replacement.ocr_metadata['repeated_effect'] = dict(
                    original_text=b.text,original_status=b.status.value,word=word,neighbor=other.box_id,neighbor_word=other_word,view_hashes=hashes)
                changed[b.box_id] = replacement
                break
    return changed


def bound_pair(lines, image, detections=()):
    """Recheck both independent detector footprints before using a low-confidence source."""
    from ..translation.contracts import strict_json
    if not lines or any(r['reason']!='corroborated_repeated_effect' for r in lines):
        return False
    if len(lines)==1:
        # Each printed effect is lettered independently. Its sibling is still
        # required to corroborate the transcription in the original page.
        row = lines[0]
        sibling = strict_json(row['metadata_json']).get('repeated_effect',{}).get('neighbor')
        other = next((r for r in detections if r['source_id']==sibling),None)
        if other is None or other.get('reason')!='corroborated_repeated_effect':
            return False
        from ..translation.contracts import canonical
        lines = [row,dict(other,metadata_json=canonical(other['ocr_metadata']))]
    by_id = {r['source_id']:r for r in lines}
    for r in lines:
        proof = strict_json(r['metadata_json']).get('repeated_effect') or {}
        other = by_id.get(proof.get('neighbor'))
        if (not other or other is r or proof.get('word')!=r['text'] or proof.get('neighbor_word')!=other['text']
                or proof.get('view_hashes')!=[image_fingerprint(c) for c in views(image,r['polygon'])]):
            return False
    return True


def extended_footprint(gray, bounds):
    """Follow separated dark strokes below a partially recognized vertical kana."""
    x,y,r,b = bounds
    w,h = r-x+1,b-y+1
    l,t = max(0,x-w//2),max(0,y-h//5)
    patch = gray[t:min(gray.shape[0],b+2*h),l:min(gray.shape[1],r+w)]
    _,_,stats,_ = cv2.connectedComponentsWithStats(np.uint8(patch<145))
    strokes = sorted(((l+a,t+c,l+a+cw,t+c+ch) for a,c,cw,ch,area in stats[1:]
                     if a>0 and c>0 and a+cw<patch.shape[1] and c+ch<patch.shape[0]
                     and .14*w<cw<=1.3*w and 2<=ch<=1.25*h and area>=.13*w),
                     key=lambda box:box[1])
    parts = [v for v in strokes if v[1]<b and v[3]>y and x-.2*w<(v[0]+v[2])/2<r+.2*w]
    if not parts:
        return
    end = max(v[3] for v in parts)
    for v in strokes:
        if v[1]<b-2 or v in parts:
            continue
        if v[1]-end>.46*h:
            break
        middle = (parts[-1][0]+parts[-1][2])/2
        if abs((v[0]+v[2])/2-middle)<.85*w:
            parts.append(v)
            end=max(end,v[3])
    if end-b < .85*h:
        return
    return (max(0,min(v[0] for v in parts)-3),max(0,min(v[1] for v in parts)-3),
            min(gray.shape[1],max(v[2] for v in parts)+3),min(gray.shape[0],end+3))


def printed_twins(image, boxes, reader):
    """Extend a clipped effect only when an independent copy confirms its whole reading."""
    from PIL import ImageOps
    if not boxes or any(type(b.box_id) is not int for b in boxes):
        return {}
    seeds = [b for b in boxes if re.fullmatch(r'[ァ-ヺ]',b.text) and b.confidence<.85 and
             13<=b.bbox[2]-b.bbox[0]<=75 and 16<=b.bbox[3]-b.bbox[1]<=95]
    if not seeds:
        return {}
    gray = np.asarray(image.convert('L'))
    changes = {}
    next_id = max(v.box_id for v in boxes)+1
    for seed in seeds[:6]:
        x,y,r,b = seed.bbox
        w,h = r-x+1,b-y+1
        left,top = max(0,x-8*w),max(0,y-h//2)
        area = np.uint8(gray[top:b+h//2+1,left:r+8*w+1]<145)
        glyph = np.uint8(gray[y:b+1,x:r+1]<145)
        if not .035<glyph.mean()<.6 or min(area.shape)<1 or any(a<g for a,g in zip(area.shape,glyph.shape)):
            continue
        matches = cv2.matchTemplate(area,glyph,cv2.TM_CCOEFF_NORMED)
        matches[:,max(0,x-left-2*w):x-left+2*w+1] = 0
        _,score,_,(dx,dy) = cv2.minMaxLoc(matches)
        if score<.55:
            continue
        sx,sy = left+dx,top+dy
        footprints = [extended_footprint(gray,(a,c,a+w-1,c+h-1)) for a,c in ((x,y),(sx,sy))]
        if not all(footprints):
            continue
        readings = []
        existing = None
        for idx,box in enumerate(footprints):
            a,c,right,end = box
            overlaps = []
            for other in [*boxes,*changes.values()]:
                if other is seed:
                    continue
                ox,oy,orr,ob = other.bbox
                shared = max(0,min(right,orr+1)-max(a,ox))*max(0,min(end,ob+1)-max(c,oy))
                if shared:
                    overlaps.append((other,shared/max((right-a)*(end-c),(orr+1-ox)*(ob+1-oy))))
            if overlaps:
                if idx==0 or len(overlaps)!=1 or overlaps[0][1]<.7:
                    break
                existing = overlaps[0][0]
            chip = image.crop(box)
            try:
                word = reader(chip)
                if (not isinstance(word,str) or not word.startswith(seed.text) or not re.fullmatch(r'[ァ-ヺー]{2,6}',word)
                        or (existing is not None and existing.text!=word)
                        or reader(ImageOps.expand(chip,border=3,fill='white'))!=word):
                    break
            except (OSError, RuntimeError, ValueError):
                break
            readings.append(word)
        if len(readings)!=2 or readings[0]!=readings[1]:
            continue
        peer = existing.box_id if existing is not None else next_id
        for idx,(identity,(a,c,right,end)) in enumerate(zip((seed.box_id,peer),footprints)):
            if idx and existing is not None:
                break
            member=Text_Box(identity,seed.confidence,[(a,c),(right-1,c),(right-1,end-1),(a,end-1)],readings[idx],review=1)
            member.selected_engine=OCREngine.HAYAI
            member.reason='corroborated_printed_twin'
            member.ocr_metadata={**(seed.ocr_metadata if not idx else {}),'candidates':{
                **(seed.ocr_metadata.get('candidates',{}) if not idx else {}),'hayai':member.text},'printed_twin':{
                'seed_id':seed.box_id,'seed_text':seed.text,'original_polygon':seed.poly.tolist(),
                'peer':peer if idx==0 else seed.box_id,'crop_sha256':image_fingerprint(image.crop(footprints[idx]))}}
            changes[identity]=member
        if existing is None:
            next_id+=1
    return changes


def curved_repeats(image, boxes, reader):
    """Join a sound that turns a corner, when both printed segments account for the full reading."""
    from .crop_readings import views as group_views
    changes = {}
    heads = [box for box in boxes if re.fullmatch(r'[ぁ-ゖァ-ヺー]{1,5}([ぁ-ゖァ-ヺ])\1',box.text)
             and box.bbox[2]-box.bbox[0]>1.35*(box.bbox[3]-box.bbox[1])]
    tails = [box for box in boxes if box.confidence<.85 and re.fullmatch(r'([0-9ぁ-ゖァ-ヺ])\1{1,3}',box.text)
             and box.bbox[3]-box.bbox[1]>1.2*(box.bbox[2]-box.bbox[0])]
    for head in heads[:8]:
        choices = []
        for tail in tails:
            if tail is head or tail.bbox[1]<=head.bbox[1] or tail.bbox[0]<=sum(head.bbox[::2])/2:
                continue
            shared,_ = cv2.intersectConvexConvex(head.poly,tail.poly)
            area = min(cv2.contourArea(head.poly),cv2.contourArea(tail.poly))
            if .025*area<shared<.40*area:
                choices.append(tail)
        if len(choices)!=1:
            continue
        tail = choices[0]
        points = np.vstack([head.poly,tail.poly])
        bounds = (*np.floor(points.min(0)).astype(int).tolist(),*np.ceil(points.max(0)).astype(int).tolist())
        glyph = head.text[-1]
        try:
            ending = reader(next(views(image,tail.poly)))
            if not isinstance(ending,str) or not re.fullmatch(re.escape(glyph)+r'{2,4}',ending):
                continue
            leading = reader(next(views(image,head.poly)))
            if not isinstance(leading,str):
                continue
            prefix = leading.rstrip('っッ')
            if (len(prefix)!=len(head.text) or prefix[-2:]!=glyph*2
                    or sum(a!=b for a,b in zip(prefix,head.text))>1):
                continue
            passage = prefix+ending
            if any(reader(view)!=passage for view in group_views(image,bounds,[head.poly,tail.poly])):
                continue
        except (ValueError,OSError,RuntimeError):
            continue
        anchor = deepcopy(head)
        anchor.ocr_metadata['curved_repeat'] = dict(ids=[head.box_id,tail.box_id],reading=passage)
        changes[head.box_id] = anchor
    return changes
