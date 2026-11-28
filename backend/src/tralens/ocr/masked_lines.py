"""Read lettering on a second detector pass with nearby artwork suppressed."""
import re
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from . import detached_kana, reader_geometry
from .automatic_units import propose_page_from_boxes
from .crop_readings import views
from .recognition_units import image_fingerprint
from .sound_readings import repeat
from .types import OCREngine, Text_Box


WORD = re.compile(r'[ぁ-ゖァ-ヺー一-鿿々]{2,22}')
READ_ERRORS = (ValueError, OSError, RuntimeError, cv2.error)


def source_views(image, polygons):
    points = np.vstack(polygons)
    # Paddle can place a vertex on the page boundary. PIL's crop already
    # treats that coordinate as the exclusive end.
    right,bottom = points.max(0).astype(int).tolist()
    bounds = (*points.min(0).astype(int).tolist(),
              right if right == image.width else right+1,
              bottom if bottom == image.height else bottom+1)
    return views(image, bounds, polygons)


def read_line(image, isolated, poly, native, score, reader):
    mapping = reader_geometry.plan(image.size, poly)
    word = reader(reader_geometry.pixels(isolated, mapping))
    if not isinstance(word, str) or not WORD.fullmatch(word):
        return
    original = reader_geometry.pixels(image, mapping)
    if reader(original) != word:
        return
    if score >= .45 and native == word and reader(source_views(image, [poly])[0]) == word:
        return word
    motif = repeat(word)
    if not motif or word != motif*2 or min(original.size) < 16:
        return
    # On drawings two parallel strokes alone can look like kana. A repeated
    # inscription must give the same multi-kana beat on each separate half.
    w, h = original.size
    if max(w,h) < 1.4*min(w,h):
        return
    halves = ((0,0,w,h//2),(0,h//2,w,h)) if h>w else ((0,0,w//2,h),(w//2,0,w,h))
    if all(reader(original.crop(box)) == motif for box in halves):
        return word


def corrected(identity, score, poly, word, paddle, previous=None, *, masked=False):
    member = Text_Box(identity, score, poly, word, review=1)
    member.reason = 'corroborated_mask_line' if masked else 'corroborated_native_line'
    member.selected_engine = OCREngine.HAYAI
    prior = previous.ocr_metadata if previous else {}
    member.ocr_metadata = {
        **prior, 'candidates': {**prior.get('candidates', {}), 'masked_paddle' if masked else 'paddle': paddle, 'hayai': word},
        'masked_line' if masked else 'native_line': {'reading': word, 'original_text': previous.text if previous else None,
                        'original_polygon': previous.poly.tolist() if previous else None}}
    return member


def bind(lines, reading, crops, direction):
    lines[0].ocr_metadata['masked_passage'] = {
        'ids': [v.box_id for v in lines], 'reading': reading, 'direction': direction,
        'view_hashes': [image_fingerprint(crop) for crop in crops]}


def read_predicate(image, reader, head, rest):
    polygons = [v.poly for v in rest]
    tail = [reader(v) for v in source_views(image, polygons)]
    if len(set(tail)) != 1 or not re.fullmatch(r'[ぁ-ゖァ-ヺ一-鿿]{2,15}[！!？?]?', tail[0] or ''):
        return
    combined = source_views(image, [head.poly, *polygons])
    readings = [reader(v) for v in combined]
    whole = head.text + tail[0]
    if readings[:2] != [whole,whole] or (readings[2] or '').rstrip('!！?？') != whole.rstrip('!！?？'):
        return
    prefix = reader(source_views(image, polygons[:1])[0])
    if not prefix or not tail[0].startswith(prefix) or len(prefix) >= len(tail[0]):
        return
    parts = [corrected(v.box_id, v.confidence, v.poly, part, v.paddle_text, v)
             for v,part in zip(rest, (prefix,tail[0][len(prefix):]))]
    bind([head,*parts], whole, combined, 'vertical')
    return parts


def crossed_rows(image, isolated, reader, original, rows):
    """Two tall intersecting boxes can actually contain two lines read across."""
    rows = sorted(rows, key=lambda row: row[2][:,1].mean())
    polygons = [p for _,_,p in rows]
    widths, heights = np.array([np.ptp(p,axis=0) for p in polygons]).T
    dx, dy = np.abs(polygons[0].mean(0) - polygons[1].mean(0))
    if (any(widths < 1.4*heights) or dx > .25*min(widths)
            or not .55*min(heights) <= dy <= 1.5*max(heights)):
        return
    words = []
    for paddle,score,poly in rows:
        mapping = reader_geometry.plan(image.size, poly)
        word = reader(reader_geometry.pixels(isolated, mapping))
        if (score < .6 or not WORD.fullmatch(word or '') or reader(reader_geometry.pixels(image, mapping)) != word
                or any(reader(crop) != word for crop in source_views(image,[poly]))):
            return
        words.append(word)
    if not any(row[0] == word for row,word in zip(rows,words)):
        return
    whole = ''.join(words)
    combined = source_views(image, polygons)
    if any(reader(crop) != whole for crop in [*combined,*source_views(image,[v.poly for v in original])]):
        return
    lines = [corrected(v.box_id, score, poly, word, paddle, v, masked=True)
             for v,(paddle,score,poly),word in zip(original,rows,words)]
    bind(lines, whole, combined, 'horizontal')
    return lines


def slanted_continuation(image, reader, detector, head, neighbors, next_id):
    """The native bounding rectangle of an angled line can include the next column."""
    poly = head.poly
    if poly.shape!=(4,2) or not .38<head.confidence<.87 or not WORD.fullmatch(head.text):
        return
    x,y,r,b = head.bbox
    w,h = r-x+1,b-y+1
    def slope(points):
        top,base = np.array_split(points[np.argsort(points[:,1])],2)
        dx,dy = base.mean(0)-top.mean(0)
        return dx/max(1,dy)
    angle = slope(poly)
    if not (45<=w<=240 and 1.3*w<h<3.2*w and .20<abs(angle)<.63):
        return
    context = reader(image.crop((x,y,r+1,b+1)))
    if not isinstance(context,str) or not context.startswith(head.text):
        return
    ending = context[len(head.text):].rstrip('!！?？')
    if not WORD.fullmatch(ending) or len(ending)>10:
        return
    head_read = reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,poly)))
    if head_read!=head.text:
        return
    l,t = max(0,x-round(.32*w)),max(0,y-round(.25*h))
    bounds = (l,t,min(image.width,r+round(.5*w)),min(image.height,b+round(.9*h)))
    result = detector.predict(np.array(image.crop(bounds)))
    if not result:
        return
    candidates = []
    for points,paddle,score in zip(result[0]['rec_polys'],result[0]['rec_texts'],result[0]['rec_scores']):
        side = np.float32(points)+np.float32([l,t])
        if side.shape!=(4,2) or score<.3 or abs(slope(side)-angle)>.25:
            continue
        dx,dy = poly.mean(0)-side.mean(0)
        if not (.20*w<dx<.93*w and abs(dy)<.45*h and max(side[:,1])-min(side[:,1])>.55*h):
            continue
        if any(cv2.intersectConvexConvex(cv2.convexHull(v.poly),side)[0] > .04*cv2.contourArea(side)
               for v in neighbors if v is not head):
            continue
        reads = [reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,side,view=view)))
                 for view in ('tight','axial','context')]
        if not all(isinstance(v,str) and v.rstrip('!！?？')==ending for v in reads):
            continue
        crops = source_views(image,[poly,side])
        whole = head.text+ending
        if all((reader(crop) or '').rstrip('!！?？')==whole for crop in crops):
            first = corrected(head.box_id,head.confidence,poly,head.text,head.paddle_text,head)
            tail = corrected(next_id,float(score),side,ending,paddle)
            bind([first,tail],whole,crops,'vertical')
            candidates.append((first,tail))
    if len(candidates)==1:
        return candidates[0]


def discover(engine, image, reader, probability):
    boxes = engine.raw_detections[0]
    if len(boxes)>256 or any(type(b.box_id) is not int for b in boxes):
        return
    support = cv2.dilate(np.uint8(probability>28),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))) != 0
    pixels = np.where(support[:,:,None],np.asarray(image),255).astype(np.uint8)
    result = engine.ocr.predict(pixels.copy())
    output = result[0] if result else {}
    isolated = Image.fromarray(pixels)
    detected = list(boxes)
    by_id = {v.box_id:v for v in boxes}
    batch = engine.automatic_unit_proposals[0]
    proposals = batch.proposals if batch is not None else ()
    pairs = {frozenset(p.main_ids):[by_id[i] for i in p.main_ids]
             for p in proposals
             if len(p.main_ids)==2 and p.main_ids==p.member_ids and p.direction=='vertical'
             and any(not by_id[i].text for i in p.main_ids) and all(by_id[i].confidence<.85 for i in p.main_ids)}
    row_candidates = defaultdict(list)
    next_id = max(by_id,default=-1)+1
    attempts = 0
    for text,score,points in zip(output.get('rec_texts', ()), output.get('rec_scores', ()), output.get('rec_polys', ())):
        poly = np.asarray(points,dtype=np.float32)
        if poly.shape!=(4,2) or not np.isfinite(poly).all():
            continue
        area = cv2.contourArea(poly)
        w,h = np.ptp(poly,axis=0)
        if area<160 or min(w,h)<10 or max(w,h)>460:
            continue
        overlaps = []
        for other in detected:
            shared,_ = cv2.intersectConvexConvex(poly,cv2.convexHull(other.poly))
            if shared>.01*area:
                overlaps.append((other,shared/area,shared/max(1,cv2.contourArea(other.poly))))
        linked = frozenset(v.box_id for v,a,_ in overlaps if a>.025)
        blank = None
        if len(overlaps)==1 and not overlaps[0][0].text and min(overlaps[0][1:])>.67:
            blank = overlaps[0][0]
        elif any(a>.18 or b>.45 for _,a,b in overlaps):
            if (linked in pairs and len(overlaps)==2 and w>1.4*h and score>=.6
                    and all(a>.30 and b>.22 for _,a,b in overlaps)):
                row_candidates[linked].append((text,float(score),poly))
            continue
        # Bound the number of crop-recognition attempts on dense illustrations.
        normal = score>=.45 and WORD.fullmatch(text or '')
        if attempts>=16 or (not normal and (min(w,h)<35 or max(w,h)<1.35*min(w,h))):
            continue
        attempts += 1
        try:
            geometry = blank.poly if blank else poly
            word = read_line(image,isolated,geometry,text,score,reader)
            if not word and blank and not np.array_equal(poly,geometry):
                geometry = poly
                word = read_line(image,isolated,geometry,text,score,reader)
            if not word:
                continue
            member = corrected(blank.box_id if blank else next_id,float(score),geometry,word,text,blank,masked=True)
            if blank:
                detected[detected.index(blank)] = member
            else:
                next_id += 1
                detected.append(member)
                rest = pairs.get(linked)
                if rest and all(v in detected for v in rest):
                    parts = read_predicate(image,reader,member,rest)
                    if parts:
                        for before,after in zip(rest,parts):
                            detected[detected.index(before)] = after
        except READ_ERRORS:
            continue
    for ids,rows in list(row_candidates.items())[:4]:
        rest = pairs[ids]
        if len(rows)!=2 or not all(v in detected for v in rest):
            continue
        try:
            lines = crossed_rows(image,isolated,reader,rest,rows)
        except READ_ERRORS:
            continue
        if lines:
            for before,after in zip(rest,lines):
                detected[detected.index(before)] = after
    singles = {p.main_ids[0] for p in proposals if len(p.member_ids)==1}
    for head in [v for v in boxes if v.box_id in singles and v in detected]:
        try:
            parts = slanted_continuation(image,reader,engine.ocr,head,detected,next_id)
        except READ_ERRORS:
            continue
        if parts:
            detected[detected.index(head)] = parts[0]
            detected.append(parts[1])
            next_id += 1
    seeds=[v for v in boxes if v.box_id in singles and v in detected]
    for head in ([v for v in seeds if re.fullmatch(r'[一-鿿]',v.text)][:12]
                 +[v for v in seeds if re.fullmatch(r'[ぁ-ゖァ-ヺ]',v.text)][:8]):
        try:
            aside=handwritten_aside(image,probability,head,detected,reader)
        except READ_ERRORS:
            continue
        if aside is not None:
            detected[detected.index(head)]=aside
    detected = colored_effects(image, probability, detected, reader)
    from .sideways_sound import pale_effect
    for i, box in enumerate(detected):
        if box.confidence < .65:
            found = pale_effect(image, box, reader)
            if found is not None:
                detected[i] = found
    detected = nearby_repeated_kana(image, probability, detected, reader)
    next_id = max((v.box_id for v in detected), default=-1)+1
    detected.extend(drawn_kana_rows(image, probability, detected, reader, next_id))
    next_id = max((v.box_id for v in detected), default=-1)+1
    readings = detached_kana.readings(image, probability, [v.poly for v in detected], reader)
    for identity, (word, bounds, score, island, dot) in enumerate(readings, next_id):
        left, top, right, bottom = bounds
        poly = np.float32([[left, top], [right-1, top], [right-1, bottom-1], [left, bottom-1]])
        member = corrected(identity, score, poly, word, '', masked=True)
        member.ocr_metadata['detached_kana'] = {'reading': word, 'source_bounds': list(bounds), 'island': list(island)}
        if dot:
            member.ocr_metadata['detached_kana']['terminal_dot'] = list(dot)
        detected.append(member)
    if detected!=boxes:
        batch = propose_page_from_boxes(detected,engine.image_path,complementary_ownership=True,regroup_companions=True)
        engine.raw_detections,engine.automatic_unit_proposals = [detected],[batch]


def painted_outline(pixels, probability):
    _, components, stats, _ = cv2.connectedComponentsWithStats(np.uint8(pixels.max(2)<100))
    votes = np.bincount(components[probability>20],minlength=len(stats))
    retained = (votes>.32*stats[:,4]) & (stats[:,4]>=3)
    retained[0]=False
    return retained[components] if 2<=retained.sum()<=55 else None


def colored_effects(image, probability, boxes, reader):
    """Read weak colored kana from both the original paint and the isolated outline."""
    from PIL import ImageOps
    result = list(boxes)
    attempts = 0
    for head in boxes:
        if (head not in result or not head.text or head.confidence >= .75 or len(head.text)>7
                or not re.search(r'[ぁ-ゖァ-ヺ]',head.text)):
            continue
        x,y,r,b=head.bbox
        w,h=r-x+1,b-y+1
        if not (25<=min(w,h) and 65<=max(w,h)<=320):
            continue
        parts=[head]
        for other in result:
            if other is head or other.text:
                continue
            a,t,z,d=other.bbox
            if (b>t>y and d>b and abs((a+z-x-r)/2)<.4*w and z-a<1.4*w and d-t<1.2*h):
                parts.append(other)
        if len(parts)>2:
            continue
        points=np.vstack([v.poly for v in parts])
        left,top=np.maximum(0,points.min(0).astype(int)-8)
        right,bottom=np.minimum(image.size,points.max(0).astype(int)+9)
        bounds=tuple(map(int,(left,top,right,bottom)))
        native=image.crop(bounds)
        pixels=np.asarray(native)
        if np.mean(np.ptp(pixels,axis=2)>18)<.07:
            continue
        gate=probability[top:bottom,left:right]
        ink=painted_outline(pixels,gate)
        if ink is None or not .012<=ink.mean()<=.35:
            continue
        yy,xx=np.where(ink)
        if np.ptp(xx)<.40*w or np.ptp(yy)<.40*h:
            continue
        if attempts>=12:
            break
        attempts+=1
        try:
            binary=Image.fromarray(np.uint8(~ink)*255).convert('RGB')
            isolated=ImageOps.expand(binary.crop((int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))),
                                     border=5,fill='white')
            word=reader(isolated)
            if (not isinstance(word,str) or not re.fullmatch(r'[ぁ-ゖァ-ヺー]{3,10}[.。!！?？…]{0,4}',word)
                    or word==head.text or reader(native)!=word
                    or reader(native.rotate(-12,expand=True,fillcolor='white'))!=word):
                continue
        except READ_ERRORS:
            continue
        hull=cv2.convexHull(np.float32(np.column_stack((xx+left,yy+top)))).reshape(-1,2)
        if any(v not in parts and cv2.intersectConvexConvex(hull,cv2.convexHull(v.poly))[0]>3
               for v in result):
            continue
        polygon=np.clip(cv2.boxPoints(cv2.minAreaRect(hull)),[0,0],[image.width-1,image.height-1]).astype('float32')
        member=corrected(head.box_id,head.confidence,polygon,word,head.text,head,masked=True)
        member.ocr_metadata['painted_effect']={'reading':word,'bounds':list(bounds),'source_digest':image_fingerprint(native),
            'reads':[word]*3}
        result=[member if v is head else v for v in result if v is head or v not in parts]
    return result


def peer_strokes(pixels, probability):
    """Only include ink components supported by the local text probability."""
    gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
    _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(gray<146))
    hits=np.bincount(labels[probability>65],minlength=len(stats))
    keep=(hits>.58*stats[:,4])&(stats[:,4]>=2)&(stats[:,4]<.35*gray.size)
    keep[0]=False
    return keep[labels]


def nearby_repeated_kana(image, probability, boxes, reader):
    """Recover small repeated effects when separate native copies give the same reading."""
    from PIL import ImageOps
    seeds=[v for v in boxes if v.confidence<.70 and 'repeated_kana' not in v.ocr_metadata
           and re.fullmatch(r'[ぁ-ゖ]{2,4}',v.text)
           and 15<=v.bbox[2]-v.bbox[0]<=90 and 18<=v.bbox[3]-v.bbox[1]<=100]
    if not seeds or max(image.size)>4096:
        return boxes
    grouped=cv2.dilate(np.uint8(probability>=110),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
    _,_,components,_=cv2.connectedComponentsWithStats(grouped)
    readings=defaultdict(list)
    attempts=0
    for x,y,w,h,_ in components[1:]:
        if not (25<=w<=92 and 28<=h<=110 and x>=1 and y>=1):
            continue
        l,t,r,b=map(int,(x+3,y+3,x+w-3,y+h-3))
        cx,cy=(l+r)/2,(t+b)/2
        if not any(abs(cx-(v.bbox[0]+v.bbox[2])/2)<6*w and abs(cy-(v.bbox[1]+v.bbox[3])/2)<2*h for v in seeds):
            continue
        owners=[v for v in boxes if min(r,v.bbox[2])-max(l,v.bbox[0])>1 and min(b,v.bbox[3])-max(t,v.bbox[1])>1]
        if len(owners)>1 or (owners and owners[0] not in seeds):
            continue
        native=image.crop((l,t,r,b))
        ink=peer_strokes(np.asarray(native),probability[t:b,l:r])
        if not .035<=ink.mean()<=.38:
            continue
        attempts+=1
        isolated=Image.fromarray(np.uint8(~ink)*255).convert('RGB')
        try:
            word=reader(isolated)
            if (isinstance(word,str) and re.fullmatch(r'[ぁ-ゖ]{2,4}',word)
                    and reader(native)==word and reader(ImageOps.expand(isolated,border=3,fill='white'))==word):
                readings[word].append(((l,t,r,b), owners[0] if owners else None, ink, image_fingerprint(native)))
        except READ_ERRORS:
            pass
        if attempts>=24:
            break
    result=list(boxes)
    next_id=max(v.box_id for v in boxes)+1
    for word,copies in readings.items():
        # Ordinary typeset dialogue can repeat the same short response in two
        # balloons. An angled source or an independently found unboxed copy
        # distinguishes the lettering this pass is intended to recover.
        if not any(v is None or abs((cv2.minAreaRect(v.poly)[2]+45)%90-45)>6 for _,v,_,_ in copies):
            continue
        for (l,t,r,b),owner,ink,digest in copies:
            peers=[c for c in copies if c[0]!=(l,t,r,b) and abs(c[0][0]-l)<7*(r-l)
                   and abs(c[0][1]-t)<2.5*(b-t) and .55<(c[0][3]-c[0][1])/(b-t)<1.8]
            if owner and owner not in result:
                continue
            if not peers or not any(v and set(v.text)&set(word) for v in [owner,*[c[1] for c in peers]]):
                continue
            yy,xx=np.where(ink)
            points=np.float32(np.column_stack((xx+l,yy+t)))
            poly=cv2.boxPoints(cv2.minAreaRect(points))
            member=corrected(owner.box_id if owner else next_id,owner.confidence if owner else .75,
                             poly,word,owner.text if owner else '',owner,masked=True)
            member.ocr_metadata['repeated_kana']={'reading':word,'bounds':[l,t,r,b],
                'source_digest':digest,'reads':[word]*3,'peer_bounds':list(peers[0][0]),'peer_digest':peers[0][3]}
            if owner:
                result[result.index(owner)]=member
            else:
                result.append(member)
                next_id+=1
    return result


def drawn_kana_ink(ink, probability):
    """Separate the two end glyphs and the long middle stroke from the artwork."""
    _, labels, parts, _ = cv2.connectedComponentsWithStats(np.uint8(ink))
    keep = []
    for j,(x,y,w,h,area) in enumerate(parts[1:],1):
        if (min(w,h)>=4 and area>=40 and x>0 and y>0 and x+w<ink.shape[1] and y+h<ink.shape[0]
                and np.mean(probability[labels==j]>12)>.6):
            keep.append(j)
    if len(keep)!=3:
        return None
    keep.sort(key=lambda j:parts[j,0])
    a,bar,z=parts[keep]
    if (bar[2]>3*bar[3] and a[0]+a[2]<=bar[0] and bar[0]+bar[2]<=z[0]
            and min(a[3],z[3])>=1.4*bar[3]):
        return np.isin(labels,keep)
    return None


def drawn_kana_rows(image, probability, boxes, reader, next_id):
    """Look for widely spaced kana where the text mask fires but Paddle found no line."""
    if max(image.size) > 4096:
        return []
    gray = np.asarray(image.convert('L'))
    claimed = np.zeros_like(gray)
    for box in boxes:
        cv2.fillPoly(claimed, [np.int32(box.poly)], 1)
    seeds = (probability > 12) & (gray < 160) & (claimed == 0)
    grouped = cv2.dilate(np.uint8(seeds), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 23)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped)
    result = []
    attempts = 0
    for i in range(1, count):
        x,y,w,h,_ = map(int,stats[i])
        if not (60 <= w <= 410 and 20 <= h <= 140):
            continue
        region = np.s_[y:y+h,x:x+w]
        yy,xx = np.nonzero((labels[region] == i) & seeds[region])
        if not xx.size:
            continue
        l,t,r,b = x+int(xx.min()), y+int(yy.min()), x+int(xx.max())+1, y+int(yy.max())+1
        width,height = r-l,b-t
        if not (60<width<350 and 16<height<110 and 2<width/height<8 and
                xx.size>.022*width*height and probability[t:b,l:r].max()>=140 and
                not claimed[t:b,l:r].any() and l>=8 and t>=8 and r+8<image.width and b+8<image.height):
            continue
        region = np.s_[t-8:b+8,l-8:r+8]
        ink = gray[region] < 140
        mask = drawn_kana_ink(ink, probability[region])
        if mask is None:
            continue
        gx,gy,gw,gh = cv2.boundingRect(mask.astype(np.uint8))
        left,top = int(l-8+gx),int(t-8+gy)
        rect = (max(0,left-8),max(0,top-8),min(image.width,left+gw+8),min(image.height,top+gh+8))
        native = image.crop(rect)
        isolated = Image.new('RGB',native.size,'white')
        isolated.paste(Image.fromarray(np.uint8(~mask)*255), (l-8-rect[0], t-8-rect[1]))
        attempts += 1
        try:
            word = reader(isolated)
            if (isinstance(word,str) and re.fullmatch(r'[ぁ-ゖァ-ヺ][ー〜～]{1,3}[ぁ-ゖァ-ヺ][っッ]?',word)
                    and reader(native)==word):
                highlight = Image.new('L', native.size)
                highlight.paste(Image.fromarray(np.uint8(mask)*255), (l-8-rect[0], t-8-rect[1]))
                halo=cv2.dilate(np.asarray(highlight),np.ones((7,7),np.uint8))>0
                painted = Image.fromarray(np.where(halo[:,:,None],np.asarray(native),245).astype('uint8'))
                if reader(painted)==word:
                    poly = np.float32([[left-2,top-2],[left+gw+2,top-2],[left+gw+2,top+gh+2],[left-2,top+gh+2]])
                    member=corrected(next_id+len(result),.75,poly,word,'',masked=True)
                    member.ocr_metadata['drawn_kana'] = {'reading':word,'source_bounds':list(rect),
                        'views':[image_fingerprint(v) for v in (native,isolated,painted)]}
                    result.append(member)
        except READ_ERRORS:
            pass
        if attempts>=8:
            break
    return result


def diagonal_ink(gray, probability, scale, *, touching=False):
    from .handwritten import isolated_ink
    ink=isolated_ink(gray, scale) & (cv2.dilate(np.uint8(probability > 12), np.ones((11,11), np.uint8)) != 0)
    return ink | ((gray<150) & (probability>=140)) if touching else ink


def handwritten_aside(image, probability, box, neighbors, reader):
    """Follow lettering down and right from a single detected glyph on pale paper."""
    x,y,r,b=box.bbox
    scale=b-y+1
    kana=bool(re.fullmatch(r'[ぁ-ゖァ-ヺ]',box.text))
    if (not (kana or re.fullmatch(r'[一-鿿]',box.text)) or box.confidence<.88 or not 17<=scale<=55
            or not .5*scale<=r-x<=1.45*scale):
        return None
    pad,dx,dy=(.12,1.25,1.55) if kana else (.25,3.6,2.8)
    left,top=max(0,x-round(pad*scale)),max(0,y-round(pad*scale))
    right,bottom=min(image.width,r+round(dx*scale)),min(image.height,b+round(dy*scale))
    native=image.crop((left,top,right,bottom))
    gray=np.asarray(native.convert('L'))
    if np.mean(gray>235)<.4 or np.mean(np.ptp(np.asarray(native),axis=2)>16)>.025:
        return None
    ink = diagonal_ink(gray, probability[top:bottom,left:right], scale, touching=kana)
    yy,xx=np.nonzero(ink)
    if len(xx)<40 or not ((1.25 if kana else 1.65)*scale < np.ptp(xx) < 4*scale and 1.45*scale < np.ptp(yy) < 3.3*scale):
        return None
    ring=(cv2.dilate(np.uint8(ink),np.ones((7,7),np.uint8)) != 0) & ~ink
    if np.mean(gray[ring]>235)<.8:
        return None
    if np.corrcoef(xx,yy)[0,1]<.28 or int(xx.min())+left < x-.25*scale or int(yy.min())+top<y-.3*scale:
        return None
    hull=cv2.convexHull(np.float32(np.column_stack((xx+left,yy+top)))).reshape(-1,2)
    # Follow the lettering diagonally to keep adjacent figures outside the detection.
    polygon=cv2.boxPoints(cv2.minAreaRect(hull))
    if (polygon<0).any() or (polygon>=[image.width,image.height]).any():
        return None
    occupied=np.zeros_like(gray)
    for other in neighbors:
        if other is not box and cv2.intersectConvexConvex(cv2.convexHull(other.poly),polygon)[0]>0:
            cv2.fillPoly(occupied,[np.int32(other.poly-[left,top])],1)
    if kana:
        # A neighboring detector may already overlap the original anchor.
        # Do not extend the new reading into any further occupied pixels.
        cv2.fillPoly(occupied,[np.int32(box.poly-[left,top])],0)
    if np.any(ink & (occupied!=0)):
        return None
    if not ink[max(0,y-top):b-top+1,max(0,x-left):r-left+1].any():
        return None
    isolated=Image.fromarray(np.uint8(~ink)*255).convert('RGB')
    word=reader(isolated)
    pattern=r'([ぁ-ゖァ-ヺー]{2,3})\1[っつッツ]{0,2}' if kana else r'[一-鿿][ぁ-ゖァ-ヺ一-鿿ー]{3,18}[!?！?。…]?'
    if not isinstance(word,str) or not word.startswith(box.text) or not (match:=re.fullmatch(pattern,word)) or reader(native)!=word:
        return None
    if kana and reader(image.crop((x-3,y-3,x+round(1.18*scale),y+round(.95*scale))))!=match[1]:
        return None
    tight=isolated.crop((max(0,int(xx.min())-3),max(0,int(yy.min())-3),
                         min(isolated.width,int(xx.max())+4),min(isolated.height,int(yy.max())+4)))
    if reader(tight)!=word:
        return None
    member=corrected(box.box_id,box.confidence,polygon,word,box.paddle_text,box,masked=True)
    member.ocr_metadata['handwritten_aside']={'reading':word,'bounds':[left,top,right,bottom],
            'scale':scale,'views':[image_fingerprint(v) for v in (native,isolated,tight)]}
    if kana:
        member.ocr_metadata['handwritten_aside']['touching']=True
    return member
