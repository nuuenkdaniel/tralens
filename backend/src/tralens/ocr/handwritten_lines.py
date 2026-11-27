"""Recover adjacent handwritten rows when OCR stops on a Japanese connective."""
from difflib import SequenceMatcher
import re

import cv2
import numpy as np
from PIL import Image

from .automatic_units import propose_page_from_boxes
from .types import OCREngine, Text_Box


def lines(view, scale, detector, reader):
    zoom = view.resize((view.width*scale,view.height*scale),Image.Resampling.LANCZOS)
    results = detector.predict(np.asarray(zoom))
    if not results:
        return []
    found = []
    output = results[0]
    for polygon,native,score in zip(output['rec_polys'],output['rec_texts'],output['rec_scores']):
        if score<.65:
            continue
        pts = np.asarray(polygon)
        (x,y),(r,b) = pts.min(0),pts.max(0)
        if r-x<1.1*(b-y):
            continue
        chip = zoom.crop((max(0,x-2),max(0,y-2),min(zoom.width,r+3),min(zoom.height,b+3)))
        reading = reader(chip).strip()
        stem = reading.rstrip('ーっッ！!…') or reading
        if (re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿々っッ！!…]{1,24}',reading)
                and SequenceMatcher(None,native,stem).ratio()>=.62):
            found.append((pts.astype('float32')/scale,reading,float(score)))
    return found


def continuation(image, anchor, detector, reader):
    x,y,r,b = anchor.bbox
    h = b-y+1
    l,t = max(0,x-round(.25*h)),max(0,y-round(.25*h))
    chip = image.crop((l,t,min(image.width,r+round(1.1*h)),min(image.height,b+round(1.95*h))))
    first,second = (lines(chip,s,detector,reader) for s in (1,2))
    agreed = []
    for polygon,text,score in second:
        center = polygon.mean(0)
        matches = [v for p,v,_ in first if np.max(abs(p.mean(0)-center))<.25*h]
        if not matches:
            # A scale change can reveal a line the full-size detector misses.
            # Read that footprint again from native pixels with different padding.
            (a,c),(right,bottom) = polygon.min(0),polygon.max(0)
            native = chip.crop((max(0,int(a)-3),max(0,int(c)-3),
                                min(chip.width,int(right)+4),min(chip.height,int(bottom)+4)))
            matches = [reader(native).strip()]
        if len(matches)!=1 or matches[0].rstrip('っッ')!=text.rstrip('っッ'):
            continue
        other = matches[0]
        agreed.append((polygon+np.float32([l,t]),max((text,other),key=len),score))
    agreed.sort(key=lambda row:row[0][:,1].mean())
    if not 2<=len(agreed)<=4:
        return
    head = agreed[0]
    if (np.max(abs(head[0].mean(0)-anchor.poly.mean(0)))>.25*h
            or not re.search(r'ながら$|つつ$|けれども?$',head[1])
            or SequenceMatcher(None,anchor.text,head[1]).ratio()<.5):
        return
    for (upper,_,_),(lower,_,_) in zip(agreed,agreed[1:]):
        if (not .45*h<lower[:,1].mean()-upper[:,1].mean()<1.25*h
                or abs(lower[:,0].min()-head[0][:,0].min())>.6*h):
            return
    # Require a grammatical completion. A new label under an unfinished line
    # can be close enough to share the same detector window.
    if not re.search(r'(?:ないで|するな|しよう|したい|できる|ください|だった|でした)[ーっッ!！…。]*$',
                     ''.join(row[1] for row in agreed[1:])):
        return
    return agreed


def discover(engine,image,reader):
    boxes = engine.raw_detections[0]
    if not boxes or len(boxes)>256 or any(type(b.box_id) is not int for b in boxes):
        return
    old = engine.automatic_unit_proposals[0]
    singles = {p.main_ids[0] for p in old.proposals if len(p.member_ids)==1}
    anchors = [b for b in boxes if b.box_id in singles and b.bbox[2]-b.bbox[0]>1.6*(b.bbox[3]-b.bbox[1])
               and 22<=b.bbox[3]-b.bbox[1]<=110 and re.search(r'(?:な[かが]ら|つつ|けれども?)$',b.text)]
    for anchor in anchors[:4]:
        try:
            rows = continuation(image,anchor,engine.ocr,reader)
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
        if not rows or any(cv2.intersectConvexConvex(cv2.convexHull(b.poly),polygon)[0]>0
                for polygon,_,_ in rows for b in boxes if b is not anchor):
            continue
        batch = []
        for index,(poly,text,score) in enumerate(rows):
            member = Text_Box(anchor.box_id if index==0 else max(b.box_id for b in boxes)+index,score,poly,text,review=1)
            member.selected_engine = OCREngine.HAYAI
            member.reason = 'confirmed_handwritten_line'
            member.ocr_metadata = {'candidates':{'hayai':text},'handwritten_anchor':anchor.box_id,
                                   'line_index':index,'original_anchor_text':anchor.text}
            batch.append(member)
        candidates = [b for b in boxes if b is not anchor]+batch
        new = propose_page_from_boxes(candidates,engine.image_path,complementary_ownership=True,regroup_companions=True)
        members = tuple(b.box_id for b in batch)
        if not any(p.main_ids==members and p.member_ids==members for p in new.proposals):
            continue
        def other_parts(proposals):
            return {(p.main_ids,p.member_ids,p.annotation_ids) for p in proposals if anchor.box_id not in p.member_ids}
        if other_parts(new.proposals)==other_parts(old.proposals):
            engine.raw_detections, engine.automatic_unit_proposals = [candidates],[new]
            boxes,old = candidates,new


def full_width_annotation(engine,image,reader):
    """A full sentence above larger handwriting can be mistaken for phonetic ruby."""
    from . import reader_geometry
    from .masked_lines import bind,corrected,source_views
    batch = engine.automatic_unit_proposals[0]
    boxes = engine.raw_detections[0]
    by_id = {v.box_id:v for v in boxes}
    changed = list(boxes)
    for proposal in batch.proposals:
        if (proposal.direction!='horizontal' or len(proposal.member_ids)!=2
                or len(proposal.annotation_parents)!=1):
            continue
        child_id,parent_id = proposal.annotation_parents[0]
        child,parent = by_id[child_id],by_id[parent_id]
        if (len(child.text)<8 or len(re.findall('[一-鿿]',child.text))<2
                or not re.search('[がをにはで]',child.text) or min(child.confidence,parent.confidence)<.75):
            continue
        x,y,r,b = child.bbox
        l,t,right,_ = parent.bbox
        if not (abs(x-l)<.15*(r-x) and abs(r-right)<.15*(r-x) and y<t and abs(t-b)<.35*(b-y)):
            continue
        try:
            rows = [child,parent]
            words = []
            for row in rows:
                reads = [reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,row.poly,view=v)))
                         for v in ('tight','axial')]
                if (not reads[0] or reads[0]!=reads[1] or SequenceMatcher(None,row.text,reads[0]).ratio()<.64):
                    break
                words.append(reads[0])
            if len(words)!=2:
                continue
            crops = source_views(image,[row.poly for row in rows])
            reads = [reader(crop) for crop in crops]
            whole = ''.join(words)
            if (not all(isinstance(v,str) and v==reads[0] for v in reads)
                    or not reads[0].endswith(words[0][-4:]+words[1])
                    or SequenceMatcher(None,whole,reads[0]).ratio()<.87):
                continue
            parts = [corrected(row.box_id,row.confidence,row.poly,word,row.paddle_text,row) for row,word in zip(rows,words)]
            bind(parts,whole,crops,'horizontal')
            for before,after in zip(rows,parts):
                changed[changed.index(before)] = after
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    if changed!=boxes:
        batch=propose_page_from_boxes(changed,engine.image_path,complementary_ownership=True,regroup_companions=True)
        engine.raw_detections,engine.automatic_unit_proposals=[changed],[batch]


def outlined_vowel(image, box):
    """Locate the hollow long-vowel stroke cut by the right edge of a title."""
    from . import reader_geometry
    poly = np.float32(box.poly)
    width = np.linalg.norm(poly[1]-poly[0])
    height = np.linalg.norm(poly[3]-poly[0])
    if not (18<=height<=110 and 1.6*height<=width<=6*height and poly[1,0]>poly[0,0]):
        return None
    trial = poly.copy()
    trial[[1,2]] += (poly[[1,2]]-poly[[0,3]])*(.9*height/width)
    if (trial<0).any() or (trial>=image.size).any():
        return None
    mapping = reader_geometry.plan(image.size,trial)
    gray = np.asarray(reader_geometry.pixels(image,mapping)).min(2)
    contours,hierarchy = cv2.findContours(np.uint8(gray<145),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    edge = mapping['padding'][0]+(mapping['seed_size'][0]-1)/(1+.9*height/width)
    ends = []
    for i,contour in enumerate(contours):
        child = hierarchy[0,i,2]
        if child<0:
            continue
        x,y,w,h = cv2.boundingRect(contour)
        if (edge-.4*height<x<edge and edge+3<=x+w<edge+.7*height and .3*height<=w<=1.3*height
                and .08*height<=h<=.38*height and w>=1.45*h and .25*height<=y+h/2<=.84*height
                and any(cv2.contourArea(contours[j])>.4*cv2.contourArea(contour)
                        for j in np.flatnonzero(hierarchy[0,:,3]==i))):
            ends.append(x+w)
    if len(ends)!=1:
        return None
    extension = (ends[0]+1-edge)/(edge-mapping['padding'][0])
    result = poly.copy()
    result[[1,2]] += (poly[[1,2]]-poly[[0,3]])*extension
    return result


def magazine_headings(engine,image,reader):
    """Join a periodical prefix to its neighboring kana masthead."""
    from . import reader_geometry
    from .masked_lines import corrected,bind,source_views
    boxes = engine.raw_detections[0]
    if not boxes or len(boxes)>256:
        return
    singles = {p.main_ids[0] for p in engine.automatic_unit_proposals[0].proposals if len(p.member_ids)==1}
    changed = list(boxes)
    for head in [b for b in boxes if b.box_id in singles and re.fullmatch('[一-鿿]刊',b.text)][:5]:
        x,y,r,b = head.bbox
        height = b-y+1
        titles = [v for v in changed if v.box_id in singles and re.fullmatch('[ァ-ヺー]{2,12}',v.text)
                  and x<v.bbox[0]<r+.5*height and abs(v.bbox[1]-y)<.6*height and v.bbox[2]>r]
        if len(titles)!=1:
            continue
        title = titles[0]
        try:
            reads = [reader(ch) for ch in source_views(image,[head.poly])]
            prefix = reads[0]
            if prefix not in ('週刊','月刊','季刊','日刊','夕刊','朝刊','増刊') or reads!=[prefix]*3:
                continue
            whole = source_views(image,[head.poly,title.poly])
            if [reader(ch) for ch in whole]!=[prefix+title.text]*3:
                continue
            term = title.text
            poly = title.poly
            # A terminal long-vowel stroke can extend past a tilted detector
            # edge. Require the same single extension at two independent pads.
            extended = []
            for fraction in (.10,.28):
                trial = title.poly.copy()
                trial[[1,2]] += (trial[[1,2]]-trial[[0,3]])*fraction
                trial = np.clip(trial,[0,0],np.asarray(image.size)-1).astype(np.float32)
                # Only the narrower sample can become cleanup geometry. The
                # larger confirmation may include the adjoining cover feature.
                shared_limit = .10 if extended else .03
                if any(cv2.intersectConvexConvex(cv2.convexHull(v.poly),trial)[0]>shared_limit*cv2.contourArea(trial)
                       for v in boxes if v not in (head,title)):
                    break
                if reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,trial)))!=title.text+'ー':
                    break
                context = reader_geometry.pixels(image,reader_geometry.plan(image.size,trial,view='context'))
                if reader(context)!=prefix+title.text+'ー':
                    break
                extended.append(trial)
            if len(extended)==2:
                term += 'ー'
                poly = extended[0]
            rows = [corrected(head.box_id,head.confidence,head.poly,prefix,head.paddle_text,head),
                    corrected(title.box_id,title.confidence,poly,term,title.paddle_text,title)]
            bind(rows,prefix+term,source_views(image,[head.poly,poly]),'horizontal')
            rows[0].ocr_metadata['periodical_title'] = term
            changed[changed.index(head)],changed[changed.index(title)] = rows
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    # A standalone outlined word may have no publisher or frequency label
    # nearby. Only extend it as far as a measured hollow stroke, after OCR
    # independently reads the same extra vowel in tight and surrounding views.
    for box in [v for v in boxes if v.box_id in singles and v in changed and v.confidence>=.85
                and re.fullmatch('[ァ-ヺ]{2,9}',v.text)][:24]:
        try:
            poly = outlined_vowel(image,box)
            if poly is None or any(cv2.intersectConvexConvex(cv2.convexHull(v.poly),poly)[0]>0 for v in boxes if v is not box):
                continue
            term = box.text+'ー'
            if any(reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,poly,view=view)))!=term
                   for view in ('tight','context')):
                continue
            fixed = corrected(box.box_id,box.confidence,poly,term,box.paddle_text,box)
            fixed.ocr_metadata['outlined_vowel'] = True
            changed[changed.index(box)] = fixed
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    if changed!=boxes:
        batch=propose_page_from_boxes(changed,engine.image_path,complementary_ownership=True,regroup_companions=True)
        engine.raw_detections,engine.automatic_unit_proposals=[changed],[batch]


def missing_middle_column(image, boxes, reader):
    """Recover a missing vertical line between a possessive and a short trailing particle."""
    from .masked_lines import bind, corrected, source_views
    if not boxes or len(boxes)>256 or any(type(v.box_id) is not int for v in boxes):
        return boxes
    result = list(boxes)
    for right in [v for v in boxes if re.fullmatch(r'[一-鿿ぁ-ゖァ-ヺー]{1,8}の',v.text)]:
        for left in [v for v in boxes if re.fullmatch(r'[ぁ-ゖ]{1,2}(?:…+|\.{2,}|[!?！？])',v.text)]:
            if right not in result or left not in result:
                continue
            x,y,r,b = right.bbox
            l,t,z,end = left.bbox
            width = min(r-x,z-l)
            if (width<12 or not .85*width < (x+r-l-z)/2 < 1.8*width
                    or min(b-y,end-t)<1.45*width or abs(y-t)>.4*width):
                continue
            hull = cv2.convexHull(np.vstack([right.poly,left.poly]).astype(np.float32))
            inside = [v for v in result if v not in (right,left)
                      and cv2.intersectConvexConvex(hull,v.poly.astype(np.float32))[0]>2]
            if inside and (len(inside)!=1 or inside[0].reason!='frontier_discovery'):
                continue
            edges = [np.array_split(v.poly[np.argsort(v.poly[:,1])],2) for v in (right,left)]
            top,base = np.mean(edges,axis=(0,2))
            dx = np.float32([.30*width,0])
            base[1] -= .14*width
            middle = np.float32([top-dx,top+dx,base+dx,base-dx])
            extra = inside[0] if inside else None
            if extra and cv2.pointPolygonTest(middle,tuple(map(float,extra.poly.mean(0))),False)<0:
                continue
            try:
                reads = [reader(v) for v in source_views(image,[middle])]
                word = reads[0]
                if (not re.fullmatch(r'[一-鿿ぁ-ゖァ-ヺー]{2,8}',word or '') or reads!=[word]*3
                        or (extra and extra.text not in word)):
                    continue
                crops = source_views(image,[right.poly,middle,left.poly])
                sentence = right.text+word+left.text
                if any((reader(v) or '').replace('...','…')!=sentence.replace('...','…') for v in crops):
                    continue
            except (ValueError,RuntimeError,OSError,cv2.error):
                continue
            lead = corrected(right.box_id,right.confidence,right.poly,right.text,right.paddle_text,right)
            tail = corrected(left.box_id,left.confidence,left.poly,left.text,left.paddle_text,left)
            center = corrected(extra.box_id if extra else max(v.box_id for v in result)+1,
                               extra.confidence if extra else 0,middle,word,extra.paddle_text if extra else '',extra)
            bind([lead,center,tail],sentence,crops,'vertical')
            result[result.index(right)],result[result.index(left)] = lead,tail
            if extra:
                result[result.index(extra)] = center
            else:
                result.append(center)
    return result


def small_stacked_labels(image, boxes, reader):
    """Read a compact two-line label whose lower row was detected character by character."""
    from . import reader_geometry
    from .masked_lines import bind, corrected, source_views
    result = list(boxes)
    for top in boxes:
        x,y,r,b = top.bbox
        w,h = r-x,b-y
        if (not 18<=h<=54 or not 1.8*h<=w<=5*h or top.confidence>.88
                or not re.fullmatch(r'[ぁ-ゖァ-ヺー]{3,12}',top.text) or top not in result):
            continue
        cells = [v for v in result if len(v.text)==1 and x-.18*w<v.poly[:,0].mean()<r+.18*w
                 and .34*h<v.poly[:,1].mean()-top.poly[:,1].mean()<1.15*h]
        cells.sort(key=lambda v:v.poly[:,0].mean())
        if (not 3<=len(cells)<=6 or not all(re.fullmatch('[一-鿿]',v.text) for v in cells)
                or abs(cells[0].bbox[0]-x)>.22*w or abs(cells[-1].bbox[2]-r)>.23*w
                or any(abs(a.bbox[2]-c.bbox[0])>.4*h for a,c in zip(cells,cells[1:]))):
            continue
        clipped = [v for v in result if v is not top and v not in cells and len(v.text)==1
                   and r-.25*w<v.poly[:,0].mean()<r+.18*w and abs(v.poly[:,1].mean()-top.poly[:,1].mean())<.34*h]
        if len(clipped)>1:
            continue
        groups = [[top,*clipped],cells]
        try:
            rows = []
            for group in groups:
                polygons = [v.poly for v in group]
                corners = cv2.boxPoints(cv2.minAreaRect(np.vstack(polygons)))
                plan = reader_geometry.plan(image.size,corners)
                word = reader(reader_geometry.pixels(image,plan))
                if not word or any(reader(c)!=word for c in source_views(image,polygons)):
                    break
                first = group[0]
                row = corrected(first.box_id,first.confidence,np.float32(plan['ordered_polygon']),word,first.paddle_text,first)
                row.ocr_metadata['label_fragments'] = [dict(id=v.box_id,polygon=v.poly.tolist(),text=v.text) for v in group]
                rows.append(row)
            if (len(rows)!=2 or not re.fullmatch(r'[ぁ-ゖァ-ヺー]{3,14}',rows[0].text)
                    or SequenceMatcher(None,rows[0].text,top.text).ratio()<.55
                    or not re.fullmatch(r'[一-鿿]{3,6}',rows[1].text) or len(rows[1].text)!=len(cells)
                    or not rows[1].text.startswith(cells[0].text) or not rows[1].text.endswith(cells[-1].text)):
                continue
            source = ''.join(v.text for v in rows)
            crops = source_views(image,[v.poly for v in rows])
            if any(reader(c)!=source for c in crops):
                continue
            hull = cv2.convexHull(np.vstack([v.poly for v in rows]))
            if any(cv2.intersectConvexConvex(hull,v.poly)[0]>2 for v in result if all(v not in g for g in groups)):
                continue
            bind(rows,source,crops,'horizontal')
            result = [v for v in result if all(v not in group for group in groups)] + rows
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    return result


def forum_header(image, boxes, reader):
    """Read a navigation strip and its aligned account field separately."""
    from . import reader_geometry
    from .masked_lines import bind, corrected, source_views
    changes = {}
    for field in boxes:
        # The Latin account identifier is much more reliable than the small
        # printed field name. Require it to survive every reading unchanged.
        account = re.search(r'[:：]([A-Za-z][\w.-]{1,31})$',field.text,re.ASCII)
        if not account or not re.search('[ぁ-ヺ一-鿿]',field.text):
            continue
        x,y,r,b = field.bbox
        h = b-y
        above = [v for v in boxes if v is not field and abs(v.bbox[0]-x)<.6*h
                 and abs(v.bbox[3]-y)<.45*h and v.bbox[2]-v.bbox[0]>7*h
                 and re.search('[ァ-ヺー]{3}',v.text)]
        if len(above)!=1 or not 9<=h<=70:
            continue
        try:
            menu = above[0]
            crops = source_views(image,[field.poly])
            label = reader(crops[0])
            if (not re.fullmatch(r'(?:ログイン)?(?:ユーザ[ー]?名|アカウント名)[:：]'+re.escape(account[1]),label or '')
                    or not all(reader(ch)==label for ch in crops[1:])):
                continue
            chips = source_views(image,[menu.poly])
            text = reader(chips[1])
            if not text or len(text)>75 or reader(chips[0])!=text:
                continue
            items = re.findall(r'インデックス|ユーザ[ー]?リスト|検索|プロフィール|ヘルプ|メンバー|ホーム|設定|ログアウト|通知|メッセージ',text)
            if len(items)<3 or ''.join(items)!=text or len(set(items))!=len(items):
                continue
            mapped = reader_geometry.pixels(image,reader_geometry.plan(image.size,menu.poly))
            if reader(mapped)!=text:
                continue
            for row,word,views,role in ((menu,text,chips,'navigation'),(field,label,crops,'account')):
                copy = corrected(row.box_id,row.confidence,row.poly,word,row.paddle_text,row)
                bind([copy],word,views,'horizontal')
                copy.ocr_metadata['screen_field'] = role
                if row is menu:
                    copy.ocr_metadata['navigation_items'] = items
                changes[row.box_id] = copy
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    return changes


def clipped_signature(image, box, reader):
    """Look left of a clipped second row for a consistently printed longer suffix."""
    from . import reader_geometry
    from .masked_lines import corrected
    if not re.fullmatch('[一-鿿]{3,10}',box.text):
        return None
    extension = box.poly[1]-box.poly[0]
    if extension[0]<=0 or abs(extension[1])<.15*extension[0]:
        return None
    # The row leaves a panel through a long horizontal or vertical boundary.
    x,y,r,b=box.bbox
    gray=np.asarray(image.crop((x,y,r+1,b+1)).convert('L'))
    if max((gray<140).mean(1).max(),(gray<140).mean(0).max())<.85:
        return None
    words=[]
    for extra in (.3,.46):
        trial=box.poly.copy()
        trial[[0,3]]-=extra*extension
        mapped=reader_geometry.plan(image.size,trial)
        words.append(reader(reader_geometry.pixels(image,mapped)))
    if (not isinstance(words[0],str) or words[0]!=words[1] or not words[0].startswith('ー')
            or not re.fullmatch(r'[ァ-ヺー]{3,}[一-鿿]{2,}',words[0]) or not words[0].endswith(box.text[1:])):
        return None
    word='…'+words[0][1:]+'…'
    copy=corrected(box.box_id,box.confidence,box.poly,word,box.paddle_text,box)
    copy.ocr_metadata['clipped_signature']=words
    return copy


def wrapped_auxiliaries(image, boxes, proposals, reader):
    """Read a tall vertical verb beside the short columns completing its auxiliary."""
    from .masked_lines import bind,corrected,source_views
    by_id={b.box_id:b for b in boxes}
    mains=[p for p in proposals if p.direction=='vertical' and p.member_ids==p.main_ids]
    leads=[by_id[p.main_ids[0]] for p in mains if len(p.main_ids)==1
           and re.search(r'[ぁ-ゖ一-鿿][てで]$',by_id[p.main_ids[0]].text)]
    changes={}
    for p in mains:
        if p.direction!='vertical' or not 2<=len(p.main_ids)<=3:
            continue
        tail=[by_id[i] for i in p.main_ids]
        if min(v.confidence for v in tail)>.82 or not re.search(r'(?:ない|んだ|ます|せんか|さい)[ーっッ!！…。]*$',tail[-1].text):
            continue
        l,t,r,b=p.crop_bounds
        width=tail[0].bbox[2]-tail[0].bbox[0]
        if width<=0 or any(abs(v.bbox[1]-t)>.55*width for v in tail):
            continue
        anchors=[v for v in leads if abs(v.bbox[3]-b)<width and v.bbox[1]<t-1.5*width
                 and -.55*width<v.bbox[0]-r<.65*width and v.bbox[2]>r]
        if len(anchors)!=1:
            continue
        rows=[anchors[0],*tail]
        hull=cv2.convexHull(np.vstack([v.poly for v in rows]))
        if any(cv2.intersectConvexConvex(hull,v.poly)[0]>0 for v in boxes if v not in rows):
            continue
        try:
            chips=source_views(image,[v.poly for v in rows])
            whole=[reader(v) for v in chips]
            base=(whole[0] or '').rstrip('ーっッ!！。')
            if not base or any((v or '').rstrip('ーっッ!！。')!=base for v in whole):
                continue
            words=[]
            for row in rows:
                readings=[reader(chip) for chip in source_views(image,[row.poly])]
                word=(readings[1] or '').rstrip('ーっッ!！。')
                if (not word or any((v or '').rstrip('ーっッ!！。')!=word for v in readings)
                        or SequenceMatcher(None,word,row.text).ratio()<.55):
                    break
                words.append(word)
            if (len(words)!=len(rows) or not words[0].endswith(('て','で')) or ''.join(words)!=base
                    or not re.match(r'(?:くれ|もら|いただ|くださ|しま)', ''.join(words[1:]))):
                continue
            # Only retain expressive marks seen in every full presentation.
            ending=next((v for v in sorted(whole,key=len) if all(s.startswith(v) for s in whole)),base)
            words[-1]+=ending[len(base):]
            parts=[corrected(v.box_id,v.confidence,v.poly,word,v.paddle_text,v) for v,word in zip(rows,words)]
            bind(parts,ending,chips,'vertical')
            changes.update((v.box_id,v) for v in parts)
        except (OSError,ValueError,RuntimeError,cv2.error):
            continue
    return changes


def polite_continuations(image, boxes, proposals, reader):
    """Bind a clipped polite auxiliary to the line from which it wraps."""
    from . import reader_geometry
    from .masked_lines import bind,corrected,source_views
    singles={p.member_ids[0] for p in proposals if len(p.member_ids)==1}
    by_id={b.box_id:b for b in boxes}
    changes={}
    for tail in boxes:
        if tail.box_id not in singles or not re.fullmatch(r'せて(?:頂|いただ)[ぁ-ゖ…]*',tail.text):
            continue
        x,y,r,b=tail.bbox
        groups=[]
        for group in proposals:
            if group.direction!='horizontal' or group.annotation_ids or not 1<=len(group.member_ids)<=3:
                continue
            rows=[by_id[i] for i in group.member_ids]
            l,t,right,bottom=group.crop_bounds
            if (abs(x-l)<b-y and abs(y-bottom)<.55*(b-y) and t<y and right-r>2*(b-y)
                    and rows[-1].text.endswith('さ')):
                groups.append(rows)
        if len(groups)!=1:
            continue
        rows=groups[0]
        last=rows[-1]
        try:
            crops=source_views(image,[last.poly])
            ending=reader(crops[1])
            if (not isinstance(ending,str) or not re.search(r'[をに][一-鿿]{2,4}さ$',ending)
                    or reader(crops[0])!=ending or reader(crops[2])!=ending
                    or reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,last.poly)))!=ending
                    or any(reader(ch)!=tail.text for ch in source_views(image,[tail.poly]))):
                continue
            parts=[corrected(row.box_id,row.confidence,row.poly,ending if row is last else row.text,row.paddle_text,row)
                   for row in [*rows,tail]]
            bind(parts,''.join(v.text for v in parts),source_views(image,[v.poly for v in parts]),'horizontal')
            changes.update((v.box_id,v) for v in parts)
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    return changes


def printed_versions(image, boxes, proposals, reader):
    """Retain a decimal version when the original line reader lost its point."""
    from . import reader_geometry
    from .masked_lines import bind,corrected,source_views
    by_id={b.box_id:b for b in boxes}
    changes={}
    for group in proposals:
        if group.direction!='horizontal' or group.annotation_ids or not 2<=len(group.member_ids)<=6:
            continue
        rows=[by_id[i] for i in group.member_ids]
        for row in rows:
            raw=re.search(r'([A-Za-z]{3,18})(\d{1,3})[、，][一-鿿]$',row.text)
            if not raw:
                continue
            try:
                crops=source_views(image,[row.poly])
                reads=[reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,row.poly,view=view)))
                       for view in ('tight','context')]+[reader(crops[2])]
                found=[re.search(r'([A-Za-z]{3,18})(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)$',s or '') for s in reads]
                if (not all(found) or len({m[2] for m in found})!=1 or not found[0][2].startswith(raw[2]+'.')
                        or not any(m[1]==raw[1] for m in found)
                        or any(SequenceMatcher(None,m[1],raw[1]).ratio()<.8 for m in found)):
                    continue
                word=row.text[:raw.start()]+raw[1]+found[0][2]
                parts=[corrected(v.box_id,v.confidence,v.poly,word if v is row else v.text,v.paddle_text,v) for v in rows]
                bind(parts,''.join(v.text for v in parts),source_views(image,[v.poly for v in parts]),'horizontal')
                changes.update((v.box_id,v) for v in parts)
                break
            except (ValueError,RuntimeError,OSError,cv2.error):
                continue
    return changes


def forum_post_readings(image, boxes, proposals, reader):
    """Use a printed forum masthead and repeated kana to disambiguate a slang-filled post."""
    from copy import copy
    import unicodedata
    from . import reader_geometry
    from .masked_lines import bind,corrected,source_views
    by_id={v.box_id:v for v in boxes}
    mastheads=[re.match(r'([A-Z][A-Za-z0-9.+-]{2,18})(?:日本語)?フ?ォーラム',b.text,re.I)
               for b in boxes]
    products={m[1] for m in mastheads if m}
    if len(products)!=1:
        return {}
    product=products.pop()
    changes={}
    for group in proposals:
        if group.direction!='horizontal' or group.annotation_ids or not 3<=len(group.member_ids)<=8:
            continue
        rows=sorted((by_id[i] for i in group.main_ids),key=lambda v:v.poly[0,1])
        if sum(bool(re.search('w{2,}',row.text,re.I)) for row in rows)<2:
            continue
        x,y,r,b=rows[0].bbox
        labels=[by_id[g.main_ids[0]] for g in proposals if len(g.member_ids)==1 and len(g.main_ids)==1]
        heads=[v for v in labels if abs(v.bbox[0]-x)<.2*(r-x) and .45*(r-x)<v.bbox[2]-v.bbox[0]<1.1*(r-x)
               and 0<y-v.bbox[1]<b-y and re.search('[ぁ-ゖ].*[一-鿿]{2,}$',v.text)]
        if len(heads)!=1:
            continue
        head=heads[0]
        def readings(row):
            return [reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,row.poly,view=view)))
                    for view in ('tight','axial')]
        try:
            head_reads=readings(head)
            first,second=head_reads
            match=SequenceMatcher(None,first,second).find_longest_match()
            kana=re.search('[ぁ-ゖ]{4,12}',first[match.a:match.a+match.size])
            original=re.fullmatch(r'([ぁ-ゖ、]+)([一-鿿]{2,6})',head.text)
            if not kana or not original:
                continue
            def plain(word):
                return ''.join(c for c in unicodedata.normalize('NFD',word) if not unicodedata.combining(c) and c!='、')
            if plain(kana[0])!=plain(original[1]):
                continue
            lines=[readings(row) for row in rows]
            if not any(all(kana[0] in v for v in pair) for pair in lines):
                continue
            heading=kana[0]+original[2]
            corrected_head=corrected(head.box_id,head.confidence,head.poly,heading,head.paddle_text,head)
            bind([corrected_head],heading,source_views(image,[head.poly]),'horizontal')
            # These rows are noisy. Supply independent visible phrases alongside
            # the original rather than certifying a transcription for the whole post.
            phrases=[]
            for row,(first,second) in zip(rows,lines):
                match=SequenceMatcher(None,first,second).find_longest_match()
                shared=first[match.a:match.a+match.size]
                phrases.append(shared if len(shared)>=3 and len(shared)>=.48*min(len(first),len(second)) else row.text)
            corrected_head.ocr_metadata['forum_post']={'product':product,'name':kana[0],'heading':True,'body_id':rows[0].box_id}
            body=copy(rows[0])
            body.ocr_metadata={**rows[0].ocr_metadata,'forum_post':{'product':product,'name':kana[0],'phrases':phrases}}
            changes[head.box_id]=corrected_head
            changes[body.box_id]=body
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    return changes


def repeated_printed_passages(image, boxes, proposals, reader):
    """Recover a longer technical line from a second printing of the same message."""
    from . import reader_geometry
    from .masked_lines import bind,corrected,source_views
    by_id={b.box_id:b for b in boxes}
    groups=[sorted((by_id[i] for i in g.member_ids),key=lambda v:v.poly[0,1]) for g in proposals
            if g.direction=='horizontal' and not g.annotation_ids and 3<=len(g.member_ids)<=6]
    changes={}
    version=re.compile(r'[A-Za-z][A-Za-z.]{1,20}\d{1,3}(?:\.\d{1,3})+')
    for large in groups:
        anchors=[(j,m[0]) for j,row in enumerate(large) if (m:=version.search(row.text))]
        if len(anchors)!=1:
            continue
        index,product=anchors[0]
        for small in groups:
            if (len(small)!=len(large) or small is large or any(b.box_id in changes for b in large+small)
                    or any(len(v.poly)!=4 for v in large+small)):
                continue
            heights=[np.median([np.linalg.norm(v.poly[0]-v.poly[3]) for v in rows]) for rows in (large,small)]
            if not 1.4<heights[0]/max(1,heights[1])<5:
                continue
            # Independently matching openings on other lines identify the same
            # paragraph, even when a foreground balloon clips its longer lines.
            if sum(len(a.text)>=5 and b.text.startswith(a.text[:5]) for j,(a,b) in enumerate(zip(large,small))
                   if j!=index)<2:
                continue
            big, little=large[index],small[index]
            def view(row,kind):
                return reader_geometry.pixels(image,reader_geometry.plan(image.size,row.poly,view=kind))
            try:
                short=reader(view(big,'tight'))
                long=reader(view(little,'tight'))
                if (product not in short or len(short)<6 or not long.startswith(short)
                        or len(long)-len(short)<4 or len(long)>80 or re.search(r'[Ａ-Ｚａ-ｚ]',long)):
                    continue
                # The extended suffix must also survive a changed presentation.
                suffix=long[long.index(product)+len(product):]
                if (len(suffix)<5 or not reader(view(little,'axial')).endswith(suffix)
                        or reader(view(big,'axial'))!=short):
                    continue
                for rows,target,reading in ((large,big,short),(small,little,long)):
                    parts=[corrected(v.box_id,v.confidence,v.poly,reading if v is target else v.text,
                                     v.paddle_text,v) for v in rows]
                    bind(parts,''.join(v.text for v in parts),source_views(image,[v.poly for v in rows]),'horizontal')
                    parts[0].ocr_metadata['repeated_printed_line']={'excerpt':short,'full':long}
                    changes.update((v.box_id,v) for v in parts)
            except (ValueError,RuntimeError,OSError,cv2.error):
                continue
    return changes


def printed_fields(engine,image,reader):
    """Read a short profile field or a printed gloss within its existing grouping."""
    from . import reader_geometry
    from .masked_lines import bind,corrected,source_views
    from .recognition_units import image_fingerprint
    boxes = engine.raw_detections[0]
    if not boxes or len(boxes)>256:
        return
    changes = forum_header(image,boxes,reader)
    changes.update(forum_post_readings(image,boxes,engine.automatic_unit_proposals[0].proposals,reader))
    changes.update(wrapped_auxiliaries(image,boxes,engine.automatic_unit_proposals[0].proposals,reader))
    changes.update(polite_continuations(image,boxes,engine.automatic_unit_proposals[0].proposals,reader))
    changes.update(printed_versions(image,boxes,engine.automatic_unit_proposals[0].proposals,reader))
    changes.update(repeated_printed_passages(image,boxes,engine.automatic_unit_proposals[0].proposals,reader))
    changed = [changes.get(b.box_id,b) for b in boxes]
    by_id = {b.box_id:b for b in boxes}
    for proposal in engine.automatic_unit_proposals[0].proposals:
        if len(proposal.member_ids)!=2 or proposal.direction!='horizontal':
            continue
        rows = sorted((by_id[i] for i in proposal.member_ids),key=lambda b:b.poly[:,1].mean())
        top,base = rows
        if any(row.box_id in changes for row in rows):
            continue
        # Stroke lettering can turn an entire katakana heading into visually
        # similar kanji. Check the original shape of every glyph before
        # separating this heading from a smaller, independently printed row.
        kana_shapes = str.maketrans('才三工口力夕卜二八', 'オシエロカタトニハ')
        kana = top.text.translate(kana_shapes)
        top_size = np.linalg.norm(top.poly[3]-top.poly[0])
        lower_size = np.linalg.norm(base.poly[3]-base.poly[0])
        if (not proposal.annotation_ids and .38<top.confidence<.85 and top_size>1.15*lower_size
                and re.fullmatch('[ァ-ヺー]{3,9}',kana) and re.fullmatch('[一-鿿]+',top.text)):
            try:
                crops = source_views(image,[top.poly])
                word = reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,top.poly)))
                if word==kana and sum(reader(ch)==word for ch in crops)>=2:
                    copy = corrected(top.box_id,top.confidence,top.poly,word,top.paddle_text,top)
                    bind([copy],word,crops,'horizontal')
                    copy.ocr_metadata['display_heading']=True
                    changed[changed.index(top)] = copy
                    signature=clipped_signature(image,base,reader)
                    if signature is not None:
                        changed[changed.index(base)]=signature
                    continue
            except (ValueError,RuntimeError,OSError,cv2.error):
                pass
        gloss = (proposal.annotation_parents==((top.box_id,base.box_id),) and top.confidence>.96
                 and re.fullmatch(r"[A-Za-z]+(?: [A-Za-z'-]+){1,5}",top.text)
                 and re.fullmatch(r'「[^「」]{2,40}」',base.text))
        field = (not proposal.annotation_ids and re.match(r'From\s*[:：]',base.text,re.I) and
                 re.fullmatch(r'[ァ-ヺー×ぁ-ゖ八工口力]{2,9}',top.text)
                 and re.search(r'[ァ-ヺぁ-ゖ]',top.text) and top.confidence<.85)
        if not (gloss or field):
            continue
        try:
            whole = source_views(image,[b.poly for b in rows])
            joined = [reader(ch) for ch in whole]
            if not all(isinstance(v,str) for v in joined):
                continue
            if gloss:
                expected = (top.text+base.text).replace(' ','').casefold()
                if (all(v.replace(' ','').casefold()==expected for v in joined)
                        and all(reader(ch)==top.text for ch in source_views(image,[top.poly]))):
                    copy = corrected(base.box_id,base.confidence,base.poly,base.text,base.paddle_text,base)
                    copy.ocr_metadata['printed_gloss'] = {'text':top.text,'source_id':top.box_id,
                                                          'views':[image_fingerprint(ch) for ch in whole]}
                    changed[changed.index(base)] = copy
                continue
            top_reads = [reader(ch) for ch in source_views(image,[top.poly])]
            label = top_reads[0]
            if (not isinstance(label,str) or not re.fullmatch('[ァ-ヺー]{2,9}',label)
                    or not all(v==label for v in top_reads) or not all(v.startswith(label) for v in joined)):
                continue
            location = reader(reader_geometry.pixels(image,reader_geometry.plan(image.size,base.poly,view='context')))
            tail = re.search(r'From[:：]([一-鿿]{2,6}|[A-Za-z]{2,12})$',location or '',re.I)
            if not tail or sum(v.casefold().endswith(tail[0].casefold()) for v in joined)<2:
                continue
            values = [label,tail[0]]
            for old,value in zip(rows,values):
                copy = corrected(old.box_id,old.confidence,old.poly,value,old.paddle_text,old)
                copy.ocr_metadata['printed_field'] = {'rows':values,'views':[image_fingerprint(ch) for ch in whole]}
                changed[changed.index(old)] = copy
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    if changed!=boxes:
        batch=propose_page_from_boxes(changed,engine.image_path,complementary_ownership=True,regroup_companions=True)
        engine.raw_detections,engine.automatic_unit_proposals=[changed],[batch]


def command_lists(engine,image,reader):
    """Join stacked shell instructions when a printed Japanese particle connects them."""
    from .masked_lines import corrected,bind,source_views
    boxes = engine.raw_detections[0]
    if not boxes or len(boxes)>256:
        return
    command = re.compile(r'(?:wget|curl|git|make|install|cmake|configure|pip|npm|apt|sudo|docker)(?=$|[^A-Za-z]|[A-Za-z][^A-Za-z0-9])',re.I)
    anchors = [v for v in boxes if command.match(v.text)]
    if len(anchors)<2:
        return
    changed = list(boxes)
    for tail in [v for v in boxes if re.fullmatch(r'(?:で|して|してね|です)[！!…。]*',v.text)][:8]:
        x,y,r,b = tail.bbox
        height = b-y+1
        if not 12<height<115:
            continue
        above = sorted((v for v in anchors if v in changed and x-2.5*height<v.bbox[0]<r
                        and x<v.bbox[2]+height and y-5*height<v.bbox[1]<y),key=lambda v:v.bbox[1])
        if not 2<=len(above)<=4 or not all(0<b.bbox[1]-a.bbox[1]<1.8*height for a,b in zip(above,[*above[1:],tail])):
            continue
        rows = [*above,tail]
        try:
            crops = source_views(image,[v.poly for v in rows])
            whole = [reader(crop) for crop in crops]
            if not all(isinstance(v,str) and v.casefold()==whole[0].casefold() for v in whole):
                continue
            # Capture only the actual written particles between known program
            # names. This also preserves which Latin letters belong to each line.
            terms = [command.match(v.text)[0] for v in above]
            pattern = ''.join('('+re.escape(term)+r'(?:で|を|して|し|と|に)?)' for term in terms)+re.escape(tail.text)
            match = re.fullmatch(pattern,whole[0],re.I)
            if not match or not any(re.search('[ぁ-ゖ]',part) for part in match.groups()):
                continue
            parts = [*match.groups(),tail.text]
            lines = [corrected(v.box_id,v.confidence,v.poly,part,v.paddle_text,v) for v,part in zip(rows,parts)]
            bind(lines,whole[0],crops,'horizontal')
            lines[0].ocr_metadata['command_list'] = terms
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
        for before,after in zip(rows,lines):
            changed[changed.index(before)] = after
    if changed!=boxes:
        batch=propose_page_from_boxes(changed,engine.image_path,complementary_ownership=True,regroup_companions=True)
        engine.raw_detections,engine.automatic_unit_proposals=[changed],[batch]


def tier_gap(box, neighbors, gray):
    """A detector column spans the ends and starts of two adjacent tiers of text."""
    x,y,r,b = box.bbox
    w,h = r-x,b-y
    if w<14 or h<3.7*w or not re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿々]{4,22}',box.text):
        return None
    ends,starts = [],[]
    for other in neighbors:
        l,t,right,bottom = other.bbox
        if other is box or not 0 < (x+r-l-right)/2 < 1.7*w or bottom-t < .9*(right-l) or len(other.text)<2:
            continue
        if abs(t-y)<.5*w:
            ends.append(bottom)
        elif y+.5*h<t<b-.12*h and abs(bottom-b)<.35*h:
            starts.append(t)
    if not ends or not starts:
        return None
    dark_rows = (gray[y:b+1,x:r+1]<160).any(1)
    edges = np.flatnonzero(np.diff(np.r_[True,dark_rows,True]))
    cuts = []
    for a,z in zip(edges[::2],edges[1::2]):
        cut = y+(a+z)//2
        if (z-a >= max(4,.19*h/(len(box.text)+1)) and y+.55*h<cut<b-.12*h
                and any(0<cut-end<.44*w for end in ends) and any(0<=start-cut<.22*w for start in starts)):
            cuts.append(int(cut))
    return cuts[0] if len(cuts)==1 else None


def corroborate_columns(image, anchor, boxes, reader):
    x,y,r,b = anchor.bbox
    w = r-x
    candidates = sorted((v for v in boxes if v is not anchor and v.text and
                         v.bbox[2]<r and abs(v.bbox[1]-y)<.4*w and
                         v.bbox[3]-v.bbox[1]>.9*(v.bbox[2]-v.bbox[0])),
                        key=lambda v: -(v.bbox[0]+v.bbox[2]))
    columns = [anchor]
    for other in candidates:
        a,_,c,_ = columns[-1].bbox
        l,t,right,end = other.bbox
        if right<a-.28*w:
            break
        if (not .32*w<(a+c-l-right)/2<1.5*w or
                not .4*w<right-l<1.75*w or len(columns)>=6):
            return
        columns.append(other)
    if len(columns)<2:
        return
    left,top = min(v.bbox[0] for v in columns),min(v.bbox[1] for v in columns)
    bottom = max(v.bbox[3] for v in columns)
    native = ''.join(v.text for v in columns)
    reads = [reader(image.crop((max(0,left-k),max(0,top-k),min(image.width,r+k+1),
                               min(image.height,bottom+k+1)))) for k in (0,3)]
    if (isinstance(reads[0],str) and reads[0]==reads[1] and reads[0].startswith(anchor.text)
            and SequenceMatcher(None,native,reads[0]).ratio()>=.82):
        return {'ids':[v.box_id for v in columns],'reading':reads[0]}


def split_tiers(engine,image,reader):
    boxes = engine.raw_detections[0]
    if not boxes or len(boxes)>256 or any(type(b.box_id) is not int for b in boxes):
        return
    gray = np.asarray(image.convert('L'))
    revised = list(boxes)
    for box,cut in [(b,at) for b in boxes if (at:=tier_gap(b,boxes,gray)) is not None][:4]:
        try:
            x,y,r,b = box.bbox
            w = r-x
            l,t = max(0,x-round(1.25*w)),max(0,y-round(.27*w))
            chip = image.crop((l,t,min(image.width,r+round(.45*w)),min(image.height,b+round(.29*w))))
            results = engine.ocr.predict(np.asarray(chip))
            if not results:
                continue
            output = results[0]
            parts = []
            for poly,native,score in zip(output['rec_polys'],output['rec_texts'],output['rec_scores']):
                pts = np.asarray(poly,dtype=np.float32)+[l,t]
                cx,cy = pts.mean(0)
                if not (x+.24*w<cx<r and y<cy<b and score>=.75 and native):
                    continue
                (ax,ay),(bx,by) = pts.min(0),pts.max(0)
                # Both resulting columns must end at the same visible break.
                if not (ay<cut<=by+.2*w and by<cut+.2*w or ay>cut-.2*w and by>cut):
                    continue
                (ax,ay),(bx,by) = np.floor([ax,ay]).astype(int),np.ceil([bx,by]).astype(int)
                parts.append((pts, native, float(score), (ax,ay,bx,by)))
            parts.sort(key=lambda part:part[3][1])
            if (len(parts)!=2 or not (parts[0][3][1]<cut<parts[1][3][3]
                    and parts[0][3][3]<=cut+.2*w and parts[1][3][1]>=cut-.2*w) or ''.join(part[1] for part in parts)!=box.text):
                continue
            members = []
            for index,(pts,native,score,(ax,ay,bx,by)) in enumerate(parts):
                reads = [reader(image.crop((max(0,ax-k),max(0,ay-k),min(image.width,bx+k+1),
                                           min(image.height,by+k+1)))) for k in (0,3)]
                if (not isinstance(reads[0],str) or reads[0]!=reads[1] or (reads[0]!=native and
                        not (index and re.fullmatch(re.escape(native)+r'[ぁ-ゖ]{1,2}',reads[0])))):
                    break
                pts[:,1] = np.minimum(pts[:,1],cut-1) if index==0 else np.maximum(pts[:,1],cut+1)
                identity = box.box_id if index==0 else max(v.box_id for v in revised)+1
                piece = Text_Box(identity,score,pts,reads[0],review=1)
                piece.selected_engine = OCREngine.HAYAI
                piece.reason = 'confirmed_tier_column'
                piece.ocr_metadata = {'candidates':{'paddle':native,'hayai':reads[0]},'tier_split':{
                    'original_id':box.box_id,'original_text':box.text,'original_polygon':box.poly.tolist(),
                    'cut':cut,'part':index,'views':reads}}
                members.append(piece)
            if len(members)==2:
                sequence = corroborate_columns(image,members[1],[v for v in revised if v is not box],reader)
                if sequence:
                    members[1].ocr_metadata['corroborated_columns'] = sequence
                i = revised.index(box)
                revised[i:i+1] = members
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    if len(revised)!=len(boxes):
        engine.automatic_unit_proposals = [propose_page_from_boxes(revised,engine.image_path,
            complementary_ownership=True,regroup_companions=True)]
        engine.raw_detections = [revised]


def possessive_column(image, parent, child, reader):
    from . import masked_lines
    # A short possessive by the same speaker can occupy the usual ruby
    # position. Check each column on its own as well as their reading order.
    try:
        child_views = masked_lines.source_views(image,[child.poly])
        prefix = reader(child_views[1])
        if (not isinstance(prefix,str) or not re.fullmatch(r'(?:オレ|おれ|ボク|ぼく|私|わたし|あたし|俺|僕)の',prefix)
                or any(reader(child_views[k])!=prefix for k in (0,2))
                or reader(masked_lines.source_views(image,[parent.poly])[1])!=parent.text):
            return
        views = masked_lines.source_views(image,[child.poly,parent.poly])
        readings = [reader(view) for view in views]
        if not all(isinstance(v,str) and v.startswith(prefix) and v.endswith(parent.text[-2:])
                   and len(v)==len(prefix+parent.text) for v in readings):
            return
        members = [masked_lines.corrected(row.box_id,row.confidence,row.poly,word,row.paddle_text,row)
                   for row,word in ((child,prefix),(parent,parent.text))]
        masked_lines.bind(members,prefix+parent.text,views,'vertical')
        return members
    except masked_lines.READ_ERRORS:
        return


def join_slanted_columns(engine,image,reader):
    boxes = engine.raw_detections[0]
    if not boxes or len(boxes)>256:
        return
    by_id = {v.box_id:v for v in boxes}
    proposals = engine.automatic_unit_proposals[0].proposals
    singles = {p.main_ids[0] for p in proposals if len(p.member_ids)==1}
    def tilt(box):
        top,base = np.array_split(box.poly[np.argsort(box.poly[:,1])],2)
        dx,dy = base.mean(0)-top.mean(0)
        return dx/max(dy,1)
    changed = False
    for p in proposals:
        if len(p.main_ids)==1 and len(p.annotation_ids)==1 and len(p.member_ids)==2:
            parent,child = by_id[p.main_ids[0]],by_id[p.annotation_ids[0]]
            if (child.confidence<.82 and 2<=len(child.text)<=5
                    and re.fullmatch(r'[一-鿿][ぁ-ゖァ-ヺ一-鿿]{1,10}[にをはが]',parent.text)
                    and .13<abs(tilt(parent))<.6 and abs(tilt(child)-tilt(parent))<.17
                    and child.poly[:,0].mean()>parent.poly[:,0].mean()
                    and not re.match(r'私|僕|俺|わたし|ぼく|おれ',parent.text)):
                pair = possessive_column(image,parent,child,reader)
                if pair is not None:
                    for original,revised in zip((child,parent),pair):
                        boxes[boxes.index(original)] = revised
                        by_id[original.box_id] = revised
                    changed = True
        if (p.member_ids!=p.main_ids or not 2<=len(p.main_ids)<=4 or not singles):
            continue
        group = [by_id[i] for i in p.main_ids]
        group.sort(key=lambda b:-b.poly[:,0].mean())
        if (not all(1.4*(b.bbox[2]-b.bbox[0])<b.bbox[3]-b.bbox[1] for b in group)
                or min(b.confidence for b in group)>.80 or
                not .13<abs(tilt(group[0]))<.6 or
                any(abs(tilt(b)-tilt(group[0]))>.15 for b in group)):
            continue
        anchor,last = group[0],group[-1]
        width = last.bbox[2]-last.bbox[0]
        tails = [by_id[i] for i in singles if by_id[i].bbox[2]<last.bbox[2] and
                 0<last.poly[:,0].mean()-by_id[i].poly[:,0].mean()<1.4*width and
                 abs(by_id[i].bbox[1]-anchor.bbox[1])<.4*width and abs(tilt(by_id[i])-tilt(last))<.15]
        if len(tails)!=1 or len(tails[0].text)<2:
            continue
        tail = tails[0]
        try:
            # The native crop of the first group must already read into the
            # next column before two full crops may corroborate their union.
            prefix = reader(image.crop(p.crop_bounds))
            if not isinstance(prefix,str) or not prefix.endswith(tuple(tail.text[:n] for n in range(2,len(tail.text)+1))):
                continue
            sequence = corroborate_columns(image,anchor,boxes,reader)
        except (OSError,ValueError,RuntimeError,cv2.error):
            continue
        if sequence and sequence['ids']==[b.box_id for b in [*group,tail]]:
            anchor.ocr_metadata['corroborated_columns'] = sequence
            singles.remove(tail.box_id)
            changed = True
    if changed:
        engine.automatic_unit_proposals = [propose_page_from_boxes(boxes,engine.image_path,
            complementary_ownership=True,regroup_companions=True)]


def emphatic_ending(engine,image,reader):
    """Join a short rhetorical ending when an anger mark was read as its final kana."""
    from .masked_lines import bind, corrected, source_views
    from .reader_geometry import plan,pixels
    import unicodedata
    boxes = engine.raw_detections[0]
    if len(boxes)>256:
        return
    batch = engine.automatic_unit_proposals[0]
    by_id = {v.box_id:v for v in boxes}
    changed = list(boxes)
    for tail in boxes:
        text = unicodedata.normalize('NFKC',tail.text)
        if not re.fullmatch(r'[かよぞねな][っッ]?[!?][ハ八]',text):
            continue
        x,y,r,b = tail.bbox
        if not 1.5*(r-x)<b-y<5*(r-x):
            continue
        heads = [p for p in batch.proposals if p.direction=='vertical' and 1<=len(p.main_ids)<=5 and
                 p.main_ids==p.member_ids and all(by_id[i].confidence>=.85 for i in p.main_ids) and
                 re.search(r'[るうすくぐむぶつぬい]$',by_id[p.main_ids[-1]].text) and
                 -.25*(r-x)<by_id[p.main_ids[-1]].bbox[0]-r<.65*(r-x) and
                 abs(by_id[p.main_ids[-1]].bbox[1]-y)<.5*(r-x)]
        if len(heads)!=1:
            continue
        crop = np.asarray(image.crop((x,y,r+1,b+1))).min(2)
        mask = np.zeros_like(crop)
        cv2.fillConvexPoly(mask,np.int32(tail.poly-[x,y]),1)
        empty = ~np.any((crop<140)&(mask!=0),axis=1)
        runs = np.flatnonzero(np.diff(np.r_[False,empty,False])).reshape(-1,2)
        gaps = [(z-a,(a+z)//2) for a,z in runs if .49*(b-y)<a<.8*(b-y) and z<.9*(b-y)]
        if not gaps or max(gaps)[0]<.065*(b-y):
            continue
        cut = y+max(gaps)[1]
        poly = tail.poly.copy()
        for bottom,top in ((3,0),(2,1)):
            span=poly[bottom]-poly[top]
            if span[1]<=0:
                break
            poly[bottom]=poly[top]+span*(cut-poly[top,1])/span[1]
        else:
            previous=[by_id[i] for i in heads[0].main_ids]
            expect=text[:-1]
            whole=''.join(v.text for v in previous)+expect
            try:
                tight=reader(pixels(image,plan(image.size,tail.poly)))
                if tight!=expect or any(reader(v)!=expect for v in source_views(image,[poly])):
                    continue
                views=source_views(image,[v.poly for v in previous]+[poly])
                if any(reader(v)!=whole for v in views):
                    continue
            except (ValueError,OSError,RuntimeError,cv2.error):
                continue
            members=[corrected(v.box_id,v.confidence,v.poly,v.text,v.paddle_text,v) for v in previous]
            members.append(corrected(tail.box_id,tail.confidence,poly,expect,tail.paddle_text,tail))
            bind(members,whole,views,'vertical')
            for before,after in zip([*previous,tail],members):
                changed[changed.index(before)]=after
    if changed!=boxes:
        engine.automatic_unit_proposals = [propose_page_from_boxes(changed,engine.image_path,
            complementary_ownership=True,regroup_companions=True)]
        engine.raw_detections=[changed]

def printed_hesitations(engine,image):
    """Refine separately lettered onsets after the reader's original OCR pass."""
    from .automatic_units import join_stuttered_onsets
    from .types import Text_Group
    boxes = engine.raw_detections[0]
    by_id = {v.box_id:v for v in boxes}
    if not any(len(v.text)==1 and re.fullmatch('[ぁ-ゖァ-ヺ]',v.text) for v in boxes):
        return
    proposals = engine.automatic_unit_proposals[0].proposals
    groups = [Text_Group([by_id[i] for i in p.main_ids],direction=p.direction) for p in proposals
              if p.main_ids and p.main_ids==p.member_ids]
    joined = join_stuttered_onsets(groups,np.asarray(image),set())
    onsets = [g.group[0] for g in joined if g.method=='stuttered_onset']
    if onsets:
        for first in onsets:
            first.ocr_metadata = {**first.ocr_metadata,'printed_onset':True}
        engine.automatic_unit_proposals = [propose_page_from_boxes(boxes,engine.image_path,
            complementary_ownership=True,regroup_companions=True)]


def terminal_kana(image, box, reader):
    """Find a small final っ that falls across the outer edge of a handwritten column."""
    from .recognition_units import image_fingerprint

    if not re.fullmatch(r'[ぁ-ゖァ-ヺ一-鿿々]{1,10}っ', box.text):
        return None
    x, y, r, b = box.bbox
    width, height = r-x, b-y
    if not (20 <= width <= 110 and 1.5*width < height < 5*width):
        return None
    left, top = max(0, x-width//4), max(0, y+height//2)
    right, bottom = min(image.width, x+width//2), min(image.height, b+2)
    region = image.crop((left, top, right, bottom))
    gray = np.asarray(region.convert('L'))
    if not gray.size or np.mean(gray > 235) < .75:
        return None
    _, _, stats, _ = cv2.connectedComponentsWithStats((gray < 150).astype(np.uint8))
    eligible = []
    for i, j, w, h, area in stats[1:]:
        if (not (.12*width <= w <= .38*width and .12*width <= h <= .4*width
                 and .55 <= w/h <= 1.5 and .17*w*h < area < .65*w*h)
                or i < 2 or j < 2 or i+w+2 > gray.shape[1] or j+h+2 > gray.shape[0]):
            continue
        l, t = int(left+i), int(top+j)
        # An isolated glyph should stay legible at two scales and after
        # thresholding. One lobe of an adjacent anger mark can read as っ alone.
        bounds = (l-2, t-2, int(l+w+2), int(t+h+2))
        chip = image.crop(bounds)
        binary = chip.convert('L').point(lambda v: 255 if v >= 150 else 0).convert('RGB')
        enlarged = chip.resize((chip.width*2, chip.height*2), Image.Resampling.BICUBIC)
        if all(reader(view) == 'っ' for view in (chip, binary, enlarged)):
            eligible.append({'bounds': list(bounds), 'digest': image_fingerprint(chip)})
    return eligible[0] if len(eligible) == 1 else None


def mark_terminal_kana(engine, image, reader):
    changed = False
    by_id = {box.box_id: box for box in engine.raw_detections[0]}
    for proposal in engine.automatic_unit_proposals[0].proposals:
        if proposal.direction != 'vertical' or not 2 <= len(proposal.main_ids) <= 5:
            continue
        last = min((by_id[i] for i in proposal.main_ids), key=lambda box: box.bbox[0])
        found = terminal_kana(image, last, reader)
        if found:
            last.ocr_metadata = {**last.ocr_metadata, 'terminal_kana': found}
            changed = True
    if changed:
        engine.automatic_unit_proposals = [propose_page_from_boxes(engine.raw_detections[0], engine.image_path,
            complementary_ownership=True, regroup_companions=True)]


def printed_balloon_columns(engine,image,reader):
    """Join top-aligned columns split between OCR groups on the same enclosed paper."""
    from collections import defaultdict
    from .masked_lines import source_views
    boxes=engine.raw_detections[0]
    if not boxes or len(boxes)>256 or image.width*image.height>4_000_000:
        return
    batch=engine.automatic_unit_proposals[0]
    owners={i:p for p in batch.proposals for i in p.member_ids}
    available=[v for v in boxes if v.box_id in owners and owners[v.box_id].member_ids==owners[v.box_id].main_ids
               and len(owners[v.box_id].main_ids)<=6 and re.fullmatch(r'[ぁ-ゖァ-ヺ一-鿿々ー0-9。.〜～…！!?？]{1,30}',v.text)]
    if len(available)<2:
        return
    paper=np.uint8(np.asarray(image.convert('L'))>=235)
    _,labels,stats,_=cv2.connectedComponentsWithStats(paper)
    groups=defaultdict(list)
    for v in available:
        x,y,r,b=v.bbox
        width,height=r-x,b-y
        if not (14<=width<=100 and 1.0*width<=height<10*width):
            continue
        chip=labels[y:b+1,x:r+1]
        mask=np.zeros(chip.shape,np.uint8)
        cv2.fillPoly(mask,[np.int32(v.poly-[x,y])],1)
        ids,counts=np.unique(chip[mask!=0],return_counts=True)
        if not len(ids):continue
        j=counts.argmax()
        if ids[j] and counts[j]>.58*mask.sum():
            groups[int(ids[j])].append(v)
    changed=False
    # An open neck between two speech balloons can join their paper. Separate
    # distinct baselines before comparing neighboring columns on each one.
    baselines=[]
    for label, columns in groups.items():
        for column in sorted(columns, key=lambda v: v.bbox[1]):
            width=column.bbox[2]-column.bbox[0]
            inline_mark = (column.text in ('2','？','?') and baselines and baselines[-1][0]==label
                and any(re.search(r'か[.…!！]*$',v.text) and -.3*width< v.bbox[0]-column.bbox[2]<.7*width
                        and v.bbox[1]<=column.bbox[1]<column.bbox[3]<=v.bbox[3]
                        for v in baselines[-1][1]))
            if not baselines or baselines[-1][0]!=label or (not inline_mark and column.bbox[1]-baselines[-1][1][0].bbox[1]>.65*width):
                baselines.append((label, []))
            baselines[-1][1].append(column)
    for label,cols in baselines:
        if not 2<=len(cols)<=6:
            continue
        x,y,w,h,area=stats[label]
        if (x==0 or y==0 or x+w>=image.width or y+h>=image.height or area>.06*paper.size
                or w*h>320_000 or area<.4*w*h):
            continue
        cols.sort(key=lambda v:-v.poly[:,0].mean())
        ids=[v.box_id for v in cols]
        existing={owners[i].proposal_id for i in ids}
        if len(existing)<2 or any(not set(owners[i].member_ids).issubset(ids) for i in ids):
            continue
        widths=[v.bbox[2]-v.bbox[0] for v in cols]
        width=float(np.median(widths))
        question = len(cols)>=2 and cols[-1].text in ('2','？','?') and re.search(r'か[.…!！]*$',cols[-2].text)
        hesitation = (len(cols)>=2 and re.fullmatch(r'(?:え[えー]?と|あの+)[…。]*',cols[-1].text)
                      and re.search(r'(?:のは|とは|のが)$',cols[-2].text))
        aligned=cols[:-1] if question else cols
        overlap=.85 if hesitation else .65
        if (max(widths)>2.1*min(widths) or np.ptp([v.bbox[1] for v in aligned])>.65*width
                or any(not -overlap*width<a.bbox[0]-b.bbox[2]<.8*width for a,b in zip(cols,cols[1:]))):
            continue
        # Only connect pieces with a grammatical dependency across a former
        # boundary. Independent labels can share the same paper and baseline.
        if not question and not any(owners[a.box_id] is not owners[b.box_id] and (
                re.search(r'(?:[てでなをがのは]|[ぁ-ゖ一-鿿]く)$',a.text) or a.text in ('私','僕','俺') or
                re.match(r'(?:あげ|くれ|もら)',b.text)) for a,b in zip(cols,cols[1:])):
            continue
        polys=[v.poly for v in cols]
        try:
            views=source_views(image,polys)
            first,isolated=reader(views[0]),reader(views[1])
            native=''.join(v.text for v in cols)
            if question:
                stem=''.join(v.text for v in cols[:-1]).rstrip('…。.!！')
                if (not first or not first.endswith(('?','？')) or first.rstrip('!?！？')!=stem
                        or first!=isolated or reader(views[2])!=first
                        or any(v not in ('?','？') for v in (reader(c) for c in source_views(image,[cols[-1].poly])))):
                    continue
                native=first
            if hesitation and (not first or first!=native or (reader(views[2]) or '').rstrip('。')!=first):
                continue
            if any(v.confidence<.6 and v.text.isdecimal() for v in cols):
                # A stray digit can be the detector's reading of an entire
                # handwritten column. Confirm the column on native paper.
                components=[]
                for v in cols:
                    readings=[reader(view) for view in source_views(image,[v.poly])[:2]]
                    # An angled single-column rectangle can include ink from
                    # its neighbor. Compare its isolated view with the complete
                    # passage read three ways, including an expanded margin.
                    if not readings[1] or (v.confidence>=.6 and readings[0]!=readings[1]):
                        break
                    components.append(readings[1])
                if ''.join(components)!=first or reader(views[2])!=first:
                    continue
                native=first
            if (not isinstance(first,str) or first!=isolated or not 3<=len(first)<=96 or
                    SequenceMatcher(None,first,native).ratio()<.74 or not first.startswith(cols[0].text)):
                continue
            cols[0].ocr_metadata['corroborated_columns']={'ids':ids,'reading':first}
            changed=True
        except (OSError,ValueError,RuntimeError,cv2.error):
            continue
    if changed:
        engine.automatic_unit_proposals=[propose_page_from_boxes(boxes,engine.image_path,complementary_ownership=True,regroup_companions=True)]
