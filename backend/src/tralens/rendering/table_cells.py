"""Place small print inside measured rules on a monochrome paper form."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..ocr.table_rows import interior, tab_interiors, title_surface
from . import natural


def lettering(target, font_data, width, height, minimum=10):
    """Choose the largest readable fit, allowing one break at a word boundary."""
    if not 0<len(target)<=120 or '\n' in target:
        return
    words = target.split(' ')
    lines = [[target]] + [[' '.join(words[:i]),' '.join(words[i:])] for i in range(1,len(words))]
    for size in range(min(max(28,minimum),height+1),minimum-1,-1):
        font = ImageFont.truetype(BytesIO(font_data),size)
        for attempt in lines:
            boxes = [font.getbbox(v) for v in attempt]
            w = max(rr-l for l,t,rr,bb in boxes)
            h = sum(bb-t for l,t,rr,bb in boxes)+len(attempt)-1
            if not (0<w<=width*min(1.23,size/minimum) and 0<h<=height):
                continue
            letters = Image.new('L',(w,h))
            offset = 0
            for line,(l,t,rr,bb) in zip(attempt,boxes):
                ImageDraw.Draw(letters).text((-l,offset-t),line,font=font,fill=255)
                offset += bb-t+1
            if w>width:
                letters = letters.resize((width,h), Image.Resampling.LANCZOS)
            return letters,size,attempt


def title_repair(original, gray, probability, cell, extended):
    """Remove supported glyph strokes on a gray title bar while retaining the panel rules."""
    import cv2
    surface = title_surface(gray,cell)
    if surface is None or probability is None:
        return
    xx,tops,height,tone = surface
    x,r = xx[0],xx[-1]+1
    y,b = tops.min()-4,tops.max()+height+4
    crop = gray[y:b,x:r]
    if np.quantile(np.ptp(original[y:b,x:r],axis=2),.95)>8:
        return
    yy = np.arange(y,b)[:,None]
    inside = (yy>=tops-2)&(yy<=tops+height+1)
    # A fitted strip can overlap a neighboring white balloon. Its white paper
    # and adjoining outline must remain intact even beyond the OCR bounds.
    left,top = max(0,x-12),max(0,y-12)
    white = np.uint8(original[top:b+12,left:r+12].min(2)>tone+10)
    white = cv2.morphologyEx(white,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    guard = cv2.dilate(white,np.ones((9,9),np.uint8))[y-top:b-top,x-left:r-left]!=0
    for expected in (tops-3,tops+height+2):
        best = None
        for slope in (-.025,0,.025):
            for offset in range(-5,6):
                line = np.rint(expected+offset+slope*(xx-xx.mean())).astype(int)
                if line.min()<0 or line.max()>=gray.shape[0]:
                    continue
                score = (gray[line,xx]<110).mean()-.007*abs(offset)
                if best is None or score>best[0]:
                    best = score,line
        if best and best[0]>=.40:
            guard |= abs(yy-best[1])<=2
    eligible = inside & ~guard
    _,labels,stats,_ = cv2.connectedComponentsWithStats(np.uint8((crop<tone-16)&eligible))
    support = np.bincount(labels[probability[y:b,x:r]>30],minlength=len(stats))
    bx,_,width,_,area = stats.T
    # Only an independently reread longer title can supply a missing final glyph.
    extra = extended & (x+bx+width-1>cell[0][-1]-.15*height) & (bx+width<r-x) & (area>=4)
    selected = (width<1.85*height) & ((support>=.10*area)|extra)
    selected[0] = False
    mask = (cv2.dilate(np.uint8(selected[labels]),np.ones((3,3),np.uint8))!=0)&eligible
    if mask.any():
        return (x,y,r,b),mask,tone


def checklist_line(clean, polygon, existing, probability):
    """Find isolated lettering left between a previously set line and a checklist rule."""
    import cv2
    pts = np.asarray(polygon)
    x,y = pts.min(0).astype(int)
    r,b = np.ceil(pts.max(0)).astype(int)
    if not (18<=b-y<=65 and 2*(b-y)<r-x<350 and probability is not None):
        return
    pad = 6
    x,y = max(0,x-pad),max(0,y-pad)
    r,b = min(clean.width,r+pad+1),min(clean.height,b+pad+1)
    patch = np.asarray(clean.crop((x,y,r,b)))
    inside = cv2.fillConvexPoly(np.zeros(patch.shape[:2],np.uint8),np.int32(pts-[x,y]),1)!=0
    if np.ptp(patch[inside],axis=1).max()>6:
        return
    gray = patch.max(2)
    if (gray[inside]>=250).mean()<.55:
        return
    _,labels,stats,_ = cv2.connectedComponentsWithStats(np.uint8(gray<230))
    area = stats[:,-1]
    contained = np.bincount(labels[inside],minlength=len(area))
    support = np.bincount(labels[probability[y:b,x:r]>45],minlength=len(area))
    letters = (contained==area) & (support>=.25*area) & (area>=3)
    letters[0] = False
    if letters.sum()<3:
        return
    core = letters[labels]
    erase = (cv2.dilate(np.uint8(core),np.ones((3,3),np.uint8))!=0) & inside
    ring = (cv2.dilate(np.uint8(erase),np.ones((3,3),np.uint8))!=0) & ~erase
    if not ring.any() or np.quantile(gray[ring],.05)<250:
        return
    retained = (gray<230)&~core
    if np.any(erase & (retained|existing[y:b,x:r])):
        return
    # The dark rule may enter the OCR polygon at its edge. Interior ink needs
    # to be accounted for before replacing any part of the reading.
    inner = cv2.erode(np.uint8(inside),np.ones((7,7),np.uint8))!=0
    if np.any(retained & inner):
        return
    yy,xx = np.where(core)
    if np.ptp(xx)<.65*np.ptp(pts[:,0]):
        return
    forbidden = ~inside | retained | existing[y:b,x:r]
    forbidden = cv2.dilate(np.uint8(forbidden),np.ones((3,3),np.uint8))!=0
    return (x,y,r,b),erase,forbidden,(xx.min(),xx.max(),yy.min(),yy.max())


def place_checklist_lines(clean,text,erase,layout,rows,by_id,existing,font_resolver,probability,minimum):
    headings = [np.asarray(d['polygon']) for d in by_id.values() if d.get('ocr_metadata',{}).get('checkbox_heading')]
    if not headings or probability is None:
        return False
    changed = False
    for row in rows:
        binding = row.get('binding') or {}
        members = binding.get('member_ids',[])
        if row.get('status')!='skipped' or row.get('reason_code')!='coverage_uncertain_geometry' or len(members)!=1:
            continue
        detection = by_id.get(members[0],{})
        if detection.get('reason')!='neighbor_overlap':
            continue
        poly = np.asarray(detection['polygon'])
        cx,cy = poly.mean(0)
        h = np.ptp(poly[:,1])
        if not any(v[:,0].min()<cx<v[:,0].max() and 0<v[:,1].mean()-cy<2.2*h for v in headings):
            continue
        trial = checklist_line(clean,poly,existing,probability)
        if trial is None:
            continue
        (x,y,r,b),strokes,forbidden,(left,right,top,bottom) = trial
        try:
            font,coverage = font_resolver(row)
            natural.validate_text(binding['text'],coverage)
        except ValueError:
            continue
        for height in range(bottom-top+1,9,-1):
            fit = lettering(binding['text'],font,right-left+1,height,minimum)
            if not fit:
                continue
            glyphs,size,lines = fit
            gx = (left+right+1-glyphs.width)//2
            gy = (top+bottom+1-glyphs.height)//2
            area = np.s_[gy:gy+glyphs.height,gx:gx+glyphs.width]
            if not np.any(forbidden[area] & (np.asarray(glyphs)>0)):
                break
        else:
            continue
        clean.paste((255,255,255),(x,y,r,b),Image.fromarray(strokes))
        patch = Image.new('RGBA',glyphs.size)
        patch.putalpha(glyphs)
        text.alpha_composite(patch,dest=(x+gx,y+gy))
        erase[y:b,x:r] |= strokes
        for mask in (existing,layout):
            mask[y:b,x:r][area] |= np.asarray(glyphs)>0
        row.update(status='automatic_composed',reason_code=None,reason=None,
                   cleanup_method='checklist_line_paper',render_quality='best_effort',
                   ruled_cell={'bounds':list(map(int,(x,y,r,b))),'font_size':size,'lines':lines,'text':binding['text']})
        changed = True
    return changed


def place(rgb, clean, text, erase, layout, rows, proposals, detections, font_resolver, *, probability=None, minimum=10):
    candidates = [row for row in rows if row['status']=='skipped' and row.get('binding')
                  and row.get('reason_code') not in ('uncorroborated_source','evaluation_cap','runtime_locality',
                        'alpha_containment','erasure_obstacle','typography_containment','replacement_unchanged_text')
                  and proposals[row['unit_id']].get('grouping_method')=='separate_printed_table_rows'
                  and len(row['binding']['member_ids'])==1]
    has_checklist = probability is not None and any(d.get('ocr_metadata',{}).get('checkbox_heading') for d in detections)
    if not (candidates or has_checklist):
        return False
    original = np.asarray(rgb)
    gray = original.max(2)
    tab_gray = None
    existing = np.asarray(text.getchannel('A'))!=0
    by_id = {v['source_id']:v for v in detections}
    outlines = {v['source_id']: np.asarray(v['polygon']) for v in detections}
    any_placed = place_checklist_lines(clean,text,erase,layout,rows,by_id,existing,font_resolver,probability,minimum)
    for row in candidates:
        detection = by_id[row['binding']['member_ids'][0]]
        strip = detection.get('ocr_metadata',{}).get('tab_strip')
        if strip:
            if tab_gray is None:
                tab_gray = np.asarray(rgb.convert('L'))
            cells = tab_interiors(tab_gray,strip['polygon'],strip['count'])
            cell = cells[strip['index']][-1] if cells else None
        else:
            cell = interior(gray,detection['polygon'])
        if cell is None:
            continue
        xx,top,bottom,slope = cell
        x,y,r,b = int(xx.min()),int(top.min()),int(xx.max()+1),int(bottom.max()+1)
        # Avoid borrowing width from a second label in the same row.
        right = max(point[0] for point in detection['polygon'])
        if detection.get('reason')=='confirmed_extended_printed_cell':
            right = max(right,detection.get('ocr_metadata',{}).get('printed_cell_readings',{}).get('right',right))
        if strip:
            right = r-1
        middle = (int(top[0])+int(bottom[0]))/2
        for other,poly in outlines.items():
            if other!=detection['source_id'] and poly[:,1].min()<middle<poly[:,1].max() and poly[:,0].min()>right:
                r = min(r,int(poly[:,0].min())-1)
        if r-x<16:
            continue
        top,bottom,xx = top[:r-x],bottom[:r-x],xx[:r-x]
        yy = np.arange(y,b)[:,None]
        inside = (yy>=top)&(yy<=bottom)
        bounds = np.s_[y:b,x:r]
        if (inside & (erase[bounds] | existing[bounds])).any():
            continue
        paper = original[bounds]
        if (paper[inside].min(1)>210).mean()<.40 or np.ptp(paper[inside],axis=1).mean()>8:
            continue
        # Estimate the exposed paper separately in every column. Vertical
        # hatching visible above and below a glyph remains in the repair.
        backdrop = np.maximum(original[top,xx],original[bottom,xx])
        if (backdrop.min(1)<190).mean()>.25:
            # Ink can touch both sampled edges in a slanted, tightly printed
            # row. Look for exposed paper in the same column before giving up.
            height = int(np.min(bottom-top))+1
            samples = original[top+np.arange(height)[:,None],xx]
            backdrop = np.maximum(backdrop,np.percentile(samples,82,axis=0).astype(np.uint8))
            if (backdrop.min(1)<190).mean()>.25:
                continue
        ink = inside & (xx<=right+2) & (paper.astype(np.int16)+8 < backdrop[None,:,:]).any(2)
        if ink.sum()<8:
            continue
        target = row['binding']['text']
        try:
            font_data, coverage = font_resolver(row)
            natural.validate_text(target,coverage)
        except ValueError:
            continue
        occupied = inside & (paper.astype(np.int16)+36<backdrop[None,:,:]).any(2) & ~ink
        beyond = np.flatnonzero(occupied.any(0) & (xx>right+2))
        width = int(beyond[0]) if beyond.size else r-x
        height = int(np.min(bottom-top))+1
        # Prefer the unobstructed span. Ink at the far edge of a row can
        # also fall in the blank space above or between English letters.
        for span in dict.fromkeys((width,r-x)):
            fit = lettering(target,font_data,span-1,height,minimum)
            if fit is None:
                continue
            letters,size,lines = fit
            glyphs = np.asarray(letters)
            mask = np.zeros(inside.shape,np.uint8)
            for j in range(letters.width):
                k = int(top[j]+(bottom[j]-top[j]+1-letters.height)//2-y)
                mask[k:k+letters.height,j] = glyphs[:,j]
            if mask.any() and not np.any((mask!=0)&(~inside|occupied)):
                break
        else:
            continue
        if strip and strip['count']==1:
            repair = title_repair(original,tab_gray,probability,cell,strip.get('extended',False))
            if repair is None:
                continue
            (left,top_edge,right_edge,bottom_edge),strokes,tone = repair
            patch = np.s_[top_edge:bottom_edge,left:right_edge]
            strokes &= ~(erase[patch] | existing[patch])
            if not strokes.any():
                continue
            clean.paste((tone,)*3,(left,top_edge,right_edge,bottom_edge),Image.fromarray(strokes))
            erase[patch] |= strokes
        else:
            tile = Image.fromarray(np.broadcast_to(backdrop,paper.shape).copy())
            clean.paste(tile,(x,y),Image.fromarray(ink))
            erase[bounds] |= ink
        layer = Image.new('RGBA',(r-x,b-y),(0,0,0,0))
        layer.putalpha(Image.fromarray(mask))
        text.alpha_composite(layer,dest=(x,y))
        layout[bounds] |= inside
        existing[bounds] |= mask!=0
        row.update(status='automatic_composed',reason_code=None,reason=None,
            cleanup_method='ruled_cell_paper',render_quality='best_effort',
            ruled_cell={'bounds':[x,y,r,b],'font_size':size,'slope':float(slope),'lines':lines,'text':target})
        any_placed = True
    return any_placed
