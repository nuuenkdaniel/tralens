"""Separate OCR rows where parallel printed rules continue into an adjacent cell."""
import re
import unicodedata

import numpy as np

from .types import OCREngine, Text_Box, Text_Group


def _bands(pixels, left, right, middle, height, pitch, tilt, *, wide):
    lo = max(0, int(left - (7 if wide else 2) * height))
    hi = min(pixels.shape[1], int(right + height))
    if hi - lo < 55:
        return False
    x = np.unique(np.linspace(lo, hi - 1, min(150, hi - lo)).astype(int))
    outside = x < left
    if outside.sum() < 9 or (wide and left-lo < 4*height):
        return False
    top = max(0, int(middle - 2*pitch - 30))
    bottom = min(pixels.shape[0], int(middle + 2*pitch + 30))
    gray = pixels[top:bottom, lo:hi].max(2).astype(np.int16)
    xx = x-lo
    # Narrow OCR rectangles on skewed scans can be axis-aligned even when
    # the longer, unobscured horizontal rules in the neighboring cell tilt.
    slopes = np.arange(-.15,.151,.025) if wide else (tilt, tilt-.025, tilt+.025)
    for slope in slopes:
        drift = (x - (left+right)/2) * slope
        bands = []
        for k in range(-2,3):
            centers = np.arange(round(middle+(k-.30)*pitch), round(middle+(k+.30)*pitch)+1)
            ys = np.rint(centers[:,None]+drift-top).astype(int)
            ys = ys[(ys.min(1)>=3)&(ys.max(1)<len(gray)-3)]
            if not len(ys):
                bands.append(False)
                continue
            stroke = np.minimum(gray[ys,xx],gray[ys+1,xx])
            sides = np.minimum(gray[ys-2,xx],gray[ys+3,xx])
            ink = sides-stroke >= 15
            continuation = ink[:,~outside].mean(1)>=.33 if wide else ink.mean(1)>=.68
            bands.append(np.any(continuation & (ink[:,outside].mean(1)>=(.69 if wide else .68))))
        if bands[2] and any(all(bands[i:i+3]) for i in range(3)):
            return True
    return False


def tab_interiors(gray, polygon, count):
    """Find a row of separately enclosed paper labels beneath one OCR polygon."""
    import cv2
    from PIL import Image

    pts = np.asarray(polygon)
    x,y = np.floor(pts.min(0)).astype(int)
    r,b = np.ceil(pts.max(0)).astype(int)
    h = b-y
    title = count==1
    if not (1<=count<=6 and 15<=h<=80 and 2.2*h<=r-x<=600):
        return []
    pad = 2*h if title else h//2
    left,top = max(0,x-pad),max(0,y-h//2)
    crop = gray[top:b+h//2,left:r+pad]
    _,labels,stats,centers = cv2.connectedComponentsWithStats(np.uint8(crop>180),connectivity=4)
    cells = []
    for i,(cx,cy,w,height,area) in enumerate(stats[1:],1):
        if not (.48*h<=height<=1.4*h and .65*height<=w<=5*height and area>(.35 if title else .40)*w*height
                and 0<cx and 0<cy and cx+w<crop.shape[1] and cy+height<crop.shape[0]):
            continue
        if cv2.pointPolygonTest(pts.astype(np.float32),tuple(centers[i]+[left,top]),False)<0:
            continue
        bounds = np.s_[cy:cy+height,cx:cx+w]
        contours,_ = cv2.findContours(np.uint8(labels[bounds]==i),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours,key=cv2.contourArea)
        if title:
            # Title-bar letters and motion lines sometimes touch the printed border.
            # Its convex inner face still follows the two long window rules.
            contour = cv2.convexHull(contour)
        if cv2.contourArea(contour)<(.55 if title else .74)*w*height:
            continue
        inside = cv2.drawContours(np.zeros((height,w),np.uint8),[contour],-1,1,-1).astype(bool)
        if title:
            inside = cv2.erode(np.uint8(inside),np.ones((3,3),np.uint8),
                               borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        chip = np.where(inside,crop[bounds],255)
        yy,xx = np.where(chip<140)
        if len(xx)<8:
            continue
        offset = np.array([left+cx,top+cy])
        glyph = cv2.boxPoints(cv2.minAreaRect(np.float32(np.column_stack((xx,yy)))))+offset
        # Leave a pixel of clearance at the rounded ends and along each rule.
        spans = inside.sum(0)
        if title:
            columns = np.flatnonzero((spans>.68*spans.max()) & (np.arange(w)+offset[0]>=x)
                                     & (np.arange(w)+offset[0]<=r))
        else:
            columns = np.flatnonzero(spans>.66*height)
        if len(columns)<12 or np.any(np.diff(columns)!=1):
            continue
        columns = columns[2:-2]
        upper = np.argmax(inside[:,columns],axis=0)+offset[1]+1
        lower = height-2-np.argmax(inside[::-1,columns],axis=0)+offset[1]
        frame = columns+offset[0],upper,lower,float((upper[-1]-upper[0])/max(1,len(columns)-1))
        cells.append((int(offset[0]),w,Image.fromarray(np.pad(chip,5,constant_values=255)).convert('RGB'),glyph,frame))
    cells.sort(key=lambda c:c[0])
    edge = 1.15*h if title else .45*h
    if (len(cells)!=count or abs(cells[0][0]-x)>edge or abs(cells[-1][0]+cells[-1][1]-r)>edge
            or any(abs(a[0]+a[1]-b[0])>.22*h for a,b in zip(cells,cells[1:]))):
        return []
    return cells


def title_surface(gray, cell):
    """Follow the paper between the long rules to the end of a clipped label."""
    import cv2
    columns,upper,lower,_ = cell
    center = columns.mean()
    step = max(1,len(columns)//4)
    slope = np.median((lower[step:]-lower[:-step])/step)
    if abs(slope)>.20:
        return
    low = np.median(lower-(columns-center)*slope)
    high = np.percentile(upper-(columns-center)*slope,20)
    height = int(low-high+1)
    if not 8<=height<=90:
        return
    xx = np.arange(max(0,columns[0]-int(height*.35)),min(gray.shape[1],columns[-1]+2*height))
    tops = np.rint(high+(xx-center)*slope).astype(int)
    if tops.min()<5 or tops.max()+height+5>gray.shape[0]:
        return
    strip = gray[tops+np.arange(height)[:,None],xx]
    exposed = strip[(strip>175)&(strip<248)]
    if exposed.size<.30*strip.size:
        return
    tone = int(np.median(exposed))
    same = abs(strip.astype(float)-tone)<6
    if same.mean()<.40:
        return
    columns_on_paper = np.uint8(same.mean(0)>.30)
    continuous = cv2.morphologyEx(columns_on_paper[None,:],cv2.MORPH_CLOSE,np.ones((1,7),np.uint8))[0]
    stops = np.flatnonzero((xx>columns[-1])&(continuous==0))
    end = int(stops[0])+3 if len(stops) else len(xx)
    return xx[:end],tops[:end],height,tone


def split_tab_labels(image, boxes, reader):
    """Split a merged OCR line only when each enclosed label reads independently."""
    from difflib import SequenceMatcher
    import cv2
    from .masked_lines import corrected

    gray = None
    result = []
    next_id = max((box.box_id for box in boxes if type(box.box_id) is int),default=-1)+1
    for box in boxes:
        words = box.text.split()
        result.append(box)
        if not (2<=len(words)<=6 and any(re.search('[ぁ-ヺ一-鿿]',v) for v in words)
                and all(re.fullmatch('[ぁ-ヺー一-鿿々A-Za-z0-9]{1,12}',v) for v in words)):
            continue
        if gray is None:
            gray = np.asarray(image.convert('L'))
        cells = tab_interiors(gray,box.poly,len(words))
        if any(cv2.pointPolygonTest(poly.astype(np.float32),tuple(other.poly.mean(0).astype(float)),False)>=0
               for _,_,_,poly,_ in cells for other in boxes if other is not box):
            continue
        parts = []
        for index,(_,_,chip,poly,_) in enumerate(cells):
            word = reader(chip)
            if (not word or SequenceMatcher(None,word,words[index]).ratio()<.60
                    or word!=reader(chip.resize((chip.width*3,chip.height*3)))):
                break
            part = corrected(next_id+index-1 if index else box.box_id,box.confidence,poly,word,words[index],box)
            part.reason = 'confirmed_tab_label'
            part.ocr_metadata['tab_strip'] = {'polygon':box.poly.tolist(),'count':len(words),'index':index}
            parts.append(part)
        if len(parts)==len(words):
            result[-1:] = parts
            next_id += len(parts)-1
    # Nearby window titles often occupy their own ruled strip. Use the same
    # physical enclosure, with a separate reading of the detector crop.
    anchors = [v for v in result if v.ocr_metadata.get('tab_strip') or v.reason=='confirmed_checkbox_item']
    if anchors:
        from PIL import Image
        from .masked_lines import source_views
        if gray is None:
            gray = np.asarray(image.convert('L'))
        for i,box in enumerate(result):
            if box in anchors or not re.fullmatch(r'[ァ-ヺーA-Za-z0-9ぁ-ゖ]{3,15}',box.text) or not re.search('[ァ-ヺ]',box.text):
                continue
            x,y,r,b = box.bbox
            if not any(abs(x-v.bbox[0]) < 2*(r-x) and abs(y-v.bbox[1]) < 6*(b-y) for v in anchors):
                continue
            cells = tab_interiors(gray,box.poly,1)
            if not cells:
                continue
            chip = cells[0][2]
            word = reader(chip)
            if (not word or not re.fullmatch('[ァ-ヺーぁ-ゖ]{3,18}',word)
                    or SequenceMatcher(None,box.text,word).ratio()<.4
                    or reader(chip.resize((chip.width*3,chip.height*3)))!=word
                    or not any(reader(view)==word for view in source_views(image,[box.poly]))):
                continue
            surface = title_surface(gray,cells[0][-1])
            if surface is None:
                continue
            xx,tops,height,_ = surface
            paper = np.asarray(image.crop((int(xx[0]),int(tops.min()-4),
                                          int(xx[-1]+1),int(tops.max()+height+4))))
            if np.quantile(np.ptp(paper,axis=2),.95)>8:
                continue
            full = gray[tops+np.arange(height)[:,None],xx]
            view = Image.fromarray(np.pad(full,4,constant_values=255)).convert('RGB')
            reading = reader(view)
            if (not reading or not reading.startswith(word) or len(reading)>len(word)+3
                    or not re.fullmatch('[ァ-ヺーぁ-ゖ]+',reading)
                    or reader(view.resize((view.width*3,view.height*3)))!=reading):
                continue
            polygon = box.poly
            if reading!=word:
                polygon = np.asarray([(xx[0],tops[0]),(xx[-1],tops[-1]),
                                      (xx[-1],tops[-1]+height-1),(xx[0],tops[0]+height-1)])
            title = corrected(box.box_id,box.confidence,polygon,reading,box.text,box)
            title.reason = 'confirmed_tab_label'
            title.ocr_metadata['tab_strip'] = {'polygon':box.poly.tolist(),'count':1,'index':0,'extended':reading!=word}
            result[i] = title
    return result


def checkbox_headings(image, boxes, reader):
    """Keep a section title independent of the checked software names below it."""
    import cv2
    from copy import copy
    from .masked_lines import corrected, source_views

    items = [v for v in boxes if v.confidence>.88 and re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{2,35}',v.text)]
    if len(items)<2:
        return boxes
    gray = np.asarray(image.convert('L'))
    marks = []
    for item in items:
        x,y,r,b = item.bbox
        height = min(np.linalg.norm(edge) for edge in np.roll(item.poly,1,axis=0)-item.poly)
        if not 14<=height<=72 or r-x<1.7*height:
            continue
        left,top = max(0,int(x-1.3*height)),max(0,int(y-.52*height))
        patch = gray[top:int(y+1.36*height),left:int(x+1.35*height)]
        shapes,_ = cv2.findContours(np.uint8(patch<150),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for shape in shapes:
            (cx,cy),(w,h),_ = cv2.minAreaRect(shape)
            if not (.42*height<=min(w,h)<=max(w,h)<=.95*height and min(w,h)>.82*max(w,h)
                    and cv2.contourArea(shape)>.64*w*h):
                continue
            bx,by,bw,bh = cv2.boundingRect(shape)
            core = patch[by+bh//4:by+3*bh//4,bx+bw//4:bx+3*bw//4]
            if core.size and .06<np.mean(core<150)<.64:
                candidates.append((w*h,cx+left,cy+top))
        if candidates:
            _,cx,cy = max(candidates)
            marks.append((cy,cx,height,item))
    changed = {}
    marks.sort(key=lambda m:m[0])
    for (y,x,h,first),(yy,xx,hh,second) in zip(marks,marks[1:]):
        if not (.75*h<yy-y<2.15*h and abs(x-xx)<.45*h and .65<hh/h<1.4):
            continue
        headers = [v for v in boxes if 0<y-v.bbox[3]<.95*h and abs(v.bbox[0]-x)<.85*h
                   and v.bbox[2]-v.bbox[0]>1.8*h and re.search('[ぁ-ヺ一-鿿]',v.text)]
        if len(headers)!=1:
            continue
        head = headers[0]
        views = source_views(image,[head.poly])
        word = reader(views[0])
        if (not isinstance(word,str) or not re.fullmatch('[ぁ-ヺー一-鿿々]{2,22}',word)
                or word[0]!=head.text[0] or not all(reader(v)==word for v in views[1:])):
            continue
        changed[head.box_id] = corrected(head.box_id,head.confidence,head.poly,word,head.paddle_text,head)
        changed[head.box_id].ocr_metadata['checkbox_heading'] = True
        for item in (first,second):
            marked = copy(item)
            marked.reason = 'confirmed_checkbox_item'
            changed[item.box_id] = marked
    return [changed.get(v.box_id,v) for v in boxes] if changed else boxes


def printed_grid(first, second, pixels):
    upper, lower = sorted((first, second), key=lambda box: sum(box.bbox[1::2]))
    ax, ay, ar, ab = upper.bbox
    bx, by, br, bb = lower.bbox
    left, right = max(ax, bx), min(ar, br)
    height = min(ab - ay, bb - by)
    pitch = ((by + bb) - (ay + ab)) / 2
    if height < 9 or right-left < max(25, 1.4*height) or not .60*height <= pitch <= 2.2*height:
        return False
    slopes = []
    for box in (upper,lower):
        edges = np.roll(box.poly,-1,axis=0)-box.poly
        dx,dy = max(edges,key=lambda edge: abs(edge[0]))
        slopes.append(dy/dx)
    tilt = np.median(slopes)
    if abs(tilt) > .2:
        return False
    args = (pixels, left, right, (ay+ab+by+bb)/4, height, pitch, tilt)
    return ((pitch <= 1.6*height and _bands(*args, wide=False))
            or _bands(*args, wide=True))


def split(groups, pixels, excluded, *, confirmed=()):
    result = []
    for group in groups:
        boxes = group.group
        if any(box.box_id in confirmed for box in boxes):
            # Each recovered row was read independently between its own rules.
            # Keep it distinct even when an axis-aligned OCR box crosses a rule.
            rest = []
            for box in boxes:
                if box.box_id in confirmed:
                    if rest:
                        result.append(Text_Group(rest,method=group.method,direction=group.direction))
                        rest = []
                    result.append(Text_Group([box],method='separate_printed_table_rows',direction='horizontal'))
                else:
                    rest.append(box)
            if rest:
                result.append(Text_Group(rest,method=group.method,direction=group.direction))
            continue
        if (group.direction != 'horizontal' or not 2 <= len(boxes) <= 12
                or any(box.box_id in excluded for box in boxes)):
            result.append(group)
            continue
        cuts = [i for i in range(1,len(boxes)) if printed_grid(boxes[i-1],boxes[i],pixels)]
        if not cuts or (len(boxes)>2 and len(cuts)<2):
            result.append(group)
            continue
        for start,end in zip([0,*cuts],[*cuts,len(boxes)]):
            result.append(Text_Group(boxes[start:end],method='separate_printed_table_rows',direction='horizontal'))
    anchors = [g.group[0] for g in result if g.method=='separate_printed_table_rows' and len(g.group)==1]
    def center(v):
        return (v.bbox[1]+v.bbox[3])/2
    for i, group in enumerate(result):
        if not anchors or group.method=='separate_printed_table_rows' or len(group.group)!=1:
            continue
        box = group.group[0]
        x,y,r,b = box.bbox
        if box.box_id in excluded or r-x<1.4*(b-y):
            continue
        aligned = [v for v in anchors if abs(v.bbox[0]-x)<.35*(b-y)
                   and abs(center(v)-center(box))<9*(b-y)]
        if len(aligned)>=3:
            # A new singleton can be several missing OCR rows down. Require
            # the same parallel grid at its own location before tagging it.
            near = min(aligned,key=lambda v:abs(center(v)-center(box)))
            if printed_grid(near,box,pixels):
                result[i] = Text_Group([box],method='separate_printed_table_rows',direction='horizontal')
            else:
                pitches = np.diff(sorted(center(v) for v in aligned))
                pitch = np.median(pitches)
                if .65*(b-y)<=pitch<=1.9*(b-y) and _bands(pixels,x,r,center(box)-pitch/2,b-y,pitch,0,wide=True):
                    result[i] = Text_Group([box],method='separate_printed_table_rows',direction='horizontal')
    return result


def interior(gray, polygon):
    """Fit two neighboring rules from their long continuation to the left."""
    points = np.asarray(polygon)
    x, y = points.min(axis=0).astype(int)
    r, b = points.max(axis=0).astype(int)
    height = b-y
    if not (10 <= height <= 44 and 24 <= r-x <= 180 and x>=50):
        return
    mid = (y+b)//2
    left = np.arange(max(2,x-5*height),x-4)
    if len(left)<48:
        return
    offsets = np.arange(-height,height+1)
    best = None
    for slope in np.arange(-.15,.151,.0125):
        yy = np.rint(mid+offsets[:,None]+(left-(x+r)/2)*slope).astype(int)
        if yy.min()<3 or yy.max()+3>=gray.shape[0]:
            return
        stroke = np.minimum(gray[yy,left],gray[yy+1,left])
        sides = np.minimum(gray[yy-2,left],gray[yy+3,left])
        ink = sides.astype(np.int16)-stroke>=15
        score = np.maximum(ink.mean(1), ink[:,-min(82,len(left)):].mean(1))
        for upper in np.flatnonzero((offsets < -2)&(score>=.66)):
            for lower in np.flatnonzero((offsets > 2)&(score>=.66)):
                pitch = offsets[lower]-offsets[upper]
                if not max(11,.58*height)<=pitch<=min(39,1.85*height):
                    continue
                rank = min(score[upper],score[lower]) - .012*abs(offsets[upper]+offsets[lower])
                if best is None or rank > best[0]:
                    best = rank,mid+offsets[upper],mid+offsets[lower],slope
    if best is None:
        return
    _, upper, lower, slope = best
    # Confirm the near column edge independently over several row heights.
    ys = np.arange(max(2,mid-height*2),min(gray.shape[0]-2,mid+height*2))
    xx = np.arange(max(2,x-6),min(x+7,gray.shape[1]-2))
    column = gray[np.ix_(ys,xx)].astype(np.int16)
    sides = np.minimum(gray[np.ix_(ys,xx-2)],gray[np.ix_(ys,xx+2)])
    support = (sides-column>=15).mean(0)
    if not len(support) or support.max()<.48:
        return
    start = xx[support.argmax()]+2
    end = min(gray.shape[1]-2,r+3)
    # A heavy panel edge can end the column sooner than the OCR rectangle.
    # A faint vertical divider can also bound spare space in a short cell.
    xx = np.arange(start,min(gray.shape[1]-2,end+2*height))
    if len(xx)<16:
        return
    slab = gray[np.ix_(ys,xx)].astype(np.int16)
    frame = (slab<80).mean(0)>.90
    heavy = np.flatnonzero(frame[:-2] & frame[1:-1] & frame[2:])
    contrast = np.minimum(gray[np.ix_(ys,xx-2)],gray[np.ix_(ys,xx+2)])-slab
    dividers = np.flatnonzero((contrast>=15).mean(0)>.82)
    # Before the end of the recognized word, a thin vertical stroke may be
    # hatching crossing the cell. Only a solid frame truncates that word.
    ends = [int(xx[i])-1 for i in dividers if xx[i]>=r]
    if heavy.size:
        ends.append(int(xx[heavy[0]])-1)
    if ends:
        end = min(ends)
    if end-start<16:
        return
    xx = np.arange(start,end)
    top = np.rint(upper+(xx-(x+r)/2)*slope).astype(int)+2
    bottom = np.rint(lower+(xx-(x+r)/2)*slope).astype(int)-2
    if top.min()<0 or bottom.max()>=gray.shape[0] or (bottom-top).min()<8:
        return
    return xx,top,bottom,slope


def missing_cells(proposals, boxes, pixels):
    """Interpolate an interior gap in an otherwise measured column of short print."""
    singles = [p.crop_bounds for p in proposals if p.grouping_method=='separate_printed_table_rows'
               and len(p.member_ids)==1]
    if len(singles)<4:
        return
    gray = pixels.max(2)
    emitted = set()
    for x,y,r,b in singles:
        height = b-y
        column = sorted((a for a in singles if abs(a[0]-x)<.35*height),key=lambda a:a[1]+a[3])
        if len(column)<4:
            continue
        centers = np.array([(t+bot)/2 for l,t,right,bot in column])
        gaps = np.diff(centers)
        regular = gaps[(gaps>=.62*height)&(gaps<=1.7*height)]
        if len(regular)<3:
            continue
        pitch = np.median(regular)
        for j,gap in enumerate(gaps):
            steps = round(gap/pitch)
            if not 2<=steps<=6 or abs(gap/steps-pitch)>.15*pitch:
                continue
            for k in range(1,steps):
                mid = centers[j]+k*gap/steps
                # Both bounding anchors must share a column. Do not claim a
                # row if any real detector already covers its middle.
                l = round(column[j][0]+k*(column[j+1][0]-column[j][0])/steps)
                width = max(right-left for left,_,right,_ in column)
                if any(box.bbox[1]<=mid<=box.bbox[3] and box.bbox[0]<l+width and box.bbox[2]>l+3 for box in boxes):
                    continue
                key = l,round(mid)
                if key in emitted or len(emitted)>=12:
                    continue
                emitted.add(key)
                t,bot = round(mid-.5*pitch),round(mid+.5*pitch)
                cell = interior(gray,[(l,t),(l+width,t),(l+width,bot),(l,bot)])
                if cell is not None:
                    yield cell


def reading_extent(gray, cell):
    """Read up to the inside edge of a thick frame when it bounds the last cell."""
    xx,top,bottom,slope = cell
    start,end = int(xx[0]),int(xx[-1])+1
    height = int(np.median(bottom-top))+1
    mid = int(np.median(top+bottom)/2)
    cols = np.arange(max(start+8,end-2),min(gray.shape[1],end+height+5))
    strip = gray[max(0,mid-height*2):mid+height*2,cols]
    solid = (strip<96).mean(0)>.9
    edge = np.flatnonzero(solid[:-3]&solid[1:-2]&solid[2:-1]&solid[3:])
    if edge.size:
        end = max(end,int(cols[edge[0]]))
    xx = np.arange(start,end)
    top = np.rint(top[0]+(xx-start)*slope).astype(int)
    if top.min()<0 or top.max()+height>=gray.shape[0]:
        return cell
    return xx,top,top+height-1,slope


def chip(pixels, cell, *, normalize=False):
    from PIL import Image
    xx,top,bottom,_ = cell
    crop = pixels[top+np.arange(int(np.min(bottom-top))+1)[:,None],xx]
    if normalize:
        # Estimate the bright paper independently in every column so vertical
        # speed lines through small print do not become additional kana.
        tone = np.maximum(64,np.percentile(crop,82,axis=0))
        crop = np.clip(crop.astype(float)*255/tone,0,255).astype(np.uint8)
    bordered = np.pad(crop,((4,4),(4,4),(0,0)),constant_values=255)
    return Image.fromarray(bordered).resize((bordered.shape[1]*4,bordered.shape[0]*4),Image.Resampling.BICUBIC)


def loanword_reading(source, alternate, known):
    """Resolve a voicing mark only if another reader saw it and the word exists."""
    if len(source)!=len(alternate) or source==alternate:
        return
    m = re.fullmatch(r'([ァ-ヺー]{3,12})([ぁ-ゖ]{0,4})',source)
    if not m or known(m[1]) or not alternate.endswith(m[2]):
        return
    word = m[1]
    diffs = [(i,a,b) for i,(a,b) in enumerate(zip(word,alternate)) if a!=b]
    if not 1<=len(diffs)<=2:
        return
    verified = []
    for i,a,b in diffs:
        # Restrict substitutions to the same base kana. Other characters
        # that only one reader recognized still come from the original OCR.
        if unicodedata.normalize('NFD',a)[0]!=unicodedata.normalize('NFD',b)[0]:
            continue
        candidate = word[:i]+b+word[i+1:]
        if known(candidate):
            verified.append(candidate+m[2])
    return verified[0] if len(verified)==1 else None


def corroborated_box(box, text, reason, readings):
    # Preview binds the source text and geometry together. Preserve the
    # detector's original reading and provenance beside the corroborated one.
    new = Text_Box(box.box_id,box.confidence,box.poly,text,review=1)
    new.selected_engine = OCREngine.HAYAI
    new.reason = reason
    new.ocr_metadata = {**box.ocr_metadata,'printed_cell_readings':{
        'original':box.paddle_text,'original_engine':box.selected_engine.value,'original_reason':box.reason,
        **readings}}
    return new


def printed_loanwords(boxes, proposals, pixels, reader):
    words = {p.member_ids[0] for p in proposals if p.grouping_method=='separate_printed_table_rows'
             and len(p.member_ids)==1}
    candidates = [b for b in boxes if b.box_id in words and b.selected_engine==OCREngine.PADDLE
                  and re.fullmatch(r'[ァ-ヺー]{3,12}[ぁ-ゖ]{0,4}',b.text)]
    if not candidates:
        return {}
    try:
        from fugashi import Tagger
        import unidic_lite
        tagger = Tagger(f'-d "{unidic_lite.DICDIR}"')
    except (ImportError, RuntimeError):
        return {}
    def known(word):
        entries = list(tagger(word))
        return len(entries)==1 and not entries[0].is_unk and (entries[0].feature.lemma or '').startswith(word+'-')
    gray = pixels.max(2)
    corrected = {}
    for box in candidates[:12]:
        if known(re.match(r'[ァ-ヺー]+',box.text)[0]):
            continue
        cell = interior(gray,box.poly)
        if cell is None:
            continue
        try:
            reading = reader(chip(pixels,reading_extent(gray,cell))).strip()
        except (OSError,RuntimeError,ValueError):
            continue
        selected = loanword_reading(box.text,reading,known)
        if selected:
            corrected[box.box_id] = corroborated_box(box,selected,'corroborated_printed_voicing_mark',
                {'native':box.text,'hayai':reading,'dictionary_reading':selected})
    return corrected


def discover(engine, image, reader):
    from .automatic_units import propose_page_from_boxes

    boxes = engine.raw_detections[0]
    if not boxes or len(boxes)>256 or any(type(b.box_id) is not int for b in boxes):
        return
    old = engine.automatic_unit_proposals[0]
    if not any(p.grouping_method=='separate_printed_table_rows' for p in old.proposals):
        return
    pixels = np.asarray(image)
    corrected = printed_loanwords(boxes,old.proposals,pixels,reader)
    gray = pixels.max(2)
    def agreement(cell):
        zoom = chip(pixels,cell,normalize=True)
        try:
            result = engine.ocr.predict(np.asarray(zoom))
            if not result:
                return
            readings = result[0]
            if len(readings['rec_texts'])==1 and len(readings['rec_scores'])==1 and readings['rec_scores'][0]>=.80:
                native = readings['rec_texts'][0].strip()
                if re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿々]{2,14}',native) and reader(zoom).strip()==native:
                    return native,float(readings['rec_scores'][0])
        except (OSError,RuntimeError,ValueError):
            return
    ids = {p.member_ids[0] for p in old.proposals if p.grouping_method=='separate_printed_table_rows'
           and len(p.member_ids)==1}
    incomplete = [b for b in boxes if b.box_id in ids and (b.text.endswith('の') or b.selected_engine==OCREngine.FRONTIER)]
    detector_ids = {b.box_id for b in boxes if b.selected_engine!=OCREngine.FRONTIER}
    for box in incomplete[:12]:
        x,y,r,b = box.bbox
        # Use native detector coordinates to measure the column. An extra
        # model-supplied box can cross its border by a pixel or two.
        usual_right = max((p.crop_bounds[2] for p in old.proposals if p.grouping_method=='separate_printed_table_rows'
                           and len(p.member_ids)==1 and p.member_ids[0] in detector_ids
                           and abs(p.crop_bounds[0]-x)<.35*(b-y) and abs(p.crop_bounds[1]-y)<9*(b-y)),default=r)
        cell = interior(gray,[(x,y),(max(r,usual_right),y),(max(r,usual_right),b),(x,b)])
        if cell is None:
            continue
        cell = reading_extent(gray,cell)
        check = agreement(cell)
        if check is None:
            continue
        native,_ = check
        if native==box.text or (not native.startswith(box.text) and not
                (box.selected_engine==OCREngine.FRONTIER and native[:3]==box.text[:3])):
            continue
        corrected[box.box_id] = corroborated_box(box,native,'confirmed_extended_printed_cell',
            {'paddle':native,'hayai':native,'right':int(cell[0][-1])})
    added = []
    for cell in missing_cells(old.proposals,boxes,pixels):
        cell = reading_extent(gray,cell)
        check = agreement(cell)
        if check is None:
            continue
        native,score = check
        xx,top,bottom,_ = cell
        polygon = [(int(xx[0]-1),int(top[0]-1)),(int(xx[-1]+1),int(top[-1]-1)),
                   (int(xx[-1]+1),int(bottom[-1]+1)),(int(xx[0]-1),int(bottom[0]+1))]
        new = Text_Box(max(b.box_id for b in boxes)+len(added)+1,score,polygon,native)
        new.selected_engine = OCREngine.HAYAI
        new.reason = 'confirmed_printed_cell'
        new.ocr_metadata = {'candidates':{'paddle':native,'hayai':native},'source':'measured_ruled_cell'}
        added.append(new)
    if added or corrected:
        boxes = [corrected.get(b.box_id,b) for b in boxes]
        original_ids = {b.box_id for b in boxes}
        updated = propose_page_from_boxes(boxes+added,engine.image_path,complementary_ownership=True,regroup_companions=True)
        def membership(batch):
            return {(p.main_ids,p.member_ids,p.annotation_ids) for p in batch.proposals if original_ids.intersection(p.member_ids)}
        if membership(old)==membership(updated) and all(len(p.member_ids)==1 for p in updated.proposals
                if not original_ids.intersection(p.member_ids)):
            engine.raw_detections = [boxes+added]
            engine.automatic_unit_proposals = [updated]
