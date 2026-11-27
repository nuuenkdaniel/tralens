"""Look for missing copies of an independently read sound in painted panels."""
import cv2
import numpy as np
from PIL import ImageOps

from . import reader_geometry, sound_readings
from .automatic_units import propose_page_from_boxes
from .paddle_result import associate_paddle_result
from .types import OCRStatus, Text_Box


POLICY = 'local_rotated_outline_sound_v1'


def zones(rgb, polygons):
    height,width=rgb.shape[:2]
    if height*width>8_000_000 or (np.ptp(rgb[::8,::8],axis=2)>24).mean()<.025:
        return []
    occupied=np.zeros((height,width),np.uint8)
    for polygon in polygons:
        cv2.fillPoly(occupied,[np.asarray(polygon,np.int32)],1)
    contours,tree=cv2.findContours((rgb.max(2)<72).astype('uint8'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if tree is None or len(contours)>60000:
        return []
    white=(rgb.min(2)>160)&(np.ptp(rgb,axis=2)<50)
    seeds=[]
    for contour,parent in zip(contours,tree[0,:,3]):
        if parent<0:
            continue
        x,y,w,h=cv2.boundingRect(contour)
        span=max(w,h)
        if not (8<=span<=85 and min(w,h)>=3 and 1.6<=span/min(w,h)<=18 and w*h<=1800):
            continue
        shape=np.zeros((h,w),np.uint8)
        cv2.drawContours(shape,[contour-np.int32([x,y])],-1,1,-1)
        if (white[y:y+h,x:x+w] & (shape!=0)).sum()>=max(6,w*h*.10):
            seeds.append((x+w//2,y+h//2,span))
    if not 7<=len(seeds)<=8192:
        return []
    field=np.zeros_like(occupied)
    for x,y,span in seeds:
        dx,dy=min(35,2*span),min(17,span)
        cv2.rectangle(field,(x-dx,y-dy),(x+dx,y+dy),1,-1)
    _,labels,stats,_=cv2.connectedComponentsWithStats(field,8)
    centers=np.asarray(seeds)
    xs,ys=centers[:,:2].T
    groups=labels[ys,xs]
    totals=np.bincount(groups,minlength=len(stats))
    covered=np.bincount(groups,weights=occupied[ys,xs],minlength=len(stats))
    result=[]
    for i,(x,y,w,h,_) in enumerate(stats[1:],1):
        if (totals[i]<7 or covered[i]<2 or totals[i]-covered[i]<4 or not 100<=w<=1400 or h>600
                or (np.ptp(rgb[y:y+h,x:x+w],axis=2)>24).mean()<.10):
            continue
        result.append(tuple(map(int,(max(0,x-22),max(0,y-22),min(width,x+w+22),min(height,y+h+22)))))
    return result[:4]


def views(image, polygon):
    tight=reader_geometry.pixels(image,reader_geometry.plan(image.size,polygon))
    axial=reader_geometry.pixels(image,reader_geometry.plan(image.size,polygon,view='axial'))
    w,h=tight.size
    return (tight,axial,ImageOps.expand(tight,border=max(2,round(h*.12)),fill='white'),
            tight.crop((0,0,w//2,h)),tight.crop((w//2,0,w,h)))


def confirmed_text(readings):
    if len(readings)!=5 or any(not isinstance(s,str) or len(s)>32 for s in readings):
        return None
    first,axial,padded,left,right=readings
    motif=sound_readings.repeat(first)
    if not motif or any(sound_readings.repeat(s)!=motif for s in (axial,padded)):
        return None
    lengths=list(map(len,(first,axial,padded)))
    spread=max(lengths)-min(lengths)
    if first in (axial,padded) and spread<=len(motif):
        return first
    if spread>3*len(motif) or min(len(left),len(right))<len(motif):
        return None
    joined=left+right
    for target in (first,axial,padded):
        if joined==target:
            return target
        # A clipped edge can introduce one kana. Each disjoint half must
        # still contain a full beat, and a whole view must supply the wording.
        if (motif in left and motif in right and len(joined)==len(target)+1
                and ((joined[1:]==target and 'ァ'<=joined[0]<='ヺ')
                     or (joined[:-1]==target and joined[-1]==motif[0]))):
            return target
    return None


def anchors(image, boxes, zone, reader):
    left,top,right,bottom=zone
    found={}
    candidates=[]
    for box in boxes:
        x,y,r,b=box.bbox
        if (not left<=(x+r)/2<=right or not top<=(y+b)/2<=bottom or r-x<1.15*(b-y)
                or not 18<=b-y<=220 or box.confidence>.94):
            continue
        candidates.append(box)
    for box in sorted(candidates,key=lambda b:-cv2.contourArea(b.poly))[:8]:
        try:
            tight,axial,*_=views(image,box.poly.tolist())
        except ValueError:
            continue
        a,b=reader(tight),reader(axial)
        motif=sound_readings.repeat(a)
        if motif and sound_readings.repeat(b)==motif and max(len(a),len(b))<=32:
            edges=np.roll(box.poly,-1,axis=0)-box.poly
            dx,dy=max(edges,key=lambda edge:abs(edge[0]))
            found.setdefault(motif,[]).append((box.box_id,-1 if dx*dy<=0 else 1,a,b))
    return found


def outlined_surface(rgb):
    """Present light glyph interiors as dark print on a plain field."""
    dark=(rgb.max(2)<72).astype(np.uint8)
    contours,tree=cv2.findContours(dark,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    gray=rgb.min(2)
    surface=np.full(gray.shape,255,np.uint8)
    if tree is None:
        return np.repeat(surface[:,:,None],3,axis=2)
    for c,parent in zip(contours,tree[0,:,3]):
        if parent<0:
            continue
        x,y,w,h=cv2.boundingRect(c)
        if not 6<=w*h<=4000 or max(w,h)>100:
            continue
        body=np.zeros((h,w),np.uint8)
        cv2.drawContours(body,[c-np.int32([x,y])],-1,1,-1)
        loc=np.s_[y:y+h,x:x+w]
        keep=(body!=0)&(gray[loc]>100)&(np.ptp(rgb[loc],axis=2)<55)
        surface[loc][keep]=255-gray[loc][keep]
    return np.repeat(surface[:,:,None],3,axis=2)


def hollow_letter_shapes(gray):
    """Isolate enclosed light lettering, excluding pen lines connected to the surrounding drawing."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray < 128))
    if count > 4000:
        return
    mask = np.zeros_like(gray)
    pieces = 0
    for label, (x,y,w,h,area) in enumerate(stats[1:],1):
        if (min(x,y) <= 1 or x+w >= gray.shape[1]-1 or y+h >= gray.shape[0]-1
                or min(w,h)<12 or area<120):
            continue
        tile = np.s_[y:y+h,x:x+w]
        contours, tree = cv2.findContours(np.uint8(labels[tile]==label), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [cv2.contourArea(c) for c,parent in zip(contours,tree[0,:,3]) if parent>=0]
        if sum(a for a in holes if a>30) < .55*area:
            continue
        cv2.drawContours(mask[tile],[c for c,parent in zip(contours,tree[0,:,3]) if parent<0],-1,255,-1)
        pieces += 1
    if 2<=pieces<=18:
        return mask


def blank_hollow_repeat(image, box, reader):
    """Read a repeated effect as a whole and confirm each beat in disjoint slices."""
    import re
    from PIL import Image
    from .masked_lines import bind, corrected, source_views
    from .recognition_units import image_fingerprint
    if box.text or box.reason != 'blank_paddle':
        return
    l,t,r,b = box.bbox
    if not 80<=min(r-l,b-t)<=max(r-l,b-t)<=500:
        return
    gray = np.asarray(image.crop((l,t,r+1,b+1)).convert('L'))
    mask = hollow_letter_shapes(gray)
    if mask is None:
        return
    x,y,w,h = cv2.boundingRect(mask)
    if not 1.35*w <= h <= 4*w or not .12 < np.count_nonzero(mask)/(w*h) < .8:
        return
    native = image.crop((l+x,t+y,l+x+w,t+y+h))
    margin = max(3,round(w*.05))
    tile = np.s_[y:y+h,x:x+w]
    isolated = Image.fromarray(np.where(mask[tile]!=0,gray[tile],255)).convert('RGB')
    isolated = ImageOps.expand(isolated,margin,fill='white')
    word = reader(isolated)
    if not re.fullmatch(r'([ぁ-ゖァ-ヺー]{2,4})\1',word or '') or reader(native)!=word:
        return
    motif = word[:len(word)//2]
    if len(set(motif)) < 2:
        return
    # Drawn repetitions often have uneven heights. Never present the same ink in both halves.
    for fraction in (.5,.57,.43):
        split = round(isolated.height*fraction)
        if (reader(isolated.crop((0,0,isolated.width,split)))==motif
                and reader(isolated.crop((0,split,isolated.width,isolated.height)))==motif):
            break
    else:
        return
    left,top,right,bottom = l+x,t+y,l+x+w,t+y+h
    poly = [(left,top),(right-1,top),(right-1,bottom-1),(left,bottom-1)]
    crops = source_views(image,[np.asarray(poly)])
    if any(reader(crop)!=word for crop in crops):
        return
    bounds = [max(l,left-8),max(t,top-8),min(r+1,right+8),min(b+1,bottom+8)]
    region = image.crop(bounds)
    local = np.s_[bounds[1]-t:bounds[3]-t,bounds[0]-l:bounds[2]-l]
    if not np.array_equal(hollow_letter_shapes(np.asarray(region.convert('L'))),mask[local]):
        return
    member = corrected(box.box_id,box.confidence,poly,word,box.paddle_text,box)
    bind([member],word,crops,'vertical')
    member.ocr_metadata['hollow_print'] = {'bounds':bounds,'digest':image_fingerprint(region),'repeated_beat':motif}
    return member


def bordered_strokes(gray, *, fine=False):
    """Separate thick outline type from fine illustration by measuring its ink collar."""
    count,labels,stats,_ = cv2.connectedComponentsWithStats(np.uint8(gray>160),connectivity=4)
    if count>2048:
        return
    core = np.zeros_like(gray,bool)
    mask = np.zeros_like(core)
    pieces = 0
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        if min(x,y)<2 or x+w>=gray.shape[1]-1 or y+h>=gray.shape[0]-1 or min(w,h)<4 or area<32:
            continue
        tile = np.s_[max(0,y-12):y+h+12,max(0,x-12):x+w+12]
        interior = labels[tile]==label
        distance = cv2.distanceTransform(np.uint8(~interior),cv2.DIST_L2,5)
        ink = gray[tile]<160
        # A keyboard key or a speed line has a much thinner border than
        # outlined display lettering. Measure around the entire component.
        edge = 3 if fine else 4
        collar = (distance>1)&(distance<=edge)
        if np.mean(ink[collar]) < (.92 if fine else .94):
            continue
        for radius in range(edge,11):
            ring = (distance>radius)&(distance<=radius+1)
            if not ring.any() or ink[ring].mean()<.4:
                break
        else:
            continue
        core[tile] |= interior
        mask[tile] |= interior | ((distance<=radius+2)&(gray[tile]<235))
        pieces += 1
    if 4 <= pieces <= 64:
        return core,cv2.dilate(np.uint8(mask),np.ones((3,3),np.uint8))!=0


def outlined_display(image, box, reader):
    """Verify wide, outlined dialogue using its light interiors as a second reading."""
    import re,unicodedata
    from PIL import Image
    from .masked_lines import bind,corrected,source_views
    from .recognition_units import image_fingerprint
    x,y,r,b = box.bbox
    blank = not box.text and box.reason == 'blank_paddle'
    if blank:
        if not 70<=b-y<=260 or not 1.1*(b-y)<=r-x<=380:
            return
    elif (box.confidence>=.82 or not re.fullmatch(r'[ぁ-ゖァ-ヺ一-鿿ー]{2,10}[！!？?]{1,3}',box.text)
            or not 65<=b-y<=300 or not 1.7*(b-y)<=r-x<=650):
        return
    bounds=[x,y,min(r+1,image.width),min(b+1,image.height)]
    native=image.crop(bounds)
    gray=np.asarray(native.convert('L'))
    parts=bordered_strokes(gray,fine=blank)
    if parts is None:
        return
    core,mask = parts
    isolated=Image.fromarray(np.uint8(~core)*255).convert('RGB')
    l,t,w,h=cv2.boundingRect(np.uint8(mask))
    if w<.55*(r-x) or h<.40*(b-y):
        return
    views=source_views(image,[box.poly])
    word=reader(isolated.crop((l,t,l+w,t+h)))
    if blank:
        if not re.fullmatch(r'[ァ-ヺ]{2}',word or ''):
            return
        # Read two disjoint parts of the actual outline too. A filled
        # silhouette alone can turn a pair of speed lines into a kana.
        ink=np.uint8(gray<160)
        edges=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        edges=Image.fromarray(np.uint8(~(edges.astype(bool)&mask))*255).convert('RGB')
        if reader(edges)!=word:
            return
        for fraction in (.44,.50,.56):
            mid=round(native.width*fraction)
            if (reader(edges.crop((0,0,mid,native.height)))==word[0]
                    and reader(edges.crop((mid,0,native.width,native.height)))==word[1]):
                break
        else:
            return
    elif unicodedata.normalize('NFKC',word or '')!=unicodedata.normalize('NFKC',box.text):
        return
    word=word if blank else box.text
    if any(unicodedata.normalize('NFKC',reader(v) or '')!=unicodedata.normalize('NFKC',word) for v in views):
        return
    member=corrected(box.box_id,box.confidence,box.poly,word,box.paddle_text,box)
    bind([member],word,views,'horizontal')
    member.ocr_metadata['outlined_display']={'bounds':bounds,'digest':image_fingerprint(native),'reading':word,
                                            'fine_outline':blank}
    return member


def hollow_print(image, box, reader):
    """Check whether apparent Latin initials are katakana drawn with hollow strokes."""
    import re
    from PIL import Image
    from .masked_lines import bind, corrected, source_views
    from .recognition_units import image_fingerprint
    x,y,r,b = box.bbox
    w,h=r-x,b-y
    if (box.confidence>=.60 or not re.fullmatch('[A-Za-z0-9]{1,5}',box.text)
            or not 38<=min(w,h)<=max(w,h)<=260 or not .55<w/h<2.8):
        return
    bounds=(max(0,x-3),max(0,y-3),min(image.width,r+4),min(image.height,b+4))
    native=image.crop(bounds).convert('RGB')
    isolated=Image.fromarray(outlined_surface(np.asarray(native)))
    cores=np.asarray(isolated)[:,:,0]<110
    _,_,stats,_=cv2.connectedComponentsWithStats(np.uint8(cores))
    if not 3<=sum(area>=20 for area in stats[1:,4])<=18 or not .035<cores.mean()<.40:
        return
    text=reader(isolated)
    if not re.fullmatch(r'[ァ-ヺー]{2,6}',text or ''):
        return
    crops=source_views(image,[box.poly])
    if any(reader(view)!=text for view in crops):
        return
    member=corrected(box.box_id,box.confidence,box.poly,text,box.paddle_text,box)
    bind([member],text,crops,'horizontal')
    member.ocr_metadata['hollow_print']={'bounds':list(bounds),'digest':image_fingerprint(native)}
    return member


def outlined_ellipsis(gray):
    """Identify white pen strokes of uniform weight ending in three outlined dots."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray > 140), connectivity=4)
    if count > 6000:
        return
    dots, strokes = [], []
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ridge_kernel = np.ones((5, 5), np.uint8)
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if (min(w, h) < 5 or not 7 <= max(w, h) <= 370 or not 22 <= area <= 20000
                or min(x, y) <= 4 or x+w >= gray.shape[1]-4 or y+h >= gray.shape[0]-4):
            continue
        tile = np.s_[y-4:y+h+4, x-4:x+w+4]
        body = np.uint8(labels[tile] == label)
        distance = cv2.distanceTransform(body, cv2.DIST_L2, 5)
        ridge = (distance >= cv2.dilate(distance, ridge_kernel) - .1) & (distance > 0)
        low, radius, high = np.percentile(distance[ridge], [15, 50, 85])
        if not 2.5 <= radius <= 18 or high-low > .31*radius:
            continue
        rim = (cv2.dilate(body, rim_kernel) != 0) & (body == 0)
        if np.mean(gray[tile][rim] < 150) < .97:
            continue
        center = np.array([x+w/2, y+h/2])
        if .74 < w/h < 1.35 and area > .66*w*h and max(w,h) < 2.7*radius:
            dots.append((label, center, radius))
        elif max(w,h) > 7*radius:
            strokes.append((label, center, radius))
    if len(dots) != 3:
        return
    dots.sort(key=lambda d: d[1][0])
    radii = [d[2] for d in dots]
    radius = np.median(radii)
    dx, dy = np.diff([d[1] for d in dots], axis=0).T
    if (max(radii) > 1.25*min(radii) or min(dx) < 2.2*radius or max(dx) > 8*radius
            or max(dx) > 1.45*min(dx) or max(abs(dy)) > 4*radius or abs(dy[1]-dy[0]) > 2*radius):
        return
    left, baseline = dots[0][1]
    letters = [label for label, (cx, cy), weight in strokes if .82 < weight/radius < 1.2
               and 0 < left-cx < 80*radius and abs(cy-baseline) < 40*radius]
    if not 3 <= len(letters) <= 15:
        return
    right = max(stats[label,0]+stats[label,2] for label in letters)
    if not 0 <= left-right < 4*radius:
        return
    core = np.isin(labels, letters+[d[0] for d in dots])
    x, y, w, h = cv2.boundingRect(np.uint8(core))
    if .85 < w/h < 4 and sum(max(stats[label,2:4]) > 12*radius for label in letters) >= 2:
        return core, (x,y,w,h), float(radius)


def ellipsis_views(core, bounds, radius):
    from PIL import Image
    x,y,w,h = bounds
    body = Image.fromarray(255-255*np.uint8(core[y:y+h,x:x+w]))
    return [ImageOps.expand(body, round(radius*scale), fill=255).convert('RGB') for scale in (2,4)]


def read_outlined_words(image, boxes, reader):
    """Recover interrupted display words even when background handwriting crosses the detector."""
    import re
    from .masked_lines import bind, corrected
    from .recognition_units import image_fingerprint
    result = list(boxes)
    consumed = set()
    for index, box in enumerate(boxes):
        recovered=blank_hollow_repeat(image,box,reader) or hollow_print(image,box,reader) or outlined_display(image,box,reader)
        if recovered is not None:
            result[index]=recovered
            motif = recovered.ocr_metadata.get('hollow_print',{}).get('repeated_beat')
            if motif:
                for other in boxes:
                    if other.reason != 'frontier_discovery' or other.text != motif:
                        continue
                    area = cv2.contourArea(other.poly)
                    common,_ = cv2.intersectConvexConvex(other.poly,recovered.poly)
                    if area>0 and common>=.90*area:
                        consumed.add(other.box_id)
                        recovered.ocr_metadata.setdefault('contained_discoveries',[]).append({
                            'source_id':other.box_id,'text':other.text,'polygon':other.poly.tolist()})
            continue
        x,y,r,b = box.bbox
        w,h = r-x,b-y
        if (box.confidence >= .55 or not 80 <= h <= 350 or not 1.05*h <= w <= min(500,3*h)
                or not 1 <= len(box.text.strip()) <= 5):
            continue
        l,t = max(0, x-round(.12*w)), max(0, y-round(.56*h))
        region = image.crop((l,t,min(image.width,r+round(.34*w)),min(image.height,b+round(.2*h))))
        found = outlined_ellipsis(np.asarray(region.convert('L')))
        if found is None:
            continue
        _, (a,c,width,height), radius = found
        if not (x < l+a+.5*width < r and y < t+c+.5*height < b):
            continue
        crops = ellipsis_views(*found)
        word = reader(crops[0])
        if (not re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿]{1,8}(?:\.{3}|…)', word or '')
                or reader(crops[1]) != word):
            continue
        x0,y0 = l+a,t+c
        x1,y1 = x0+width-1,y0+height-1
        # The detector must own the black outline as well as its white center.
        pad=min(12,round(1.6*radius))
        left,top=max(0,x0-pad),max(0,y0-pad)
        right,bottom=min(image.width-1,x1+pad),min(image.height-1,y1+pad)
        poly = [(left,top),(right,top),(right,bottom),(left,bottom)]
        member = corrected(box.box_id,box.confidence,poly,word,box.paddle_text,box)
        bind([member],word,crops,'horizontal')
        bounds = [max(0,x0-12),max(0,y0-12),min(image.width,x1+13),min(image.height,y1+13)]
        member.ocr_metadata['outlined_ellipsis'] = {
            'bounds':bounds,'digest':image_fingerprint(image.crop(bounds))}
        result[index] = member
    return [box for box in result if box.box_id not in consumed]


def rotated(rgb, degrees):
    h,w=rgb.shape[:2]
    forward=cv2.getRotationMatrix2D((w/2,h/2),degrees,1)
    cosine,sine=np.abs(forward[0,:2])
    width,height=int(h*sine+w*cosine+1),int(h*cosine+w*sine+1)
    forward[:,2]+=[(width-w)/2,(height-h)/2]
    return cv2.warpAffine(rgb,forward,(width,height),borderValue=(255,255,255)),cv2.invertAffineTransform(forward)


def overlap(polygon, others):
    area=cv2.contourArea(polygon)
    total=0.
    for other in others:
        intersection,_=cv2.intersectConvexConvex(polygon,other)
        if intersection>0:
            total+=intersection
            if intersection>.24*min(area,cv2.contourArea(other)):
                return True
    return total>.24*area


def _discover(engine,image,reader):
    pages=engine.raw_detections
    if len(pages)!=1 or not pages[0] or len(pages[0])>256:
        return
    original=pages[0]
    if not all(isinstance(b.box_id,int) and len(b.poly)==4 for b in original):
        return
    rgb=np.asarray(image)
    polygons=[cv2.convexHull(box.poly.astype(np.float32)) for box in original]
    areas=zones(rgb,polygons)
    if not areas:
        return
    stage={'policy':POLICY,'native_detections':len(original),'zones':[],'added_ids':[],'reader_trials':0}
    engine.outlined_sound_discovery=stage
    added=[]
    for zone in areas:
        refs=anchors(image,original,zone,reader)
        record={'bounds':zone,'anchors':refs,'passes':[]}
        stage['zones'].append(record)
        if not refs:
            continue
        direction=sum(v[1] for values in refs.values() for v in values)
        sign=1 if direction>0 else -1
        left,top,right,bottom=zone
        surface=rgb[top:bottom,left:right]
        isolated=outlined_surface(surface)
        for degrees, ink_only in ((0,False),(sign*22,False),(sign*38,False),(0,True),(sign*22,True),(sign*38,True)):
            canvas,inverse=rotated(isolated if ink_only else surface,degrees)
            attempt={'degrees':degrees,'ink_only':ink_only,'inverse':inverse.tolist(),'detections':[]}
            record['passes'].append(attempt)
            results=list(engine.ocr.predict(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)))
            if len(results)!=1:
                continue
            boxes,report=associate_paddle_result(getattr(results[0],'res',results[0]),0)
            attempt['association_status']=report['status']
            for b in boxes[:48]:
                poly=np.rint(cv2.transform(b.poly[None],inverse)[0]+[left,top]).astype('float32')
                entry={'local_id':b.box_id,'local_polygon':b.poly.tolist(),'polygon':poly.tolist(),
                       'text':b.text,'score':float(b.confidence)}
                attempt['detections'].append(entry)
                if (len(added)>=12 or stage['reader_trials']>=64 or not b.text.strip() or cv2.contourArea(poly)<130 or not cv2.isContourConvex(poly)
                        or np.any(poly<0) or np.any(poly>=[image.width,image.height]) or overlap(poly,polygons)):
                    continue
                try:
                    crops=views(image,poly.tolist())
                except ValueError:
                    continue
                stage['reader_trials']+=1
                readings=[reader(v) for v in crops]
                entry['readings']=readings
                confirmed=confirmed_text(readings)
                if not confirmed or sound_readings.repeat(confirmed) not in refs:
                    continue
                identity=max(box.box_id for box in original)+len(added)+1
                box=Text_Box(identity,b.confidence,poly,b.text,review=1)
                box.set_status(OCRStatus.REVIEW,'supplemental_rotated_sound_region')
                box.ocr_metadata.update(b.ocr_metadata)
                box.ocr_metadata[POLICY]={'zone':zone,'degrees':degrees,'ink_only':ink_only,'local_id':b.box_id,
                    'local_polygon':b.poly.tolist(),'readings':readings,'anchors':refs[sound_readings.repeat(readings[0])]}
                added.append(box)
                polygons.append(poly)
                entry['source_id']=identity
    if added:
        combined=[*original,*added]
        batch=propose_page_from_boxes(combined,engine.image_path,complementary_ownership=True,regroup_companions=True)
        def partition(proposals):
            return {(tuple(p.member_ids),tuple(p.main_ids),tuple(p.annotation_parents),p.direction) for p in proposals}
        original_groups=partition(engine.automatic_unit_proposals[0].proposals)
        updated_groups=partition(batch.proposals)
        singletons={p.main_ids[0] for p in batch.proposals if len(p.main_ids)==len(p.member_ids)==1}
        if not original_groups.issubset(updated_groups) or any(b.box_id not in singletons for b in added):
            stage['deferred']='supplemental regions would repartition existing text'
            return
        engine.automatic_unit_proposals=[batch]
        engine.raw_detections=[combined]
        stage['added_ids']=[b.box_id for b in added]


def discover(engine,image,reader):
    """An unavailable extra pass leaves the complete native page usable."""
    try:
        _discover(engine,image,reader)
    except (OSError,RuntimeError,ValueError,cv2.error) as error:
        stage=getattr(engine,'outlined_sound_discovery',{'policy':POLICY,'added_ids':[]})
        stage['error']=type(error).__name__+': '+str(error)[:512]
        engine.outlined_sound_discovery=stage


def outlined_kana(image, box, reader):
    """Corroborate a weak or truncated native reading against outlined glyph interiors."""
    import re
    from PIL import Image
    l,t,r,b = box.bbox
    w,h = r-l,b-t
    if not (24 <= min(w,h) and max(w,h) <= 144 and len(box.text) <= 2
            and (box.confidence < .5 or re.fullmatch('[一-鿿]',box.text))):
        return
    # A single square glyph can be the beginning of a horizontal effect.
    extend = round(w*1.15) if len(box.text)==1 and .65 <= w/h <= 1.5 else 0
    bounds = max(0,l-2),max(0,t-2),min(image.width,r+extend+3),min(image.height,b+7)
    crop = image.crop(bounds)
    gray = cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2GRAY)
    contours,tree = cv2.findContours(np.uint8(gray<150),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if tree is None:
        return
    surface = np.full(gray.shape,255,np.uint8)
    holes = []
    for contour,parent in zip(contours,tree[0,:,3]):
        x,y,cw,ch = cv2.boundingRect(contour)
        if (parent >= 0 and 4 <= min(cw,ch) and max(cw,ch) < min(64,.85*max(w,h))
                and cv2.contourArea(contour) > .15*cw*ch):
            holes.append(contour)
            cv2.drawContours(surface,[contour],-1,0,-1)
    if not 3 <= len(holes) <= 14:
        return
    x,y,cw,ch = cv2.boundingRect(np.concatenate(holes))
    view = Image.fromarray(surface[y:y+ch,x:x+cw]).convert('RGB')
    reading = reader(view)
    if not isinstance(reading,str) or not re.fullmatch('[ぁ-ゖァ-ヺー]{1,5}[っッ]',reading) or reader(crop) != reading:
        return
    left,top = bounds[:2]
    polygon = np.float32([[left+x-2,top+y-2],[left+x+cw+2,top+y-2],
                          [left+x+cw+2,top+y+ch+2],[left+x-2,top+y+ch+2]])
    polygon = np.clip(polygon,[0,0],[image.width-1,image.height-1]).astype('float32')
    yy,xx = np.where(surface == 0)
    major,minor = (yy,xx) if ch>cw else (xx,yy)
    lo,hi = np.percentile(major,[25,75])
    slope = (np.median(minor[major>=hi])-np.median(minor[major<=lo]))/max(1,hi-lo)
    angle = float(np.clip(np.degrees(np.arctan(slope))*(1 if ch>cw else -1),-15,15))
    return reading, polygon, angle


def blank_sound(image, box, reader):
    """Check an existing blank detector footprint for a short handwritten effect."""
    import re
    if box.text:
        return None
    reading = None
    for view in reader_geometry.VIEWS:
        try:
            mapping = reader_geometry.plan(image.size,box.poly,view=view)
            crop = reader_geometry.pixels(image,mapping)
            if view == 'tight' and np.std(np.asarray(crop.convert('L'))) < 5:
                return None
            text = reader(crop)
        except (ValueError, RuntimeError):
            return None
        if not isinstance(text,str) or not re.fullmatch(r'[ぁ-ゖァ-ヺー]{1,7}[っッ][！!？?]?',text):
            return None
        if reading is not None and text != reading:
            return None
        reading = text
    return reading


def refine_outlined_kana(engine,image,reader):
    if len(engine.raw_detections)!=1 or not engine.raw_detections[0]:
        return
    original = engine.raw_detections[0]
    from .repeated_effects import read_pairs, printed_twins, curved_repeats
    pairs = read_pairs(image, original, reader)
    twins = printed_twins(image,original,reader)
    curved = curved_repeats(image,original,reader)
    changes = {**curved,**pairs,**twins}
    current = [changes.get(b.box_id,b) for b in original]
    old_ids = {b.box_id for b in original}
    current.extend(b for i,b in twins.items() if i not in old_ids)
    trials = 0
    for i,box in enumerate(original):
        if trials >= 16:
            break
        if box.box_id in pairs or box.box_id in twins or len(box.text)>2 or min(box.bbox[2]-box.bbox[0],box.bbox[3]-box.bbox[1])<24:
            continue
        trials += 1
        reading = blank_sound(image,box,reader)
        if reading:
            from .types import OCREngine
            replacement = Text_Box(box.box_id,box.confidence,box.poly,reading,review=1)
            replacement.reason = 'corroborated_blank_sound'
            replacement.selected_engine = OCREngine.HAYAI
            replacement.ocr_metadata = {**box.ocr_metadata,
                'candidates':{**box.ocr_metadata.get('candidates',{}),'hayai':reading},
                'blank_sound':{'original_text':box.text,'original_status':box.status.value,
                               'views':list(reader_geometry.VIEWS),'reading':reading}}
            current[i] = replacement
            continue
        result = outlined_kana(image,box,reader)
        if not result:
            continue
        reading,poly,angle = result
        # A sliver overlapping the rim of an adjacent label is common on physical objects.
        if overlap(poly,[cv2.convexHull(other.poly.astype('float32')) for other in original if other is not box]):
            continue
        replacement = Text_Box(box.box_id,box.confidence,poly,reading,review=1)
        from .types import OCREngine
        replacement.selected_engine = OCREngine.HAYAI
        replacement.reason = 'corroborated_outline_reading'
        replacement.ocr_metadata = {**box.ocr_metadata,'outline_reading':{'original_text':box.text,
            'original_polygon':box.poly.tolist(),'angle':angle,'reading':reading,'views':['glyph_interiors','source_crop']}}
        replacement.ocr_metadata['candidates'] = {**box.ocr_metadata.get('candidates',{}),'hayai':reading}
        current[i] = replacement
    if pairs:
        # Discovery runs before crop recognition. Its partial transcription can
        # now be wholly owned by a newly read effect. Keep it in that box's audit.
        for box in current[:]:
            if box.reason != 'frontier_discovery' or len(box.text)<2:
                continue
            area = cv2.contourArea(box.poly)
            for owner in pairs.values():
                if box.text not in owner.text or area <= 0:
                    continue
                overlap_area,_ = cv2.intersectConvexConvex(box.poly,owner.poly)
                if overlap_area >= .90*area:
                    owner.ocr_metadata.setdefault('contained_discoveries',[]).append(
                        dict(source_id=box.box_id,text=box.text,polygon=box.poly.tolist()))
                    current.remove(box)
                    break
    if len(current)!=len(original) or any(a is not b for a,b in zip(original,current)):
        engine.automatic_unit_proposals = [propose_page_from_boxes(current,engine.image_path,
            complementary_ownership=True,regroup_companions=True)]
        engine.raw_detections = [current]
