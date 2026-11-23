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
    if box.text or max(box.bbox[2]-box.bbox[0],box.bbox[3]-box.bbox[1]) > 320:
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
    current = list(original)
    trials = 0
    for i,box in enumerate(original):
        if trials >= 16:
            break
        if len(box.text)>2 or min(box.bbox[2]-box.bbox[0],box.bbox[3]-box.bbox[1])<24:
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
    if any(a is not b for a,b in zip(original,current)):
        engine.automatic_unit_proposals = [propose_page_from_boxes(current,engine.image_path,
            complementary_ownership=True,regroup_companions=True)]
        engine.raw_detections = [current]
