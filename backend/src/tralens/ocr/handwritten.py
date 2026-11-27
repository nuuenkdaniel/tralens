"""Recover a slanted handwritten aside ending in fragmented OCR boxes."""
import re

import cv2
import numpy as np
from PIL import Image

from .automatic_units import propose_page_from_boxes
from .outlined_sounds import rotated
from .types import Text_Box, OCREngine
from .hayai_adapter import REVISION


def isolated_ink(gray, scale):
    foreground = (gray < 142).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    keep = np.zeros(len(stats), bool)
    for i, (x,y,w,h,area) in enumerate(stats[1:], 1):
        if (min(w,h) < 3 or max(w,h) > 1.2*scale or not 10 <= area <= .33*scale**2
                or not .07 < area/(w*h) < .96):
            continue
        window = np.s_[y:y+h,x:x+w]
        keep[i] = distance[window][labels[window] == i].max() >= 2
    return keep[labels]


def search_window(size, bbox):
    x,y,r,b = bbox
    w = r-x+1
    scale = max(w,b-y+1)
    if not 32 <= scale <= 110 or not .35 <= w/scale <= 1:
        return
    left,top = max(0,x-round(.6*w)),max(0,y-round(3.5*scale))
    right,bottom = min(size[0],x+round(5.5*w)),min(size[1],b+1+round(.1*scale))
    return (left,top,right,bottom),scale


def read_cluster(rgb, boxes, blank, reader):
    window = search_window(rgb.size,blank.bbox)
    if window is None:
        return
    (left,top,right,bottom),scale = window
    x,y,r,b = blank.bbox
    inside = []
    for box in boxes:
        l,t,a,d = box.bbox
        if a<=left or l>=right or d<=top or t>=bottom:
            continue
        if l<left or a>=right or t<top or d>=bottom:
            return
        inside.append(box)
    leading = [v for v in inside if v.bbox[0]>r and v.bbox[3]<y and
               v.confidence>=.75 and re.fullmatch(r'[ぁ-ヺ一-鿿]{2,8}',v.text)]
    if len(leading)!=1 or not 3 <= len(inside)<=8:
        return
    first, = leading
    if any(len(v.text)>2 or (v.confidence>=.6 and not (v.text.isdigit() and v.confidence<.9))
           for v in inside if v is not first):
        return
    pixels = np.asarray(rgb.crop((left,top,right,bottom)))
    if np.mean(np.ptp(pixels,axis=2)>16)>.02:
        return
    ink = isolated_ink(cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY), scale)
    canvas = cv2.cvtColor(np.uint8(~ink)*255,cv2.COLOR_GRAY2RGB)
    readings = []
    for degrees in (-16,-20):
        view,inverse = rotated(canvas,degrees)
        transform = cv2.invertAffineTransform(inverse)
        points = cv2.transform((first.poly-[left,top]).astype('float32')[None],transform)[0]
        view[:max(0,int(points[:,1].min())-round(.1*scale))] = 255
        yy,xx = np.where(view[:,:,0]<128)
        if not len(xx):
            return
        bounds = (int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
        crop = Image.fromarray(view).crop(bounds)
        if crop.width*crop.height>300000:
            return
        reading = reader(crop)
        if (not isinstance(reading,str) or not reading.startswith(first.text)
                or not re.fullmatch(r'[ぁ-ヺ一-鿿ー！？。、…]{6,48}',reading)):
            return
        readings.append(reading)
    if readings[0]!=readings[1]:
        return
    # Confirm the remainder independently with the recognized leading column removed.
    split = int(points[:,0].min())
    profile = (view[:,:,0]<128).sum(0)
    gaps = np.flatnonzero((profile==0) & (np.arange(len(profile))<split+scale*.20)
                          & (np.arange(len(profile))>split-scale*.35))
    if not len(gaps):
        return
    edge = int(gaps[-1])+1
    if not bounds[0]<edge<bounds[2]:
        return
    remainder = Image.fromarray(view).crop((bounds[0],bounds[1],edge,bounds[3]))
    suffix = reader(remainder)
    if not isinstance(suffix,str) or first.text+suffix!=readings[0]:
        return
    polygon = cv2.boxPoints(cv2.minAreaRect(cv2.transform(
        np.column_stack((xx,yy)).astype('float32')[None],inverse)[0]))
    polygon += [left,top]
    if (polygon<0).any() or (polygon>=[rgb.width,rgb.height]).any():
        return
    for other in boxes:
        if other not in inside and cv2.intersectConvexConvex(polygon,cv2.convexHull(other.poly))[0]>0:
            return
    return readings[0],polygon,inside


def punctuated_column(image, boxes, blank, reader):
    """Read a short vertical line beside an unrecognized column of ellipsis dots."""
    l,t,r,b = blank.bbox
    w,h = r-l+1,b-t+1
    if not (16<=w<=72 and 1.35*w<=h<=144):
        return
    gray = np.asarray(image.crop((l,t,r+1,b+1)).convert('L'))
    _,_,stats,centers = cv2.connectedComponentsWithStats(np.uint8(gray<150))
    interior = [(cx,cy,area) for (x,y,ww,hh,area),(cx,cy) in zip(stats[1:],centers[1:])
                if x>1 and y>1 and x+ww<w-1 and y+hh<h-1 and area>=3]
    if len(interior) not in (3,6):
        return
    xs,ys,areas = np.asarray(sorted(interior,key=lambda v:v[1])).T
    steps = np.diff(ys)
    if (np.ptp(xs)>max(3,w*.15) or min(steps)<3 or max(steps)>2.2*min(steps)
            or max(areas)>3*min(areas) or max(areas)>.025*w*h or np.ptp(ys)<.48*h):
        return
    neighbors = [other for other in boxes if other is not blank
                 and re.fullmatch(r'[ぁ-ゖァ-ヺ一-鿿]{2,8}',other.text)
                 and 0<l-other.bbox[0]<=1.6*w and abs(other.bbox[1]-t)<.25*h
                 and abs(other.bbox[3]-b)<.25*h and abs(other.bbox[2]-l)<.5*w]
    if len(neighbors)!=1:
        return
    other, = neighbors
    dots_right = l+int(max(xs))+max(2,round(max(areas)**.5))
    rect = (other.bbox[0],min(t,other.bbox[1]),dots_right+1,max(b,other.bbox[3])+1)
    full = image.crop(rect)
    reading = reader(full)
    if not isinstance(reading,str):
        return
    match = re.fullmatch(r'[.…・]{1,12}([ぁ-ゖァ-ヺ一-鿿]{2,9})',reading)
    before_dots = gray[:,:max(0,int(min(xs))-2)]
    gap = np.flatnonzero((before_dots>=200).all(axis=0))
    if not match or not len(gap) or reader(image.crop((rect[0],rect[1],l+int(np.median(gap)),rect[3])))!=match[1]:
        return
    x,y,right,bottom = rect
    polygon = np.float32([[x,y],[right-1,y],[right-1,bottom-1],[x,bottom-1]])
    if any(cv2.intersectConvexConvex(polygon,cv2.convexHull(v.poly))[0]>0
           for v in boxes if v not in (blank,other)):
        return
    return reading,polygon,[blank,other]


def trailing_kana(image, parent, reader, detector):
    """Look below a vertical label for a smaller, separate row of repeated handwriting."""
    x,y,r,b = parent.bbox
    w,h = r-x+1,b-y+1
    if not (38<=w<=220 and 1.3*w<h<3*w and re.fullmatch(r'[一-鿿々]{1,4}',parent.text)):
        return
    left,top = max(0,x-3),max(0,y+round(.62*h))
    chip = np.asarray(image.crop((left,top,min(image.width,r+4),min(image.height,b+round(.3*h)))))
    if not chip.size:
        return
    gray = cv2.cvtColor(chip,cv2.COLOR_RGB2GRAY)
    _,labels,stats,_ = cv2.connectedComponentsWithStats(np.uint8(gray<135))
    glyphs = [(a,t,gw,gh,i) for i,(a,t,gw,gh,area) in enumerate(stats[1:],1)
              if a>0 and t>0 and a+gw<gray.shape[1] and t+gh<gray.shape[0]
              and .10*w<gh<.31*w and .29*gh<=gw<=1.2*gh and .12*gw*gh<area<.70*gw*gh
              and y+.78*h <= top+t <= b+.1*h]
    glyphs.sort()
    if not 3<=len(glyphs)<=6:
        return
    starts,tops,widths,heights,ids = map(np.array,zip(*glyphs))
    if (max(heights)>1.5*min(heights) or max(widths)>1.8*min(widths)
            or np.any(np.diff(starts)<.65*widths[:-1]) or np.any(np.diff(starts)>2.7*widths[:-1])):
        return
    slope,offset = np.polyfit(starts,tops,1)
    if abs(slope)>.65 or max(abs(tops-(slope*starts+offset)))>.23*min(heights):
        return
    visible = cv2.dilate(np.isin(labels,ids).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    yy,xx = np.where(visible)
    ink = np.where(visible[:,:,None],chip,255)[yy.min():yy.max()+1,xx.min():xx.max()+1]
    view = Image.fromarray(np.pad(ink,((5,5),(6,6),(0,0)),constant_values=255))
    view = view.resize((view.width*2,view.height*2),Image.Resampling.LANCZOS)
    reading = reader(view).strip()
    if len(reading)!=len(ids) or not re.fullmatch(r'([ぁ-ゖァ-ヺ])\1{2,5}',reading):
        return
    result = detector.predict(np.asarray(view))
    if not result or list(result[0]['rec_texts']) != [reading] or result[0]['rec_scores'][0]<.75:
        return
    polygon = cv2.boxPoints(cv2.minAreaRect(np.column_stack((xx+left,yy+top)).astype('float32')))
    upper = parent.poly.copy()
    upper[:,1] = np.minimum(upper[:,1],slope*(upper[:,0]-left)+offset+top-3)
    return reading,polygon,upper,float(result[0]['rec_scores'][0])


def small_kana_candidate(parent):
    return (parent.confidence < .8 and re.fullmatch(r'[一-鿿]{1,2}[やゆよヤユヨ]',parent.text))


def clipped_handwriting(image, parent, boxes, reader):
    """Recover a short kana ending below a detector line with a suspect Latin tip."""
    from .crop_readings import views
    x,y,r,b = parent.bbox
    w,h = r-x+1,b-y+1
    small_kana = small_kana_candidate(parent)
    if not (50<=w<=225 and 42<=h<=200 and (small_kana or re.fullmatch(r'[一-鿿ぁ-ゖァ-ヺ]{1,6}[A-Za-z]',parent.text))):
        return
    prefix = reader(image.crop((x,y,r+1,b+1)))
    expected = (r'[ぁ-ゖァ-ヺ]{1,2}'+parent.text[-1].translate(str.maketrans('やゆよヤユヨ','ゃゅょャュョ'))
                if small_kana else re.escape(parent.text[:-1])+r'[ぁ-ゖァ-ヺ]')
    if not isinstance(prefix,str) or not re.fullmatch(expected,prefix):
        return
    left,top = max(0,x-w//8),max(0,y-h//8)
    chip = image.crop((left,top,min(image.width,r+w//2),min(image.height,b+h))).convert('L')
    _,labels,stats,_ = cv2.connectedComponentsWithStats(np.uint8(np.asarray(chip)<145))
    parts = []
    for i,(a,c,cw,ch,area) in enumerate(stats[1:],1):
        ax,ay = left+a,top+c
        if not (0<a<chip.width-cw and 0<c<chip.height-ch and b-.18*h<ay<b+.66*h
                and b+.08*h<ay+ch<b+.9*h and .045*w<cw<.55*w and .08*h<ch<.74*h
                and .07*cw*ch<area<.75*cw*ch):
            continue
        rect = np.float32([(ax,ay),(ax+cw,ay),(ax+cw,ay+ch),(ax,ay+ch)])
        if not any(cv2.intersectConvexConvex(v.poly,rect)[0]>0 for v in boxes if v is not parent):
            parts.append(i)
    if not 2<=len(parts)<=8:
        return
    yy,xx = np.where(np.isin(labels,parts))
    points = np.column_stack((xx+left,yy+top)).astype('float32')
    ax,ay = points.min(0).astype(int)-3
    rx,bt = points.max(0).astype(int)+4
    suffix = reader(image.crop((max(0,ax),max(0,ay),min(image.width,rx),min(image.height,bt))))
    if not isinstance(suffix,str) or not re.fullmatch(r'[ぁ-ゖァ-ヺ]{1,4}',suffix):
        return
    poly = cv2.convexHull(np.vstack((parent.poly,points))).reshape(-1,2)
    for v in boxes:
        if v is parent:
            continue
        before = cv2.intersectConvexConvex(v.poly,parent.poly)[0]
        after = cv2.intersectConvexConvex(v.poly,poly)[0]
        if after>before+min(2,.02*before):
            return
    x,y = np.floor(poly.min(0)).astype(int).tolist()
    r,b = (np.ceil(poly.max(0)).astype(int)+1).tolist()
    if not all(reader(crop)==prefix+suffix for crop in views(image,(x,y,r,b),[poly])):
        return
    full = Text_Box(parent.box_id,parent.confidence,poly,prefix+suffix,review=1)
    full.selected_engine,full.reason = OCREngine.HAYAI,'corroborated_handwritten_ending'
    full.ocr_metadata = {**parent.ocr_metadata,'candidates':{**parent.ocr_metadata.get('candidates',{}),'hayai':full.text},
                         'clipped_ending':{'original_polygon':parent.poly.tolist(),'prefix':prefix,'suffix':suffix}}
    return full


def discover(engine, image, reader):
    if len(engine.raw_detections)!=1:
        return
    original = engine.raw_detections[0]
    if not original or len(original)>256 or any(type(v.box_id) is not int for v in original):
        return
    current = list(original)
    next_id = max(v.box_id for v in original)+1
    for blank in [v for v in original if not v.text.strip()][:4]:
        if blank not in current:
            continue
        try:
            found = read_cluster(image,current,blank,reader)
            reason = 'confirmed_handwritten_cluster'
            if found is None:
                found = punctuated_column(image,current,blank,reader)
                reason = 'confirmed_punctuated_column'
        except (ValueError,OSError,RuntimeError,cv2.error):
            continue
        if found is None:
            continue
        reading,polygon,group = found
        box = Text_Box(next_id,0.,polygon,reading,review=1)
        next_id += 1
        box.selected_engine = OCREngine.HAYAI
        box.reason = reason
        box.ocr_metadata = {'candidates':{'hayai':reading},'reader_revision':REVISION,'original_fragments':[
            {'source_id':v.box_id,'text':v.paddle_text,'confidence':float(v.paddle_confidence),
             'polygon':v.paddle_poly.tolist()} for v in group]}
        current = [v for v in current if v not in group]+[box]
    for parent in [box for box in original if box in current and re.fullmatch(r'[一-鿿々]{1,4}',box.text)][:36]:
        try:
            found = trailing_kana(image,parent,reader,engine.ocr)
        except (ValueError,OSError,RuntimeError,cv2.error):
            continue
        if found is None:
            continue
        reading,polygon,upper,confidence = found
        if any(cv2.intersectConvexConvex(cv2.convexHull(box.poly),polygon)[0]>0 for box in current if box is not parent):
            continue
        heading = Text_Box(parent.box_id,parent.confidence,upper,parent.text,parent.review)
        heading.selected_engine,heading.reason = parent.selected_engine,parent.reason
        heading.ocr_metadata = {**parent.ocr_metadata,'before_separating_small_print':parent.poly.tolist()}
        small = Text_Box(next_id,confidence,polygon,reading,review=1)
        next_id += 1
        small.selected_engine,small.reason = OCREngine.HAYAI,'confirmed_trailing_kana'
        small.ocr_metadata = {'candidates':{'paddle':reading,'hayai':reading},'parent_detection':parent.box_id}
        current[current.index(parent)] = heading
        current.append(small)
    for parent in [v for v in original if v in current and (re.search(r'[一-鿿ぁ-ゖァ-ヺ][A-Za-z]$',v.text) or small_kana_candidate(v))][:6]:
        try:
            extended = clipped_handwriting(image,parent,current,reader)
        except (ValueError,OSError,RuntimeError,cv2.error):
            continue
        if extended is not None:
            current[current.index(parent)] = extended
    if current!=original:
        batch = propose_page_from_boxes(current,engine.image_path,complementary_ownership=True,regroup_companions=True)
        previous = engine.automatic_unit_proposals[0]
        retained = {v.box_id for v in current if v in original}
        def partition(proposals):
            return {(p.member_ids,p.main_ids,p.direction,p.annotation_ids) for p in proposals
                    if retained.intersection(p.member_ids)}
        if (partition(previous.proposals)!=partition(batch.proposals)
                or any(len(p.member_ids)!=1 for p in batch.proposals if not retained.intersection(p.member_ids))):
            return
        engine.raw_detections = [current]
        engine.automatic_unit_proposals = [batch]


def stroke_masks(original, bounds, detection, own, protected):
    """Reconstruct only the thick, white-rimmed ink seen by the crop reader."""
    if detection.get('reason') != 'confirmed_handwritten_cluster':
        return
    fragments = detection.get('ocr_metadata',{}).get('original_fragments',[])
    blanks = [v for v in fragments if v.get('text')=='']
    if len(blanks)!=1:
        return
    points = np.asarray(blanks[0]['polygon'])
    if points.shape!=(4,2) or not np.isfinite(points).all():
        return
    window = search_window((original.shape[1],original.shape[0]),(*points.min(0).astype(int),*points.max(0).astype(int)))
    if window is None:
        return
    (left,top,right,bottom),scale = window
    gray = cv2.cvtColor(original[top:bottom,left:right],cv2.COLOR_RGB2GRAY)
    ink = isolated_ink(gray,scale)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    distance = cv2.distanceTransform(ink.astype(np.uint8),cv2.DIST_L2,5)
    keep = np.zeros(len(stats),bool)
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if min(x,y)<2 or x+w>=gray.shape[1]-1 or y+h>=gray.shape[0]-1:
            continue
        loc = np.s_[y-2:y+h+2,x-2:x+w+2]
        component = labels[loc]==i
        if distance[loc][component].max()<2.75:
            continue
        m = component.astype(np.uint8)
        rim = cv2.dilate(m,np.ones((5,5),np.uint8))!=cv2.dilate(m,np.ones((3,3),np.uint8))
        keep[i] = (gray[loc][rim]>247).mean()>.83
    strokes = keep[labels].astype(np.uint8)
    core = cv2.dilate(strokes,np.ones((3,3),np.uint8)).astype(bool)
    halo = cv2.dilate(strokes,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))).astype(bool)
    x,y,r,b = bounds
    l,t,a,d = max(left,x),max(top,y),min(right,r),min(bottom,b)
    if l>=a or t>=d:
        return
    src = np.s_[t-top:d-top,l-left:a-left]
    dest = np.s_[t-y:d-y,l-x:a-x]
    masks = []
    for pixels in (core,core | (halo & (gray>145))):
        mask = np.zeros_like(own)
        mask[dest] = pixels[src]
        masks.append(mask & own & ~protected)
    if masks[0].any():
        return masks
