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
        except (ValueError,OSError,RuntimeError,cv2.error):
            continue
        if found is None:
            continue
        reading,polygon,group = found
        box = Text_Box(next_id,0.,polygon,reading,review=1)
        next_id += 1
        box.selected_engine = OCREngine.HAYAI
        box.reason = 'confirmed_handwritten_cluster'
        box.ocr_metadata = {'candidates':{'hayai':reading},'reader_revision':REVISION,'original_fragments':[
            {'source_id':v.box_id,'text':v.paddle_text,'confidence':float(v.paddle_confidence),
             'polygon':v.paddle_poly.tolist()} for v in group]}
        current = [v for v in current if v not in group]+[box]
    if len(current)!=len(original):
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
