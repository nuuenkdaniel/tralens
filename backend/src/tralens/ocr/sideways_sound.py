"""Read outlined display lettering printed up the page in a painted panel."""
import re

import cv2
import numpy as np
from PIL import Image, ImageOps

POLICY = 'sideways_outlined_katakana_v1'


def silhouette(pixels, threshold):
    contours,tree = cv2.findContours((pixels.max(2)<threshold).astype(np.uint8),
                                     cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if tree is None or len(contours)>4096:
        return None
    white = (pixels.min(2)>195) & (np.ptp(pixels,axis=2)<40)
    mask = np.full(white.shape,255,np.uint8)
    count = 0
    for contour,parent in zip(contours,tree[0,:,3]):
        if parent<0 or cv2.contourArea(contour)<6:
            continue
        x,y,w,h = cv2.boundingRect(contour)
        local = np.zeros((h,w),np.uint8)
        cv2.drawContours(local,[contour-np.int32([x,y])],-1,1,-1)
        ink = (local!=0) & white[y:y+h,x:x+w]
        if ink.sum()<.60*local.sum():
            continue
        mask[y:y+h,x:x+w][ink] = 0
        count += 1
    return Image.fromarray(mask) if 4<=count<=64 else None


def presentations(image,record):
    if (record['direction']!='vertical' or len(record['polygons'])!=1
            or not re.fullmatch('[a-zA-Z0-9]{2,12}',record['source'])):
        return None
    x,y,r,b = record['bounds']
    if not (70<=r-x<=320 and 1.65*(r-x)<=b-y<=550):
        return None
    pixels = np.asarray(image.crop((max(0,x-16),max(0,y-16),min(image.width,r+16),min(image.height,b+16))))
    if (np.ptp(pixels,axis=2)>24).mean()<.25:
        return None
    views = []
    for threshold,degrees in ((120,-60),(135,-70),(120,-75)):
        mask = silhouette(pixels,threshold)
        if mask is None:
            return None
        view = mask.rotate(degrees,expand=True,fillcolor=255)
        bounds = ImageOps.invert(view).getbbox()
        if bounds is None:
            return None
        views.append(ImageOps.expand(view.crop(bounds),border=8,fill=255).convert('RGB'))
    first = views[0]
    views.append(first.crop((int(first.width*.34),0,first.width,first.height)))
    return views


def matches(record):
    readings = record['readings']
    if len(readings)!=4 or len(set(readings[:3]))!=1:
        return False
    full,tail = readings[0],readings[-1]
    return bool(re.fullmatch('[ァ-ヺー]{4,14}',full) and 2<=len(tail)<len(full)
                and len(set(tail))==1 and full.endswith(tail) and len(set(full))>=2)


def pale_effect(image, box, reader):
    """Check whether a short vertical OCR error is horizontal white kana turned sideways."""
    from .masked_lines import corrected
    from .recognition_units import image_fingerprint
    x,y,r,b=box.bbox
    w,h=r-x+1,b-y+1
    if (not 20<=w<=105 or not 1.65*w<h<=290 or not re.fullmatch(r'[一-鿿a-zA-Z0-9]{1,4}[!！?？]',box.text)):
        return None
    bounds=(max(0,x-8),max(0,y-8),min(image.width,r+9),min(image.height,b+9))
    native=image.crop(bounds)
    pixels=np.asarray(native)
    own=np.zeros(pixels.shape[:2],np.uint8)
    cv2.fillPoly(own,[np.int32(box.poly-bounds[:2])],1)
    if np.mean(np.ptp(pixels[own!=0],axis=1)>24)<.25:
        return None
    mask=(own!=0) & (pixels.min(2)>=205)
    if not .05<mask.sum()/own.sum()<.55:
        return None
    binary=Image.fromarray(np.uint8(~mask)*255).convert('RGB')
    try:
        first=reader(binary.transpose(Image.Transpose.ROTATE_270))
        if (not isinstance(first,str) or not re.fullmatch(r'[ァ-ヺー]{2,8}[ッン][!！?？]?',first)
                or reader(native.transpose(Image.Transpose.ROTATE_270))!=first
                or reader(binary.rotate(-84,expand=True,fillcolor='white'))!=first):
            return None
    except (ValueError, RuntimeError, OSError):
        return None
    member=corrected(box.box_id,box.confidence,box.poly,first,box.text,box,masked=True)
    member.ocr_metadata['painted_effect']={'reading':first,'bounds':list(bounds),'source_digest':image_fingerprint(native),
                                          'reads':[first]*3,'white_sideways':True}
    return member
