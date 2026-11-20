"""Bounded reader presentations for oversized, horizontal painted lettering."""
import cv2
import numpy as np
from PIL import Image,ImageDraw


POLICY='oversized_color_ink_views_v1'


def candidate(bounds,size,direction):
    if direction!='horizontal' or len(bounds)!=4 or any(type(v) is not int for v in bounds):
        return False
    x,y,r,b=bounds
    w,h=r-x,b-y
    return (0<=x<r<=size[0] and 0<=y<b<=size[1] and w>=1.7*h
        and w<=1536 and h<=768 and 262144<w*h<=786432)


def presentations(image,record):
    x,y,r,b=record['bounds']
    if not candidate(record['bounds'],image.size,record['direction']):
        return None
    source=image.crop((x,y,r,b))
    own=Image.new('L',source.size)
    for poly in record['polygons']:
        ImageDraw.Draw(own).polygon([(a-x,c-y) for a,c in poly],fill=255)
    own=np.asarray(own)!=0
    hue,saturation,value=cv2.split(cv2.cvtColor(np.asarray(source),cv2.COLOR_RGB2HSV))
    saturated=(saturation>=150)&(value>=120)&own
    if np.count_nonzero(saturated)<.08*own.sum() or not saturated.any():
        return None
    hist=np.bincount(hue[saturated],minlength=180)
    center=int(np.argmax(sum(np.roll(hist,j) for j in range(-4,5))))
    delta=np.abs(hue.astype(np.int16)-center)
    band=saturated&(np.minimum(delta,180-delta)<=6)
    if band.sum()<.75*saturated.sum():
        return None
    count,labels,stats,_=cv2.connectedComponentsWithStats(band.astype(np.uint8))
    if count>8192:
        return None
    keep=(stats[:,cv2.CC_STAT_AREA]>=55)&(stats[:,cv2.CC_STAT_WIDTH]>3)&(stats[:,cv2.CC_STAT_HEIGHT]>3)
    keep[0]=False
    if keep.sum()<2:
        return None
    ink=keep[labels].astype(np.uint8)*255
    images=[source,Image.fromarray(255-ink).convert('RGB'),
        Image.fromarray(255-cv2.dilate(ink,np.ones((5,5),np.uint8))).convert('RGB')]
    for view in images:
        view.thumbnail((720,320))
    return images
