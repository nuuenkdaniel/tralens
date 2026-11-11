"""Source-bound approximate lettering support for aggressive removal.

Learned probabilities locate candidates; native pixels complete their strokes.
This is deliberately separate from the strict semantic/flat-background route.
It may cut touching artwork, but a coarse OCR polygon is not an erase rectangle.
"""
import cv2
import numpy as np

from . import natural as n, preparation as p


POLICY='source_bound_semantic_strokes_v1'


def derive(crop,own,protected,probability):
    n.require(probability.shape==own.shape and probability.dtype==np.uint8,
        'source support needs native uint8 probability')
    n.require(crop.shape==(*own.shape,3) and crop.dtype==np.uint8,
        'source support needs native RGB pixels')
    limit=p.dilate(own,12)&~protected
    seed=(probability>=77)&own&~protected
    n.require(int(seed.sum())>=16,'insufficient source-bound learned support')
    # Hysteresis retains weak glyph edges connected to stronger evidence without
    # promoting disconnected low-probability texture elsewhere in the source box.
    weak=(probability>=32)&limit
    count,labels,_,_=cv2.connectedComponentsWithStats(weak.astype(np.uint8),8)
    n.budget_check(count<=8192,'source support component budget')
    ids=np.unique(labels[seed]);ids=ids[ids!=0]
    semantic=np.isin(labels,ids)
    support=p.dilate(semantic,2)&limit
    nearby=p.dilate(semantic,4)&limit
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    collar=p.dilate(semantic,6)&~support&~protected
    light=not collar.any() or np.median(gray[collar])>=128
    ink=gray<210 if light else gray>60
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
    n.budget_check(count<=16384,'source support native component budget')
    chosen=0
    for label in np.unique(labels[nearby&ink]):
        if label==0:continue
        x,y,w,h,area=map(int,stats[label])
        if max(w,h)>256 or area>32768:continue
        component=labels[y:y+h,x:x+w]==label
        if (component&~limit[y:y+h,x:x+w]).any():continue
        if np.count_nonzero(component&nearby[y:y+h,x:x+w])<area*.35:continue
        # Whole native outlines, including their interiors, remove hollow glyphs
        # without erasing the empty rectangular space between adjacent characters.
        contours,_=cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        filled=np.zeros_like(component,dtype=np.uint8)
        cv2.drawContours(filled,contours,-1,1,cv2.FILLED)
        support[y:y+h,x:x+w]|=filled!=0;chosen+=1
    support=p.dilate(support,1)&limit
    evidence={'policy':POLICY,'semantic_seed_pixels':int(seed.sum()),
        'semantic_support_pixels':int(semantic.sum()),'native_components':chosen,
        'source_polygon_pixels':int(own.sum()),'support_pixels':int(support.sum()),
        'source_extent_discarded':False,'source_removal_verified':False,
        'possible_collateral_art_loss':True,'polarity':'dark_ink' if light else 'light_ink'}
    return support,evidence
