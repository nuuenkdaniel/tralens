"""Recover the recurring strokes of a printed beat across different backgrounds."""
import re
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from . import segmentation


def samples(rgb, raw, probability, model):
    """Compare only repeated kana identified independently at distinct locations."""
    groups = defaultdict(list)
    for d in raw:
        word = d.get('selected_text') or d.get('text','')
        if d.get('reason')!='frontier_discovery' or not re.fullmatch(r'[ァ-ヺぁ-ゖ]',word):
            continue
        poly = np.asarray(d['polygon'])
        x,y = poly.min(0).astype(int)
        r,b = poly.max(0).astype(int)+1
        if 36<=min(r-x,b-y)<=max(r-x,b-y)<=270:
            groups[word].append((d['source_id'],int(x),int(y),int(r),int(b)))
    masks = {}
    for group in groups.values():
        if not 4<=len(group)<=16 or len(masks)+len(group)>24:
            continue
        # A second text-mask view presents the repeated glyphs at a comparable
        # scale. A low-resolution page score can miss the entire last beat.
        atlas = Image.new('RGB',(1000,((len(group)+4)//5)*200),'white')
        frames = []
        for i,(_,x,y,r,b) in enumerate(group):
            frame = max(0,x-15),max(0,y-15),min(rgb.width,r+15),min(rgb.height,b+15)
            tile = rgb.crop(frame)
            tile.thumbnail((194,194),Image.Resampling.LANCZOS)
            left,top = i%5*200,i//5*200
            atlas.paste(tile,(left,top))
            frames.append((frame,(left,top,left+tile.width,top+tile.height)))
        atlas_prob = segmentation.TextMask(atlas,model).raw
        tiles = []
        original = np.asarray(rgb)
        for (_,x,y,r,b),(frame,box) in zip(group,frames):
            l,t,rr,bb = frame
            a,c,d,e = box
            zoomed = cv2.resize(atlas_prob[c:e,a:d],(rr-l,bb-t))[y-t:b-t,x-l:r-l]
            score = np.maximum(probability[y:b,x:r],zoomed)
            tiles.append((original[y:b,x:r],score))
        for (member,x,y,r,b),mask in zip(group,shared_ink(tiles)):
            if mask is not None:
                masks[member] = ((x,y,r,b),mask)
    return masks


def shared_ink(tiles):
    size,bins = 96,12
    features = []
    angles = 2*np.pi*np.arange(bins)/bins
    prior = np.zeros((bins,size,size),np.float32)
    for crop,prob in tiles:
        gray = cv2.resize(crop.max(2),(size,size),interpolation=cv2.INTER_AREA)
        score = cv2.resize(prob,(size,size))
        ink = gray<120
        blur = cv2.GaussianBlur(ink.astype(np.float32),(3,3),.8)
        dx,dy = cv2.Sobel(blur,-1,1,0),cv2.Sobel(blur,-1,0,1)
        xx,yy,xy = [cv2.GaussianBlur(v,(0,0),3) for v in (dx*dx,dy*dy,dx*dy)]
        angle = np.arctan2(2*xy,xx-yy)%(2*np.pi)
        coherence = np.hypot(xx-yy,2*xy)/(xx+yy+1e-4)
        features.append((ink,score,angle,coherence))
        for i,theta in enumerate(angles):
            seed = ink & (np.cos(angle-theta)>.55) & (coherence>.23) & (score>38)
            if seed.any():
                dist = cv2.distanceTransform(np.uint8(~seed),cv2.DIST_L2,5)
                prior[i] += np.exp(-(dist/9)**2)
    prior /= len(tiles)
    ys,xs = np.indices((size,size))
    for (crop,_),(ink,score,angle,coherence) in zip(tiles,features):
        support = prior[np.rint(angle*bins/(2*np.pi)).astype(int)%bins,ys,xs]
        shared = ink & ((support>.33) | ((score>68)&(support>.20)) | ((prior.max(0)>.53)&(coherence<.32)))
        core = cv2.morphologyEx(np.uint8(ink),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        _,labels,stats,_ = cv2.connectedComponentsWithStats(core)
        near = cv2.dilate(np.uint8(shared),np.ones((3,3),np.uint8))!=0
        own = np.zeros_like(ink)
        for i in np.flatnonzero(stats[1:,-1]>=6)+1:
            component = labels==i
            fraction = shared[component].mean()
            if fraction>.32:
                own |= component if fraction>.80 else component & near
        h,w = crop.shape[:2]
        native = cv2.resize(np.uint8(own),(w,h),interpolation=cv2.INTER_NEAREST)
        core = cv2.dilate(native,np.ones((3,3),np.uint8)) & np.uint8(crop.max(2)<115)
        if not .045<=core.mean()<=.55:
            yield None
            continue
        width = max(3,round(min(h,w)*.042)|1)
        yield cv2.dilate(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width,width)))!=0


def local(masks, members, bounds, crop, protected):
    if not masks or any(i not in masks for i in members):
        return
    x,y,r,b = bounds
    commit = np.zeros(protected.shape,bool)
    inference = np.zeros_like(commit)
    for i in members:
        (l,t,rr,bb),core = masks[i]
        if not (x<=l<rr<=r and y<=t<bb<=b):
            return
        binary = np.zeros(protected.shape,np.uint8)
        binary[t-y:bb-y,l-x:rr-x] = core
        width = max(7,round(min(rr-l,bb-t)*.18)|1)
        field = cv2.dilate(binary,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width,width)))!=0
        commit |= (binary!=0) | (field & (crop.min(2)>212))
        inference |= field
    commit &= ~protected
    if commit.any():
        return commit,inference & ~protected


def pale_pair(crop, lines, origin, protected):
    """Match two nearby impressions of one pale printed beat on painted art."""
    if (len(lines)!=2 or lines[0].get('text')!=lines[1].get('text')
            or not re.fullmatch(r'[ぁ-ゖァ-ヺー]{2,5}',lines[0].get('text',''))):
        return None
    polygons = [np.int32(line['polygon'])-origin for line in lines]
    if any(poly.shape!=(4,2) for poly in polygons):
        return None
    sides = [cv2.minAreaRect(poly.astype(np.float32))[1] for poly in polygons]
    thickness = min(min(size) for size in sides)
    if not 10<=thickness<=60 or max(max(size) for size in sides)>180:
        return None
    offset = np.rint(polygons[1].mean(0)-polygons[0].mean(0)).astype(int)
    if not .55*thickness<=abs(offset[1])<=2*thickness or abs(offset[0])>thickness:
        return None
    gray = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    contrast = cv2.subtract(gray,cv2.morphologyEx(gray,cv2.MORPH_OPEN,np.ones((7,7),np.uint8)))
    pale = crop.min(2)>128
    _,components = cv2.connectedComponents(np.uint8(pale))
    light = pale & (contrast>=35) & (np.ptp(crop,axis=2)<65)
    strokes = []
    for poly in polygons:
        mask = np.zeros(gray.shape,np.uint8)
        cv2.fillPoly(mask,[poly.astype(np.int32)],1)
        own = mask!=0
        if np.median(gray[own])>170 or np.mean(np.ptp(crop[own],axis=1)>25)<.5:
            return None
        # A feather, highlight or border crossing the edge of the printed
        # row cannot establish a repeated stroke, even if it occurs twice.
        outside = cv2.dilate(mask,np.ones((7,7),np.uint8))==0
        foreign = np.unique(components[outside & pale])
        ink = light & own & ~np.isin(components,foreign)
        if ink.sum()<30 or not .035<=ink[own].mean()<=.32:
            return None
        strokes.append(ink)
    x,y,w,h = cv2.boundingRect(np.uint8(strokes[0]))
    if w<.6*max(sides[0]):
        return None
    template = strokes[0][y:y+h,x:x+w].astype(np.float32)
    scores = cv2.matchTemplate(strokes[1].astype(np.float32),template,cv2.TM_CCORR)
    cx,cy = offset+[x,y]
    l,t = max(0,cx-12),max(0,cy-12)
    window = scores[t:cy+13,l:cx+13]
    if not window.size:
        return None
    j,i = np.unravel_index(window.argmax(),window.shape)
    agreement = float(window[j,i]/np.sqrt(strokes[0].sum()*strokes[1].sum()))
    if agreement<.56:
        return None
    dx,dy = int(l+i-x),int(t+j-y)
    matched = np.zeros(gray.shape,bool)
    for current,other,sx,sy in [(strokes[0],strokes[1],-dx,-dy),(strokes[1],strokes[0],dx,dy)]:
        shifted = cv2.warpAffine(np.uint8(other),np.float32([[1,0,sx],[0,1,sy]]),(gray.shape[1],gray.shape[0]))
        matched |= current & (cv2.dilate(shifted,np.ones((5,5),np.uint8))!=0)
    if not (matched & ~protected).any():
        return None
    left,right = np.array_split(polygons[0][np.argsort(polygons[0][:,0])],2)
    vx,vy = right.mean(0)-left.mean(0)
    style = {'fill':np.quantile(crop[matched],.8,axis=0).astype(int).tolist(),'outline':0,
             'rotation':float(np.clip(-np.degrees(np.arctan2(vy,vx)),-20,20))}
    clean = (cv2.dilate(np.uint8(matched),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))!=0) & ~protected
    return matched & ~protected,clean,style,round(thickness*.68)
