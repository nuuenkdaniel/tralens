"""Reconstruct monochrome printed contours intersecting a confirmed hollow effect."""
import warnings

import cv2
import numpy as np


def restore_round_edge(gray, mask, output):
    nearby = cv2.dilate(mask.astype('uint8'), np.ones((45,45),'uint8')).astype(bool)
    histogram = np.bincount(gray[nearby & ~mask], minlength=256)
    tone = 50 + histogram[50:225].argmax()
    if histogram[tone] < 64:
        return
    flat = (np.abs(gray.astype(np.int16)-tone)<4) & ~mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(flat.astype('uint8'))
    hits = np.bincount(labels[nearby], minlength=count)
    hits[0] = 0
    index = hits.argmax()
    if hits[index]<64 or not 1000<stats[index,cv2.CC_STAT_AREA]<150000:
        return
    contours,_ = cv2.findContours(np.uint8(labels==index),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    points = max(contours,key=cv2.contourArea)[:,0,:]
    if not 40 <= len(points) <= 4096:
        return
    # Ignore lettering, specular highlights and occlusion by fitting against
    # angular coverage of the surviving rim, rather than its bounding box.
    rng = np.random.default_rng(0)
    best = None
    for _ in range(800):
        sample = points[rng.choice(len(points),6,replace=False)].astype('float32')
        try:
            (cx,cy),(w,h),angle = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        if not 60 < min(w,h) <= max(w,h) < 500 or min(w,h)<.55*max(w,h):
            continue
        cos,sin = np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
        dx,dy = points[:,0]-cx,points[:,1]-cy
        u,v = 2*(cos*dx+sin*dy)/w,2*(-sin*dx+cos*dy)/h
        support = np.abs(np.hypot(u,v)-1)*min(w,h)/2 < 1.4
        sectors = np.unique(np.floor(np.arctan2(v[support],u[support])*18/np.pi)).size
        score = sectors,int(support.sum())
        if best is None or score>best[0]:
            best = score,(cx,cy,w,h,cos,sin)
    if best is None or best[0][0]<25:
        return
    cx,cy,w,h,cos,sin = best[1]
    yy,xx = np.where(mask)
    dx,dy = xx-cx,yy-cy
    u,v = 2*(cos*dx+sin*dy)/w,2*(-sin*dx+cos*dy)/h
    radius = np.hypot(u,v)
    distance = (radius-1)*min(w,h)/2
    if not ((distance<0).any() and (distance>0).any()):
        return
    take = (distance>-15)&(distance<36)
    theta = np.arctan2(v[take],u[take])
    radius,xx,yy = radius[take],xx[take],yy[take]
    samples = []
    for offset in range(-32,33):
        t = theta+np.deg2rad(offset)
        u,v = radius*np.cos(t)*w/2,radius*np.sin(t)*h/2
        x = np.rint(cx+cos*u-sin*v).astype(int)
        y = np.rint(cy+sin*u+cos*v).astype(int)
        inside = (x>=0)&(x<gray.shape[1])&(y>=0)&(y<gray.shape[0])
        x,y = x.clip(0,gray.shape[1]-1),y.clip(0,gray.shape[0]-1)
        samples.append(np.where(inside & ~mask[y,x], gray[y,x], np.nan))
    samples = np.array(samples)
    valid = np.isfinite(samples).sum(0)>10
    output[yy[valid],xx[valid]] = np.nanmedian(samples[:,valid],axis=0).astype('uint8')[:,None]


def restore_parallel_rules(gray, mask, blocked, output):
    known = ~(blocked | mask)
    lines = cv2.HoughLinesP(np.uint8((gray<80)&known)*255,1,np.pi/180,12,minLineLength=24,maxLineGap=1)
    if lines is None:
        return
    slopes = [(b-y)/(r-x) for x,y,r,b in lines[:,0].astype(float)
              if abs(r-x)>24 and abs(b-y)<.65*abs(r-x)]
    if len(slopes)<3:
        return
    counts,edges = np.histogram(slopes,np.arange(-.675,.7,.05))
    peak = counts.argmax()
    if counts[peak]<2:
        return
    slope = np.median([s for s in slopes if edges[peak]<=s<edges[peak+1]])
    yy,xx = np.where(mask)
    offsets = np.arange(-48,49)
    sy = (yy[:,None]+slope*offsets).astype('float32')
    sx = (xx[:,None]+offsets).astype('float32')
    gray = gray.astype('float32')
    values = np.stack([cv2.remap(gray,sx,sy+shift,cv2.INTER_LINEAR) for shift in range(-3,4)])
    valid = np.stack([cv2.remap(np.uint8(known),sx,sy+shift,cv2.INTER_NEAREST)!=0 for shift in range(-3,4)]).all(0)
    banks = []
    # A missing bank is expected at the panel edge. Estimate from the other
    # bank only when it supplies a stable, high-contrast cross-section.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for side in (offsets<0, offsets>0):
            available = valid & side
            distance = np.where(available,abs(offsets),999)
            available &= distance<distance.min(1)[:,None]+17
            samples = np.where(available[None],values,np.nan)
            profile = np.nanmedian(samples,axis=2)
            error = np.nanmean(abs(samples-profile[:,:,None]),axis=(0,2))
            banks.append((available.sum(1),profile,error))
        for index,(count,profile,error) in enumerate(banks):
            other_count,other_profile,_ = banks[1-index]
            consistent = (other_count<4) | (np.nanmean(abs(other_profile-profile),axis=0)<55)
            use = (count>=9)&(error<21)&(np.ptp(profile,axis=0)>95)&consistent
            output[yy[use],xx[use]] = np.rint(profile[3,use]).astype('uint8')[:,None]


def refine(original, bounds, mask, blocked, candidate):
    if not 0 < mask.sum() <= 8192:
        return
    x,y,r,b = bounds
    l,t = max(0,x-192),max(0,y-192)
    crop = original[t:min(original.shape[0],b+192),l:min(original.shape[1],r+192)]
    if np.ptp(crop[::3,::3],axis=2).max()>12:
        return
    window = np.s_[y-t:b-t,x-l:r-l]
    missing = np.zeros(crop.shape[:2],bool)
    missing[window] = mask
    plate = crop.copy()
    plate[window] = candidate
    gray = crop.min(2)
    restore_round_edge(gray,missing,plate)
    candidate[mask] = plate[window][mask]
    restore_parallel_rules(gray[window],mask,blocked,candidate)


def vertical_continuations(crop, hidden, blocked):
    """Match both ends of a thin vertical rule interrupted by hollow lettering."""
    gray = crop.min(2)
    repaired = np.zeros_like(hidden)
    tones = gray.copy()
    # Include the white rim, where the Japanese effect itself interrupted the line.
    region = cv2.dilate(np.uint8(hidden),np.ones((9,5),np.uint8)).astype(bool)
    for x in np.flatnonzero(region.any(0)):
        if x<3 or x+3>=gray.shape[1]:
            continue
        occupied = np.flatnonzero(region[:,x])
        top,bottom = occupied[0],occupied[-1]+1
        if bottom-top>74 or top<12 or bottom+12>gray.shape[0]:
            continue
        profiles = []
        for samples in (gray[top-12:top,x-3:x+4],gray[bottom:bottom+12,x-3:x+4]):
            profile = np.median(samples,axis=0)
            if not (profile[3]<205 and profile[0]>238 and profile[-1]>238
                    and np.count_nonzero(samples[:,3]<216)>=8):
                break
            profiles.append(profile)
        if len(profiles)!=2 or np.mean(abs(profiles[0]-profiles[1]))>32:
            continue
        rows = np.arange(top,bottom)
        rows = rows[~blocked[rows,x]]
        repaired[rows,x] = True
        fraction = (rows-top+.5)/(bottom-top)
        tones[rows,x] = np.rint((1-fraction)*profiles[0][3]+fraction*profiles[1][3]).astype('uint8')
    return repaired,tones


def framed_outline_mask(crop, own, protected):
    """Isolate black strokes with a white outline inside a ruled grayscale panel."""
    x,y,w,h = cv2.boundingRect(own.astype(np.uint8))
    if w < 96 or h < 40 or w < 1.4*h or np.max(np.ptp(crop[own],axis=1)) > 8:
        return None
    gray = crop[:,:,0]
    # A frame line crosses nearly the full width, unlike the sound itself.
    dark_fraction = (gray[:,x:x+w] < 55).mean(1)
    rules = np.flatnonzero(dark_fraction > .94)
    top = rules[(rules >= y-12) & (rules <= y+12)]
    bottom = rules[(rules >= y+h-13) & (rules <= y+h+12)]
    if not top.size or not bottom.size:
        return None
    top,bottom = top[-1]+1,bottom[0]
    while top < bottom and dark_fraction[top] > .5:
        top += 1
    while bottom > top and dark_fraction[bottom-1] > .5:
        bottom -= 1
    if bottom <= top:
        return None
    left,right = max(0,x-12),min(gray.shape[1],x+w+12)
    band = gray[top:bottom,left:right]
    rails = np.flatnonzero((band < 55).mean(0) > .97) + left
    left_edges,right_edges = rails[rails <= x+8],rails[rails >= x+w-8]
    if left_edges.size:
        left = left_edges[-1]+1
    if right_edges.size:
        right = right_edges[0]
    field = gray[top:bottom,left:right]
    if field.size == 0:
        return None
    # Require consistent midtone paper across the rows. White speech balloons,
    # textured paintings and halftones must use the normal restoration route.
    levels = np.array([np.bincount(row,minlength=256)[80:233].argmax()+80 for row in field])
    if np.mean((np.abs(field-levels[:,None]) <= 7).mean(1) > .23) < .9:
        return None
    dark = (field < 75).astype(np.uint8)
    light = (field > 238).astype(np.uint8)
    if not .04 < dark.mean() < .35 or light.mean() < .08:
        return None
    near = cv2.dilate(dark,np.ones((5,5),np.uint8)).astype(bool)
    if np.mean(light[near & (dark==0)]) < .65:
        return None
    mask = cv2.dilate(dark,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))).astype(bool)
    count,labels,stats,_ = cv2.connectedComponentsWithStats(light,connectivity=8)
    if count > 128:
        return None
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA] < 10:
            continue
        component = labels==i
        rim = cv2.dilate(component.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool) & ~component
        if rim.any() and dark[rim].mean() > .45:
            mask |= cv2.dilate(component.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    result = np.zeros_like(own)
    result[top:bottom,left:right] = mask
    return result & ~protected
