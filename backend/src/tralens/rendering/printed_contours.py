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


def outlined_title(crop, polygon, count, protected):
    """Isolate a word's dark outlines on white stock, including its short final dash."""
    if not 3 <= count <= 10:
        return None
    polygon = np.float32(polygon)
    hull = np.zeros(protected.shape, np.uint8)
    cv2.fillConvexPoly(hull, np.int32(np.rint(polygon)), 1)
    page = crop.min(2)
    if np.mean(page[hull != 0] > 245) < .55:
        return None
    transform = cv2.getPerspectiveTransform(polygon,np.float32([[0,0],[count,0],[count,1],[0,1]]))
    _, labels, stats, centers = cv2.connectedComponentsWithStats(np.uint8(crop.max(2)<110))
    slots = {}
    for i in np.unique(labels[hull != 0]):
        if not i or stats[i,cv2.CC_STAT_AREA] < 16:
            continue
        ys,xs = np.where(labels == i)
        if not hull[ys,xs].all():
            continue
        xy = cv2.perspectiveTransform(np.float32([[[xs.min(),ys.min()],[xs.max(),ys.max()],centers[i]]]),transform)[0]
        u,v = xy[2]
        slot = int(u)
        span = xy[1]-xy[0]
        if not (0<=slot<count and .19<=u-slot<=.84 and .30<v<.73 and
                .24<=span[0]<=1.65 and (.13 if slot==count-1 else .41)<span[1]<1.12):
            continue
        if slot in slots:
            return None
        slots[slot] = i
    if len(slots) != count:
        return None
    core = np.isin(labels,list(slots.values()))
    # Expand through antialiasing only when a pixel is closer to the
    # letter than any foreign dark ink (spine, cover rule or smaller print).
    foreign = (labels!=0) & ~core
    distance = cv2.distanceTransform(np.uint8(~core),cv2.DIST_L2,5)
    guard = cv2.distanceTransform(np.uint8(~foreign),cv2.DIST_L2,5)
    mask = (distance<=3) & (distance<guard) & (hull!=0) & ~protected
    ring = (distance>3) & (distance<=6) & (guard>2)
    if not ring.any() or np.mean(page[ring]>248)<.92 or (protected & core).any():
        return None
    return mask


def colored_title(rgb, own, protected, hue):
    """Recover the colored body, rim and printed shadow on a uniform cover stock."""
    if not isinstance(hue,int) or not 0<=hue<180 or own.sum()<4000 or own.size>1500000:
        return None
    h,s,v = cv2.split(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV))
    delta = abs(h.astype(np.int16)-hue)
    seed = own & (s>=170)&(v>=150)&(np.minimum(delta,180-delta)<5)
    count,labels,stats,_=cv2.connectedComponentsWithStats(seed.astype(np.uint8))
    if count>4096:
        return None
    keep=(stats[:,4]>max(55,.0006*own.sum()))&(stats[:,2]>6)&(stats[:,3]>8)
    keep[0]=False
    if keep.sum()<4:
        return None
    body=keep[labels]
    if not .1*own.sum()<body.sum()<.65*own.sum():
        return None
    # Flat paper dominates the gaps. Different painted artwork, including
    # fingers beside the logo, must survive even inside the original OCR box.
    swatch=rgb[own & ~body]
    background=np.median(swatch,axis=0)
    same=np.max(np.abs(rgb.astype(np.int16)-background),axis=2)<12
    if np.mean(same[own & ~body])<.46:
        return None
    # The same pigment may fill a tiny detached accent inside a larger
    # outlined glyph. Keep only islands near the independently read bodies.
    near_body=cv2.distanceTransform(np.uint8(~body),cv2.DIST_L2,5)<35
    body |= seed & near_body
    distance=cv2.distanceTransform(np.uint8(~body),cv2.DIST_L2,5)
    near=distance<=12
    light=(rgb.min(2)>185)&(s<120)
    dark=rgb.max(2)<130
    paper_edge=cv2.blur(same.astype(np.float32),(15,15))>.16
    surround=cv2.dilate(np.uint8(body|(near&light)),np.ones((9,9),np.uint8))!=0
    remove=(distance<6)|(near & (light | (dark & surround)))|((distance<15) & paper_edge
             & (np.abs(rgb.astype(np.int16)-background).max(2)<85))
    remove=(cv2.dilate(remove.astype(np.uint8),np.ones((3,3),np.uint8))!=0)&~protected
    result=rgb.copy()
    result[remove]=np.uint8(background)
    return remove,result,np.median(rgb[body],axis=0).astype(int).tolist()

def pale_speedline(crop, own, protected):
    """Separate white letters from a long, straight motion line crossing several glyphs."""
    from . import preparation as p
    _,_,width,height=cv2.boundingRect(np.uint8(own))
    if not (20<=width<=110 and 1.65*width<height<=300):
        return None
    light=crop.min(2)>205
    length=round(.74*height)
    length+=1-length%2
    half=length//2
    tracks=np.zeros_like(own)
    for angle in range(-30,31):
        off=round(half*np.tan(np.deg2rad(angle)))
        kernel=np.zeros((length,2*abs(off)+1),np.uint8)
        cv2.line(kernel,(abs(off)-off,0),(abs(off)+off,length-1),1,1)
        opened=cv2.morphologyEx(np.uint8(light),cv2.MORPH_OPEN,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        tracks |= (opened!=0)&own
    _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(tracks))
    yy,xx=np.indices(own.shape)
    guard=np.zeros_like(own)
    for j,(_,_,_,h,area) in enumerate(stats[1:],1):
        if h<.7*length or area>4*h:
            continue
        py,px=np.nonzero(labels==j)
        vx,vy,x0,y0=cv2.fitLine(np.float32(np.column_stack((px,py))),cv2.DIST_HUBER,0,.01,.01).reshape(-1)
        distance=np.abs((xx-x0)*vy-(yy-y0)*vx)
        guard |= (distance<1.05)&(yy>=py.min()-4)&(yy<=py.max()+4)&p.dilate(light,1)
    full=p.dilate(light&own,2)
    return full & ~guard & ~protected, full


def colored_letter_strokes(crop, bounds, probability, detection, protected):
    """Measure a colored fill and its dark outline on the original lettered artwork."""
    from PIL import Image
    from ..ocr.masked_lines import painted_outline
    from ..ocr.recognition_units import image_fingerprint
    from . import preparation as p

    proof = detection.get('ocr_metadata',{}).get('painted_effect')
    if not proof or proof.get('white_sideways'):
        return None
    x,y,r,b = proof['bounds']
    l,t,right,bottom = bounds
    if not (l<=x<r<=right and t<=y<b<=bottom):
        return None
    tile = np.s_[y-t:b-t,x-l:r-l]
    native = crop[tile]
    if image_fingerprint(Image.fromarray(native))!=proof['source_digest']:
        return None
    ink=painted_outline(native,probability[tile])
    if ink is None:
        return None
    distance=cv2.distanceTransform(np.uint8(~ink),cv2.DIST_L2,5)
    hue,saturation,value=cv2.split(cv2.cvtColor(native,cv2.COLOR_RGB2HSV))
    near=(distance<=3.5)&(distance>0)&(saturation>=18)&(value>=145)
    if near.sum()<15:
        return None
    peak=np.bincount(hue[near],weights=np.sqrt(saturation[near]),minlength=180).argmax()
    delta=np.abs(hue.astype(float)-peak)
    delta=np.minimum(delta,180-delta)
    cutoff=max(14,.66*np.percentile(saturation[near & (delta<5)],80))
    colored=(delta<8)&(saturation>cutoff)&(value>85)&(distance<=max(10,.17*min(ink.shape)))
    _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(colored))
    votes=np.bincount(labels[(distance<=3)&colored],minlength=len(stats))
    kept=(votes>=3)&(votes>.04*stats[:,4])&(stats[:,4]<.4*ink.size)
    kept[0]=False
    face=kept[labels]
    if face.sum()<30 or face.sum()<.15*ink.sum():
        return None
    strokes=ink|face
    if proof['reading'].endswith(('...', '…')) and native.shape[0]>1.8*native.shape[1]:
        # Faint outlined dots can vanish at the dark outline threshold. Restore
        # only a regularly spaced triplet with native detector support.
        strokes |= vertical_ellipsis(native,probability[tile])
    strokes |= p.dilate(strokes,3) & (native.max(2)<180)
    mask=np.zeros(crop.shape[:2],bool)
    mask[tile]=p.dilate(strokes,2)
    if (mask & protected).any():
        return None
    color=np.median(native[face],axis=0).astype(int).tolist()
    return mask, color


def vertical_ellipsis(native, probability):
    _,labels,stats,centers=cv2.connectedComponentsWithStats(np.uint8(native.max(2)<150))
    support=np.bincount(labels[probability>=128],minlength=len(stats))
    dots=[j for j,(x,y,w,h,area) in enumerate(stats[1:],1)
          if y>.6*len(native) and 3<=w<=12 and 3<=h<=12 and .65<=w/h<=1.5
          and support[j]>=.8*area]
    found=np.zeros(labels.shape,bool)
    if len(dots)==3:
        points=centers[sorted(dots,key=lambda j:centers[j,1])]
        steps=np.diff(points,axis=0)
        size=np.median(stats[dots,cv2.CC_STAT_HEIGHT])
        if (np.ptp(points[:,0])<=size and steps[:,1].min()>=size
                and steps[:,1].max()<=2.7*size and steps[:,1].max()<=1.4*steps[:,1].min()):
            found=np.isin(labels,dots)
    return found


def diagonal_aside_mask(crop, bounds, probability, detection, protected):
    """Recover the source-confirmed lettering while preserving adjacent illustration contours."""
    from PIL import Image
    from ..ocr.masked_lines import diagonal_ink
    from ..ocr.recognition_units import image_fingerprint

    proof = detection.get('ocr_metadata', {}).get('handwritten_aside')
    if not proof or proof.get('reading') != detection.get('text'):
        return None
    x,y,r,b = proof['bounds']
    l,t,right,bottom = bounds
    if not (l <= x < r <= right and t <= y < b <= bottom):
        return None
    tile = np.s_[y-t:b-t, x-l:r-l]
    native = Image.fromarray(crop[tile])
    if image_fingerprint(native) != proof['views'][0]:
        return None
    gray = np.asarray(native.convert('L'))
    core = diagonal_ink(gray, probability[tile], proof['scale'],touching=bool(proof.get('touching')))
    if image_fingerprint(Image.fromarray(np.uint8(~core)*255).convert('RGB')) != proof['views'][1]:
        return None
    # Complete antialiasing up to the edge of each isolated ink component.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray < 182))
    support = np.bincount(labels[core], minlength=len(stats))
    kept = support > .48*stats[:,cv2.CC_STAT_AREA]
    kept[0] = False
    # Small detached strokes inside a kana can be thinner than the native
    # recognizer's input. Require nearby confirmed strokes on multiple sides.
    hull = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(np.uint8(core))), 1)
    nearby = cv2.dilate(np.uint8(core), np.ones((7,7),np.uint8)) != 0
    supported = (hull != 0) & nearby & (probability[tile] > 12)
    counts = np.bincount(labels[supported], minlength=len(stats))
    small = (stats[:,cv2.CC_STAT_AREA] <= proof['scale']) & (counts >= .7*stats[:,cv2.CC_STAT_AREA])
    mask = np.zeros(crop.shape[:2], bool)
    strokes=(kept | small)[labels]
    if proof.get('touching'):
        strokes |= core
    mask[tile] = cv2.dilate(np.uint8(strokes), np.ones((3,3),np.uint8)) != 0
    # A one-pixel antialiasing margin can touch a neighboring detector on
    # blank paper. Leave those pixels alone while retaining the verified ink.
    if not mask.any() or (mask & protected & (crop.min(axis=2) < 250)).any():
        return None
    return mask & ~protected


def drawn_kana_outline(crop, bounds, probability, detection, protected):
    """Remove a measured white rim only around the three source-verified glyphs."""
    from PIL import Image
    from ..ocr.masked_lines import drawn_kana_ink
    from ..ocr.recognition_units import image_fingerprint

    proof = detection.get('ocr_metadata', {}).get('drawn_kana')
    if not proof:
        return None
    x,y,r,b = proof['source_bounds']
    l,t,right,bottom = bounds
    if not (l<=x<r<=right and t<=y<b<=bottom and r-x<=430 and b-y<=160):
        return None
    tile=np.s_[y-t:b-t,x-l:r-l]
    native=Image.fromarray(crop[tile])
    if image_fingerprint(native)!=proof['views'][0]:
        return None
    glyphs=drawn_kana_ink(np.asarray(native.convert('L'))<140,probability[tile])
    if glyphs is None or image_fingerprint(Image.fromarray(np.uint8(~glyphs)*255).convert('RGB'))!=proof['views'][1]:
        return None
    core=np.zeros(crop.shape[:2],bool)
    core[tile]=glyphs
    return white_letter_rim(crop, core, protected)


def detached_letter_outline(crop, bounds, probability, proof, protected):
    from PIL import Image
    from ..ocr.detached_kana import ink_views

    if not proof.get('island'):
        return None
    x, y, r, b = bounds
    left, top, right, bottom = proof['island']
    if not (x <= left < right <= r and y <= top < bottom <= b):
        return None
    views = ink_views(Image.fromarray(crop), probability, (left-x, top-y, right-x, bottom-y))
    if views is None:
        return None
    (l, t, end, base), _, _, strokes = views
    if [l+x, t+y, end+x, base+y] != proof['source_bounds']:
        return None
    core = np.zeros(crop.shape[:2], bool)
    core[t:base, l:end] = strokes
    if np.median(crop[core]) > 90:
        return None
    return white_letter_rim(crop, core, protected)


def white_letter_rim(crop, core, protected):
    distance=cv2.distanceTransform(np.uint8(~core),cv2.DIST_L2,5)
    gray=crop.min(2)
    inner=(distance>1)&(distance<=3)
    outer=(distance>=10)&(distance<=14)
    if not inner.any() or not outer.any() or np.mean(gray[inner]>=248)<.8 or np.median(gray[outer])>=245:
        return None
    # Fade-out of the printed white outline determines its width. Cap it
    # before nearby white clothing or a black frame could join the mask.
    radius=3
    for radius in range(3,11):
        ring=(distance>radius)&(distance<=radius+1)
        if not ring.any() or np.mean(gray[ring]>=248)<.45:
            break
    mask=(distance<=1.5)|((distance<=radius+3)&(gray>np.median(gray[outer])+3))
    return None if (core & protected).any() else mask & ~protected

def interrupted_ink(crop, mask, protected):
    """Count independent pen strokes entering the lettering from surrounding art."""
    from . import preparation as p
    field=p.dilate(mask,16)&~mask&~protected&(crop.max(2)<140)
    _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(field))
    near=np.unique(labels[field&p.dilate(mask,3)])
    far=np.unique(labels[field&~p.dilate(mask,9)])
    return sum(stats[i,cv2.CC_STAT_AREA]>=12 for i in np.intersect1d(near,far))>=3


def outlined_word_mask(crop, bounds, metadata, protected):
    """Recreate the OCR stroke interiors and measure the width of their black outline."""
    from PIL import Image
    from ..ocr.outlined_sounds import hollow_letter_shapes, outlined_ellipsis, outlined_surface
    from ..ocr.recognition_units import image_fingerprint
    x,y,r,b = bounds
    evidence = metadata.get('outlined_display') or metadata.get('heavy_print') or metadata.get('hollow_print') or metadata['outlined_ellipsis']
    l,t,right,bottom = evidence['bounds']
    if not x <= l < right <= r or not y <= t < bottom <= b:
        return
    tile = np.s_[t-y:bottom-y,l-x:right-x]
    native = Image.fromarray(crop[tile])
    if image_fingerprint(native) != evidence['digest']:
        return
    gray = np.asarray(native.convert('L'))
    if 'outlined_display' in metadata:
        from ..ocr.outlined_sounds import bordered_strokes
        parts=bordered_strokes(gray,fine=evidence.get('fine_outline') is True)
        if parts is not None:
            mask=np.zeros_like(protected)
            mask[tile]=parts[1]
            return mask & ~protected
        return
    if 'heavy_print' in metadata:
        from ..ocr.display_readings import heavy_print_strokes,broad_pen_ink
        radius=evidence.get('motion_radius')
        if radius is None:
            shapes=heavy_print_strokes(gray)
            ink=None if shapes is None else shapes[0]
        else:
            if type(radius) is not int or not 1<=radius<=8:
                return
            ink=broad_pen_ink(gray,radius)
        if ink is None:
            return
        mask=np.zeros_like(protected)
        mask[tile]=cv2.dilate(ink,np.ones((3,3),np.uint8))!=0
        return mask & ~protected
    if evidence.get('repeated_beat'):
        shapes = hollow_letter_shapes(gray)
        if shapes is None:
            return
        mask = np.zeros_like(protected)
        mask[tile] = cv2.dilate(shapes,np.ones((3,3),np.uint8))!=0
        # A second, white outline on shaded paper can survive outside the black letter.
        distance=cv2.distanceTransform(np.uint8(~mask),cv2.DIST_L2,5)
        light=crop.min(2)>=245
        near=(distance>0)&(distance<=2)&~protected
        far=(distance>10)&(distance<=14)&~protected
        if near.any() and far.any() and light[near].mean()>.90:
            background=light[far].mean()
            if background<.80:
                for width in range(2,11):
                    ring=(distance>width)&(distance<=width+1)&~protected
                    if ring.any() and light[ring].mean()<=background+.15:
                        mask |= (distance<=width+2)&(crop.min(2)>210)
                        break
        return mask & ~protected
    if metadata.get('hollow_print'):
        core=outlined_surface(np.asarray(native))[:,:,0]<110
        if not core.any():
            return
        radius=cv2.distanceTransform(np.uint8(core),cv2.DIST_L2,5).max()
    else:
        found = outlined_ellipsis(gray)
        if found is None:
            return
        core, _, radius = found
    previous = core
    for distance in range(1, min(16, round(2*radius)+3)):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*distance+1,2*distance+1))
        expanded = cv2.dilate(np.uint8(core),kernel) != 0
        edge = expanded & ~previous
        if not edge.any():
            return
        if np.mean(gray[edge]<200) < .2:
            mask = np.zeros_like(protected)
            mask[tile] = expanded
            return mask & ~protected
        previous = expanded
