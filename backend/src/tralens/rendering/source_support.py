"""Source-bound approximate lettering support for aggressive removal.

Learned probabilities locate candidates; native pixels complete their strokes.
This is deliberately separate from the strict semantic/flat-background route.
It may cut touching artwork, but a coarse OCR polygon is not an erase rectangle.
"""
import cv2
import numpy as np

from . import natural as n, preparation as p


POLICY='source_bound_semantic_strokes_v4_thin_contacts'
CONTEXT_POLICY='inference_only_neighbor_lettering_v1'


def restore_paper_band(crop, candidate, own, mask, protected):
    """Carry a uniform printed stripe through a white knockout around narrow lettering."""
    x,y,w,h = cv2.boundingRect(own.astype(np.uint8))
    if not 12 <= w <= min(h,120) or x < 13 or x+w+12 > crop.shape[1]:
        return mask
    sides = np.concatenate((crop[y:y+h,x-12:x-4],crop[y:y+h,x+w+3:x+w+11]),axis=1)
    tone = np.median(sides,axis=1)
    matches = np.abs(sides-tone[:,None]).max(2)<=5
    agree = (matches.mean(1)>=.75) & (matches[:,:8].mean(1)>=.625) & (matches[:,8:].mean(1)>=.625)
    gray = agree & (np.ptp(tone,axis=1)<10) & (tone.min(1)>=64) & (tone.max(1)<=247)
    # A one-row speck outside the glyph can interrupt an otherwise continuous band.
    similar = np.abs(np.diff(tone,axis=0)).max(1)<=4
    gray[1:-1] |= gray[:-2] & gray[2:] & similar[:-1] & similar[1:] & (matches[1:-1].mean(1)>=.625)
    transitions = np.r_[False,np.abs(np.diff(tone,axis=0)).max(1)>10]
    starts = np.flatnonzero(gray & (~np.r_[False,gray[:-1]] | transitions))
    ends = np.flatnonzero(gray & (~np.r_[gray[1:],False] | np.r_[transitions[1:],False]))+1
    result = mask.copy()
    collar = p.dilate(mask,8)
    cols = slice(x-3,x+w+3)
    for a,b in zip(starts,ends):
        if not 4<=b-a<=34 or np.ptp(tone[a:b],axis=0).max()>10:
            continue
        # An open gray field or a single gray edge cannot establish a stripe.
        if not any(agree[i] and tone[i].min()>249 for i in range(max(0,a-6),a)):
            continue
        if not any(agree[i] and tone[i].min()>249 for i in range(b,min(h,b+7))):
            continue
        end = b
        # Thin dark rules bordering the stripe need independent support at each row.
        while end<min(h,b+3) and matches[end].mean()>=.86 and tone[end].max()<tone[b-1].min()-15:
            end += 1
        for j in range(a,end):
            area = (y+j,cols)
            take = collar[area] & ~protected[area] & (mask[area] | (crop[area].min(1)>240))
            candidate[area][take] = np.rint(tone[j]).astype(np.uint8)
            result[area] |= take
    return result


def complete_contacts(ink,own,protected,probability,support,physical,labels,stats):
    """Separate thick native pieces from thin art contacts without following art.

    Only rejected whole components with substantial source-local text evidence
    can supply a parent. Opening supplies bounded native pieces, not new source
    geometry or a text classifier. Unpredicted art attached to the same parent
    can still be removed; committed pixels remain protected.
    """
    count=len(stats)
    owned_counts=np.bincount(labels[own&ink],minlength=count)
    weak_counts=np.bincount(labels[own&ink&(probability>=32)],minlength=count)
    strong_counts=np.bincount(labels[own&ink&(probability>=77)],minlength=count)
    rejected=np.zeros(count,bool);rejected[np.unique(labels[ink&~physical])]=True
    rejected|=(stats[:,2:4].max(axis=1)>256)|(stats[:,4]>32768)
    eligible=rejected&(strong_counts>=16)&(weak_counts>=owned_counts*.35)
    eligible[0]=False
    if not eligible.any():return support,[]
    radius=3
    count,cores,core_stats,_=cv2.connectedComponentsWithStats(p.erode(ink,radius).astype(np.uint8),8)
    n.budget_check(count<=16384,'source support contact component budget')
    support=support.copy();completed=[]
    for label in np.unique(cores[own]):
        if label==0:continue
        x,y,w,h,area=map(int,core_stats[label])
        if max(w,h)>256 or area>32768:continue
        core=cores[y:y+h,x:x+w]==label
        parent=int(labels[y:y+h,x:x+w][core][0])
        if not eligible[parent]:continue
        left,top=max(0,x-radius),max(0,y-radius)
        right,bottom=min(own.shape[1],x+w+radius),min(own.shape[0],y+h+radius)
        local=(slice(top,bottom),slice(left,right))
        component=p.dilate(cores[local]==label,radius)&ink[local]
        bx,by,bw,bh=cv2.boundingRect(component.astype(np.uint8));area=int(component.sum())
        if max(bw,bh)>256 or area>32768:continue
        if (component&~physical[local]).any() or np.count_nonzero(component&own[local])<area*.8:continue
        # Keep the one-pixel contour collar beyond the component's calculation
        # rectangle. The crop, physical limit and accepted edits still constrain it.
        lx,ty=max(0,left-1),max(0,top-1);rx,bt=min(own.shape[1],right+1),min(own.shape[0],bottom+1)
        block=np.zeros((bt-ty,rx-lx),bool);block[top-ty:bottom-ty,left-lx:right-lx]=component
        candidate=p.dilate(p.filled_outer(block),1)
        local=(slice(ty,bt),slice(lx,rx));candidate &= physical[local]&~protected[local]
        extra=candidate&~support[local]
        if not extra.any():continue
        support[local]|=candidate
        px,py,pw,ph,_=map(int,stats[parent])
        completed.append({'bounds':[left+bx,top+by,left+bx+bw,top+by+bh],'area':area,
            'parent_bounds':[px,py,px+pw,py+ph],'parent_source_pixels':int(owned_counts[parent]),
            'parent_weak_probability_pixels':int(weak_counts[parent]),
            'parent_strong_probability_pixels':int(strong_counts[parent]),'added_pixels':int(extra.sum())})
    return support,completed


def pale_sign_strokes(crop, own, protected, probability):
    """Trace a row of pale painted glyphs on a consistently colored surface."""
    points=cv2.findNonZero(own.astype(np.uint8))
    if points is None:
        return
    _,(w,h),_=cv2.minAreaRect(points)
    if not 15<=min(w,h)<=95 or max(w,h)<1.75*min(w,h) or max(w,h)>480:
        return
    x,y,width,height=cv2.boundingRect(points)
    if width<1.4*height:
        return
    pixels=crop[own]
    tone=np.median(pixels,axis=0)
    if not (18<np.ptp(tone)<135 and 60<tone.mean()<190
            and np.median(np.abs(pixels-tone).max(1))<22):
        return
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    cutoff=max(175,np.median(gray[own])+46)
    bright=(gray>cutoff) & (crop.min(2)>150)
    _,labels,stats,_=cv2.connectedComponentsWithStats(bright.astype(np.uint8))
    components=[]
    for label in np.unique(labels[bright & own]):
        a,b,cw,ch,area=stats[label]
        box=np.s_[b:b+ch,a:a+cw]
        component=labels[box]==label
        if (area<12 or ch<.25*height or cw>height or ch>height or not own[box][component].all()):
            continue
        components.append(label)
    if not 2<=len(components)<=18:
        return
    body=np.isin(labels,components)
    if (body.sum()<.06*own.sum() or body.sum()>.40*own.sum() or np.count_nonzero(bright & own & ~body)>.12*body.sum()
            or np.count_nonzero(body & (probability>=77))<12):
        return
    # Each letter needs to sit on the same colored material. A white window,
    # motion stripe or shirt inside the box will border a different surface.
    for label in components:
        ring=p.dilate(labels==label,5) & ~p.dilate(body,2)
        if not ring.any() or (np.abs(crop[ring]-tone).max(1)<40).mean()<.77:
            return
    radius=int(np.clip(round(min(w,h)*.06),2,4))
    mask=p.dilate(body,radius) & ~protected
    return mask, np.rint(np.median(crop[body],axis=0)).astype(int).tolist()


def outlined_interiors(crop, own, protected, probability):
    """Follow a white glyph interior to its tip where text probability fades."""
    light = crop.min(2)>180
    _, labels, stats, _ = cv2.connectedComponentsWithStats(light.astype(np.uint8))
    dark = crop.max(2)<135
    result = np.zeros_like(own)
    for label in np.unique(labels[own & (probability>=77) & light]):
        if not label:
            continue
        x,y,w,h,area = stats[label]
        if not (65<=area<=9000 and min(w,h)>=5 and max(w,h)<=230):
            continue
        pad = 8
        ys = slice(max(0,y-pad),min(own.shape[0],y+h+pad))
        xs = slice(max(0,x-pad),min(own.shape[1],x+w+pad))
        body = labels[ys,xs]==label
        if not own[ys,xs][body].all() or (probability[ys,xs][body]>=45).mean()<.70:
            continue
        ring = cv2.dilate(body.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool) & ~body
        if not ring.any() or dark[ys,xs][ring].mean()<.85:
            continue
        width = cv2.distanceTransform(body.astype(np.uint8),cv2.DIST_L2,5)[body]
        radius = int(np.clip(np.ceil(np.percentile(width,90)),2,7))
        rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
        result[ys,xs] |= cv2.dilate(body.astype(np.uint8),rim).astype(bool)
    return result & ~protected


def hollow_effect_masks(crop, own, protected, probability):
    """Measure each outlined stroke's dark rim without collecting touching illustration lines."""
    light = crop.min(2) > 170
    dark = crop.max(2) < 135
    _, labels, stats, _ = cv2.connectedComponentsWithStats(light.astype(np.uint8))
    result = np.zeros_like(own)
    widths = []
    for i in np.unique(labels[own & (probability >= 77)]):
        x,y,w,h,area = stats[i]
        if not i or not (40 < area < 15000 and 4 < min(w,h) and max(w,h) < 250):
            continue
        region = np.s_[max(0,y-12):y+h+12,max(0,x-12):x+w+12]
        body = labels[region] == i
        if own[region][body].mean() < .78 or (probability[region][body] >= 77).mean() < .60:
            continue
        distance = cv2.distanceTransform(np.uint8(~body), cv2.DIST_L2, 5)
        ring = (distance > 0) & (distance <= 2)
        if not ring.any() or dark[region][ring].mean() < .75:
            continue
        # Most of the next ring becomes page color just outside the outline.
        for radius in range(2,11):
            ring = (distance >= radius) & (distance < radius+1)
            if ring.any() and dark[region][ring].mean() < .5:
                result[region] |= distance <= radius+1
                widths.append(radius)
                break
    seeds = own & (probability >= 150) & ~protected
    if len(widths) >= 3 and seeds.any() and result[seeds].mean() >= .57:
        # The white outer border is part of outlined manga type. Leaving it
        # as known model context produces pale letter-shaped gaps. Preserve
        # original dark illustration strokes inside this additional collar.
        radius = int(np.median(widths))
        inference = p.dilate(result,radius)
        commit = (result | (inference & (crop.min(2)>238))) & ~protected
        return commit,inference


def derive(crop,own,protected,probability):
    n.require(probability.shape==own.shape and probability.dtype==np.uint8,
        'source support needs native uint8 probability')
    n.require(crop.shape==(*own.shape,3) and crop.dtype==np.uint8,
        'source support needs native RGB pixels')
    physical=p.dilate(own,12)
    limit=physical&~protected
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
    contours,_=cv2.findContours(own.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Measure slanted source columns in their own orientation. This changes only
    # a geometry measurement; masks and model inputs keep native page pixels.
    source_rect = cv2.minAreaRect(np.concatenate(contours))
    thickness=min(source_rect[1])+1
    narrow=thickness<=96
    line_limit=max(12,thickness*2)
    chosen=0;completed=0;weak_completed=[]
    for label in np.unique(labels[(nearby | own)&ink]):
        if label==0:continue
        x,y,w,h,area=map(int,stats[label])
        if max(w,h)>256 or area>32768:continue
        component=labels[y:y+h,x:x+w]==label
        # Test physical containment before subtracting accepted output. An old
        # edit touching one end of a glyph must not veto removal of its other end.
        if (component&~physical[y:y+h,x:x+w]).any():continue
        associated=np.count_nonzero(component&nearby[y:y+h,x:x+w])>=area*.35
        # A narrow source line can contain entire trailing characters missed by
        # CTD. Require native bounded components mostly owned by that line; do
        # not generalize this completion to oversized detector rectangles.
        owned=np.count_nonzero(component&own[y:y+h,x:x+w])>=area*.8
        in_line=narrow and max(w,h)<=line_limit and owned
        # A disconnected part of a large glyph can have weak CTD evidence but
        # no strong seed. Complete its ORIGINAL bounded component, not the whole
        # low-probability field or an oversized detector rectangle. Measure before
        # subtracting protection so old edits cannot strand the rest of a stroke.
        weak_pixels=int(np.count_nonzero(component&(probability[y:y+h,x:x+w]>=32)))
        weak_component=owned and weak_pixels>=16
        if not associated and not in_line and not weak_component:continue
        # Whole native outlines, including their interiors, remove hollow glyphs
        # without erasing the empty rectangular space between adjacent characters.
        contours,_=cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        filled=np.zeros_like(component,dtype=np.uint8)
        cv2.drawContours(filled,contours,-1,1,cv2.FILLED)
        support[y:y+h,x:x+w]|=filled!=0;chosen+=1
        completed+=not associated and in_line
        if weak_component and not associated and not in_line:
            weak_completed.append({'bounds':[x,y,x+w,y+h],'area':area,'weak_probability_pixels':weak_pixels})
    support=p.dilate(support,1)&limit
    support,contacts=complete_contacts(ink,own,protected,probability,support,physical,labels,stats)
    evidence={'policy':POLICY,'semantic_seed_pixels':int(seed.sum()),
        'semantic_support_pixels':int(semantic.sum()),'native_components':chosen,
        'line_completed_components':completed,'weak_completed_components':weak_completed,
        'contact_completed_components':contacts,'contact_opening_radius':3,
        'contact_added_pixels':sum(v['added_pixels'] for v in contacts),
        'narrow_source_line':narrow,'source_line_thickness':float(thickness),
        'source_polygon_pixels':int(own.sum()),'support_pixels':int(support.sum()),
        'source_extent_discarded':False,'source_removal_verified':False,
        'possible_collateral_art_loss':True,'polarity':'dark_ink' if light else 'light_ink'}
    # On pale painted gradients, gray<210 includes the background itself.
    # Prefer isolated high-contrast strokes when a slanted line has clear paper
    # between the glyphs and the original semantic mask corroborates them.
    angle = source_rect[2] % 90
    if (light and 12 < angle < 78 and thickness <= 50 and np.median(gray[own]) >= 155
            and np.median(np.ptp(crop[own], axis=1)) >= 7):
        core = gray < min(120, np.median(gray[own]) - 75)
        _, components, sizes, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8))
        near_line = p.dilate(own, 3)
        stroke = np.zeros_like(own)
        for label in np.unique(components[core & own]):
            x,y,w,h,area = sizes[label]
            local = np.s_[y:y+h,x:x+w]
            part = components[local] == label
            if (max(w,h) <= 1.6*thickness and (part & own[local]).sum() >= .85*area
                    and not (part & ~near_line[local]).any()):
                stroke[local] |= part
        if ((stroke & ~protected).sum() >= 16 and np.mean(support[stroke & ~protected]) >= .70
                and stroke.sum() >= .82*(core & own).sum()
                and not (core & own & ~stroke & ~p.dilate(protected, 9)).any()):
            refined = p.dilate(stroke,2) & ~protected
            if refined.sum() < support.sum():
                support = refined
                evidence.update(painted_stroke_support=True, support_pixels=int(refined.sum()))
    return support,evidence


def painted_stroke_fill(crop, mask, obstacle, *, slanted=False, pale_ink=False, measured_outline=False):
    """Reconstruct thin lettering on softly varying paint from nearby paper colors."""
    height,width = mask.shape
    strip = width >= 4*height and width <= 2048 and mask.size <= 512*512
    if (max(mask.shape) > 512 and not strip) or mask.sum() < 32:
        return None
    radius = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)[mask].max()
    if radius > (24 if strip or measured_outline else 12):
        return None
    ring = p.dilate(mask, 8) & ~mask & ~obstacle
    if not ring.any():
        return None
    samples = crop[ring].astype(np.float32)
    center = np.median(samples, axis=0)
    distance = np.abs(crop.astype(np.float32) - center).max(axis=2)
    smooth_strip = (strip or measured_outline) and np.percentile(distance[ring],90) <= 18
    if (center.min() < (55 if pale_ink else 115) or (radius > 12 and not smooth_strip)
            or (np.percentile(np.ptp(samples, axis=1), 50) < 7 and not smooth_strip)
            or np.mean(distance[ring] < 45) < (.62 if slanted else .72)):
        return None
    # Exclude outlying ink from neighboring characters even if OCR missed it.
    weights = ((~mask & ~obstacle) & (distance < 55)).astype(np.float32)
    sigma = max(3., float(radius) * .85)
    support = cv2.GaussianBlur(weights, (0,0), sigma)
    if support[mask].min() < .06:
        return None
    colors = cv2.GaussianBlur(crop * weights[:,:,None], (0,0), sigma)
    result = crop.copy()
    paper = np.clip(np.rint(colors[mask] / support[mask,None]), 0, 255)
    if smooth_strip:
        # Broad semantic masks include visible paper between printed strokes.
        # Preserve its texture wherever the measured local shade still agrees.
        matching = np.max(np.abs(crop[mask] - paper), axis=1) <= 10
        paper[matching] = crop[mask][matching]
    result[mask] = paper.astype(np.uint8)
    return result


def painted_keyline_fill(crop, ink, protected):
    """Continue matching paint bands across a vertical glyph with a white outer rim."""
    x,y,width,height = cv2.boundingRect(np.uint8(ink))
    if height < 1.6*width or width>96 or height>512 or ink.sum()<60:
        return None
    reach = cv2.dilate(np.uint8(ink), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))) != 0
    reach &= ~protected
    repair = crop.copy()
    supported = np.zeros_like(ink)
    for row in range(max(0,y-3),min(len(ink),y+height+3)):
        cols = np.flatnonzero(reach[row])
        for run in np.split(cols,np.flatnonzero(np.diff(cols)>1)+1):
            if not len(run):
                continue
            l,r = run[0]-1,run[-1]+1
            if l<2 or r+2>=crop.shape[1] or r-l>64 or protected[row,l-2:r+3].any():
                continue
            left,right = (np.median(samples,axis=0) for samples in (crop[row,l-2:l+1],crop[row,r:r+3]))
            if min(left.min(),right.min())<110 or np.abs(left-right).max()>26:
                continue
            weight = ((run-l)/(r-l))[:,None]
            repair[row,run] = np.rint(left*(1-weight)+right*weight)
            supported[row,run] = True
    edge = ink & ~p.erode(ink,2) & supported
    difference = crop.astype(np.int16)-repair
    bright = edge & (crop.min(2)>215) & (difference.mean(2)>8)
    if bright.sum()<24 or bright.sum()<.12*edge.sum() or (ink & supported).sum()<.88*ink.sum():
        return None
    # Antialiasing along this rim also picks up some of the colored face.
    # Keep high-contrast lines and untouched source pixels outside the glyph.
    fringe = reach & ~ink & supported & (difference.min(2)>-32) & (np.abs(difference).max(2)>3)
    commit = ink | fringe
    missing = ink & ~supported
    if missing.any():
        repair[missing] = cv2.inpaint(crop,np.uint8(reach),3,cv2.INPAINT_NS)[missing]
    return commit, repair


def inference_context(commit,probability):
    """Hide nearby lettering from reconstruction without authorizing its erasure.

    Protected pixels may be hidden from the model. Composition still uses only
    the independently bound commit mask, so their output pixels remain exact.
    Inputs are already cropped in native coordinates; no resize or layout input.
    """
    n.require(commit.ndim==2 and commit.dtype==bool and probability.shape==commit.shape
        and probability.dtype==np.uint8,'inference context needs native masks')
    inference=commit | p.dilate(probability>=32,4)
    return inference,{'policy':CONTEXT_POLICY,'inference_only_pixels':int((inference&~commit).sum()),
        'commit_support_expanded':False,'source_removal_verified':False}


def adjoining_solid_ink(crop, own, support, probability):
    """Protect the low-confidence edge of a solid illustration beside open dialogue."""
    if support.sum()<1500 or not own.any() or (crop.min(2)[own]>240).mean()<.55:
        return None
    gray = crop.max(2)
    dark = gray<125
    if (dark & ~own).sum()<400:
        return None
    distance = cv2.distanceTransform(dark.astype(np.uint8),cv2.DIST_L2,5)
    _, components = cv2.connectedComponents(np.uint8(distance>3.5))
    outside = ~cv2.dilate(own.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
    seeds = np.unique(components[outside & (distance>6)])
    seeds = seeds[seeds!=0]
    if not len(seeds):
        return None
    # Disconnected dark glyphs and thin text that touches the hair at a tip
    # cannot propagate this seed. Keep the source mask where the model is confident.
    near = cv2.dilate(np.uint8(np.isin(components,seeds)),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))!=0
    guard = near & support & (gray<245) & (probability<105)
    if guard.sum()>=40 and probability[guard].mean()<55:
        return guard
