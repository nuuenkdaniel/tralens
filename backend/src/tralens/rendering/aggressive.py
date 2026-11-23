"""Crop-local removal after precise routes, before the independent visibility pass.

Art contact is a soft risk. Foreign bindings, overwritten committed pixels and
target text loss are hard errors. Geometry masks are never called semantic masks.
"""
from copy import deepcopy
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw

from . import natural as n, preparation as p, typography
from . import reconstruction as rec, source_support, source_extents


POLICY='crop_local_aggressive_v10_corroborated_extents'
PROTECTION_POLICY='exact_commits_with_separate_layout_clearance_v1'
MAX_REGIONS=64
MAX_LOCAL_WORK=16_777_216
HARD_REASONS={'runtime_locality','alpha_containment','erasure_obstacle','typography_containment',
    'evaluation_cap','replacement_unchanged_text','uncorroborated_source'}


def settings(removal,backend,config):
    n.require(removal in ('legacy','preserve','aggressive'),'unknown removal policy')
    n.require(backend in ('auto','classical','lama','sdxl'),'unknown reconstruction policy')
    n.require(backend=='classical' or (removal=='aggressive' and (config is not None or backend=='auto')),
        'Learned reconstruction requires aggressive removal and explicit approved local reconstruction configuration')
    n.require(config is None or backend!='classical','reconstruction configuration supplied while backend disabled')


class Geometry:
    """Cache bounds and a spatial grid; rasterize only intersecting local polygons."""
    def __init__(self,raw,size,pixels=None):
        self.rules = None
        if pixels is not None:
            ink = np.uint8(np.max(pixels, axis=2) < 50)
            # Long uninterrupted page rules survive text overlap and separate adjacent panels.
            self.rules = (cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(96,size[1]//8),1),np.uint8)),
                          cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1,max(96,size[0]//8)),np.uint8)))
        self.size=size;self.raw={};self.boxes={};self.grid={}
        for d in raw:
            key=(type(d['source_id']).__name__,d['source_id'])
            rec.require(key not in self.raw,'duplicate raw source ID')
            points=np.asarray(d['polygon']);rec.require(points.ndim==2 and points.shape[1]==2 and np.isfinite(points).all(),
                'invalid raw source polygon')
            x,y=map(int,np.floor(points.min(axis=0)));r,b=map(int,np.ceil(points.max(axis=0))+1)
            rec.require(0<=x<r<=size[0]+1 and 0<=y<b<=size[1]+1,'source polygon outside page')
            self.raw[key]=d;self.boxes[key]=(x,y,min(size[0],r),min(size[1],b))
            for tx in range(x//128,(r-1)//128+1):
                for ty in range(y//128,(b-1)//128+1):self.grid.setdefault((tx,ty),set()).add(key)

    def local(self,member_ids,reserved,margin=48,maximum_side=512,source_priority=False,cleanup_polygon=None,cleaned_for_layout=()):
        keys={(type(i).__name__,i) for i in member_ids}
        rec.require(keys and keys<=self.raw.keys() and len(keys)==len(member_ids),'foreign/duplicate recovery membership')
        boxes=[self.boxes[k] for k in keys]
        if cleanup_polygon is not None:
            # Extra context follows a separately bound detector hypothesis. Raw
            # ownership and all placement geometry below remain the originals.
            points=np.asarray(cleanup_polygon)
            rec.require(points.ndim==2 and points.shape[1]==2 and 3<=len(points)<=128 and
                np.isfinite(points).all() and (points>=0).all() and
                (points<=np.asarray(self.size)).all(),'invalid cleanup extent polygon')
            left,top=map(int,np.floor(points.min(axis=0)));right,bottom=map(int,np.ceil(points.max(axis=0))+1)
            boxes.append((left,top,min(self.size[0],right),min(self.size[1],bottom)))
        x=max(0,min(b[0] for b in boxes)-margin);y=max(0,min(b[1] for b in boxes)-margin)
        r=min(self.size[0],max(b[2] for b in boxes)+margin);b=min(self.size[1],max(b[3] for b in boxes)+margin)
        if r-x>maximum_side:
            left,right=min(v[0] for v in boxes),max(v[2] for v in boxes)
            n.budget_check(right-left<=maximum_side,'native source extent exceeds context scope')
            x=max(0,min(left-(maximum_side-(right-left))//2,self.size[0]-maximum_side));r=x+maximum_side
        if b-y>maximum_side:
            top,bottom=min(v[1] for v in boxes),max(v[3] for v in boxes)
            n.budget_check(bottom-top<=maximum_side,'native source extent exceeds context scope')
            y=max(0,min(top-(maximum_side-(bottom-top))//2,self.size[1]-maximum_side));b=y+maximum_side
        n.budget_check(max(r-x,b-y)<=maximum_side,'native reconstruction crop exceeds requested context scope')
        nearby=set()
        for tx in range(x//128,(r-1)//128+1):
            for ty in range(y//128,(b-1)//128+1):nearby.update(self.grid.get((tx,ty),()))
        own=Image.new('L',(r-x,b-y));other=own.copy()
        for key in nearby:
            if key in cleaned_for_layout and key not in keys:
                continue
            bx,by,br,bb=self.boxes[key]
            if br<=x or bb<=y or bx>=r or by>=b:continue
            target=own if key in keys else other
            ImageDraw.Draw(target).polygon([(px-x,py-y) for px,py in self.raw[key]['polygon']],fill=255)
        protected=p.dilate(np.asarray(other)!=0,2)
        if source_priority:
            # An oversized neighboring detection is not authority to leave the
            # current source's ink intact. Accepted edits always remain protected.
            protected &= ~(np.asarray(own)!=0)
        if self.rules is not None:
            yy,xx = np.nonzero(np.asarray(own))
            for axis,center,cross in ((0,int(np.median(xx)),int(np.median(yy))),
                                      (1,int(np.median(yy)),int(np.median(xx)))):
                rule = self.rules[axis][y:b,x:r]
                section = rule if axis == 0 else rule.T
                # A panel rule has thickness and runs past the source centerline.
                stripe = section[max(0,cross-8):cross+9].all(axis=0)
                across = yy if axis == 0 else xx
                # The rule must continue beyond the lettering at both ends.
                a,bottom = int(across.min())-4,int(across.max())+4
                if a < 0 or bottom >= len(section):
                    continue
                stripe &= section[a].astype(bool) & section[bottom].astype(bool)
                edges = np.flatnonzero(stripe & (np.roll(stripe,1) | np.roll(stripe,-1)))
                lower, upper = edges[edges < center],edges[edges > center]
                barrier = protected if axis == 0 else protected.T
                if lower.size:
                    barrier[:,:lower[-1]+2] = True
                if upper.size:
                    barrier[:,max(0,upper[0]-1):] = True
            if any(self.raw[key].get('reason') == 'corroborated_blank_sound' for key in keys):
                # Oversized handwritten glyph boxes can terminate on a long panel rule.
                vertical,horizontal = self.rules
                protected |= p.dilate((vertical[y:b,x:r] | horizontal[y:b,x:r]).astype(bool),2)
        protected |= reserved[y:b,x:r]
        return [x,y,r,b],np.asarray(own)!=0,protected


def hollow_strokes(crop, own, protected):
    """Recover the stroke interiors used by the two-view kana recognizer."""
    gray = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    contours,tree = cv2.findContours(np.uint8(gray<150),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if tree is None:
        return
    out = np.zeros_like(gray)
    for contour,parent in zip(contours,tree[0,:,3]):
        x,y,w,h = cv2.boundingRect(contour)
        if parent >= 0 and 4 <= min(w,h) and max(w,h)<64 and own[y:y+h,x:x+w].all():
            cv2.drawContours(out,[contour],-1,1,-1)
    if out.any():
        return cv2.dilate(out,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))).astype(bool) & ~protected


def adjacent_ellipsis(crop, own, protected):
    """Recover isolated three-dot runs just outside a horizontal detector line."""
    added = np.zeros_like(own)
    x, y, w, h = cv2.boundingRect(own.astype(np.uint8))
    if w < 2*h or h < 8 or np.mean(crop[own].min(axis=1) >= 244) < .55:
        return added
    for start, end in ((max(0,x-round(1.5*h)), x), (x+w, min(crop.shape[1],x+w+round(1.5*h)))):
        if end-start < 5:
            continue
        strip = crop[y:y+h,start:end]
        if protected[y:y+h,start:end].any():
            continue
        dark = strip.max(axis=2) < 180
        count, labels, stats, centers = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
        if count != 4:
            continue
        boxes = stats[1:]
        if (boxes[:,2:4].min() < 2 or boxes[:,2:4].max() > max(3,h*.24)
                or boxes[:,4].max() > boxes[:,4].min()*1.7):
            continue
        points = centers[1:]; gaps = np.diff(np.sort(points[:,0]))
        if np.ptp(points[:,1]) > 1.5 or gaps.min() < 3 or gaps.max() > min(.5*h,1.25*gaps.min()):
            continue
        mask = p.dilate(labels != 0, 2)
        collar = p.dilate(mask,2) & ~mask
        if (dark[0].any() or dark[-1].any() or dark[:,0].any() or dark[:,-1].any()
                or not collar.any() or np.mean(strip[collar].min(axis=1) >= 244) < .98):
            continue
        added[y:y+h,start:end] |= mask
    return added


def masks(crop,own,protected,*,probability=None):
    """Return distinct evidence and removal candidates; uncertain cuts are allowed."""
    # Multichannel local contrast captures light outlines and colored lettering;
    # it is only image evidence, NOT a learned semantic prediction.
    smooth=cv2.GaussianBlur(crop,(5,5),0)
    contrast=np.max(np.abs(crop.astype(np.int16)-smooth.astype(np.int16)),axis=2)>12
    contrast &= p.dilate(own,4)
    candidates=[]
    if probability is not None:
        rec.require(probability.shape==own.shape and probability.dtype==np.uint8,'local learned mask shape/type')
        try:
            support,details=source_support.derive(crop,own,protected,probability)
            # Artwork contrast elsewhere in an oversized detector polygon is not
            # evidence that a narrower lettering mask failed to remove its text.
            contrast &= support
            return contrast,[(source_support.POLICY,support),
                (source_support.POLICY+'_expanded',p.dilate(support,2)&~protected)],details
        except ValueError:
            pass
    candidates.extend([('padded_source_polygons',p.dilate(own,1)&~protected),
        ('expanded_source_polygons',p.dilate(own,5)&~protected)])
    return contrast,candidates,{'predicted_lettering':'learned_probability' if probability is not None else 'unavailable',
        'native_evidence':'multichannel_local_contrast_only','native_component_route':'attempted separately before recovery',
        'omitted_ladder_rungs':[] if probability is not None else ['learned_glyph_support','learned_support_plus_geometry'],
        'foreground_is_text_certified':False,'possible_collateral_art_loss':True,
        'bounded_cut_across_art_contact':True,'source_pixels_protected':int((own&protected).sum())}


def assess(original,candidate,mask,evidence,protected):
    """Residual/seam/edge observations, not a learned quality or truth score."""
    delta=np.max(np.abs(candidate.astype(np.int16)-original.astype(np.int16)),axis=2)
    remaining=evidence & ~protected & (delta<=3)
    outside=p.dilate(mask,3)&~mask&~protected
    edge=cv2.Canny(cv2.cvtColor(candidate,cv2.COLOR_RGB2GRAY),60,150)>0
    before_edge=cv2.Canny(cv2.cvtColor(original,cv2.COLOR_RGB2GRAY),60,150)>0
    count,_,stats,_=cv2.connectedComponentsWithStats((edge&mask).astype(np.uint8),8)
    text_like=sum(3<=int(s[4])<=180 and 2<=int(s[2])<=48 and 2<=int(s[3])<=48 for s in stats[1:])
    ring=mask & p.dilate(outside,1)
    jump=float(np.abs(cv2.Laplacian(candidate.astype(np.float32),cv2.CV_32F)).mean(axis=2)[ring].mean()) if ring.any() else 0
    return {'unchanged_source_contrast_pixels':int(remaining.sum()),'evidence_pixels':int((evidence&~protected).sum()),
        'outer_residual_pixels':int((remaining&~mask).sum()),'seam_laplacian_mean':jump,
        'boundary_edge_count_before':int((before_edge&outside).sum()),'boundary_edge_count_after':int((edge&outside).sum()),
        'new_text_like_components':int(text_like),'text_like_components_are_semantic_ocr':False,
        'ocr_silence_used':False,'quality_verified':False}


def cheap(crop,mask,protected):
    ring=p.dilate(mask,4)&~p.dilate(mask,1)&~protected
    samples=crop[ring]
    if len(samples)>=32:
        color=np.median(samples,axis=0).astype(np.uint8)
        if np.max(np.abs(samples.astype(np.int16)-color))<=3:
            candidate=crop.copy();candidate[mask]=color
            return candidate,'measured_uniform_fill'
        # A few clipped glyph edges used to send otherwise flat balloons through
        # inpainting, which can copy those very edges back into the cleanplate.
        # Require a dominant neutral collar AND matching interior background.
        # This is an explicitly approximate fill, not the exact-collar route.
        difference=np.max(np.abs(samples.astype(np.int16)-color),axis=1)
        interior=np.max(np.abs(crop[mask].astype(np.int16)-color),axis=1)
        if (int(color.max())-int(color.min())<=12 and (int(color.min())>=210 or int(color.max())<=35)
                and len(interior) and ((np.mean(difference<=3)>=.85 and np.mean(interior<=5)>=.45)
                    or (np.mean(difference<=3)>=.55 and np.mean(interior<=5)>=.7))):
            candidate=crop.copy();candidate[mask]=color
            return candidate,'dominant_surface_fill'
    candidate=cv2.inpaint(crop,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
    return candidate,'classical_telea_unverified'


def light_dialogue_frame(crop, own, allowed, *, enclosed=False):
    """Estimate local placement space for dark print on a white balloon."""
    light = crop.min(axis=2) >= 244
    if np.mean(light[own]) < .55:
        return None
    # A long straight rule near the source suggests a form or panel. Its white
    # field is a poor guide to dialogue placement.
    line_width = max(48, round(crop.shape[1]*.5))
    rules = cv2.morphologyEx((crop.max(axis=2)<120).astype(np.uint8), cv2.MORPH_OPEN,
        np.ones((1,line_width),np.uint8))
    near_rule = (rules.astype(bool) & p.dilate(own,24)).any()
    if near_rule and (not enclosed or (rules.astype(bool) & own).any()):
        return None
    # Bridge the original letters only inside their detector footprints.
    # Choose the connected surface covering most of this passage.
    _, labels = cv2.connectedComponents((light | own).astype(np.uint8), connectivity=4)
    ids, counts = np.unique(labels[own], return_counts=True)
    largest = counts.argmax()
    if counts[largest] < .9 * counts.sum():
        return None
    surface = labels == ids[largest]
    if enclosed and (surface[0].any() or surface[-1].any() or surface[:,0].any() or surface[:,-1].any()):
        return None
    if near_rule:
        # A closed round balloon may sit against a panel divider. A rectangular
        # form or panel contains the corners of its own inset bounding box.
        x,y,w,h = cv2.boundingRect(surface.astype(np.uint8))
        dx,dy = w//12,h//12
        if surface[np.ix_([y+dy,y+h-1-dy],[x+dx,x+w-1-dx])].any():
            return None
    if enclosed:
        # Isolated JPEG speckles in the paper can veto a whole line of type.
        # Keep larger holes and reapply neighboring text and reserved pixels.
        _, holes, stats, _ = cv2.connectedComponentsWithStats((~surface).astype(np.uint8))
        speck = stats[:,cv2.CC_STAT_AREA] <= 4
        speck[0] = False
        surface |= speck[holes] & (crop.min(axis=2) >= 232)
    frame = surface & allowed
    return frame if frame.any() else None


def scalloped_balloon(surface):
    contours,_ = cv2.findContours(surface.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours,key=cv2.contourArea)
    if len(contour)<16:
        return False
    hull = cv2.convexHull(contour,returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour,hull)
    except cv2.error:
        return False
    if defects is None:
        return False
    scale = min(cv2.boundingRect(contour)[2:])
    depths = defects[:,0,3]/256
    return np.count_nonzero((depths>.018*scale)&(depths<.09*scale))>=8


def slanted_layout(text, font_data, coverage, own, blocked, style):
    points = cv2.findNonZero(own.astype(np.uint8))
    vx,vy,_,_ = cv2.fitLine(points,cv2.DIST_L2,0,.01,.01).reshape(-1)
    angle = float(np.clip(-np.degrees(np.arctan2(vy,vx)), -20,20))
    thickness = min(cv2.minAreaRect(points)[1])+1
    cap = min(style['max_size'],max(style['min_size'],round(.67*thickness)))
    lettering = {**style,'rotation':angle,'max_size':cap,'fill':[0]*3,'outline':1,'outline_fill':[255]*3}
    region = p.dilate(own,4)
    content = Image.fromarray((region & ~blocked).astype(np.uint8)*255)
    try:
        layer,proof = n.layout(text,font_data,coverage,content,lettering)
    except n.ResourceLimit:
        raise
    except ValueError:
        # Nearby effects can leave a clear slot just off the polygon center.
        # Rasterize once, then test the actual glyphs against protected pixels.
        mask = Image.fromarray(region.astype(np.uint8)*255)
        layer,proof = n.layout(text,font_data,coverage,mask,lettering)
        box = layer.getchannel('A').getbbox()
        tile = layer.crop(box)
        ink = np.asarray(tile.getchannel('A')) != 0
        allowed = p.dilate(own,12) & ~blocked
        offsets = [(dx,dy) for dx in range(-12,13,2) for dy in range(-12,13,2)]
        for dx,dy in sorted(offsets,key=lambda point:point[0]**2 + point[1]**2):
            x,y,r,b = box[0]+dx,box[1]+dy,box[2]+dx,box[3]+dy
            if 0<=x<r<=layer.width and 0<=y<b<=layer.height and allowed[y:b,x:r][ink].all():
                layer = Image.new('RGBA',mask.size)
                layer.paste(tile,(x,y))
                proof['ink_bounds'] = [x,y,r,b]
                proof['placement_adjustment'] = [dx,dy]
                content = Image.fromarray(allowed.astype(np.uint8)*255)
                break
        else:
            raise
    return layer,proof,content


def place_on_clean_paper(layer, proof, plate, allowed, font_data, coverage, style):
    """Fit the entire aside into a contiguous patch of paper on the clean plate."""
    x,y,r,b = layer.getchannel('A').getbbox()
    width,height = r-x,b-y
    dark = p.dilate(np.min(plate,axis=2)<180,2)
    original_center = ((x+r)/2,(y+b)/2)
    radius = min(96,max(width,height))
    left,top = max(0,x-radius),max(0,y-radius)
    right,bottom = min(layer.width,r+radius),min(layer.height,b+radius)
    artwork = dark[top:bottom,left:right].astype('float32')
    obstacles = (~allowed[top:bottom,left:right]).astype('float32')
    options = [(layer,proof)]
    lettering = {**style,'fill':[0]*3,'outline':1,'outline_fill':[255]*3}
    for factor,aspect,taller in ((.85,1.3,1),(.72,1.6,1),(.78,.78,1.5)):
        w,h = round(width*aspect),round(height*taller)
        frame = Image.new('L', layer.size)
        ImageDraw.Draw(frame).rectangle((x,y,x+w-1,y+h-1),fill=255)
        try:
            options.append(n.layout(proof['original_text'],font_data,coverage,frame,
                {**lettering,'max_size':max(style['min_size'],round(proof['size']*factor))}))
        except n.ResourceLimit:
            raise
        except ValueError:
            continue
    best = None
    for candidate,metrics in options:
        box = candidate.getchannel('A').getbbox()
        tile = candidate.crop(box)
        alpha = np.asarray(tile.getchannel('A'))!=0
        if tile.width>artwork.shape[1] or tile.height>artwork.shape[0]:
            continue
        def overlaps(field,template):
            return cv2.matchTemplate(field,template.astype('float32'),cv2.TM_CCORR)
        valid = overlaps(obstacles,alpha)<.5
        if not valid.any():
            continue
        yy,xx = np.indices(valid.shape)
        displacement = ((xx+left+tile.width/2-original_center[0])**2+
                        (yy+top+tile.height/2-original_center[1])**2)/radius**2
        cost = (4*overlaps(artwork,alpha)/alpha.sum()
                + overlaps(artwork,np.ones_like(alpha))/alpha.size
                + .04*displacement + .12*(1-metrics['size']/proof['size']))
        cost[~valid] = np.inf
        j,i = np.unravel_index(cost.argmin(),cost.shape)
        px,py = int(i)+left,int(j)+top
        if (best is None or cost[j,i]<best[0]) and allowed[py:py+tile.height,px:px+tile.width][alpha].all():
            best = cost[j,i],px,py,tile,box,metrics
    if best is None:
        return layer,proof
    _,px,py,tile,box,metrics = best
    shifted = Image.new('RGBA',layer.size)
    shifted.paste(tile,(px,py))
    metrics['ink_bounds'] = [px,py,px+tile.width,py+tile.height]
    if metrics.get('origin') is not None:
        metrics['origin'] = [metrics['origin'][0]+px-box[0],metrics['origin'][1]+py-box[1]]
    metrics['placement_frame'] = 'clean_paper'
    return shifted,metrics


def tall_layout(layer,proof):
    """Only retry automatic wrapping, never an explicitly supplied line layout."""
    if '\n' in proof['original_text'] or len(proof['draw_text'].splitlines())<4:return False
    x,y,r,b=layer.getchannel('A').getbbox()
    return b-y>1.35*(r-x)


def recover(rgb,source,raw,rows,clean,text,erase,layout,font_data,coverage,style,*,backend=None,budget=None,probability=None,extents=None,font_resolver=None):
    """Independent bounded pass; native successes are retained, then visibility runs.

    Every region gets one cheap attempt before any retry. Learned quota is separate
    and never consumes baseline/display budget. Per-region work is local, not page².
    """
    started=time.monotonic();budget=budget or rec.Budget();geometry=Geometry(raw,rgb.size,np.asarray(rgb))
    original=np.asarray(rgb);plate=np.asarray(clean).copy();new_text=text.copy()
    if probability is not None:
        rec.require(type(probability) is np.ndarray and probability.dtype==np.uint8
            and probability.shape==original.shape[:2],'native page text probability shape/type mismatch')
    erase=erase.copy();layout=layout.copy();rows=deepcopy(rows)
    alpha=np.asarray(text.getchannel('A'))!=0
    # An unedited clearance pixel can still contain the next source's lettering.
    # Cleanup protects exact commits; English placement keeps its wider spacing.
    reserved=erase|alpha;clearance=p.dilate(reserved,2)
    initial_reserved=reserved.copy();native_plate=plate.copy();native_text=np.asarray(text).copy()
    summary={'policy':POLICY,'protection_policy':PROTECTION_POLICY,'attempted':0,'learned_calls':0,'learned_failures':0,'retries':0,'local_work_pixels':0,
        'budget_exclusions':0,'selected_learned':0,'selected_classical':0,'selected_fill':0,
        'model_config':getattr(backend,'config',None),'limits':{'regions':MAX_REGIONS,'learned_calls':budget.maximum_calls,
        'learned_seconds':budget.maximum_seconds,'local_work_pixels':MAX_LOCAL_WORK},'artwork_recovered':False}
    artifacts={};jobs=[];backend_errors={};extents=extents or {}
    members={(type(m).__name__,m) for row in rows if row['status']=='automatic_composed'
        for m in row.get('binding',{}).get('member_ids',[])}
    for i,row in enumerate(rows):
        if row['status']=='automatic_composed' or 'binding' not in row or row.get('reason_code') in HARD_REASONS:continue
        entry={'status':'unresolved','prior_reason':row.get('reason_code'),'attempts':[]};row['reconstruction']=entry
        membership={(type(m).__name__,m) for m in row['binding']['member_ids']}
        if membership & members:
            entry['reason']='membership_already_displayed';continue
        if len(jobs)>=MAX_REGIONS:
            entry['reason']='region_budget';summary['budget_exclusions']+=1;continue
        try:
            extent=extents.get(row['unit_id'])
            bounds,own,protected=geometry.local(row['binding']['member_ids'],reserved,source_priority=True,margin=80,
                cleanup_polygon=extent['polygon'] if extent else None)
            x,y,r,b=bounds;crop=original[y:b,x:r].copy()
            placement_protected=protected|clearance[y:b,x:r]
            rec.require(crop.size and own.any(),'empty recovery source')
            predicted,ladder,evidence=masks(crop,own,protected,
                probability=probability[y:b,x:r] if probability is not None else None)
            handwriting = None
            ids = row['binding']['member_ids']
            if len(ids)==1:
                from ..ocr.handwritten import stroke_masks
                handwriting = stroke_masks(original,bounds,geometry.raw[(type(ids[0]).__name__,ids[0])],own,protected)
            if handwriting is not None:
                core,halo = handwriting
                predicted = core
                target = halo if backend is not None else core
                ladder = [('outlined_handwriting',target)]*2
                evidence = {'policy':'outlined_handwriting','source_removal_verified':False}
            outlined = None
            continuations = None
            if len(ids)==1 and geometry.raw[(type(ids[0]).__name__,ids[0])].get('reason')=='corroborated_outline_reading':
                outlined = hollow_strokes(crop,own,protected)
                if outlined is not None:
                    from . import printed_contours
                    continuations = printed_contours.vertical_continuations(crop,outlined,protected)
                    outlined |= continuations[0]
                    predicted = outlined
                    ladder = [('corroborated_hollow_strokes',outlined)]*2
            dots = np.zeros_like(own)
            for member in row['binding']['member_ids']:
                points = np.asarray(geometry.raw[(type(member).__name__,member)]['polygon'])
                line = Image.new('L',(r-x,b-y))
                ImageDraw.Draw(line).polygon([(px-x,py-y) for px,py in points],fill=255)
                line = np.asarray(line) != 0
                dots |= adjacent_ellipsis(crop,line,protected | (own & ~line))
            if dots.any() and handwriting is None:
                ladder = [(name+'+ellipsis',mask | dots) for name,mask in ladder]
                evidence = {**evidence, 'adjacent_ellipsis_pixels':int(dots.sum())}
            if extent and handwriting is None:
                ladder=[(name+'+'+source_extents.POLICY,source_extents.apply(mask,extent,bounds,protected))
                    for name,mask in ladder]
                evidence={**evidence,'cleanup_extent':extent}
            framed = None
            if (backend is not None and backend.config.get('backend') == 'auto' and len(ids)==1
                    and geometry.raw[(type(ids[0]).__name__,ids[0])].get('reason')=='frontier_discovery'):
                from .printed_contours import framed_outline_mask
                framed = framed_outline_mask(crop,own,protected)
                if framed is not None:
                    predicted = framed
                    ladder = [('framed_outlined_ink',framed)]*2
            # Presentation space is independent of the larger inference context.
            # A failed layout still reaches the final removal-only transaction.
            allowed=p.dilate(own,24)&~placement_protected
            light_dialogue_enclosed = False
            if len(row['binding']['text'].split()) >= 5:
                interior = light_dialogue_frame(crop,own,~placement_protected,enclosed=True)
                if interior is not None:
                    allowed = interior
                    light_dialogue_enclosed = True
                    if scalloped_balloon(interior):
                        row['lettering_role'] = 'thought'
            layer=proof=content=None;retry=False
            dark=np.median(crop[own])<80
            try:
                if font_resolver is not None:font_data,coverage=font_resolver(row)
                n.validate_text(row['binding']['text'],coverage)
                if evidence.get('painted_stroke_support') and not style.get('rotation'):
                    # Keep neighboring diagonal effects in their original reading order.
                    try:
                        layer,proof,content = slanted_layout(row['binding']['text'],font_data,coverage,
                            own,placement_protected,style)
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        entry['placement_frame'] = 'original_slanted_lettering'
                if layer is None:
                    layer,proof,content=typography.layout_dialogue(row['binding']['text'],font_data,coverage,
                        Image.fromarray(allowed.astype(np.uint8)*255),{**style,'fill':[255 if dark else 0]*3,
                            'outline':1,'outline_fill':[0 if dark else 255]*3,
                            **({'preset':'thought','outline':0,'leading':.12,'fill':[48,48,52]}
                               if row.get('lettering_role')=='thought' else {}),**row.get('effect_style',{})})
                rec.require(proof['original_text']==row['binding']['text'] and proof['size']>=style['min_size'],
                    'reconstruction layout dropped wording or reduced requested floor')
                frame = light_dialogue_frame(crop, own, allowed)
                compact_effect = False
                if not row.get('effect_style') and not dark and np.mean(crop[own].min(axis=1)>=244) > .55:
                    # Connect through source ink only when its whole contour is
                    # local. A balloon or panel border must separate paper fields.
                    paper = crop.min(axis=2) >= 235
                    _, ink_labels = cv2.connectedComponents((~paper).astype(np.uint8))
                    crossing = np.unique(ink_labels[~p.dilate(own,3)])
                    edge = np.isin(ink_labels,crossing[crossing!=0])
                    if (np.asarray(layer.getchannel('A')).astype(bool) & p.dilate(edge,2)).any():
                        _, fields = cv2.connectedComponents((paper | (own & ~edge)).astype(np.uint8),connectivity=4)
                        votes = np.bincount(fields[own & paper])
                        if len(votes)>1:
                            votes[0] = 0
                            surface = (fields == votes.argmax()) & ~p.dilate(edge,1)
                            frame = (frame if frame is not None else allowed) & surface
                            if len(row['binding']['text'].split()) == 1:
                                # A tapered gap can be much wider at one end.
                                # Center a short effect in the gap at its source height.
                                center_y = round(np.median(np.nonzero(own)[0]))
                                band = abs(np.arange(len(frame)) - center_y) <= style['min_size']
                                local_frame = frame & band[:, None]
                                if local_frame.any():
                                    frame = local_frame
                                    compact_effect = True
                if not row.get('effect_style') and frame is not None and (np.asarray(layer.getchannel('A')).astype(bool) & ~frame).any():
                    try:
                        mask = Image.fromarray(frame.astype(np.uint8)*255)
                        lettering = {**style,'fill':[0]*3,'outline':1,'outline_fill':[255]*3}
                        if compact_effect:
                            # The contour already provides clearance around a narrow effect.
                            target, metrics = n.layout(row['binding']['text'],font_data,coverage,mask,lettering)
                            fitted = target, metrics, mask
                        else:
                            fitted = typography.layout_dialogue(row['binding']['text'],font_data,coverage,mask,lettering)
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass  # Keep the already readable target if the balloon is too small.
                    else:
                        layer,proof,content = fitted
                        entry['placement_frame'] = 'local_light_dialogue_surface'
                retry=tall_layout(layer,proof) and 'placement_frame' not in entry
            except rec.BindingError:raise
            except ValueError as error:
                entry['placement_failure']=str(error);retry=True
            if layer is None and entry.get('placement_failure', '').startswith('overflow:'):
                # A closed balloon has a real boundary. Use its full interior
                # before expanding a source-centered rectangle over nearby art.
                frame = light_dialogue_frame(crop, own, ~placement_protected, enclosed=True)
                if frame is not None:
                    try:
                        layer,proof,content = typography.layout_dialogue(row['binding']['text'],
                            font_data,coverage,Image.fromarray(frame.astype(np.uint8)*255),
                            {**style,'fill':[0]*3,'outline':0,'leading':min(.1,style.get('leading',.2))})
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        entry['placement_retry'] = 'closed_light_balloon'
                        retry = False
            if retry:
                try:
                    # A narrow Japanese column can require a wider English block.
                    # Find that readable local placement before deciding whether
                    # reconstruction would be wasted on an undisplayable target.
                    n.require(evidence.get('policy')==source_support.POLICY or extent is not None,
                        'source-centered reconstruction retry needs source-bound stroke support')
                    from .replacement import patch_layout
                    if font_resolver is not None:font_data,coverage=font_resolver(row)
                    n.validate_text(row['binding']['text'],coverage)
                    sx,sy,sw,sh=cv2.boundingRect(own.astype(np.uint8))
                    dark=np.median(crop[own])<80
                    tile,xy,alternative=patch_layout(row['binding']['text'],font_data,coverage,
                        (sx,sy,sx+sw,sy+sh),placement_protected,{**style,'fill':[255 if dark else 0]*3,
                            'outline_fill':[0 if dark else 255]*3},outline=1)
                    trial=Image.new('RGBA',(crop.shape[1],crop.shape[0]));trial.alpha_composite(tile,dest=xy)
                    if layer is not None:
                        before=layer.getchannel('A').getbbox();after=trial.getchannel('A').getbbox()
                        cx,cy=sx+sw/2,sy+sh/2
                        distance=lambda box:np.hypot((box[0]+box[2])/2-cx,(box[1]+box[3])/2-cy)
                        n.require(after[3]-after[1]<before[3]-before[1] and alternative['size']>=proof['size']
                            and len(alternative['draw_text'].splitlines())<len(proof['draw_text'].splitlines())
                            and distance(after)<=max(distance(before),style['max_size']),
                            'compact retry did not improve wrapping at the source')
                        entry['prior_placement']=proof
                    rec.require(alternative['original_text']==row['binding']['text'] and alternative['size']>=style['min_size'],
                        'reconstruction retry dropped wording or reduced requested floor')
                    entry['placement_retry']='compact_source_centered_rectangle' if layer is not None else 'source_centered_local_rectangle'
                    layer=trial;proof=alternative;content=None
                except rec.BindingError:raise
                except ValueError as failure:
                    entry['placement_retry_failure']=str(failure)
            entry.update(bounds=bounds,layout=proof,mask_evidence=evidence)
            artifacts[f'recovery-{i:03d}-original.png']=Image.fromarray(crop)
            artifacts[f'recovery-{i:03d}-lettering-evidence.png']=Image.fromarray(predicted.astype(np.uint8)*255)
            if content is not None:artifacts[f'recovery-{i:03d}-layout.png']=content
            jobs.append({'i':i,'bounds':bounds,'crop':crop,'protected':protected,'predicted':predicted,
                'ladder':ladder,'layer':layer,'proof':proof,'best':None,'retry':False,
                'outlined':outlined is not None,'continuations':continuations,
                'source_box':cv2.boundingRect(own.astype(np.uint8)),
                'handwriting_core':handwriting[0] if handwriting is not None else None,
                'handwriting_layout':(own & ~placement_protected if handwriting is not None else
                    p.dilate(own,10) & ~placement_protected if not light_dialogue_enclosed
                    and outlined is None and not dark and np.mean(crop[own].min(1)>238)>.60
                    and len(row['binding']['text'].split())>=2 else None),
                'probability':probability[y:b,x:r] if framed is None and handwriting is None and probability is not None and
                    (evidence.get('policy')==source_support.POLICY or extent is not None) else None})
        except rec.BindingError:raise
        except (ValueError,OSError,cv2.error) as error:entry.update(reason=str(error),status='layout_or_crop_unavailable')
    for round_index in range(2):
        for job in jobs:
            if round_index and not job['retry']:continue
            i=job['i'];row=rows[i];entry=row['reconstruction'];crop=job['crop'];protected=job['protected']
            if summary['local_work_pixels']+crop.shape[0]*crop.shape[1]>MAX_LOCAL_WORK:
                entry['reason']='local_work_budget';summary['budget_exclusions']+=1;continue
            summary['local_work_pixels']+=crop.shape[0]*crop.shape[1]
            if round_index:summary['retries']+=1
            else:summary['attempted']+=1
            name,mask=job['ladder'][round_index]
            if not mask.any():entry['reason']='protected_source_overlap';continue
            inference=mask;context=None
            if job['probability'] is not None:
                inference,context=source_support.inference_context(mask,job['probability'])
            candidate,method = cheap(crop,mask,protected)
            if job['outlined'] and method=='classical_telea_unverified':
                # Shift-map copies intact texture across hollow glyphs while
                # keeping the original art between adjacent letters.
                shifted = np.empty_like(crop)
                cv2.xphoto.inpaint(crop,np.uint8(~mask)*255,shifted,cv2.xphoto.INPAINT_SHIFTMAP)
                candidate[mask] = shifted[mask]
                from . import printed_contours
                printed_contours.refine(original,job['bounds'],mask,protected,candidate)
                method = 'classical_shiftmap'
            if job['handwriting_core'] is not None:
                candidate = crop.copy()
                candidate[job['handwriting_core']] = 255
                method = 'outlined_handwriting_fill'
            if ((backend is None or backend.config.get('backend')=='auto') and method=='classical_telea_unverified'
                    and entry['mask_evidence'].get('painted_stroke_support')):
                painted = source_support.painted_stroke_fill(crop,mask,protected,slanted=True)
                if painted is not None:
                    candidate,method = painted,'local_painted_strokes'
            selected = None if method=='local_painted_strokes' else rec.region_backend(backend,crop,inference)
            req=rec.request(source,row['binding'],job['bounds'],crop,inference,mask,protected,attempt=round_index,
                model=rec.model_identity(selected.config) if selected else None)
            prefix=f'recovery-{i:03d}-{round_index}'
            artifacts[prefix+'-request.json']=rec.canonical(req).encode()
            artifacts[prefix+'-inference-mask.png']=Image.fromarray(inference.astype(np.uint8)*255)
            artifacts[prefix+'-commit-mask.png']=Image.fromarray(mask.astype(np.uint8)*255)
            artifacts[prefix+'-protected-mask.png']=Image.fromarray(protected.astype(np.uint8)*255)
            attempt={'mask':name,'request_revision':req['revision'],'method':method,'status':'candidate'}
            if context is not None:attempt['inference_context']=context
            cheap_signals=assess(crop,candidate,mask,job['predicted'],protected)
            defer_learned=round_index==0 and cheap_signals['outer_residual_pixels']>0
            if method not in ('measured_uniform_fill','dominant_surface_fill') and selected is not None:
                if job['layer'] is None:attempt['learned_skipped']='placement_unavailable_final_surface_cleanup'
                elif defer_learned:attempt['learned_skipped']='deferred_until_residual_mask_expansion'
                elif selected.config.get('backend') in backend_errors:
                    attempt['learned_skipped']='backend_circuit_open: '+backend_errors[selected.config.get('backend')]
                elif not budget.available():attempt['learned_skipped']='separate_learned_budget';summary['budget_exclusions']+=1
                else:
                    budget.calls+=1;clock=time.monotonic()
                    try:
                        result=selected.reconstruct(req,crop,inference,mask,protected,
                            timeout=min(req['limits']['seconds'],max(.1,budget.maximum_seconds-budget.seconds)))
                        req,inference,result = rec.review_outline(backend,req,result,crop,inference,mask,protected,
                            budget,clock + budget.maximum_seconds - budget.seconds)
                        if result.get('outline_review', {}).get('selected') == 'lama-onnx-512':
                            artifacts[prefix+'-request.json']=rec.canonical(req).encode()
                            artifacts[prefix+'-inference-mask.png']=Image.fromarray(inference.astype(np.uint8)*255)
                            attempt['request_revision']=req['revision']
                        candidate=rec.compose(req,crop,inference,mask,protected,result)
                        artifacts[prefix+'-backend.png']=Image.fromarray(result['pixels'])
                        method='learned_sdxl' if req['model'].get('backend')=='sdxl-inpaint' else 'learned_lama';attempt['method']=method
                        attempt['backend_result']={k:v for k,v in result.items() if k!='pixels'}
                    except rec.BindingError:raise
                    except Exception as error:
                        backend_error=type(error).__name__+': '+str(error)[:1000]
                        backend_errors[selected.config.get('backend')]=backend_error
                        attempt['learned_failure']=backend_error;summary['learned_failures']+=1
                    finally:budget.seconds+=time.monotonic()-clock
            if method == 'classical_telea_unverified' and inference is not mask:
                # A white outline around Japanese ink can seed a pale ghost on
                # gray paper. Sample beyond the detected lettering before
                # deciding whether nearby pixels are a reliable fill source.
                collar = p.dilate(mask, 8) & ~inference & ~protected
                samples = crop[collar]
                if len(samples) >= 32 and (inference & ~mask).any():
                    distance = np.abs(samples - np.median(samples, axis=0)).max(axis=1)
                    if np.mean(distance <= 8) >= .85:
                        candidate = cv2.inpaint(crop, inference.astype(np.uint8), 3, cv2.INPAINT_TELEA)
                        attempt['classical_inference_context'] = True
            if job['continuations'] is not None:
                strokes,tones = job['continuations']
                candidate[strokes] = tones[strokes,None]
            # Even classical/native output is explicitly composited through commit.
            candidate=rec.compose(req,crop,inference,mask,protected,{'request_revision':req['revision'],'pixels':candidate})
            signals=assess(crop,candidate,mask,job['predicted'],protected)
            attempt['assessment']=signals;entry['attempts'].append(attempt)
            artifacts[prefix+'-cleanplate.png']=Image.fromarray(candidate)
            priority={'measured_uniform_fill':0,'dominant_surface_fill':0,'learned_lama':1,'learned_sdxl':1,'local_painted_strokes':1,'outlined_handwriting_fill':1,'classical_shiftmap':1,'classical_telea_unverified':2}[method]
            # Residual evidence precedes footprint cost. Do not optimize smoothness
            # or claim an aesthetic oracle; a valid learned result precedes cheap
            # classical output on equally complete masks.
            rank=(signals['outer_residual_pixels'],priority,int(mask.sum()))
            if job['best'] is None or rank<job['best'][0]:job['best']=(rank,candidate,mask,method,round_index,signals)
            job['retry']=round_index==0 and signals['outer_residual_pixels']>0
    for job in jobs:
        if job['best'] is None:continue
        i=job['i'];row=rows[i];entry=row['reconstruction']
        _,candidate,mask,method,chosen,signals=job['best'];x,y,r,b=job['bounds'];layer=job['layer']
        if (layer is not None and job['handwriting_layout'] is not None
                and (job['handwriting_core'] is not None or np.any(
                    (np.asarray(layer.getchannel('A'))!=0) & p.dilate(candidate.min(2)<185,2)))):
            if font_resolver is not None:
                font_data,coverage = font_resolver(row)
            layer,job['proof'] = place_on_clean_paper(layer,job['proof'],candidate,
                job['handwriting_layout'] & ~clearance[y:b,x:r],font_data,coverage,style)
            entry['layout'] = job['proof']
        sx,sy,sw,sh = job['source_box']
        if (layer is not None and entry.get('placement_frame') == 'local_light_dialogue_surface'
                and not row.get('effect_style') and not style.get('rotation') and job['handwriting_core'] is None
                and len(row['binding']['member_ids']) >= 2 and sw > 1.5*sh
                and '\n' not in job['proof']['original_text'] and len(job['proof']['draw_text'].splitlines()) >= 5):
            # Fit the English on the restored paper once the Japanese strokes are gone.
            frame = np.zeros_like(mask)
            padx,pady = min(28,style['max_size']), min(10,sh//4)
            frame[max(0,sy-pady):sy+sh+pady,max(0,sx-padx):sx+sw+padx] = True
            frame &= (candidate.min(2)>=244) & ~job['protected'] & ~clearance[y:b,x:r]
            _,labels,stats,_ = cv2.connectedComponentsWithStats(frame.astype(np.uint8),connectivity=4)
            if len(stats)>1:
                frame &= labels == 1 + np.argmax(stats[1:,cv2.CC_STAT_AREA])
            if font_resolver is not None:
                font_data,coverage = font_resolver(row)
            try:
                alternative,metrics,_ = typography.layout_dialogue(job['proof']['original_text'],font_data,coverage,
                    Image.fromarray(frame.astype('uint8')*255),{**style,'fill':[0]*3,'outline':0,'leading':min(.12,style.get('leading',.2))})
            except n.ResourceLimit:
                raise
            except ValueError:
                pass
            else:
                if metrics['size'] > job['proof']['size'] and len(metrics['draw_text'].splitlines()) < len(job['proof']['draw_text'].splitlines()):
                    layer = alternative
                    entry['layout'] = job['proof'] = metrics
                    entry['placement_frame'] = 'restored_dialogue_surface'
        local_alpha=np.asarray(layer.getchannel('A'))!=0 if layer is not None else np.zeros_like(mask)
        membership={(type(m).__name__,m) for m in row['binding']['member_ids']}
        # Reservations may have grown since layout planning. Reject atomically;
        # neither old English nor earlier cleanup can be overwritten by a later job.
        if membership & members:
            entry.update(status='collision_retained_primary',reason='membership_already_displayed');continue
        if method=='classical_telea_unverified' and entry['attempts'][chosen].get('learned_skipped')=='separate_learned_budget':
            # Exhausting this pass must not freeze a broad classical smear before
            # the removal-first pass can use its separate reconstruction budget.
            # That pass binds a fresh crop to the then-current protected support.
            entry.update(status='cleanup_candidate_only',reason='deferred_to_final_reconstruction_budget',
                selected_attempt=chosen,method=method,assessment=signals)
            continue
        if layer is None or (mask&reserved[y:b,x:r]).any() or (local_alpha&clearance[y:b,x:r]).any():
            # Retain candidate evidence. Automatic reuse in final cleanup was
            # rejected after visual trials showed copied/hallucinated lettering.
            entry.update(status='cleanup_candidate_only',reason=entry.get('placement_failure','earlier_output_reserved'),
                selected_attempt=chosen,method=method,assessment=signals)
            continue
        plate[y:b,x:r][mask]=candidate[mask];erase[y:b,x:r]|=mask;layout[y:b,x:r]|=local_alpha
        new_text.alpha_composite(layer,dest=(x,y));alpha[y:b,x:r]|=local_alpha
        occupied=mask|local_alpha
        reserved[y:b,x:r]|=occupied
        clearance[y:b,x:r]|=p.dilate(occupied,2)
        members.update(membership)
        entry.update(status='composed_unverified',selected_attempt=chosen,method=method,assessment=signals,
            source_removal_verified=False,artwork_loss_possible=True)
        row.update(status='automatic_composed',reason_code=None,reason=None,render_quality='best_effort',
            cleanup_method=method,layout={**job['proof'],'crop_origin':[x,y]},
            best_effort={'warnings':['Bounded aggressive removal may damage local artwork; reconstructed pixels are speculative.']})
        artifacts[f'recovery-{i:03d}-text.png']=layer
        summary[{'learned_lama':'selected_learned','learned_sdxl':'selected_learned','classical_telea_unverified':'selected_classical','classical_shiftmap':'selected_classical','local_painted_strokes':'selected_classical',
            'outlined_handwriting_fill':'selected_fill','measured_uniform_fill':'selected_fill','dominant_surface_fill':'selected_fill'}[method]]+=1
    final_clean=Image.fromarray(plate);composite=Image.alpha_composite(final_clean.convert('RGBA'),new_text).convert('RGB')
    rec.require(np.array_equal(plate[initial_reserved],native_plate[initial_reserved]),'recovery overwrote committed cleanup')
    rec.require(np.array_equal(np.asarray(new_text)[initial_reserved],native_text[initial_reserved]),'recovery overwrote accepted English')
    n.verify_pixel_audit(n.pixel_audit(rgb,final_clean,composite,erase,alpha,~(erase|layout)))
    summary.update(learned_calls=budget.calls,learned_seconds=budget.seconds,seconds=time.monotonic()-started)
    return rows,final_clean,new_text,composite,erase,layout,artifacts,summary
