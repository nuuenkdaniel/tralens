"""Crop-local removal after precise routes, before the independent visibility pass.

Art contact is a soft risk. Foreign bindings, overwritten committed pixels and
target text loss are hard errors. Geometry masks are never called semantic masks.
"""
from copy import deepcopy
import re
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
    'evaluation_cap','replacement_unchanged_text','uncorroborated_source','ruled_table_cell'}


def settings(removal,backend,config):
    n.require(removal in ('legacy','preserve','aggressive'),'unknown removal policy')
    n.require(backend in ('auto','classical','lama','sdxl'),'unknown reconstruction policy')
    n.require(backend=='classical' or (removal=='aggressive' and (config is not None or backend=='auto')),
        'Learned reconstruction requires aggressive removal and explicit approved local reconstruction configuration')
    n.require(config is None or backend!='classical','reconstruction configuration supplied while backend disabled')


class Geometry:
    """Cache bounds and a spatial grid; rasterize only intersecting local polygons."""
    def __init__(self,raw,size,pixels=None):
        from ..ocr.types import is_printed_literal
        self.file_listing = len({d.get('text','') for d in raw if '.' in d.get('text','') and is_printed_literal(d['text'])})>=2
        self.rules = None
        self.paper_components = None
        if pixels is not None:
            ink = np.uint8(np.max(pixels, axis=2) < 50)
            self.paper = np.min(pixels, axis=2) >= 245
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

    def nearby(self, x, y, right, bottom):
        neighbors=set()
        for tx in range(max(0,x)//128,(min(self.size[0],right)-1)//128+1):
            for ty in range(max(0,y)//128,(min(self.size[1],bottom)-1)//128+1):
                neighbors.update(self.grid.get((tx,ty),()))
        return neighbors

    def inset_balloon(self, own, bounds, members):
        """Protect a smaller speech balloon drawn over lines of printed text."""
        x,y,r,b=bounds
        guard=np.zeros_like(own)
        for key in self.nearby(x-96,y-96,r+96,b+96)-members:
            l,t,right,base=self.boxes[key]
            if (right-l>80 or base-t>90 or right<x-96 or l>r+96 or base<y-96 or t>b+96):
                continue
            if self.paper_components is None:
                _,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(self.paper),connectivity=4)
                self.paper_components=labels,stats
            labels,stats=self.paper_components
            region=labels[t:base,l:right]
            ids,counts=np.unique(region[region!=0],return_counts=True)
            if not len(ids) or counts.max()<.45*region.size:
                continue
            label=ids[counts.argmax()]
            px,py,w,h,area=stats[label]
            if (not 30<=w<=320 or not 30<=h<=320 or not .45*w*h<=area<=60000
                    or px<=0 or py<=0 or px+w>=self.size[0] or py+h>=self.size[1]):
                continue
            local=labels[y:b,x:r]==label
            if not (local & own).any() or (local & own).sum()>.35*own.sum():
                continue
            # Include the ink boundary and holes such as lettering inside the
            # balloon. The text of the enclosing paragraph lies outside it.
            hull=np.uint8(local)
            contours=cv2.findContours(hull,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(hull,contours,-1,1,-1)
            guard|=p.dilate(hull!=0,4)
        return guard

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
        nearby=self.nearby(x,y,r,b)
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
            # A profile name or caption can share the edge of a much larger
            # glyph's detection. Its printed Latin letters survive the edit.
            for key in nearby-keys:
                neighbor = self.raw[key]
                word=neighbor.get('text','')
                ornament=neighbor.get('reason')=='corroborated_native_line' and word in ('♡','♥','☆','★')
                if not ornament and (neighbor.get('confidence',0)<.9 or
                        not re.fullmatch(r'[A-Za-z][\w.-]{2,}',word,re.ASCII)):
                    continue
                points = np.asarray(neighbor['polygon'],np.int32)-[x,y]
                keep = np.zeros(protected.shape,np.uint8)
                cv2.fillPoly(keep,[points.astype(np.int32)],1)
                protected |= keep.astype(bool)
            if self.rules is not None and len(keys)>1:
                # Small asides and emotion marks can overlap the corner of a
                # longer line's OCR polygon. Keep complete ink components of
                # the smaller detection, even inside that geometric overlap.
                passage = np.asarray(own)!=0
                if (len(keys)>=3 and sum(len(self.raw[k].get('text','')) for k in keys)>=25
                        and max(v[2]-v[0] for v in boxes)>220
                        and all(v[2]-v[0]>2*(v[3]-v[1]) for v in boxes)):
                    protected |= self.inset_balloon(passage,(x,y,r,b),keys)
                for key in nearby-keys-set(cleaned_for_layout):
                    bx,by,br,bb = self.boxes[key]
                    if (max(br-bx,bb-by)>80 or (br-bx)*(bb-by)>.25*passage.sum()
                            or bx<x+2 or by<y+2 or br>r-2 or bb>b-2):
                        continue
                    local = np.s_[by-y-2:bb-y+2,bx-x-2:br-x+2]
                    if not passage[local].any() or passage[local].all():
                        continue
                    ink = ~self.paper[by-2:bb+2,bx-2:br+2]
                    if ink.mean()>.40:
                        continue
                    poly = np.asarray(self.raw[key]['polygon'],np.int32)-[bx-2,by-2]
                    support = np.zeros(ink.shape,np.uint8)
                    cv2.fillPoly(support,[poly.astype('int32')],1)
                    _,labels,stats,_ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
                    contained = (stats[:,-1]>=3) & (stats[:,-1]<=max(12,.15*support.sum()))
                    contained[np.unique(labels[support==0])] = False
                    contained[0] = False
                    protected[local] |= p.dilate(contained[labels],1)
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
                endpoints = section[a].astype(bool) & section[bottom].astype(bool)
                # A detector can start in the gutter above a panel corner.
                # A rule reaching one end still bounds the label if its outside
                # face is blank and the rule spans almost the entire source.
                across_source = xx if axis == 0 else yy
                face = self.paper[y:b,x:r] if axis == 0 else self.paper[y:b,x:r].T
                inset = max(3,min(18,(bottom-a)//9))
                for edge in np.flatnonzero(stripe & ~endpoints &
                        (section[a].astype(bool) | section[bottom].astype(bool))):
                    if edge >= across_source.max()-inset and edge+14 < face.shape[1]:
                        gutter = face[a+inset:bottom-inset+1,edge+6:edge+14]
                    elif edge <= across_source.min()+inset and edge >= 14:
                        gutter = face[a+inset:bottom-inset+1,edge-13:edge-5]
                    else:
                        continue
                    if (gutter.size and gutter.mean()>=.98
                            and section[a+inset:bottom-inset+1,edge].all()):
                        endpoints[edge] = True
                stripe &= endpoints
                wide = stripe & (np.roll(stripe,1) | np.roll(stripe,-1))
                # A downsampled frame can have just one solid black pixel.
                # Accept it when the outside faces a continuous paper gutter.
                paper = self.paper[y:b,x:r]
                face = paper if axis == 0 else paper.T
                along = xx if axis == 0 else yy
                for edge in np.flatnonzero(stripe & ~wide):
                    if edge < along.min() and edge >= 9:
                        gutter = face[a:bottom+1, edge-9:edge-1]
                    elif edge > along.max() and edge+9 < face.shape[1]:
                        gutter = face[a:bottom+1, edge+2:edge+10]
                    else:
                        continue
                    if gutter.mean() >= .97:
                        wide[edge] = True
                edges = np.flatnonzero(wide)
                lower, upper = edges[edges < center],edges[edges > center]
                barrier = protected if axis == 0 else protected.T
                if lower.size:
                    barrier[:,:lower[-1]+2] = True
                if upper.size:
                    barrier[:,max(0,upper[0]-1):] = True
            if any(self.raw[key].get('reason') == 'corroborated_blank_sound'
                   or self.raw[key].get('ocr_metadata',{}).get('empty_detection')
                   or self.raw[key].get('ocr_metadata',{}).get('diagonal_effect')
                   or any(k in self.raw[key].get('ocr_metadata',{}) for k in ('outlined_ellipsis','hollow_print','heavy_print','outlined_display'))
                   or (self.raw[key].get('confidence',1)<.7 and self.raw[key].get('text','').isdigit()
                       and min(self.boxes[key][2]-self.boxes[key][0],self.boxes[key][3]-self.boxes[key][1])>=80)
                   for key in keys):
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


def printed_menu_fill(crop, own, mask):
    """Replace dark menu type with nearby shading from the same screen column."""
    gray = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    paper = gray[own].astype(float)
    tone = np.percentile(paper,70)
    if (tone < 95 or np.mean(np.ptp(crop[own],axis=1)>14) > .05
            or np.mean(np.abs(paper-tone)<27) < .45):
        return None
    band = p.dilate(own,3)
    donors = band & (np.abs(gray.astype(float)-tone)<29)
    height = cv2.boundingRect(own.astype(np.uint8))[3]
    border = cv2.morphologyEx(np.uint8(gray<tone*.44), cv2.MORPH_OPEN,
                              np.ones((1,max(21,round(1.3*height))),np.uint8))
    border = cv2.dilate(border,np.ones((3,1),np.uint8)).astype(bool)
    ink = mask & band & ~border & (gray<tone-26)
    result = crop.copy()
    for col in np.flatnonzero(ink.any(0)):
        samples = np.flatnonzero(donors[:,col])
        if len(samples)<2:
            result[ink[:,col],col] = round(tone)
        else:
            ys = np.flatnonzero(ink[:,col])
            nearest = np.abs(ys[:,None]-samples).argmin(1)
            result[ys,col] = crop[samples[nearest],col]
    return result


def cheap(crop,mask,protected,paper_aside=False):
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
        # For optically joined handwriting beside a figure, the surrounding
        # collar can include a thin black contour and a light gray sleeve.
        flat_paper = (paper_aside and color.min()>=248 and (difference<=3).mean()>=.55
                      and len(interior) and (interior<=5).mean()>=.55
                      and (samples.min(1)>=200).mean()>=.84 and np.quantile(np.ptp(samples,axis=1),.95)<=8)
        if flat_paper or (int(color.max())-int(color.min())<=12 and (int(color.min())>=210 or int(color.max())<=35)
                and len(interior) and ((np.mean(difference<=3)>=.85 and np.mean(interior<=5)>=.45)
                    or (np.mean(difference<=3)>=.55 and np.mean(interior<=5)>=.7))):
            candidate=crop.copy();candidate[mask]=color
            return candidate,'dominant_surface_fill'
    candidate=cv2.inpaint(crop,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
    return candidate,'classical_telea_unverified'


def ruled_caption_band(own, rules, blocked):
    """Find paper inside a shallow label cropped against a horizontal panel rule."""
    x,y,w,h=cv2.boundingRect(own.astype(np.uint8))
    if not 12<=h<=80 or w<3.5*h:
        return
    rows=np.flatnonzero((rules[y:y+h,x:x+w]!=0).mean(1)>.9)
    if not len(rows):
        return
    band=own & ~blocked
    inset=max(3,round(.16*h))
    band[:,:x+inset]=False
    band[:,x+w-inset:]=False
    if rows[0]>h*.78:
        band[:y+round(.22*h)]=False
        band[y+rows[0]-1:]=False
    elif rows[-1]<h*.22:
        band[:y+rows[-1]+2]=False
        band[y+round(.78*h):]=False
    else:
        return
    return Image.fromarray(np.uint8(band)*255) if band.any() else None


def crossing_ink(crop,footprint):
    """Protect contours that enter the text region from adjoining artwork."""
    _,components,stats,_ = cv2.connectedComponentsWithStats((crop.max(axis=2)<135).astype(np.uint8))
    beyond = np.bincount(components[~p.dilate(footprint,3)],minlength=len(stats))
    crossing = (beyond>=18) & (beyond>.52*stats[:,cv2.CC_STAT_AREA])
    crossing[0] = False
    return p.dilate(crossing[components],1)


def split_touching_glyph(crop,guard,glyph,rules):
    """Keep short pieces of an existing contour supported on both sides of a glyph."""
    kept = guard & ~glyph
    ink = crop.max(axis=2)<160
    banks = p.dilate(ink & kept,1).astype(np.uint8)
    chords = np.zeros_like(banks)
    for angle in range(0,180,12):
        radians = np.deg2rad(angle)
        offset = np.rint(8*np.array([np.cos(radians),np.sin(radians)])).astype(int)
        kernel = np.zeros((17,17),np.uint8)
        cv2.line(kernel,tuple(8-offset),tuple(8+offset),1,1)
        chords |= cv2.morphologyEx(banks,cv2.MORPH_CLOSE,kernel)
    # A solid panel rule already has its own protection. A little spur
    # extending out of a continuous rule is not a gap in that rule.
    gaps = p.dilate((chords!=0) & (banks==0),1)
    bridge = guard & glyph & ink & gaps & ~p.dilate(rules,4)
    return kept | (guard & p.dilate(bridge,1))


def placement_obstacles(crop, own):
    """Expand onto connected paper around the text, keeping panel lines and art clear."""
    paper = crop.min(axis=2) >= 240
    nearby = p.dilate(own, 3)
    # On painted panels only the lettering footprint establishes a landing area.
    if paper[own].mean() < .5:
        collar = p.dilate(own, 5) & ~own
        if not collar.any():
            return ~nearby
        color = np.median(crop[collar], axis=0)
        paper = np.max(np.abs(crop.astype(np.int16) - color), axis=2) <= 12
        if paper[collar].mean() < .8:
            return ~nearby
    _, labels = cv2.connectedComponents((paper | own).astype(np.uint8), connectivity=4)
    ids = np.unique(labels[own])
    return ~(nearby | (paper & np.isin(labels, ids)))


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


def painted_balloon_paper(plate, own, protected, size, *, closed=False):
    """Find the white paper underneath this passage on a reconstructed color panel."""
    if not closed and np.mean(np.ptp(plate,axis=2)>24)<.08:
        return None
    white = plate.min(2)>=(230 if closed else 240)
    if np.mean(white[own])<.8:
        return None
    _,labels = cv2.connectedComponents(white.astype(np.uint8),connectivity=4)
    inside,counts = np.unique(labels[own & white],return_counts=True)
    largest = counts.argmax()
    if counts[largest]<.9*own.sum():
        return None
    surface = labels==inside[largest]
    if closed:
        if surface[0].any() or surface[-1].any() or surface[:,0].any() or surface[:,-1].any():
            return None
        # A rectangular caption or a field sharing white space with the panel
        # cannot supply the curved boundary of a small speech balloon.
        x,y,w,h = cv2.boundingRect(surface.astype(np.uint8))
        dx,dy = max(1,w//12),max(1,h//12)
        if surface[np.ix_([y+dy,y+h-1-dy],[x+dx,x+w-1-dx])].sum()>1:
            return None
    # Reserve a type-relative inset even in a balloon clipped at the crop edge.
    distance = cv2.distanceTransform(surface.astype(np.uint8),cv2.DIST_L2,5)
    return (distance >= max(4,round(size*.38))) & ~protected


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


def light_effect_style(crop, own, polygon, source):
    """Sample separated pale strokes along the baseline of a painted sound or end mark."""
    if not re.fullmatch(r'(?:[ぁ-ゖァ-ヺー]{2,12}[!！?？]*|[-–—][ぁ-ゖァ-ヺー]{2,12}[-–—])',source):
        return None
    if np.mean(np.ptp(crop[own],axis=1)>26)<.48:
        return None
    points = np.asarray(polygon)
    left,right = np.array_split(points[np.argsort(points[:,0])],2)
    dx,dy = right.mean(axis=0)-left.mean(axis=0)
    angle = -np.degrees(np.arctan2(dy,dx))
    if dx<20 or abs(angle)>49:
        return None
    _,labels,stats,_ = cv2.connectedComponentsWithStats((crop.min(2)>=234).astype(np.uint8))
    counts = np.bincount(labels[own],minlength=len(stats))
    sizes = stats[:,cv2.CC_STAT_AREA]
    eligible = (counts==sizes) & (sizes>=8) & (sizes<=.14*own.sum())
    eligible[0] = False
    if eligible.sum()<3:
        return None
    ink = eligible[labels]
    if not .045<=ink.sum()/own.sum()<=.4:
        return None
    yy,xx = np.nonzero(ink)
    covariance = np.cov(xx,yy)
    values,axes = np.linalg.eigh(covariance)
    baseline = np.degrees(np.arctan2(axes[1,-1],axes[0,-1]))
    if values[-1]<2.2*max(1,values[0]) or abs((baseline+angle+90)%180-90)>15:
        return None
    # Highlights in a corner can share a slope. Require strokes along most
    # of the printed row before borrowing their color or orientation.
    if np.ptp(xx)<.65*dx:
        return None
    rim = p.dilate(ink,2) & ~ink
    bordered = np.mean(crop[rim].max(axis=1)<65)
    return {'fill':np.median(crop[ink],axis=0).astype(int).tolist(),
            'outline_fill':[24]*3,'outline':2 if bordered>.38 else int(bordered>.12),
            'rotation':float(np.clip(angle,-20,20))}


def printed_effect_color(crop, own, predicted, source, target):
    """Estimate stable saturated ink for a wide, single-word printed effect."""
    if len(target.split()) != 1 or not re.fullmatch(r'[ァ-ヺー]{2,16}[!！?？]{1,5}',source):
        return None
    _,_,width,height = cv2.boundingRect(own.astype(np.uint8))
    samples = crop[predicted & own]
    if width < 3*height or len(samples) < 40:
        return None
    values = samples.mean(axis=1)
    core = samples[values <= np.percentile(values,20)]
    color = np.median(core,axis=0).astype(int)
    if (np.ptp(color) >= 32 and np.median(crop[own].mean(axis=1))-color.mean() >= 65
            and np.median(np.abs(core-color).max(axis=1)) <= 22):
        return color.tolist()
    return None


def caption_band(plate, own, blocked):
    """Find the continuous face of a ribbon or narrow sign around a printed line."""
    _, _, sw, sh = cv2.boundingRect(own.astype(np.uint8))
    if sw < 1.8*sh or sh < 8:
        return None
    lab = cv2.cvtColor(plate, cv2.COLOR_RGB2LAB).astype(np.int16)
    center = np.median(lab[own], axis=0)
    close = np.linalg.norm(lab-center, axis=2) <= 27
    _, labels, stats, _ = cv2.connectedComponentsWithStats(close.astype(np.uint8), connectivity=4)
    counts = np.bincount(labels[own], minlength=len(stats))
    if len(counts) < 2:
        return None
    label = 1 + counts[1:].argmax()
    _, top, w, h, area = stats[label]
    if (counts[label] < .8*own.sum() or top == 0 or top+h == plate.shape[0] or
            w < max(1.5*sw, 3*h) or h > 2*sh or area < .8*w*h):
        return None
    return p.erode(labels == label, max(2, round(h*.08))) & ~blocked


def narrow_strip_layout(text, font_data, coverage, paper, style):
    """Fit a rotated caption inside the clean vertical paper left by its source."""
    # Work horizontally during shaping so breaks follow the long axis. A
    # quarter turn is an exact pixel permutation and preserves every glyph.
    upright = np.ascontiguousarray(np.rot90(paper))
    frame = typography.interior_box(upright)
    lettering = {'outline': 0, 'fill': [0, 0, 0], **style,
                 'min_size': min(13, style['min_size']), 'leading': .08}
    layer, proof = n.layout(text, font_data, coverage, frame, lettering)
    layer = layer.transpose(Image.Transpose.ROTATE_270)
    proof.update(ink_bounds=layer.getchannel('A').getbbox(), origin=None, quarter_turn_clockwise=True)
    return layer, proof


def vertical_effect_layout(text, font_data, coverage, room, own, style):
    """Place a complete quarter-turned word beside a rule or character outline."""
    yy,xx=np.nonzero(own)
    cx,cy=xx.mean(),yy.mean()
    x,y,w,h=cv2.boundingRect(np.uint8(room))
    if min(w,h)<4:
        raise ValueError('empty effect surface')
    template=Image.new('L',(h,w),255)
    obstacles=np.float32(~room[y:y+h,x:x+w])
    best=None
    for cap in (15,13,11,10):
        try:
            layer,proof=n.layout(text,font_data,coverage,template,{**style,'min_size':cap,'max_size':cap},
                                 fixed_breaks=())
        except n.ResourceLimit:
            raise
        except ValueError:
            continue
        tile=layer.crop(layer.getchannel('A').getbbox()).transpose(Image.Transpose.ROTATE_270)
        ink=np.float32(np.asarray(tile.getchannel('A'))!=0)
        overlap=cv2.matchTemplate(obstacles,ink,cv2.TM_CCORR)
        rows,cols=np.nonzero(overlap<.5)
        if not len(rows):
            continue
        distances=np.hypot(x+cols+tile.width/2-cx,y+rows+tile.height/2-cy)
        nearest=distances.argmin()
        cost=distances[nearest]+2*(15-cap)
        if best is None or cost<best[0]:
            best=cost,x+int(cols[nearest]),y+int(rows[nearest]),tile,proof
    if best is None:
        raise ValueError('effect does not fit on unobstructed paper')
    _,x,y,tile,proof=best
    layer=Image.new('RGBA',(room.shape[1],room.shape[0]))
    layer.paste(tile,(x,y))
    proof.update(ink_bounds=layer.getchannel('A').getbbox(),origin=None,quarter_turn_clockwise=True)
    return layer,proof


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


def terminal_kana_mask(crop, bounds, detection):
    """Recover the verified native component only, including its antialiasing."""
    from ..ocr.recognition_units import image_fingerprint

    record = detection.get('ocr_metadata', {}).get('terminal_kana')
    if not record or not detection['text'].endswith('っ'):
        return None
    x, y, r, b = record['bounds']
    l, t, right, bottom = bounds
    if not (l <= x < r <= right and t <= y < b <= bottom and max(r-x, b-y) <= 48):
        return None
    patch = crop[y-t:b-t, x-l:r-l]
    if image_fingerprint(Image.fromarray(patch)) != record['digest']:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    # The two-pixel ring belongs to the whitespace around the component.
    support = np.zeros(gray.shape, bool)
    support[2:-2, 2:-2] = gray[2:-2, 2:-2] < 150
    mask = np.zeros(crop.shape[:2], bool)
    mask[y-t:b-t, x-l:r-l] = p.dilate(support, 1) & (gray < 250)
    return mask


def recover(rgb,source,raw,rows,clean,text,erase,layout,font_data,coverage,style,*,backend=None,budget=None,probability=None,extents=None,font_resolver=None,repetitions=None,max_title_size=None):
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
    twin_ids = set()
    for detection in geometry.raw.values():
        if detection.get('reason')=='corroborated_printed_twin':
            twin_ids.add(detection['source_id'])
            twin_ids.add(detection.get('ocr_metadata',{}).get('printed_twin',{}).get('peer'))
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
            extended_strokes = None
            ids = row['binding']['member_ids']
            hollow_inference = None
            if probability is not None and all(geometry.raw[(type(m).__name__,m)].get('reason')
                    == 'corroborated_repeated_effect' for m in ids):
                strokes = source_support.hollow_effect_masks(crop,own,protected,probability[y:b,x:r])
                if strokes is not None:
                    predicted,hollow_inference = strokes
                    ladder = [('measured_hollow_effect',predicted)]*2
                    evidence = {'policy':'measured_hollow_effect','source_removal_verified':False}
                    row['effect_style'] = {'fill':[255]*3,'outline_fill':[0]*3,'outline':3}
            if len(ids)>1:
                for m in ids:
                    tip = terminal_kana_mask(crop, bounds, geometry.raw[(type(m).__name__,m)])
                    if tip is None or not tip.any():
                        continue
                    if (tip & protected).any():
                        # A weak second box sometimes reads the same っ plus a digit
                        # from an adjacent anger mark. Release only the verified ink.
                        yy, xx = np.nonzero(tip)
                        center = float(x+xx.mean()), float(y+yy.mean())
                        duplicates = {key for key, other in geometry.raw.items() if key[1] not in ids
                                      and other.get('confidence',1) < .55
                                      and re.fullmatch(r'っ[0-9０-９?？]{1,2}', other.get('text',''))
                                      and cv2.pointPolygonTest(np.float32(other['polygon']), center, False) >= 0}
                        if len(duplicates)==1:
                            _, _, other_guards = geometry.local(ids, reserved, source_priority=True, margin=80,
                                cleanup_polygon=extent['polygon'] if extent else None, cleaned_for_layout=duplicates)
                            protected = protected & ~(tip & ~other_guards)
                    if not (tip & protected).any():
                        predicted |= tip
                        ladder = [(name, mask | tip) for name, mask in ladder]
                        evidence['corroborated_terminal_kana_pixels'] = int(tip.sum())
            if len(ids)==1:
                from ..ocr.handwritten import stroke_masks
                detection = geometry.raw[(type(ids[0]).__name__,ids[0])]
                handwriting = stroke_masks(original,bounds,detection,own,protected)
                if probability is not None and detection.get('ocr_metadata',{}).get('painted_effect'):
                    from .printed_contours import colored_letter_strokes, pale_speedline
                    colored = colored_letter_strokes(crop, bounds, probability[y:b,x:r], detection, protected)
                    if colored is not None:
                        strokes, color = colored
                        predicted = hollow_inference = strokes
                        ladder = [('measured_painted_outline', strokes)]*2
                        evidence = {'policy':'measured_painted_outline','source_removal_verified':False}
                        row['effect_style'] = {'fill':color,'outline_fill':[36]*3,'outline':2}
                    elif detection['ocr_metadata']['painted_effect'].get('white_sideways'):
                        from ..ocr.recognition_units import image_fingerprint
                        note=detection['ocr_metadata']['painted_effect']
                        l,t,right,bottom=note['bounds']
                        if (x<=l<right<=r and y<=t<bottom<=b and image_fingerprint(
                                Image.fromarray(crop[t-y:bottom-y,l-x:right-x]))==note['source_digest']):
                            white=pale_speedline(crop,own,protected)
                            if white is not None and white[0].any():
                                predicted,hollow_inference=white
                                ladder=[('white_print_across_motion',predicted)]*2
                                evidence={'policy':'white_print_across_motion','source_removal_verified':False}
                                row['effect_style']={'fill':[255]*3,'outline_fill':[40]*3,'outline':1}
                metadata = detection.get('ocr_metadata',{})
                if any(k in metadata for k in ('outlined_ellipsis','hollow_print','heavy_print','outlined_display')):
                    from .printed_contours import outlined_word_mask
                    rim = outlined_word_mask(crop,bounds,metadata,protected)
                    if rim is not None and rim.any():
                        predicted = hollow_inference = rim
                        ladder = [('verified_outlined_word',rim)]*2
                        evidence = {'policy':'verified_outlined_word','source_removal_verified':False}
                        if 'outlined_display' in metadata:
                            from .printed_contours import interrupted_ink
                            if interrupted_ink(crop,rim,protected):
                                hollow_inference=p.dilate(rim,5)
                                evidence['interrupted_ink']=True
                        if 'heavy_print' not in metadata:
                            row['effect_style'] = {'fill':[255]*3,'outline_fill':[0]*3,'outline':3}
                if probability is not None and detection.get('ocr_metadata',{}).get('handwritten_aside'):
                    from .printed_contours import diagonal_aside_mask
                    strokes = diagonal_aside_mask(crop, bounds, probability[y:b,x:r], detection, protected)
                    if strokes is not None:
                        predicted = hollow_inference = strokes
                        ladder = [('verified_diagonal_aside', strokes)]*2
                        evidence = {'policy':'verified_diagonal_aside','source_removal_verified':False}
                        row['lettering_role'] = 'handwriting'
                if probability is not None and detection.get('ocr_metadata',{}).get('drawn_kana'):
                    from .printed_contours import drawn_kana_outline
                    rim = drawn_kana_outline(crop,bounds,probability[y:b,x:r],detection,protected)
                    if rim is not None:
                        predicted = hollow_inference = rim
                        ladder = [('bound_drawn_outline',rim)]*2
                        evidence = {'policy':'bound_drawn_outline','source_removal_verified':False}
                if (probability is not None and detection.get('reason') in ('corroborated_blank_sound','frontier_discovery')
                        and re.fullmatch(r'[ぁ-ゖァ-ヺー]{2,12}[！!？?]?',detection.get('selected_text',''))
                        and evidence.get('source_line_thickness',0)>96):
                    tips = source_support.outlined_interiors(crop,own,protected,probability[y:b,x:r])
                    if tips.any():
                        extended_strokes = tips & ~ladder[0][1]
                        ladder = [(name+'+outlined_interiors',mask|tips) for name,mask in ladder]
                        predicted |= tips
                        evidence = {**evidence,'outlined_interior_pixels':int(tips.sum())}
                        row.setdefault('effect_style',{'fill':[255]*3,'outline_fill':[0]*3,
                            'outline':max(2,min(5,round(style['min_size']*.15)))})
                if (probability is not None and backend is not None and backend.config.get('backend')=='auto'
                        and re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿々]{2,14}',detection.get('text',''))):
                    pale = source_support.pale_sign_strokes(crop,own,protected,probability[y:b,x:r])
                    if pale is not None:
                        predicted,color = pale
                        ladder=[('pale_painted_ink',predicted)]*2
                        evidence={**evidence,'pale_printed_sign':True}
                        row['effect_style']={'fill':color,'outline_fill':[50]*3,'outline':1}
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
            title_mask = None
            if len(ids)==1 and backend is not None and backend.config.get('backend')=='auto':
                detection = geometry.raw[(type(ids[0]).__name__,ids[0])]
                if detection.get('ocr_metadata',{}).get('outlined_vowel'):
                    from .printed_contours import outlined_title
                    title_mask = outlined_title(crop,np.asarray(detection['polygon'])-[x,y],
                                                len(detection['selected_text']),protected)
                    if title_mask is not None:
                        predicted = title_mask
                        ladder = [('measured_title_outlines',title_mask)]*2
                        evidence = {'policy':'measured_title_outlines','source_removal_verified':False}
            dots = np.zeros_like(own)
            for member in row['binding']['member_ids']:
                points = np.asarray(geometry.raw[(type(member).__name__,member)]['polygon'])
                line = Image.new('L',(r-x,b-y))
                ImageDraw.Draw(line).polygon([(px-x,py-y) for px,py in points],fill=255)
                line = np.asarray(line) != 0
                dots |= adjacent_ellipsis(crop,line,protected | (own & ~line))
            if dots.any() and handwriting is None and title_mask is None:
                ladder = [(name+'+ellipsis',mask | dots) for name,mask in ladder]
                evidence = {**evidence, 'adjacent_ellipsis_pixels':int(dots.sum())}
            if extent and handwriting is None and title_mask is None:
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
            from . import repeated_strokes
            pair = None
            if len(ids)==2 and backend is not None and backend.config.get('backend')=='auto':
                pair = repeated_strokes.pale_pair(crop,[geometry.raw[(type(i).__name__,i)] for i in ids],
                    np.array([x,y]),protected)
                if pair is not None:
                    predicted,commit,row['effect_style'],_ = pair
                    ladder = [('recurring_pale_strokes',commit)]*2
                    evidence = {'policy':'recurring_pale_strokes','pale_repeated_ink':True,'source_removal_verified':False}
            repeated = repeated_strokes.local(repetitions,ids,bounds,crop,protected)
            if repeated is not None:
                predicted,inference = repeated
                ladder = [('recurring_print_strokes',predicted)]*2
                evidence = {'policy':'recurring_print_strokes','source_removal_verified':False}
                row['effect_style'] = {'fill':[0]*3,'outline_fill':[255]*3,'outline':2}
            if (probability is not None and backend is not None and backend.config.get('backend')=='auto'
                    and len(ids)>=2 and pair is None and repeated is None and handwriting is None and outlined is None):
                guard = source_support.adjoining_solid_ink(crop,own,ladder[0][1],probability[y:b,x:r])
                if guard is not None:
                    ladder = [(name,mask & ~guard) for name,mask in ladder]
                    predicted = predicted & ~guard
                    placement_protected |= guard
                    evidence = {**evidence,'adjoining_ink_preserved':int(guard.sum())}
            first = geometry.raw[(type(ids[0]).__name__,ids[0])]
            passage = first.get('ocr_metadata',{}).get('masked_passage',{})
            wrapped_aside = (2 <= len(ids) <= 4 and len(row['binding']['text'].split()) <= 9
                             and passage.get('direction') == 'vertical' and passage.get('ids') == list(ids))
            # Presentation space is independent of the larger inference context.
            # A failed layout still reaches the final removal-only transaction.
            allowed=p.dilate(own,24)&~placement_protected
            if repeated is not None:
                # The glyph may cross a panel rule. Its English can still sit
                # above or below that rule without covering a second time.
                vertical,horizontal = geometry.rules
                stripes = (vertical[y:b,x:r] | horizontal[y:b,x:r]).astype(bool)
                allowed &= ~p.dilate(stripes,3)
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
            outline = 1
            if not row.get('effect_style') and len(row['binding']['text'].split())<=3:
                # A thin halo disappears among the black dots of a screentone.
                # Test the surrounding print so dark source letters on white
                # paper do not trigger an unnecessarily heavy outline.
                ring = (p.dilate(own,7) & ~own & ~placement_protected)
                surround = crop[ring]
                if len(surround)>=80 and np.quantile(np.ptp(surround,axis=1),.9)<12:
                    gray = surround[:,1]
                    if (np.mean(gray<55)>.16 and np.mean(gray>215)>.16
                            and np.mean((gray>=55)&(gray<=215))>.18 and np.median(gray)<205):
                        outline = min(4,max(2,round(style['min_size']*.15)))
            try:
                if font_resolver is not None:font_data,coverage=font_resolver(row)
                n.validate_text(row['binding']['text'],coverage)
                native_frame = 'outlined_title' if title_mask is not None else None
                if wrapped_aside and len(ids)==2:
                    native_frame = 'paired_handwritten_columns'
                if len(ids)==1 and detection.get('ocr_metadata',{}).get('screen_field') in ('account','navigation'):
                    native_frame = 'printed_menu_strip'
                elif len(ids)==1 and detection.get('ocr_metadata',{}).get('forum_post',{}).get('heading'):
                    native_frame = 'slanted_printed_heading'
                elif len(ids)==1 and geometry.raw[(type(ids[0]).__name__,ids[0])].get('reason')=='confirmed_trailing_kana':
                    native_frame = 'small_handwritten_aside'
                elif len(ids)>=2 and re.search(r'\([A-Z]\)',row['binding']['text']):
                    from ..ocr.types import menu_shortcuts
                    source_menu = ''.join(geometry.raw[(type(m).__name__,m)].get('text','') for m in row['binding'].get('order',ids))
                    keys = menu_shortcuts(source_menu)
                    if len(keys)>=2 and keys==re.findall(r'\(([A-Z])\)',row['binding']['text']):
                        native_frame = 'printed_menu_strip'
                if native_frame is None and len(ids)<=3:
                    fields = [geometry.raw[(type(i).__name__,i)] for i in ids]
                    labels = [d.get('text','') for d in fields]
                    profile = any(re.match(r'[A-Za-z]{2,12}\s*[:：]',v) for v in labels)
                    if len(fields)==1:
                        metadata=fields[0].get('ocr_metadata',{})
                        profile=profile or metadata.get('printed_field') or metadata.get('display_heading') or metadata.get('clipped_signature')
                        # Mixed-script product labels often sit on a cover or
                        # screen cut off by a panel rule. Keep the English in
                        # the shallow printed strip when the rule spans it.
                        if (not profile and geometry.rules is not None and re.search('[A-Za-z]{2,}',labels[0])
                                and re.search('[ぁ-ヺ一-鿿]',labels[0]) and len(labels[0])<=30):
                            frame=ruled_caption_band(own,geometry.rules[1][y:b,x:r],placement_protected)
                            if frame is not None:
                                # The original ink can touch the rule. Only the English
                                # layout changes here, leaving the stroke cleanup intact.
                                try:
                                    layer,proof=n.layout(row['binding']['text'],font_data,coverage,frame,
                                        {**style,'min_size':min(10,style['min_size']),'outline':0,'fill':[0]*3},
                                        fixed_breaks=())
                                except n.ResourceLimit:
                                    raise
                                except ValueError:
                                    pass
                                else:
                                    native_frame=entry['placement_frame']='ruled_printed_caption'
                                    content=frame
                    stacked = (len(labels)==2 and re.fullmatch(r'[A-Za-z][\w.-]{2,}',labels[0],re.ASCII)
                               and re.search(r'[ぁ-ヺ一-鿿]',labels[1]))
                    widget = False
                    if geometry.file_listing and len(labels)==1 and re.fullmatch(r'(?:[一-鿿]{2,4}|[［\[][ァ-ヺー]{2,12}[.…・]*)',labels[0]):
                        w,h = np.ptp(fields[0]['polygon'],axis=0)
                        widget = 6<=h<=32 and h<=w<=84 and len(row['binding']['text'].split())<=3
                    if widget or (profile or stacked) and all(
                            np.ptp(d['polygon'],axis=0)[0]>1.45*np.ptp(d['polygon'],axis=0)[1] for d in fields):
                        native_frame = 'printed_field'
                        # Small rows often sit directly under a toolbar icon
                        # or avatar. Its contour can enter the top of the OCR
                        # polygon, even when the icon has no detection itself.
                        footprint = p.dilate(own,1)
                        guard = crossing_ink(crop,footprint)
                        protected |= guard
                        placement_protected |= guard
                        footprint &= ~guard
                        predicted &= footprint
                        ladder = [(name,mask & footprint) for name,mask in ladder]
                        # Detector rows can have a gap at the baseline. A
                        # compact field occupies the quadrilateral spanning
                        # its rows, including the interline whitespace.
                        points = cv2.findNonZero(own.astype(np.uint8))
                        field = np.zeros_like(own,dtype=np.uint8)
                        cv2.fillConvexPoly(field,cv2.convexHull(points),1)
                        field = p.dilate(field!=0,4) & ~placement_protected
                        mask = Image.fromarray(field.astype(np.uint8)*255)
                        edge = np.asarray(fields[0]['polygon'])
                        left,right = np.array_split(edge[np.argsort(edge[:,0])],2)
                        dx,dy = right.mean(axis=0)-left.mean(axis=0)
                        rotation = float(np.clip(-np.degrees(np.arctan2(dy,dx)),-20,20))
                        try:
                            layer,proof = n.layout(row['binding']['text'],font_data,coverage,mask,
                                {**style,'min_size':min(10,style['min_size']),'rotation':rotation,
                                 'leading':.06,'outline':0,'fill':[0]*3})
                        except n.ResourceLimit:
                            raise
                        except ValueError:
                            pass
                        else:
                            content = mask
                            entry['placement_frame'] = native_frame
                painted_effect = False
                if (len(ids)==1 and backend is not None and backend.config.get('backend')=='auto'
                        and not row.get('effect_style') and len(row['binding']['text'].split())<=4):
                    effect = light_effect_style(crop,own,detection['polygon'],detection.get('text',''))
                    if effect:
                        row['effect_style'] = effect
                        painted_effect = True
                if pair is not None:
                    mask = Image.fromarray((p.dilate(own,2) & ~placement_protected).astype(np.uint8)*255)
                    words = row['binding']['text'].split()
                    breaks = (len(words[0])+1,) if len(words)==2 and words[0].lower()==words[1].lower().rstrip('.!…') else None
                    low = min(10,style['min_size'])
                    try:
                        layer,proof = n.layout(row['binding']['text'],font_data,coverage,mask,
                            {**style,**row['effect_style'],'min_size':low,'max_size':max(low,min(pair[3],style['max_size'])),
                             'leading':.05},fixed_breaks=breaks)
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        content = mask
                        native_frame = entry['placement_frame'] = 'paired_painted_effect'
                if painted_effect or evidence.get('pale_printed_sign'):
                    mask=Image.fromarray((own & ~placement_protected).astype(np.uint8)*255)
                    points=cv2.findNonZero(own.astype(np.uint8))
                    vx,vy,_,_=cv2.fitLine(points,cv2.DIST_L2,0,.01,.01).reshape(-1)
                    rotation=float(np.clip(-np.degrees(np.arctan2(vy,vx)),-20,20))
                    try:
                        layer,proof=n.layout(row['binding']['text'],font_data,coverage,mask,
                            {**style,'rotation':rotation,**row['effect_style'],'min_size':min(10,style['min_size'])})
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        content=mask
                        native_frame=entry['placement_frame']='light_painted_effect' if painted_effect else 'pale_printed_sign'
                if evidence.get('policy') in ('measured_painted_outline','white_print_across_motion'):
                    _,_,width,height=cv2.boundingRect(np.uint8(own))
                    if height>2.25*width:
                        # The original upright Japanese occupied a slender
                        # column, often immediately beside a face or a hand.
                        field=own & ~placement_protected
                        try:
                            layer,proof=narrow_strip_layout(row['binding']['text'],font_data,coverage,field,
                                                            {**style,**row['effect_style']})
                        except n.ResourceLimit:
                            raise
                        except ValueError:
                            pass
                        else:
                            content=Image.fromarray(np.uint8(field)*255)
                            native_frame=entry['placement_frame']='vertical_painted_effect'
                if len(ids)==2:
                    head = geometry.raw[(type(ids[0]).__name__,ids[0])]
                    passage = head.get('ocr_metadata',{}).get('masked_passage',{})
                    if head.get('ocr_metadata',{}).get('periodical_title') and passage.get('ids')==list(ids):
                        native_frame = 'printed_masthead'
                    elif passage.get('direction')=='horizontal' and passage.get('ids')==list(ids):
                        # Two recovered rows in a printed diagram or sign give
                        # the label its own footprint, independent of nearby art.
                        mask = Image.fromarray((own & ~placement_protected).astype(np.uint8)*255)
                        try:
                            layer,proof = n.layout(row['binding']['text'],font_data,coverage,mask,
                                {**style,'min_size':min(10,style['min_size']),'fill':[0]*3,'outline':0,'leading':.08})
                        except n.ResourceLimit:
                            raise
                        except ValueError:
                            pass
                        else:
                            content = mask
                            native_frame = entry['placement_frame'] = 'stacked_printed_label'
                if native_frame and layer is None:
                    try:
                        layer,proof,content = slanted_layout(row['binding']['text'],font_data,coverage,
                            own,placement_protected,{**style,'min_size':min(10,style['min_size'])})
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        entry['placement_frame'] = native_frame
                if backend is not None and backend.config.get('backend') == 'auto' and len(ids) == 1 and not row.get('effect_style'):
                    detection = geometry.raw[(type(ids[0]).__name__,ids[0])]
                    color = printed_effect_color(crop,own,predicted,detection.get('text',''),row['binding']['text'])
                    if color is not None:
                        # Keep a narrow effect within its original strip along a panel edge.
                        mask = Image.fromarray((own & ~placement_protected).astype(np.uint8)*255)
                        try:
                            layer,proof = n.layout(row['binding']['text'],font_data,coverage,mask,
                                {**style,'fill':color,'outline':0})
                        except n.ResourceLimit:
                            raise
                        except ValueError:
                            pass
                        else:
                            content = mask
                            entry['placement_frame'] = 'original_colored_effect'
                if layer is None and evidence.get('painted_stroke_support') and not style.get('rotation'):
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
                literal = re.match(r'^["“]([A-Za-z0-9][A-Za-z0-9 ._+/:\-]{2,64})["”]', first.get('text',''))
                if layer is None and len(ids)==1 and literal and literal[1] in row['binding']['text']:
                    _, _, width, height = cv2.boundingRect(own.astype(np.uint8))
                    if width >= 3 * height:
                        # A command printed on a single line under an illustration
                        # should stay together and clear the drawing above it.
                        strip = Image.fromarray((own & ~placement_protected).astype(np.uint8)*255)
                        try:
                            layer,proof = n.layout(row['binding']['text'], font_data, coverage, strip,
                                {**style,'fill':[255 if dark else 0]*3,'outline':0}, fixed_breaks=())
                        except n.ResourceLimit:
                            raise
                        except ValueError:
                            pass
                        else:
                            content = strip
                            entry['placement_frame'] = 'quoted_literal_line'
                if layer is None:
                    display_style = {}
                    if (max_title_size and evidence.get('policy') == 'verified_outlined_word'
                            and first.get('ocr_metadata', {}).get('outlined_display')):
                        # Measure the printed word itself. An oversized detection box
                        # can also include the keyboard, speed lines or panel gutter.
                        _, _, ink_width, ink_height = cv2.boundingRect(predicted.astype(np.uint8))
                        if ink_width >= ink_height and ink_height > style['max_size']:
                            display_style['max_size'] = max(style['max_size'], min(max_title_size, ink_height))
                            allowed = own & ~placement_protected
                    layer,proof,content=typography.layout_dialogue(row['binding']['text'],font_data,coverage,
                        Image.fromarray(allowed.astype(np.uint8)*255),{**style,'fill':[255 if dark else 0]*3,
                            'outline':outline,'outline_fill':[0 if dark else 255]*3,
                            **({'preset':'thought','outline':0,'leading':.12,'fill':[48,48,52]}
                               if row.get('lettering_role')=='thought' else {}),**row.get('effect_style',{}),**display_style})
                floor = min(10,style['min_size']) if native_frame and entry.get('placement_frame')==native_frame else style['min_size']
                rec.require(proof['original_text']==row['binding']['text'] and proof['size']>=floor,
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
                if layer is None and not row.get('effect_style') and len(row['binding']['text'].split()) >= 2:
                    # A borderless balloon can open onto the gutter. Limit the
                    # surface to nearby paper and try tighter leading and margins.
                    frame = light_dialogue_frame(crop, own, p.dilate(own,25) & ~placement_protected)
                    if frame is not None:
                        try:
                            content = Image.fromarray(frame.astype(np.uint8)*255)
                            layer,proof = n.layout(row['binding']['text'],font_data,coverage,content,
                                {**style,'fill':[0]*3,'outline':0,'leading':min(.08,style.get('leading',.2))})
                        except n.ResourceLimit:
                            raise
                        except ValueError:
                            content = None
                        else:
                            entry['placement_retry'] = 'source_paper'
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
                        (sx,sy,sx+sw,sy+sh),placement_protected | placement_obstacles(crop,own),{**style,'fill':[255 if dark else 0]*3,
                            'outline_fill':[0 if dark else 255]*3},outline=outline)
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
            if (layer is None and len(ids)<=2 and len(row['binding']['text'].split())<=4
                    and entry.get('placement_failure','').startswith('overflow:') and
                    np.mean(crop[own].min(axis=1)>=238)>.56):
                lines = [geometry.raw[(type(i).__name__,i)] for i in ids]
                words = ''.join(d.get('text','') for d in lines)
                repeat = re.fullmatch(r'([ぁ-ゖァ-ヺ]{1,4})(?:\1){1,4}[っッ]?',words)
                short_aside = (len(ids)==2 and all(d.get('confidence',0)>.70 for d in lines)
                               and re.fullmatch(r'[ぁ-ゖァ-ヺ一-鿿々ー]{4,14}',words))
                if repeat or short_aside:
                    guard = crossing_ink(crop,own)
                    if probability is not None:
                        # A glyph can end exactly on a cheek or a panel edge.
                        # Let the text detector separate the letter from the
                        # connected contour. Existing neighbor/rule protection
                        # and the low-probability side of the contact survive.
                        glyph = p.dilate(probability[y:b,x:r]>=80,1) & own & ~protected
                        vertical,horizontal = geometry.rules
                        guard = split_touching_glyph(crop,guard,glyph,(vertical[y:b,x:r]|horizontal[y:b,x:r])!=0)
                    area = p.dilate(own,9) & ~placement_protected & ~guard & ~placement_obstacles(crop,own)
                    mask = Image.fromarray(area.astype(np.uint8)*255)
                    try:
                        layer,proof = n.layout(row['binding']['text'],font_data,coverage,mask,
                            {**style,'min_size':min(12,style['min_size']),'max_size':style['min_size'],
                             'leading':.06,'outline':0,'fill':[0]*3})
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        protected |= guard
                        predicted &= ~guard
                        ladder = [(name,mask & ~guard & p.dilate(own,3)) for name,mask in ladder]
                        content = mask
                        entry['placement_frame'] = 'small_ink_aside'
            if (layer is None and backend is not None and len(ids) in (2,3)
                    and entry.get('placement_failure','').startswith('overflow:')
                    and probability is not None and predicted[own].any() and np.median(np.ptp(crop[own],axis=1)) > 24
                    and len(row['binding']['text'].split()) <= 6):
                # Compact stacked lettering on a painted sign can fit at its
                # printed height. Establish a layout before requesting inpainting.
                boxes = [cv2.boundingRect(np.int32(geometry.raw[(type(k).__name__,k)]['polygon'])) for k in ids]
                if all(w >= 1.35*h for _,_,w,h in boxes):
                    room = p.dilate(own,3) & ~placement_protected
                    ink = 255 if np.median(crop[own] @ [.2126,.7152,.0722]) < 116 else 0
                    floor = max(10, min(style['min_size'],round(min(h for _,_,_,h in boxes)*.55)))
                    frame = Image.fromarray(np.uint8(room)*255)
                    try:
                        layer,proof = n.layout(row['binding']['text'],font_data,coverage,frame,
                            {**style,'min_size':floor,'max_size':max(floor,style['min_size']),
                             'leading':.08,'fill':[ink]*3,'outline':1,'outline_fill':[255-ink]*3})
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        content = frame
                        entry['placement_frame'] = 'stacked_painted_label'
            entry.update(bounds=bounds,layout=proof,mask_evidence=evidence)
            artifacts[f'recovery-{i:03d}-original.png']=Image.fromarray(crop)
            artifacts[f'recovery-{i:03d}-lettering-evidence.png']=Image.fromarray(predicted.astype(np.uint8)*255)
            if content is not None:artifacts[f'recovery-{i:03d}-layout.png']=content
            source_box = cv2.boundingRect(own.astype(np.uint8))
            vertical_label = (len(ids)==1 and source_box[3]>1.3*source_box[2]
                              and re.fullmatch(r'[一-鿿々]{1,4}', geometry.raw[(type(ids[0]).__name__,ids[0])].get('text','')))
            jobs.append({'i':i,'bounds':bounds,'crop':crop,'own':own,'protected':protected,'predicted':predicted,
                'ladder':ladder,'layer':layer,'proof':proof,'best':None,'retry':False,
                'wrapped_aside':wrapped_aside,'stroke_inference':repeated[1] if repeated is not None else hollow_inference,
                'outlined':outlined is not None,'continuations':continuations,'extended_strokes':extended_strokes,
                'source_box':source_box,
                'handwriting_core':handwriting[0] if handwriting is not None else None,
                'handwriting_layout':(own & ~placement_protected if handwriting is not None else
                    p.dilate(own,10) & ~placement_protected if not row.get('effect_style') and not light_dialogue_enclosed
                    and outlined is None and not dark and np.mean(crop[own].min(1)>238)>.60
                    and (len(row['binding']['text'].split())>=2 or vertical_label) else None),
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
            if job['stroke_inference'] is not None:
                inference = job['stroke_inference']
            elif job['probability'] is not None:
                inference,context=source_support.inference_context(mask,job['probability'])
            candidate,method = cheap(crop,mask,protected,job['wrapped_aside'])
            if name=='white_print_across_motion' and (backend is None or backend.config.get('backend')=='auto'):
                # Reconstruct through the full glyph, then commit around the
                # surviving speed line so its original white pixels stay sharp.
                candidate=cv2.inpaint(crop,np.uint8(inference),3,cv2.INPAINT_TELEA)
                method='local_painted_strokes'
            if name == 'measured_title_outlines':
                candidate = crop.copy()
                candidate[mask] = 255
                method = 'dominant_surface_fill'
            ids = row['binding']['member_ids']
            if (method=='dominant_surface_fill' and len(ids)==1 and ids[0] in twin_ids
                    and (backend is None or backend.config.get('backend')=='auto')):
                band = source_support.restore_paper_band(crop,candidate,job['own'],mask,protected)
                if (band & ~mask).any():
                    entry['restored_paper_band_extension'] = int((band & ~mask).sum())
                    mask = band
                    inference = inference | band
            if (entry.get('placement_frame') == 'printed_menu_strip'
                    and (backend is None or backend.config.get('backend') == 'auto')):
                menu = printed_menu_fill(crop,job['own'],mask)
                if menu is not None:
                    candidate,method = menu,'printed_menu_surface'
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
            if ((backend is None or backend.config.get('backend')=='auto') and (method=='classical_telea_unverified' or name=='measured_painted_outline')
                    and (entry['mask_evidence'].get('painted_stroke_support') or entry['mask_evidence'].get('pale_printed_sign')
                         or entry['mask_evidence'].get('pale_repeated_ink') or name=='measured_painted_outline')):
                painted = source_support.painted_stroke_fill(crop,mask,protected,slanted=True,
                        pale_ink=bool(entry['mask_evidence'].get('pale_printed_sign') or entry['mask_evidence'].get('pale_repeated_ink') or name=='measured_painted_outline'),
                        measured_outline=name=='measured_painted_outline')
                if painted is not None:
                    candidate,method = painted,'local_painted_strokes'
            if name=='measured_painted_outline' and backend is not None and backend.config.get('backend')=='auto':
                keyline = source_support.painted_keyline_fill(crop,mask,protected)
                if keyline is not None:
                    mask,candidate = keyline
                    inference = inference | mask
                    method = 'local_painted_strokes'
                    entry['painted_keyline_extension_pixels'] = int((mask & ~job['predicted']).sum())
            if job['layer'] is None and job['wrapped_aside']:
                left,top,right,bottom = job['bounds']
                # A handwritten line may wrap along a character's silhouette. Use
                # the restored paper and allow the English to move as one block.
                footprint = p.dilate(job['own'],18) & ~(job['protected'] | clearance[top:bottom,left:right])
                room = footprint & p.erode(candidate.min(2) > 224,1)
                if font_resolver is not None:
                    font_data,coverage = font_resolver(row)
                lettering = {**style,'min_size':min(12,style['min_size']), 'max_size':style['min_size'],
                             'fill':[0]*3,'outline':0,'leading':.06}
                try:
                    trial,metrics = n.layout(row['binding']['text'],font_data,coverage,typography.interior_box(footprint),lettering)
                    trial,metrics = place_on_clean_paper(trial,metrics,candidate,room,font_data,coverage,lettering)
                except n.ResourceLimit:
                    raise
                except ValueError:
                    pass
                else:
                    if not ((np.asarray(trial.getchannel('A'))!=0) & ~room).any():
                        job['layer'],job['proof'] = trial,metrics
                        entry.update(layout=metrics,placement_frame='restored_wrapped_aside')
            selected = None if method in ('local_painted_strokes','printed_menu_surface') else rec.region_backend(backend,crop,inference)
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
            if selected is not None and (method not in ('measured_uniform_fill','dominant_surface_fill')
                    or entry['mask_evidence'].get('interrupted_ink')):
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
            if name=='bound_drawn_outline':
                from . import printed_contours
                printed_contours.refine(original,job['bounds'],mask,protected,candidate)
            if (name=='measured_painted_outline' and method.startswith('learned_')
                    and backend.config.get('backend')=='auto'):
                # On shallow strokes a neural fill can redraw little outlines
                # inside the removed letters. Compare a smooth local repair.
                radius=cv2.distanceTransform(np.uint8(mask),cv2.DIST_L2,5).max()
                if radius<=12:
                    local=cv2.inpaint(crop,np.uint8(mask),3,cv2.INPAINT_NS)
                    learned_signals=assess(crop,candidate,mask,job['predicted'],protected)
                    local_signals=assess(crop,local,mask,job['predicted'],protected)
                    if (local_signals['new_text_like_components']*2+2<=learned_signals['new_text_like_components']
                            and local_signals['seam_laplacian_mean']<=.65*learned_signals['seam_laplacian_mean']):
                        candidate=local
                        attempt['method']=method='local_painted_strokes'
                        attempt['outline_repair']='navier_stokes'
            if job['continuations'] is not None:
                strokes,tones = job['continuations']
                candidate[strokes] = tones[strokes,None]
            extra = job['extended_strokes']
            if (extra is not None and extra.any() and method.startswith('learned_')
                    and backend.config.get('backend')=='auto'):
                # A learned fill can repeat the silhouette of a faint stroke tip.
                # Use agreeing background samples across that tip, leaving the
                # model's reconstructed lines farther from the extension intact.
                from .stroke_repair import interpolate
                distance = cv2.distanceTransform(np.uint8(~extra),cv2.DIST_L2,5)
                affected = mask & (distance<24)
                chords = interpolate(crop,affected,inference|protected,candidate)
                weight = (1-distance[affected,None]/24)
                candidate[affected] = np.rint(candidate[affected]*(1-weight)+chords[affected]*weight)
                attempt['outlined_tip_blend_pixels'] = int(affected.sum())
            # Even classical/native output is explicitly composited through commit.
            candidate=rec.compose(req,crop,inference,mask,protected,{'request_revision':req['revision'],'pixels':candidate})
            signals=assess(crop,candidate,mask,job['predicted'],protected)
            attempt['assessment']=signals;entry['attempts'].append(attempt)
            artifacts[prefix+'-cleanplate.png']=Image.fromarray(candidate)
            priority={'measured_uniform_fill':0,'dominant_surface_fill':0,'printed_menu_surface':0,'learned_lama':1,'learned_sdxl':1,'local_painted_strokes':1,'outlined_handwriting_fill':1,'classical_shiftmap':1,'classical_telea_unverified':2}[method]
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
        display_word = row.get('lettering_role')=='shout' or (sw > .30 * rgb.width and sh > .10 * rgb.height)
        if (layer is not None and display_word and method.startswith('learned_')
                and len(row['binding']['text'].split())==1 and (entry['mask_evidence'].get('policy')=='bound_drawn_outline'
                or np.mean(candidate[job['own']].min(1)>235)>.72)):
            allowed = job['own'] & ~job['protected'] & ~clearance[y:b,x:r]
            ink = np.asarray(layer.getchannel('A')) != 0
            if (ink & p.dilate(candidate.min(2)<160,3)).any():
                if font_resolver is not None:
                    font_data,coverage = font_resolver(row)
                layer,job['proof'] = place_on_clean_paper(layer,job['proof'],candidate,allowed,
                    font_data,coverage,{**style,'min_size':job['proof']['size'],'max_size':job['proof']['size']})
                entry['layout'] = job['proof']
        ids = row['binding']['member_ids']
        collides = layer is not None and (np.asarray(layer.getchannel('A')).astype(bool) & clearance[y:b,x:r]).any()
        repair_balloon = layer is None or collides or (job['proof']['size'] < style['min_size'] and
                                                       entry.get('placement_retry')=='restored_balloon_interior')
        short_reply = (len(ids)==1 and sh>sw and len(row['binding']['text'])<=12
                       and len(row['binding']['text'].split())<=2)
        if (repair_balloon and (2<=len(ids)<=6 or short_reply) and not row.get('effect_style')
                and entry.get('placement_failure','').startswith('overflow:') and len(row['binding']['text'])<=120):
            blocked = job['protected'] | clearance[y:b,x:r]
            # A physical balloon can use its entire rounded interior. A centered
            # inscribed rectangle can exclude both wide ends of a speech line.
            paper = painted_balloon_paper(candidate,job['own'],blocked,min(11,style['min_size']),closed=True)
            if paper is None and not short_reply:
                paper = painted_balloon_paper(candidate,job['own'],blocked,style['min_size'])
            if paper is not None:
                paper &= p.dilate(job['own'],max(36,round(min(sw,sh)*.36)))
                if font_resolver is not None:
                    font_data,coverage = font_resolver(row)
                lettering = {**style,'min_size':min(11,style['min_size']), 'max_size':style['min_size'],
                             'fill':[0]*3,'outline':0,'leading':.08}
                for frame in (Image.fromarray(np.uint8(paper)*255), typography.interior_box(paper)):
                    try:
                        trial,metrics = n.layout(row['binding']['text'],font_data,coverage,frame,lettering)
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        continue
                    if layer is None or collides or metrics['size']>job['proof']['size']:
                        layer,job['proof'] = trial,metrics
                        entry.update(layout=metrics,placement_retry='restored_balloon_interior')
                        collides=False
        target_words = len(row['binding']['text'].split())
        single_paper = None
        if (layer is None and len(ids)==1 and target_words==1 and sh>1.3*sw
                and geometry.raw[(type(ids[0]).__name__,ids[0])].get('confidence',0)>=.9):
            single_paper = painted_balloon_paper(candidate,job['own'],job['protected'] | clearance[y:b,x:r],
                                                 style['min_size'],closed=True)
        narrow_dialogue = (single_paper is not None or
                           (len(ids)==1 and 12<=sw and sh>2*sw and 2<=target_words<=8
                            and geometry.raw[(type(ids[0]).__name__,ids[0])].get('confidence',0)>=.9))
        if (layer is None and not narrow_dialogue and len(ids)==1 and 2<=target_words<=4
                and method in ('measured_uniform_fill','dominant_surface_fill') and sh>sw):
            # A tilted column can look nearly square in its axis-aligned bounds.
            # Its original edges and clean paper can still support a small aside.
            line = geometry.raw[(type(ids[0]).__name__,ids[0])]
            sides = cv2.minAreaRect(np.asarray(line['polygon'],np.float32))[1]
            narrow_dialogue = (min(sides)>=12 and max(sides)>=1.8*min(sides)
                               and line.get('confidence',0)>=.75
                               and re.fullmatch(r'[ぁ-ゖァ-ヺー]{2,6}',line.get('text','')) is not None
                               and np.mean(candidate[job['own']].min(1)>=242)>.9)
        if (layer is None and method in ('measured_uniform_fill','dominant_surface_fill')
                and entry.get('placement_failure','').startswith('overflow:')
                and (len(ids)>=2 or narrow_dialogue) and not row.get('effect_style')):
            # A handwritten aside can have ample paper beside its original strokes,
            # while its uneven footprint has no centered rectangle for English.
            pad = min(32,sw+4) if narrow_dialogue else 20
            nearby = p.dilate(job['own'],pad) & ~job['protected'] & ~clearance[y:b,x:r]
            paper = nearby & (candidate.min(2)>=240)
            if single_paper is not None:
                paper &= single_paper
            if narrow_dialogue:
                paper &= p.erode((candidate.min(2)>=240) & ~job['protected'],3)
            _,labels = cv2.connectedComponents(paper.astype(np.uint8),connectivity=4)
            ids,counts = np.unique(labels[paper & mask],return_counts=True)
            if len(ids):
                paper &= labels==ids[counts.argmax()]
                frame = Image.new('L',(r-x,b-y))
                ImageDraw.Draw(frame).rectangle((sx-pad,sy,sx+sw+pad,sy+sh),fill=255)
                lettering = {**style,'max_size':min(style['max_size'],round(style['min_size']*1.15)),
                             'fill':[0]*3,'outline':0,'leading':.05}
                if narrow_dialogue:
                    lettering['min_size'] = min(11,style['min_size'])
                if font_resolver is not None:
                    font_data,coverage = font_resolver(row)
                try:
                    caps = (lettering['max_size'],12,11) if narrow_dialogue else (lettering['max_size'],)
                    for cap in dict.fromkeys(min(cap,lettering['max_size']) for cap in caps):
                        trial,metrics = n.layout(row['binding']['text'],font_data,coverage,frame,
                            {**lettering,'max_size':max(lettering['min_size'],cap)})
                        trial,metrics = place_on_clean_paper(trial,metrics,candidate,paper,font_data,coverage,lettering)
                        if not ((np.asarray(trial.getchannel('A'))!=0) & ~paper).any():
                            break
                except n.ResourceLimit:
                    raise
                except ValueError:
                    pass
                else:
                    if not ((np.asarray(trial.getchannel('A'))!=0) & ~paper).any():
                        layer,job['proof'] = trial,metrics
                        entry.update(layout=metrics,placement_retry='restored_paper_aside')
        if (layer is None and narrow_dialogue and sh >= 3*sw and 2 <= target_words <= 12
                and method in ('dominant_surface_fill', 'measured_uniform_fill') and not row.get('effect_style')):
            # Try horizontal lettering first. A slim margin between a face and
            # a panel rule may have room only along the printed column itself.
            paper = p.dilate(job['own'], 4) & p.erode(candidate.min(2) >= 242, 2)
            paper &= ~(job['protected'] | clearance[y:b, x:r])
            if font_resolver is not None:
                font_data, coverage = font_resolver(row)
            try:
                trial, metrics = narrow_strip_layout(row['binding']['text'], font_data, coverage, paper,
                                                     {**style,'fill':[0]*3,'outline':0})
            except n.ResourceLimit:
                raise
            except ValueError:
                pass
            else:
                layer, job['proof'] = trial, metrics
                entry.update(layout=metrics, placement_retry='restored_vertical_strip')
        source_ids=row['binding']['member_ids']
        source_line = geometry.raw[(type(source_ids[0]).__name__,source_ids[0])] if len(source_ids)==1 else {}
        motion = (source_line.get('ocr_metadata',{}).get('heavy_print',{}).get('motion_radius')
                  and entry['mask_evidence'].get('policy') == 'verified_outlined_word')
        repaired_effect = bool(motion or source_line.get('ocr_metadata',{}).get('repeated_kana'))
        if (source_line.get('reason')=='corroborated_blank_sound' or source_line.get('ocr_metadata',{}).get('empty_detection')) and layer is not None:
            letters=np.asarray(layer.getchannel('A'))>0
            # An outlined effect may be printed partly over a character. Only
            # move the English when it covers surviving dark lines on light art.
            repaired_effect |= (np.mean(candidate[job['own']].min(1)>200)>.55 and letters.any()
                                and np.mean(candidate[letters].min(1)<125)>.06)
        if repaired_effect or entry['mask_evidence'].get('policy') == 'verified_diagonal_aside':
            # Use the repaired paper at commit time. An adjoining speech balloon
            # may already have taken part of the nominal OCR rectangle.
            room = (p.dilate(job['own'],32) & p.erode(candidate.min(2)>145,1) if repaired_effect
                    else p.dilate(job['own'],25) & p.erode(candidate.min(2)>195,2))
            room &= ~(job['protected'] | clearance[y:b,x:r])
            if repaired_effect:
                # A tall panel border can split the source footprint. Place the
                # new lettering on the side containing most of the source.
                _,components=cv2.connectedComponents(np.uint8(room),connectivity=4)
                inside=components[job['own'] & room]
                if len(inside):
                    room &= components==np.bincount(inside).argmax()
            if font_resolver is not None:
                font_data, coverage = font_resolver(row)
            lettering = {**style, 'min_size':min(11,style['min_size']), 'max_size':min(15,style['max_size']),
                         'fill':[0]*3, 'outline':0, 'leading':.05}
            if motion and job['proof']:
                # Broad strokes leave generous clean paper. Retain the effective
                # size of the original layout while moving off the balloon rim.
                lettering.update(min_size=min(style['min_size'],job['proof']['size']),max_size=job['proof']['size'])
            frame = Image.fromarray(np.uint8(p.dilate(job['own'],18 if repaired_effect else 6))*255)
            try:
                trial, metrics = n.layout(row['binding']['text'], font_data, coverage, frame, lettering)
                trial, metrics = place_on_clean_paper(trial, metrics, candidate, room, font_data, coverage, lettering)
                if repaired_effect and ((np.asarray(trial.getchannel('A'))!=0) & ~room).any():
                    trial,metrics=vertical_effect_layout(row['binding']['text'],font_data,coverage,room,job['own'],
                                                         {**lettering,'outline':1,'outline_fill':[255]*3})
            except n.ResourceLimit:
                raise
            except ValueError:
                pass
            else:
                if not ((np.asarray(trial.getchannel('A'))!=0) & ~room).any():
                    layer, job['proof'] = trial, metrics
                    entry.update(layout=metrics, placement_frame='repaired_effect' if repaired_effect else 'repaired_handwritten_aside')
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
        if (layer is not None and len(ids)==1 and not row.get('effect_style')
                and not style.get('rotation') and '\n' in job['proof']['draw_text']
                and '\n' not in row['binding']['text'] and 1 < len(row['binding']['text'].split()) <= 6):
            band = caption_band(candidate, job['own'], job['protected'] | clearance[y:b,x:r])
            if band is not None and band.any():
                if font_resolver is not None:
                    font_data,coverage = font_resolver(row)
                ink = 0 if np.median(candidate[band] @ [.2126,.7152,.0722]) >= 116 else 255
                size = job['proof']['size']
                try:
                    alternative,metrics = n.layout(row['binding']['text'],font_data,coverage,
                        Image.fromarray(np.uint8(band)*255),{**style,'min_size':max(8,round(.75*size)),
                        'max_size':size,'fill':[ink]*3,'outline':0},fixed_breaks=())
                except n.ResourceLimit:
                    raise
                except ValueError:
                    pass
                else:
                    layer,job['proof'] = alternative,metrics
                    entry.update(layout=metrics,placement_frame='single_line_caption_band')
        if (layer is not None and method in ('dominant_surface_fill','measured_uniform_fill')
                and re.search(r'[-‐]\n',job['proof']['draw_text']) and not row.get('effect_style')):
            # Try the restored paper between adjacent figures before splitting
            # a name at the narrow boundary of its original vertical columns.
            room = (p.dilate(job['own'], max(3,style['min_size']//2)) & (candidate.min(2)>=244)
                    & ~job['protected'] & ~clearance[y:b,x:r])
            if room.any():
                if font_resolver is not None:
                    font_data,coverage = font_resolver(row)
                ink_style = job['proof']['resolved_style']
                lettering = {**style, **{k:ink_style[k] for k in ('min_size','fill','outline','outline_fill','leading')}}
                alternative = n.unbroken_compounds(row['binding']['text'],font_data,coverage,
                    Image.fromarray(np.uint8(room)*255),lettering,job['proof'],n.LayoutWork())
                if alternative:
                    layer,job['proof'] = alternative
                    entry.update(layout=job['proof'],placement_frame='restored_paper_compound')
        local_alpha=np.asarray(layer.getchannel('A'))!=0 if layer is not None else np.zeros_like(mask)
        small_balloon = (len(row['binding']['member_ids'])==1 and sh>1.3*sw and
                         1 <= len(row['binding']['text'].split()) <= 5)
        if (layer is not None and not row.get('effect_style') and (len(row['binding']['text'].split())>=5 or small_balloon)
                and method in ('measured_uniform_fill','dominant_surface_fill') and not style.get('rotation')):
            paper = painted_balloon_paper(candidate, job['own'], job['protected'] | clearance[y:b,x:r], job['proof']['size'], closed=small_balloon)
            if paper is not None and (local_alpha & ~paper).any():
                if font_resolver is not None:
                    font_data,coverage = font_resolver(row)
                fitting = {**style,'min_size':min(12,style['min_size'])} if small_balloon else style
                trial,metrics = place_on_clean_paper(layer,job['proof'],candidate,paper,font_data,coverage,fitting)
                adjusted = np.asarray(trial.getchannel('A'))!=0
                if small_balloon and (adjusted & ~paper).any():
                    try:
                        trial,metrics = n.layout(row['binding']['text'],font_data,coverage,
                            Image.fromarray(np.uint8(paper)*255),{**fitting,'max_size':job['proof']['size'],
                            'fill':[0]*3,'outline':0,'leading':.08})
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        adjusted = np.asarray(trial.getchannel('A'))!=0
                if not (adjusted & ~paper).any():
                    layer,local_alpha = trial,adjusted
                    entry['layout'] = job['proof'] = metrics
                    entry['placement_frame'] = 'inset_closed_balloon' if small_balloon else 'inset_painted_balloon'
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
        blocked = clearance[y:b,x:r]
        if layer is not None and entry.get('placement_frame') in ('printed_field','printed_menu_strip'):
            # Adjacent slanted rows can share cleanup pixels. Keep the earlier
            # repair intact and let the next field use the already cleaned
            # surface only where its English clears all earlier lettering.
            mask = mask & ~reserved[y:b,x:r]
            blocked = p.dilate(alpha[y:b,x:r],2) | initial_reserved[y:b,x:r]
        if method.startswith('learned_') and layer is not None and (mask & reserved[y:b,x:r]).any():
            overlap = mask & reserved[y:b,x:r]
            # Neighboring cleanplates can share a border. Keep the earlier pixels
            # when the two repairs agree along their seam and the new lettering
            # clears every reservation. Never splice through prior lettering.
            seam = overlap & p.dilate(mask & ~overlap, 1)
            if seam.any() and not (overlap & alpha[y:b,x:r]).any():
                delta = np.abs(candidate[seam].astype(np.int16) - plate[y:b,x:r][seam]).max(axis=1)
                if np.mean(delta) <= 12:
                    mask = mask & ~overlap
                    entry['retained_neighbor_cleanup_pixels'] = int(overlap.sum())
        if layer is None or (mask&reserved[y:b,x:r]).any() or (local_alpha&blocked).any():
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
            'printed_menu_surface':'selected_fill','outlined_handwriting_fill':'selected_fill','measured_uniform_fill':'selected_fill','dominant_surface_fill':'selected_fill'}[method]]+=1
    final_clean=Image.fromarray(plate);composite=Image.alpha_composite(final_clean.convert('RGBA'),new_text).convert('RGB')
    rec.require(np.array_equal(plate[initial_reserved],native_plate[initial_reserved]),'recovery overwrote committed cleanup')
    rec.require(np.array_equal(np.asarray(new_text)[initial_reserved],native_text[initial_reserved]),'recovery overwrote accepted English')
    n.verify_pixel_audit(n.pixel_audit(rgb,final_clean,composite,erase,alpha,~(erase|layout)))
    summary.update(learned_calls=budget.calls,learned_seconds=budget.seconds,seconds=time.monotonic()-started)
    return rows,final_clean,new_text,composite,erase,layout,artifacts,summary
