"""Removal-first last resort for bound translations that precise routes cannot fit.

Cleanup and English placement are independent transactions. Original geometry,
cleanup support, opaque backing and actual glyph alpha remain distinct. Optional
local reconstruction uses fresh bound requests. No OCR or wording edits occur.
"""
from copy import deepcopy
import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import natural as n, preparation as p, aggressive, typography, source_support, source_extents, reconstruction as rec


POLICY = 'remove_then_source_centered_patch_v7_corroborated_extents'
MAX_REGIONS = 128
MAX_WORK = 32_000_000
MAX_LAYOUT_WORK = 256_000_000


def extent(crop, own, protected):
    """Bounded whole-stroke expansion, including light lettering on dark surfaces.

    Components intersecting source geometry may extend up to twelve pixels; a
    connected panel/face cannot pull the mask arbitrarily far away from the text.
    Attached art inside that bound remains an accepted best-effort risk.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    collar = p.dilate(own,8) & ~p.dilate(own,3) & ~protected
    light = not collar.any() or np.median(gray[collar]) >= 128
    ink = gray < 210 if light else gray > 60
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
    ids = np.unique(labels[own & ink]); ids = ids[ids != 0]
    support = p.dilate(own,4)
    limit = p.dilate(own,12)
    for i in ids:
        x,y,w,h,area = map(int,stats[i])
        if max(w,h)>128 or area>8192:
            continue
        component = labels[y:y+h,x:x+w] == i
        if not (component & ~limit[y:y+h,x:x+w]).any():
            support[y:y+h,x:x+w] |= component
    return p.dilate(support,1) & ~protected


def complete_color_ink(crop,own,mask,blocked):
    """Complete a clipped display glyph when a learned mask identifies its ink color."""
    if max(own.shape)>1024 or np.count_nonzero(mask)<4096:
        return None
    hue,saturation,value=cv2.split(cv2.cvtColor(crop,cv2.COLOR_RGB2HSV))
    vivid=(saturation>=150)&(value>=120)
    seeds=vivid&mask&own
    seed_count=int(seeds.sum())
    if seed_count<256:
        return None
    histogram=np.bincount(hue[seeds],minlength=180)
    band_counts=sum(np.roll(histogram,offset) for offset in range(-4,5))
    center=int(np.argmax(band_counts))
    delta=np.abs(hue.astype(np.int16)-center)
    band=vivid&(np.minimum(delta,180-delta)<=6)
    supported=int((band&seeds).sum())
    if supported<.85*seed_count or supported<.5*np.count_nonzero(band&own):
        return None
    count,labels,stats,_=cv2.connectedComponentsWithStats(band.astype(np.uint8))
    if count>8192:
        return None
    components=np.zeros_like(own)
    samples=0
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<50 or min(w,h)<4 or max(w,h)>280:
            continue
        local=np.s_[y:y+h,x:x+w]
        part=labels[local]==label
        if np.count_nonzero(part&own[local])<area*.88:
            continue
        components[local]|=part
        samples+=int(np.count_nonzero(part&mask[local])>=.8*area)
    if samples<2 or np.count_nonzero(components&~p.dilate(mask,4))<200:
        return None
    # Stroke and shadow surround the colored core. Bound their completion by
    # the original detector polygons and all previously reserved pixels.
    extra=p.dilate(components,10)&p.dilate(own,8)&~blocked&~mask
    if not extra.any() or extra.sum()>.5*mask.sum():
        return None
    return mask|extra,{'hue_bin':center,'supported_components':samples,'extra_pixels':int(extra.sum())}


def outlined_ink(crop, broad):
    """Find small light glyph interiors enclosed by dark ink on a painted page."""
    if max(broad.shape)>1024 or np.mean(np.ptp(crop,axis=2)>24)<.2:
        return None
    dark=crop.max(axis=2)<110
    white=(crop.min(axis=2)>178)&(np.ptp(crop,axis=2)<40)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    contours,_=cv2.findContours(white.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>8192:
        return None
    cores=np.zeros(broad.shape,np.uint8)
    count=0
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if (min(w,h)<2 or w*h>2300 or max(w,h)>88 or min(x,y)<2
                or x+w+2>broad.shape[1] or y+h+2>broad.shape[0]):
            continue
        local=np.s_[y-2:y+h+2,x-2:x+w+2]
        body=np.zeros((h+4,w+4),np.uint8)
        cv2.drawContours(body,[contour-np.int32([x-2,y-2])],-1,1,-1)
        ring=cv2.dilate(body,kernel)>body
        if not broad[local][body!=0].any() or dark[local][ring].mean()<.70:
            continue
        count+=1
        cores[local]|=body
    if count<8:
        return None
    nearby=p.dilate(cores!=0,6)
    ink=dark&broad
    supported=np.count_nonzero(nearby&ink)/max(1,np.count_nonzero(ink))
    if supported<.62:
        return None
    # Include dark strokes even where an individual interior was missed.
    # The broader original footprint already excludes protected output.
    strokes=p.dilate((dark&(broad|nearby))|(cores!=0),2)
    commit=strokes&broad
    if not commit.any() or commit.sum()>.90*broad.sum():
        return None
    return commit,strokes,{'enclosed_light_components':count,'supported_dark_fraction':supported,
        'original_pixels':int(broad.sum()),'committed_pixels':int(commit.sum())}


def patch_layout(text, font_data, coverage, bounds, blocked, style, *, outline=0):
    """Widen a source-centered backing instead of shrinking or breaking words.

    Enumerate compact rectangles with the requested font range; score actual
    occupied area and displacement. Other source polygons and committed pixels
    remain obstacles, but local artwork may be covered.
    """
    left,top,right,bottom = bounds
    height,width = blocked.shape
    cx,cy = (left+right)//2,(top+bottom)//2
    floor = style['min_size']
    preferred = max(floor,round(style['max_size']*.85))
    # English line lengths of roughly 12-25 characters are more usable than a
    # single vertical Japanese column. Never derive or shorten the wording.
    ideal = int(math.sqrt(max(1,len(text))*preferred*preferred*.85))
    widths = sorted({max(64,min(width-4,v)) for v in
        (right-left+16,ideal,round(ideal*1.3),160,224,320,448)})
    integral=cv2.integral(blocked.astype(np.uint8))
    candidates=[]
    for w in widths:
        h = min(height-4, max(96,min(512,round(len(text)*preferred*preferred*.8/max(1,w))+48)))
        if w*h>262144 or min(w,h)<20:
            continue
        mask=Image.new('L',(w,h))
        ImageDraw.Draw(mask).rectangle((6,6,w-7,h-7),fill=255)
        try:
            layer,proof,_=typography.layout_dialogue(text,font_data,coverage,mask,
                {**style,'max_size':preferred,'outline':outline})
        except ValueError:
            continue
        ink=layer.getchannel('A').getbbox()
        glyph=layer.crop(ink);tw,th=glyph.width+12,glyph.height+12
        if tw*th>131072:
            continue
        if tw>width-4 or th>height-4:continue
        # Exhaustive local rectangle occupancy avoids missing a fit between
        # obstacles merely because it falls between the old 24-pixel probes.
        occupied=integral[th:,tw:]-integral[:-th,tw:]-integral[th:,:-tw]+integral[:-th,:-tw]
        occupied[:2]=1;occupied[-2:]=1;occupied[:,:2]=1;occupied[:,-2:]=1
        ys,xs=np.where(occupied==0)
        if not len(xs):continue
        distances=(xs+tw/2-cx)**2+(ys+th/2-cy)**2
        best=int(np.argmin(distances));distance=math.sqrt(float(distances[best]))
        if distance>160:continue
        x,y=int(xs[best]),int(ys[best])
        # Location takes precedence over footprint savings: compactness alone
        # formerly preferred tall captions displaced into a neighboring panel.
        shape=max(0,th/max(tw,1)-1.35)*6000
        rank=(round(distance/12),500*(preferred-proof['size'])+shape+tw*th*.1,distance,y,x)
        candidates.append((rank,glyph,proof,(x,y),ink))
    n.require(candidates,'no readable source-centered patch avoids accepted edits and other text')
    _,glyph,proof,(x,y),ink=min(candidates,key=lambda c:c[0])
    tile=Image.new('RGBA',(glyph.width+12,glyph.height+12))
    tile.paste(glyph,(6,6))
    proof={**proof,'page_tile_offset':[x,y],'glyph_page_origin':[x+6,y+6],
        'original_layout_crop_bounds':list(ink),'placement_policy':POLICY}
    return tile,(x,y),proof


def finish(rgb, raw, rows, clean, text, erase, layout, font_data, coverage, style, *, probability=None,
           source=None, backend=None, budget=None, extents=None, font_resolver=None):
    """Remove pending translated sources even when every placement attempt fails."""
    started=time.monotonic();rows=deepcopy(rows)
    original=np.asarray(rgb);plate=np.asarray(clean).copy();text=text.copy()
    if probability is not None:
        rec.require(type(probability) is np.ndarray and probability.dtype==np.uint8
            and probability.shape==original.shape[:2],'native page text probability shape/type mismatch')
    erase=erase.copy();layout=layout.copy()
    # Spacing around accepted output is a placement constraint, not an immutable
    # strip of original pixels that can retain the next source's glyph edges.
    protected=erase | (np.asarray(text.getchannel('A'))!=0)
    clearance=p.dilate(protected,2)
    initial=protected.copy();before_text=np.asarray(text).copy();before_plate=plate.copy()
    geometry=aggressive.Geometry(raw,rgb.size)
    members={aggressive_key(m) for row in rows if row['status']=='automatic_composed'
        for m in row.get('binding',{}).get('member_ids',[])}
    summary={'policy':POLICY,'protection_policy':aggressive.PROTECTION_POLICY,'attempted':0,'removed':0,'placed':0,'unresolved':0,'work_pixels':0,
        'layout_work_pixels':0}
    cleaned_here=set()
    artifacts={};budget=budget or rec.Budget();backend_error=None;extents=extents or {}
    summary.update(learned_calls=0,learned_failures=0,learned_seconds=0.,selected_learned=0,
        limits={'learned_calls':budget.maximum_calls,'learned_seconds':budget.maximum_seconds})
    for i,row in enumerate(rows):
        if row['status']=='automatic_composed' or 'binding' not in row or row.get('reason_code') in aggressive.HARD_REASONS:
            continue
        entry={'policy':POLICY,'status':'pending','prior_reason':row.get('reason_code')}
        row['removal_fallback']=entry
        ids={aggressive_key(m) for m in row['binding']['member_ids']}
        if ids & members:
            entry.update(status='blocked',reason='duplicate_source_membership');continue
        if summary['attempted']>=MAX_REGIONS:
            entry.update(status='blocked',reason='region_budget');continue
        summary['attempted']+=1
        try:
            # Font validation precedes placement, but cannot veto cleanup.
            refinement=extents.get(row['unit_id'])
            bounds,own,obstacle=geometry.local(row['binding']['member_ids'],protected,margin=32,
                maximum_side=512 if refinement else 2048,source_priority=True,
                cleanup_polygon=refinement['polygon'] if refinement else None)
            x,y,r,b=bounds;crop=original[y:b,x:r]
            n.budget_check(summary['work_pixels']+own.size<=MAX_WORK,'removal fallback work budget')
            summary['work_pixels']+=own.size
            mask=extent(crop,own,obstacle)
            prefix=f'recovery-final-{i:03d}'
            if probability is not None:
                try:
                    mask,entry['support_evidence']=source_support.derive(crop,own,obstacle,probability[y:b,x:r])
                except ValueError as error:
                    entry['support_failure']=str(error)
            if refinement:
                mask=source_extents.apply(mask,refinement,bounds,obstacle)
                entry['cleanup_extent']=refinement
            n.require(mask.any(),'source fully overlaps protected output')
            candidate,method=aggressive.cheap(crop,mask,obstacle)
            if (method=='classical_telea_unverified' and ('support_evidence' in entry or refinement is not None)
                    and source is not None and backend is not None and backend_error is None
                    and max(crop.shape[:2])<=512 and budget.available()):
                # Fresh inference on the CURRENT support, after earlier edits have
                # been protected. Do not recycle rejected broad-mask candidates.
                inference=mask
                if probability is not None:
                    inference,entry['inference_context']=source_support.inference_context(mask,probability[y:b,x:r])
                request=rec.request(source,row['binding'],bounds,crop,inference,mask,obstacle,
                    attempt=0,model=rec.model_identity(backend.config))
                artifacts[prefix+'-request.json']=rec.canonical(request).encode()
                artifacts[prefix+'-inference-mask.png']=Image.fromarray(inference.astype(np.uint8)*255)
                artifacts[prefix+'-protected-mask.png']=Image.fromarray(obstacle.astype(np.uint8)*255)
                budget.calls+=1;summary['learned_calls']+=1;clock=time.monotonic()
                try:
                    result=backend.reconstruct(request,crop,inference,mask,obstacle,
                        timeout=min(30,max(.1,budget.maximum_seconds-budget.seconds)))
                    candidate=rec.compose(request,crop,inference,mask,obstacle,result)
                    artifacts[prefix+'-backend.png']=Image.fromarray(result['pixels'])
                    entry['backend_result']={k:v for k,v in result.items() if k!='pixels'}
                    entry['request_revision']=request['revision']
                    method='source_supported_lama'
                    summary['selected_learned']+=1
                except rec.BindingError:raise
                except Exception as error:
                    backend_error=type(error).__name__+': '+str(error)[:1000]
                    entry['learned_failure']=backend_error;summary['learned_failures']+=1
                finally:
                    seconds=time.monotonic()-clock;budget.seconds+=seconds;summary['learned_seconds']+=seconds
            if method=='classical_telea_unverified' and refinement is None:
                outlined_source=outlined_ink(crop,mask)
                if outlined_source is not None:
                    mask,inference,entry['outlined_ink']=outlined_source
                    candidate=cv2.inpaint(crop,inference.astype(np.uint8),5,cv2.INPAINT_TELEA)
                    method='outlined_ink_inpaint'
            if method=='classical_telea_unverified':
                # Broad Telea masks produce conspicuous radial smears and copied
                # glyphs. At this final fallback prefer a plain, local surface.
                if 'support_evidence' in entry and refinement is None:
                    completion=complete_color_ink(crop,own,mask,obstacle)
                    if completion is not None:
                        mask,entry['color_ink_completion']=completion
                ring=p.dilate(mask,4)&~mask&~obstacle
                samples=crop[ring] if ring.any() else crop[own]
                color=np.median(samples,axis=0).astype(np.uint8)
                candidate=crop.copy();candidate[mask]=color
                method='local_surface_occlusion'
            # Do not adopt discarded learned cleanplates merely because their
            # bindings are valid. Visual trials showed copied/hallucinated type;
            # candidate artifacts remain available without becoming an oracle.
            plate[y:b,x:r][mask]=candidate[mask];erase[y:b,x:r]|=mask
            members.update(ids)
            entry.update(status='removed',method=method,bounds=bounds,commit_pixels=int(mask.sum()),
                protected_source_pixels=int((own&obstacle).sum()),source_removal_verified=False,
                artwork_loss_possible=True)
            summary['removed']+=1
            artifacts[prefix+'-original.png']=Image.fromarray(crop)
            artifacts[prefix+'-commit-mask.png']=Image.fromarray(mask.astype(np.uint8)*255)
            artifacts[prefix+'-cleanplate.png']=Image.fromarray(candidate)
            # Blocking deliberately excludes this row's freshly cleaned pixels.
            # Earlier cleanup and English remain immutable.
            all_bounds,_,blocked=geometry.local(row['binding']['member_ids'],clearance,margin=192,maximum_side=2048,source_priority=True)
            ax,ay,ar,ab=all_bounds
            n.budget_check(summary['layout_work_pixels']+blocked.size*7<=MAX_LAYOUT_WORK,'final layout work budget')
            summary['layout_work_pixels']+=blocked.size*7
            sx,sy,sw,sh=cv2.boundingRect(own.astype(np.uint8))
            surface=0 if np.median(crop[own])<80 else 255
            if font_resolver is not None:font_data,coverage=font_resolver(row)
            frame=(x+sx-ax,y+sy-ay,x+sx+sw-ax,y+sy+sh-ay)
            colored = np.mean(np.ptp(crop,axis=2)>24) >= .2
            outlined = False
            if colored:
                try:
                    layer,xy,proof=patch_layout(row['binding']['text'],font_data,coverage,
                        frame,blocked,{**style,'fill':[255-surface]*3,'outline_fill':[surface]*3},outline=2)
                    outlined = True
                except n.ResourceLimit:
                    raise
                except ValueError:
                    pass
            if not outlined:
                try:
                    layer,xy,proof=patch_layout(row['binding']['text'],font_data,coverage,
                        frame,blocked,{**style,'fill':[255-surface]*3})
                except n.ResourceLimit:
                    raise
                except ValueError:
                    # Only open former source geometry that overlaps this target.
                    current=[geometry.boxes[key] for key in ids]
                    overlapping={key for key in cleaned_here
                        if any(x1<geometry.boxes[key][2] and x2>geometry.boxes[key][0]
                            and y1<geometry.boxes[key][3] and y2>geometry.boxes[key][1]
                            for x1,y1,x2,y2 in current)}
                    if not overlapping:
                        raise
                    # An earlier large title may enclose this smaller caption.
                    # That title's former geometry can now receive outlined type.
                    # Keep accepted edits, all actual lettering tiles, and every
                    # still-unprocessed neighbor protected.
                    occupied=p.dilate(initial|layout|(np.asarray(text.getchannel('A'))!=0),2)
                    _,_,overlay_blocked=geometry.local(row['binding']['member_ids'],occupied,
                        margin=192,maximum_side=2048,source_priority=True,cleaned_for_layout=overlapping)
                    n.budget_check(summary['layout_work_pixels']+overlay_blocked.size*7<=MAX_LAYOUT_WORK,
                        'final layout work budget')
                    summary['layout_work_pixels']+=overlay_blocked.size*7
                    layer,xy,proof=patch_layout(row['binding']['text'],font_data,coverage,
                        frame,overlay_blocked,{**style,'fill':[255-surface]*3,'outline_fill':[surface]*3},outline=2)
                    outlined=True
                    entry['placed_over_prior_cleanup']=True
            px,py=xy;px+=ax;py+=ay;h,w=layer.height,layer.width
            # The source commit above always survives. On painted backgrounds
            # a contrasting outline leaves the space between English glyphs visible.
            if not outlined:
                plate[py:py+h,px:px+w]=surface;erase[py:py+h,px:px+w]=True
            text.alpha_composite(layer,dest=(px,py));layout[py:py+h,px:px+w]=True
            proof.update(page_tile_offset=[px,py],glyph_page_origin=[px+6,py+6])
            row.update(status='automatic_composed',reason_code=None,reason=None,render_quality='best_effort',
                cleanup_method='removal_with_outlined_target' if outlined else 'removal_with_opaque_backing',layout=proof)
            entry.update(status='removed_and_placed')
            entry['outlined_tile_bounds' if outlined else 'backing_bounds']=[px,py,px+w,py+h]
            artifacts[prefix+'-text.png']=layer
            summary['placed']+=1
        except rec.BindingError:raise
        except (ValueError,OSError,cv2.error) as error:
            entry['reason']=str(error)
            if entry['status']=='removed':
                row.update(status='source_removed',source_visibility='source_removed_unverified')
            else:entry['status']='blocked'
            summary['unresolved']+=1
        if entry['status'] in ('removed','removed_and_placed'):
            cleaned_here.update(ids)
        # Include both removal and backing even if placement was unavailable.
        protected=erase | (np.asarray(text.getchannel('A'))!=0)
        clearance=p.dilate(protected,2)
    clean=Image.fromarray(plate)
    composite=Image.alpha_composite(clean.convert('RGBA'),text).convert('RGB')
    n.require(np.array_equal(plate[initial],before_plate[initial]) and
        np.array_equal(np.asarray(text)[initial],before_text[initial]),'fallback overwrote accepted edits')
    alpha=np.asarray(text.getchannel('A'))!=0
    n.verify_pixel_audit(n.pixel_audit(rgb,clean,composite,erase,alpha,~(erase|layout)))
    summary['seconds']=time.monotonic()-started
    return rows,clean,text,composite,erase,layout,artifacts,summary


def aggressive_key(value):
    return type(value).__name__,value
