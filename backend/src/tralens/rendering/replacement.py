"""Source-centered fallback for bound translations that precise routes cannot fit.

Cleanup is committed only after English placement succeeds. Original geometry,
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


POLICY = 'source_centered_atomic_replacement_v8'
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


def light_printed_surface(crop, own, protected):
    """Find light print on a connected midtone sign without touching outlines or adjacent objects."""
    if max(own.shape) > 1024 or own.sum() < 200:
        return None
    gray = crop[:,:,0]
    tone = int(np.bincount(gray[own], minlength=256).argmax())
    if not 48 <= tone <= 210:
        return None
    neutral = np.ptp(crop,axis=2) <= 5
    paper = neutral & (np.abs(gray.astype(np.int16) - tone) <= 3)
    bright = own & neutral & (gray > tone + 20)
    if (paper[own].mean() < .52 or neutral[own].mean() < .95 or bright[own].mean() < .08
            or (gray[own] < tone - 12).mean() > .04):
        return None
    _, _, components, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
    areas = components[1:, cv2.CC_STAT_AREA]
    if np.count_nonzero((areas >= 4) & (areas < own.sum() * .2)) < 4:
        return None
    # Bridge narrow light strokes in the paper. Dark borders and drawing lines
    # remain barriers even if closing would otherwise join both sides.
    field = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19,19))).astype(bool)
    field &= ~p.dilate((gray < tone - 12) | ~neutral, 2)
    _, labels = cv2.connectedComponents(field.astype(np.uint8), connectivity=4)
    samples = labels[own & paper & field]
    if not samples.size:
        return None
    field = (labels == np.bincount(samples).argmax()) & ~protected
    ink = field & p.dilate(own,7) & (gray > tone + 3)
    if ink[bright].mean() < .90:
        return None
    return ink, field, tone


def printed_surface_layout(text, font, coverage, bounds, paper, style):
    try:
        return patch_layout(text, font, coverage, bounds, ~paper, style)
    except n.ResourceLimit:
        raise
    except ValueError:
        if '\n' not in text:
            raise
    # A ruled agenda may fit its irregular paper even when no blank
    # rectangle encloses all the lines and their padding.
    mask = Image.fromarray(paper.astype(np.uint8)*255)
    layer, proof, _ = typography.layout_dialogue(text, font, coverage, mask,
        {**style, 'leading':0, 'outline':0})
    x, y, r, b = layer.getchannel('A').getbbox()
    tile = (x-6, y-6, r+6, b+6)
    n.require(tile[0]>=0 and tile[1]>=0 and tile[2]<=layer.width and tile[3]<=layer.height,
              'printed sign lettering too close to the crop boundary')
    return layer.crop(tile), tile[:2], proof


def compact_paper_label(crop, own, support, protected, text, font, coverage, style):
    """Set short text on its own uniform paper when a rectangular backing cannot fit."""
    if max(own.shape)>512 or len(text)>80 or not own.any():
        return
    gray = crop[:,:,0].astype(np.int16)
    tone = int(np.bincount(gray[own],minlength=256).argmax())
    neutral = np.ptp(crop,axis=2)<=5
    paper = neutral & (abs(gray-tone)<=3)
    if paper[own].mean()<.55 or neutral[own].mean()<.98:
        return
    dark = np.sum(own & (gray<tone-24)) >= np.sum(own & (gray>tone+24))
    strokes = neutral & ((gray<tone-3) if dark else (gray>tone+3))
    _,labels,stats,_ = cv2.connectedComponentsWithStats(np.uint8(strokes))
    areas = stats[:,cv2.CC_STAT_AREA]
    def counts(mask):
        return np.bincount(labels[mask],minlength=len(areas))
    # A neighboring column can partially enter the detector box. Only erase
    # complete components that the cleanup mask also identifies as lettering.
    selected = (counts(own)==areas) & (counts(support)>=.65*areas) & (counts(protected)==0)
    selected[np.unique(np.r_[labels[0], labels[-1], labels[:,0], labels[:,-1]])] = False
    selected[0] = False
    ink = selected[labels]
    if selected.sum()<2 or ink.sum()<.55*np.count_nonzero(own & strokes):
        return
    open_paper = (paper | ink) & p.dilate(own,12) & ~protected
    _,fields = cv2.connectedComponents(np.uint8(open_paper),connectivity=4)
    field = fields==np.bincount(fields[ink]).argmax()
    if not field[ink].all():
        return
    field = cv2.erode(np.uint8(field),np.ones((3,3),np.uint8),
                      borderType=cv2.BORDER_CONSTANT,borderValue=0)
    # The source angle matters on slanted book covers. Only rotate elongated
    # horizontal labels for which the detector actually measured a tilt.
    _,(rw,rh),angle = cv2.minAreaRect(cv2.findNonZero(np.uint8(own)))
    angle = -(angle if rw>=rh else angle-90)
    if max(rw,rh)<=1.6*min(rw,rh) or not 3<abs(angle)<30:
        angle = 0
    _,_,width,height = cv2.boundingRect(field)
    cy,cx = np.mean(np.nonzero(ink),axis=1)
    face = {**style,'fill':[0 if tone>=140 else 255]*3,'outline':0,'max_size':style['min_size']}
    best = None
    for trial_width in dict.fromkeys(round(width*f) for f in (1,.75,.55,1.25,1.5)):
        if trial_width<2 or height<2:
            continue
        try:
            layer,proof = n.layout(text,font,coverage,Image.new('L',(trial_width,height),255),face)
        except n.ResourceLimit:
            raise
        except ValueError:
            continue
        layer = layer.crop(layer.getchannel('A').getbbox())
        for aspect in (1,.9,.8,.72,.66):
            glyph = layer if aspect==1 else layer.resize((max(1,round(layer.width*aspect)),layer.height),Image.Resampling.LANCZOS)
            if angle:
                glyph = glyph.rotate(angle,Image.Resampling.BICUBIC,expand=True)
                glyph = glyph.crop(glyph.getchannel('A').getbbox())
            w,h = glyph.size
            if w>width or h>height:
                continue
            alpha = np.uint8(np.asarray(glyph.getchannel('A'))!=0)
            allowed = cv2.matchTemplate(1-field,alpha,cv2.TM_CCORR)<.5
            ys,xs = np.nonzero(allowed)
            # The compositor uses a six-pixel transparent margin around a tile.
            valid = (xs>=6)&(ys>=6)&(xs+w+6<=crop.shape[1])&(ys+h+6<=crop.shape[0])
            if not valid.any():
                continue
            xs,ys = xs[valid],ys[valid]
            distance = (xs+w/2-cx)**2+(ys+h/2-cy)**2
            i = distance.argmin()
            # Prefer ordinary proportions whenever they fit; extra-condensed
            # lettering can rescue an unbreakable word in a narrow column.
            rank = (aspect<.8,distance[i]+(1-aspect)*width)
            if best is None or rank<best[0]:
                best = rank,glyph,(int(xs[i])-6,int(ys[i])-6),dict(proof,
                    printed_rotation=float(angle),horizontal_scale=aspect)
    if best:
        _,glyph,position,proof = best
        tile = Image.new('RGBA',(glyph.width+12,glyph.height+12))
        tile.paste(glyph,(6,6))
        return ink,tone,tile,position,proof


def flat_caption(crop, own, protected, text, font, coverage, style):
    """Fit one line on tinted paper and retain every non-lettering pixel."""
    x, y, w, h = cv2.boundingRect(own.astype(np.uint8))
    if (h < 10 or w < 6*h or max(own.shape) > 512 or '\n' in text
            or len(text) > 120 or style.get('rotation')):
        return None
    tone = np.median(crop[own], axis=0).astype(int)
    if np.ptp(tone) < 10 or tone.min() < 150:
        return None
    difference = tone - crop.astype(np.int16)
    paper = np.abs(difference).max(2) <= 5
    dark = difference.mean(2) > 55
    if paper[own].mean() < .55 or dark[own].mean() < .045:
        return None
    # Reject an ink component that continues past the OCR or touches prior work.
    _, labels = cv2.connectedComponents((~paper).astype(np.uint8))
    letters = np.setdiff1d(np.unique(labels[own & dark]), np.unique(labels[~p.dilate(own,2) | protected]))
    if len(letters) < 4:
        return None
    ink = np.isin(labels, letters)
    if ink[own & dark].mean() < .97:
        return None
    band = np.zeros_like(own)
    band[max(0,y-h//3):y+h+h//3, max(0,x-w//3):x+w+w//3] = True
    usable = band & ~p.dilate((~paper & ~ink) | protected, 2)
    size = style['min_size']
    fill = np.median(crop[ink & dark], axis=0).astype(int).tolist()
    trial = Image.new('L', (crop.shape[1]-8, size*3), 255)
    layer, proof = n.layout(text, font, coverage, trial,
        {**style, 'min_size':size, 'max_size':size, 'outline':0, 'fill':fill})
    if '\n' in proof['draw_text']:
        return None
    glyph = layer.crop(layer.getchannel('A').getbbox())
    alpha = np.asarray(glyph.getchannel('A')) != 0
    if glyph.height+12 > crop.shape[0] or glyph.width+12 > crop.shape[1]:
        return None
    overlaps = cv2.matchTemplate((~usable).astype(np.float32), alpha.astype(np.float32), cv2.TM_CCORR)
    ys, xs = np.where(overlaps[6:-6,6:-6] < .5)
    if not len(xs):
        return None
    best = np.argmin((xs+6+glyph.width/2-x-w/2)**2 + (ys+6+glyph.height/2-y-h/2)**2)
    left, top = int(xs[best]), int(ys[best])
    if not usable[top+6:top+6+glyph.height,left+6:left+6+glyph.width][alpha].all():
        return None
    tile = Image.new('RGBA', (glyph.width+12, glyph.height+12))
    tile.paste(glyph, (6,6))
    return ink, tone, tile, (left,top), proof


def outlined_ink(crop, broad):
    """Find small light glyph interiors enclosed by dark ink on a painted page."""
    if max(broad.shape)>1024 or np.mean(np.ptp(crop,axis=2)>24)<.2:
        return None
    brightness=crop.max(axis=2)
    dark=brightness<110
    white=(crop.min(axis=2)>178)&(np.ptp(crop,axis=2)<40)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    contours,_=cv2.findContours(white.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>8192:
        return None
    cores=np.zeros(broad.shape,np.uint8)
    outline_histogram=np.zeros(256,np.int64)
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
        outline_histogram+=np.bincount(brightness[local][ring & dark[local]],minlength=256)
        count+=1
        cores[local]|=body
    if count<8:
        return None
    nearby=p.dilate(cores!=0,6)
    ink=dark&broad
    supported=np.count_nonzero(nearby&ink)/max(1,np.count_nonzero(ink))
    if supported<.62:
        return None
    # Away from a light glyph core, require the darkness of the measured
    # outlines. A shaded costume can otherwise look like one giant stroke.
    upper_quartile=np.searchsorted(outline_histogram.cumsum(), .75*outline_histogram.sum())
    remote_ceiling=min(109,int(upper_quartile)+12)
    remote_ink=(brightness<=remote_ceiling)&broad
    strokes=p.dilate((dark&nearby)|remote_ink|(cores!=0),2)
    commit=strokes&broad
    if not commit.any() or commit.sum()>.90*broad.sum():
        return None
    return commit,strokes,{'enclosed_light_components':count,'supported_dark_fraction':supported,
        'remote_outline_max_rgb':remote_ceiling,'original_pixels':int(broad.sum()),'committed_pixels':int(commit.sum())}


def painted_panel_limits(pixels, frame):
    """Find broad white gutters crossing the source along each axis."""
    left,top,right,bottom = frame
    white = pixels.min(2) >= 245
    h,w = white.shape
    limits = [0,0,w,h]
    for axis, profile, start, end in (
            (0,white[max(top-8,0):min(bottom+8,h)].mean(0),left,right),
            (1,white[:,max(left-8,0):min(right+8,w)].mean(1),top,bottom)):
        changes = np.flatnonzero(np.diff(np.r_[False,profile>.98,False]))
        for a,b in changes.reshape(-1,2):
            if b-a<5:
                continue
            if b<=start:
                limits[axis] = max(limits[axis],int(b))
            if a>=end:
                limits[axis+2] = min(limits[axis+2],int(a))
    # A painted word can straddle the gutter. Look across the group as
    # well as the individual crop, with the word's center inside its panel.
    neutral = white & (np.ptp(pixels,axis=2)<8)
    for axis,profile,local,center in (
            (0,neutral.mean(0),neutral[top:bottom].mean(0),(left+right)//2),
            (1,neutral.mean(1),neutral[:,left:right].mean(1),(top+bottom)//2)):
        changes = np.flatnonzero(np.diff(np.r_[False,(profile>.72)&(local>.60),False]))
        for a,b in changes.reshape(-1,2):
            if not 5<=b-a<=80:
                continue
            if b<=center:
                limits[axis]=max(limits[axis],int(b))
            if a>=center:
                limits[axis+2]=min(limits[axis+2],int(a))
    return limits


def patch_layout(text, font_data, coverage, bounds, blocked, style, *, outline=0):
    """Fit the bound text into an obstacle-free rectangle near its source."""
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
    probes = [(w, preferred) for w in widths]
    if floor < preferred:
        # The typography mask is a blank trial rectangle. Its preferred size may
        # fit that trial yet never fit the actual space among neighboring marks.
        # Try the requested floor at the source width and a compact dialogue width.
        source_width = right-left
        probes.extend((w, floor) for w in sorted({max(64,min(width-4,v)) for v in
                     (source_width+16, round(source_width*.72), round(ideal*floor/preferred))}))
    for w,cap in probes:
        if cap < preferred and any(candidate[2]['size'] > floor for candidate in candidates):
            break
        h = min(height-4, max(96,min(512,round(len(text)*cap*cap*.8/max(1,w))+48)))
        if w*h>262144 or min(w,h)<20:
            continue
        mask=Image.new('L',(w,h))
        ImageDraw.Draw(mask).rectangle((6,6,w-7,h-7),fill=255)
        try:
            layer,proof,_=typography.layout_dialogue(text,font_data,coverage,mask,
                {**style,'max_size':cap,'outline':outline})
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
           source=None, backend=None, budget=None, extents=None, font_resolver=None, repetitions=None, max_title_size=None):
    """Replace pending translated sources when a local target layout succeeds."""
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
    geometry=aggressive.Geometry(raw,rgb.size,original)
    members={aggressive_key(m) for row in rows if row['status']=='automatic_composed'
        for m in row.get('binding',{}).get('member_ids',[])}
    summary={'policy':POLICY,'protection_policy':aggressive.PROTECTION_POLICY,'attempted':0,'removed':0,'placed':0,'unresolved':0,'work_pixels':0,
        'layout_work_pixels':0}
    cleaned_here=set()
    artifacts={};budget=budget or rec.Budget();backend_errors={};extents=extents or {}
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
            # Prepare the source mask without mutating any shared page layer.
            refinement=extents.get(row['unit_id'])
            bounds,own,obstacle=geometry.local(row['binding']['member_ids'],protected,margin=32,
                maximum_side=512 if refinement else 2048,source_priority=True,
                cleanup_polygon=refinement['polygon'] if refinement else None)
            x,y,r,b=bounds;crop=original[y:b,x:r]
            n.budget_check(summary['work_pixels']+own.size<=MAX_WORK,'removal fallback work budget')
            summary['work_pixels']+=own.size
            from ..ocr.handwritten import stroke_masks
            handwriting = (stroke_masks(original,bounds,geometry.raw[next(iter(ids))],own,obstacle)
                           if len(ids)==1 else None)
            from . import printed_contours
            title = geometry.raw[next(iter(ids))].get('ocr_metadata',{}).get('colored_word') if len(ids)==1 else None
            title_paint = printed_contours.colored_title(crop,own,obstacle,title['hue']) if title else None
            mask=extent(crop,own,obstacle)
            prefix=f'recovery-final-{i:03d}'
            from . import repeated_strokes
            repeated = repeated_strokes.local(repetitions,row['binding']['member_ids'],bounds,crop,obstacle)
            hollow_inference = None
            if repeated is not None:
                mask, _ = repeated
                entry['support_evidence'] = {'policy':'recurring_print_strokes'}
            elif handwriting is not None:
                core,halo = handwriting
                mask = halo if backend is not None else core
                entry['support_evidence'] = {'policy':'outlined_handwriting'}
            elif probability is not None:
                try:
                    mask,entry['support_evidence']=source_support.derive(crop,own,obstacle,probability[y:b,x:r])
                except ValueError as error:
                    entry['support_failure']=str(error)
                if all(geometry.raw[k].get('reason')=='corroborated_repeated_effect' for k in ids):
                    strokes = source_support.hollow_effect_masks(crop,own,obstacle,probability[y:b,x:r])
                    if strokes is not None:
                        mask,hollow_inference = strokes
                        entry['support_evidence'] = {'policy':'measured_hollow_effect'}
            single = geometry.raw[next(iter(ids))].get('ocr_metadata',{}) if len(ids)==1 else {}
            if probability is not None and single.get('handwritten_aside'):
                # Rebuild the optically verified strokes on the original page.
                # At this stage neighboring translations may already cover
                # part of the lettering. Only clean the remaining ink.
                strokes = printed_contours.diagonal_aside_mask(
                    crop,bounds,probability[y:b,x:r],geometry.raw[next(iter(ids))],np.zeros_like(obstacle))
                if strokes is not None:
                    mask = hollow_inference = strokes & ~obstacle
                    entry['support_evidence'] = {'policy':'verified_diagonal_aside'}
            if any(k in single for k in ('outlined_ellipsis','hollow_print','heavy_print','outlined_display')):
                from .printed_contours import outlined_word_mask
                rim = outlined_word_mask(crop,bounds,single,obstacle)
                if rim is not None and rim.any():
                    mask = hollow_inference = rim
                    entry['support_evidence'] = {'policy':'verified_outlined_word'}
            if refinement and handwriting is None and repeated is None:
                mask=source_extents.apply(mask,refinement,bounds,obstacle)
                entry['cleanup_extent']=refinement
            n.require(mask.any(),'source fully overlaps protected output')
            candidate,method=aggressive.cheap(crop,mask,obstacle)
            if method=='classical_telea_unverified' and entry.get('support_evidence',{}).get('policy')=='verified_diagonal_aside':
                # A tightly traced pen stroke has little white inside it.
                # Sample its collar so a model cannot copy adjacent kana back.
                collar=p.dilate(mask,4)&~p.dilate(mask,1)&~obstacle
                if collar.sum()>=32 and np.mean(crop[collar].min(1)>=250)>=.94:
                    candidate=crop.copy()
                    candidate[mask]=np.median(crop[collar],axis=0)
                    method='dominant_surface_fill'
            if title_paint is not None:
                mask,candidate,_ = title_paint
                method='colored_title_paper'
            if handwriting is not None:
                candidate = crop.copy()
                candidate[core] = 255
                method = 'outlined_handwriting_fill'
            if ((backend is None or backend.config.get('backend')=='auto') and method=='classical_telea_unverified'
                    and entry.get('support_evidence',{}).get('painted_stroke_support')):
                painted = source_support.painted_stroke_fill(crop,mask,obstacle,slanted=True)
                if painted is not None:
                    candidate,method = painted,'local_painted_strokes'
            selected = rec.region_backend(backend,crop,mask)
            if (method in ('classical_telea_unverified','outlined_handwriting_fill') and ('support_evidence' in entry or refinement is not None)
                    and source is not None and selected is not None and selected.config.get('backend') not in backend_errors
                    and max(crop.shape[:2])<=512 and budget.available()):
                # Fresh inference on the CURRENT support, after earlier edits have
                # been protected. Do not recycle rejected broad-mask candidates.
                inference=mask
                if repeated is not None:
                    inference = repeated[1]
                elif hollow_inference is not None:
                    inference = hollow_inference
                elif probability is not None and handwriting is None:
                    inference,entry['inference_context']=source_support.inference_context(mask,probability[y:b,x:r])
                request=rec.request(source,row['binding'],bounds,crop,inference,mask,obstacle,
                    attempt=0,model=rec.model_identity(selected.config))
                artifacts[prefix+'-request.json']=rec.canonical(request).encode()
                artifacts[prefix+'-inference-mask.png']=Image.fromarray(inference.astype(np.uint8)*255)
                artifacts[prefix+'-protected-mask.png']=Image.fromarray(obstacle.astype(np.uint8)*255)
                previous_calls=budget.calls;budget.calls+=1;clock=time.monotonic()
                try:
                    result=selected.reconstruct(request,crop,inference,mask,obstacle,
                        timeout=min(request['limits']['seconds'],max(.1,budget.maximum_seconds-budget.seconds)))
                    request,inference,result = rec.review_outline(backend,request,result,crop,inference,mask,obstacle,
                        budget,clock + budget.maximum_seconds - budget.seconds)
                    if result.get('outline_review', {}).get('selected') == 'lama-onnx-512':
                        artifacts[prefix+'-request.json']=rec.canonical(request).encode()
                        artifacts[prefix+'-inference-mask.png']=Image.fromarray(inference.astype(np.uint8)*255)
                    candidate=rec.compose(request,crop,inference,mask,obstacle,result)
                    artifacts[prefix+'-backend.png']=Image.fromarray(result['pixels'])
                    entry['backend_result']={k:v for k,v in result.items() if k!='pixels'}
                    entry['request_revision']=request['revision']
                    method='source_supported_sdxl' if request['model'].get('backend')=='sdxl-inpaint' else 'source_supported_lama'
                except rec.BindingError:raise
                except Exception as error:
                    backend_error=type(error).__name__+': '+str(error)[:1000]
                    backend_errors[selected.config.get('backend')]=backend_error
                    entry['learned_failure']=backend_error;summary['learned_failures']+=1
                finally:
                    summary['learned_calls']+=budget.calls-previous_calls
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
                painted = source_support.painted_stroke_fill(crop,mask,obstacle) if 'support_evidence' in entry else None
                if painted is not None:
                    candidate,method = painted,'local_painted_strokes'
                else:
                    ring=p.dilate(mask,4)&~mask&~obstacle
                    samples=crop[ring] if ring.any() else crop[own]
                    color=np.median(samples,axis=0).astype(np.uint8)
                    candidate=crop.copy();candidate[mask]=color
                    method='local_surface_occlusion'
            # Do not adopt discarded learned cleanplates merely because their
            # bindings are valid. Visual trials showed copied/hallucinated type;
            # candidate artifacts remain available without becoming an oracle.
            entry.update(method=method,bounds=bounds,
                protected_source_pixels=int((own&obstacle).sum()),source_removal_verified=False,
                artwork_loss_possible=True)
            # Blocking deliberately excludes this row's freshly cleaned pixels.
            # Earlier cleanup and English remain immutable.
            all_bounds,layout_own,blocked=geometry.local(row['binding']['member_ids'],clearance,margin=192,maximum_side=2048,source_priority=True)
            ax,ay,ar,ab=all_bounds
            n.budget_check(summary['layout_work_pixels']+blocked.size*10<=MAX_LAYOUT_WORK,'final layout work budget')
            summary['layout_work_pixels']+=blocked.size*10
            sx,sy,sw,sh=cv2.boundingRect(own.astype(np.uint8))
            surface=0 if np.median(crop[own])<80 else 255
            if font_resolver is not None:font_data,coverage=font_resolver(row)
            frame=(x+sx-ax,y+sy-ay,x+sx+sw-ax,y+sy+sh-ay)
            layout_crop = original[ay:ab,ax:ar]
            sign = light_printed_surface(crop, own, obstacle) if repeated is None else None
            local_clearance = blocked[y-ay:b-ay,x-ax:r-ax]
            blocked = blocked | aggressive.placement_obstacles(layout_crop,layout_own)
            if repeated is not None:
                vertical,horizontal = geometry.rules
                blocked |= p.dilate((vertical[ay:ab,ax:ar] | horizontal[ay:ab,ax:ar]).astype(bool),3)
            colored = np.mean(np.ptp(crop,axis=2)>24) >= .2
            outlined = False
            printed_copy=any(geometry.raw[k].get('ocr_metadata',{}).get('repeated_printed_line') for k in ids)
            forum_body=any(geometry.raw[k].get('ocr_metadata',{}).get('forum_post',{}).get('phrases') for k in ids)
            if (not colored and (sw>512 and sw>3*sh or printed_copy and sw>2*sh or forum_body and len(ids)>=3)
                    and len(ids)>=2 and len(row['binding']['text'])>=80
                    and np.mean(candidate[own].min(1)>232)>.78):
                # A wide paragraph exceeds the learned model's crop width,
                # but its own lines can still provide enough clean paper.
                # Fit inside the contiguous free rectangle and avoid adjacent
                # speech balloons, panel rules and already placed lettering.
                paper = (candidate.min(2)>224) & ~local_clearance & ~obstacle
                hull=np.zeros_like(own,dtype=np.uint8)
                cv2.fillConvexPoly(hull,cv2.convexHull(cv2.findNonZero(np.uint8(own))),1)
                field=typography.interior_box(paper & (hull!=0))
                if field.getbbox():
                    try:
                        tile,proof=n.layout(row['binding']['text'],font_data,coverage,field,
                            {**style,'min_size':min(10 if printed_copy else 13,style['min_size']),
                             'outline':0,'fill':[0]*3,'leading':.10})
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        bb=tile.getchannel('A').getbbox()
                        layer=tile.crop(bb)
                        xy=(x-ax+bb[0],y-ay+bb[1])
                        outlined=True
                        entry['placement_frame']='original_printed_paragraph'
            if sign is not None and not outlined:
                sign_ink, field, tone = sign
                try:
                    layer,position,proof = printed_surface_layout(row['binding']['text'],font_data,coverage,
                        (sx,sy,sx+sw,sy+sh), field & ~local_clearance,
                        {**style,'fill':[25 if tone >= 138 else 255]*3})
                except n.ResourceLimit:
                    raise
                except ValueError:
                    pass
                else:
                    xy = (position[0]+x-ax, position[1]+y-ay)
                    mask,candidate = sign_ink,crop.copy()
                    candidate[mask] = tone
                    method = 'light_printed_surface'
                    entry.update(method=method,placement_frame='connected_printed_surface')
                    outlined = True
            if title_paint is not None and not outlined:
                # Match the measured title color. Leave the surrounding subtitle
                # and the character below available for their own placements.
                occupied = blocked[y-ay:b-ay,x-ax:r-ax]
                paper = (np.abs(candidate.astype(np.int16)-np.median(candidate[mask],axis=0)).max(2)<16)
                free = own & paper & ~occupied & ~obstacle
                frame_mask = typography.interior_box(free)
                if frame_mask.getbbox():
                    cap = min(max_title_size or style['max_size'],round(.55*(b-y)))
                    try:
                        title_layer,proof,_ = typography.layout_dialogue(row['binding']['text'],font_data,coverage,
                            frame_mask,{**style,'min_size':min(cap,style['min_size']),'max_size':cap,
                                        'fill':title_paint[2],'outline':min(5,max(2,round(cap*.035))),'outline_fill':[66,43,24]})
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        pass
                    else:
                        bb=title_layer.getchannel('A').getbbox()
                        layer=title_layer.crop(bb)
                        xy=(x-ax+bb[0],y-ay+bb[1])
                        outlined=True
                        entry['placement_frame']='original_colored_title'
            if not outlined and entry.get('support_evidence',{}).get('painted_stroke_support') and not style.get('rotation'):
                try:
                    local = blocked[y-ay:b-ay,x-ax:r-ax]
                    tile,proof,_ = aggressive.slanted_layout(row['binding']['text'],font_data,coverage,own,local,style)
                    box = tile.getchannel('A').getbbox()
                    layer = tile.crop(box)
                    xy = (x-ax+box[0],y-ay+box[1])
                    outlined = True
                    entry['placement_frame'] = 'original_slanted_lettering'
                except n.ResourceLimit:
                    raise
                except ValueError:
                    pass
            if (colored or repeated is not None) and not outlined:
                try:
                    layer,xy,proof=patch_layout(row['binding']['text'],font_data,coverage,
                        frame,blocked,{**style,'fill':[255-surface]*3,'outline_fill':[surface]*3},outline=2)
                    outlined = True
                except n.ResourceLimit:
                    raise
                except ValueError:
                    try:
                        caption = flat_caption(crop, own, obstacle | local_clearance,
                            row['binding']['text'],font_data,coverage,style)
                    except n.ResourceLimit:
                        raise
                    except ValueError:
                        caption = None
                    if caption is not None:
                        mask,tone,layer,position,proof = caption
                        candidate = crop.copy()
                        candidate[mask] = tone
                        xy = (x-ax+position[0], y-ay+position[1])
                        method = 'flat_printed_caption'
                        entry.update(method=method, placement_frame='original_caption_strip')
                        outlined = True
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
                        compact = compact_paper_label(crop,own,mask,obstacle | local_clearance,
                            row['binding']['text'],font_data,coverage,style)
                        if compact is None:
                            raise
                        mask,tone,layer,position,proof = compact
                        candidate = crop.copy()
                        candidate[mask] = tone
                        xy = (x-ax+position[0],y-ay+position[1])
                        method = 'compact_printed_label'
                        entry.update(method=method,placement_frame='connected_label_paper')
                        outlined = True
                    else:
                        # An earlier large title may enclose this smaller caption.
                        # That title's former geometry can now receive outlined type.
                        # Keep accepted edits, all actual lettering tiles, and every
                        # still-unprocessed neighbor protected.
                        occupied=p.dilate(initial|layout|(np.asarray(text.getchannel('A'))!=0),2)
                        _,_,overlay_blocked=geometry.local(row['binding']['member_ids'],occupied,
                            margin=192,maximum_side=2048,source_priority=True,cleaned_for_layout=overlapping)
                        n.budget_check(summary['layout_work_pixels']+overlay_blocked.size*10<=MAX_LAYOUT_WORK,
                            'final layout work budget')
                        summary['layout_work_pixels']+=overlay_blocked.size*10
                        overlay_blocked |= aggressive.placement_obstacles(layout_crop,layout_own)
                        layer,xy,proof=patch_layout(row['binding']['text'],font_data,coverage,
                            frame,overlay_blocked,{**style,'fill':[255-surface]*3,'outline_fill':[surface]*3},outline=2)
                        outlined=True
                        entry['placed_over_prior_cleanup']=True
            px,py=xy;px+=ax;py+=ay;h,w=layer.height,layer.width
            landing = plate[py:py+h,px:px+w].copy()
            # Evaluate the backing against the proposed cleanplate, including any
            # earlier commits outside this region's authorized mask.
            ix,iy,ir,ib=max(px,x),max(py,y),min(px+w,r),min(py+h,b)
            if ix<ir and iy<ib:
                part=np.s_[iy-y:ib-y,ix-x:ir-x]
                landing[iy-py:ib-py,ix-px:ir-px][mask[part]]=candidate[part][mask[part]]
            if not outlined:
                different = landing.min(2) < 250 if surface else landing.max(2) > 5
                if different.any():
                    alpha = np.asarray(layer.getchannel('A'))
                    rim = cv2.dilate(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                    edged = Image.new('RGBA', layer.size, (surface, surface, surface))
                    edged.putalpha(Image.fromarray(rim))
                    edged.alpha_composite(layer)
                    layer = edged
                    outlined = True
            plate[y:b,x:r][mask]=candidate[mask];erase[y:b,x:r]|=mask
            members.update(ids)
            cleaned_here.update(ids)
            entry['commit_pixels']=int(mask.sum())
            summary['removed']+=1
            summary['selected_learned']+=int(method in ('source_supported_lama', 'source_supported_sdxl'))
            artifacts[prefix+'-original.png']=Image.fromarray(crop)
            artifacts[prefix+'-commit-mask.png']=Image.fromarray(mask.astype(np.uint8)*255)
            artifacts[prefix+'-cleanplate.png']=Image.fromarray(candidate)
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
            entry['status']='blocked'
            summary['unresolved']+=1
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
