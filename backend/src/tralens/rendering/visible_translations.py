"""Draw bound translations after native composition, including source-local cleanup in replace-all mode."""
import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import compact_lettering, natural as n, preparation as p


POLICY = 'opaque_patch_then_associated_caption_v1'
WARNING = 'Prioritizes visible translations; fallback patches may cover artwork.'
MAX_ATTEMPTS = 32
MAX_WORK = 3_072_000_000
MAX_LOCAL_PIXELS = 262144
MAX_FOOTPRINT_PIXELS = 65536
MAX_DISTANCE = 144
PADDING = 6


def settings(policy):
    n.require(type(policy) is str and policy in ('replacement-only','always-show'), 'unknown translation visibility policy')


def display_counts(rows):
    counts = {name:sum(row.get('display_status')==status for row in rows)for name,status in (
        ('clean_replacements','clean'),('patch_backed_translations','patch'),('nearby_captions','caption'),
        ('intentionally_unchanged','unchanged'),('available_not_displayed','undisplayed'),('missing_translations','missing'))}
    if any('reconstruction' in row or row.get('display_status') == 'aggressive' for row in rows):
        counts.update(learned_reconstructions=sum(row.get('display_status')=='learned' for row in rows),
            aggressive_replacements=sum(row.get('display_status')=='aggressive' for row in rows))
    return counts


def composed_status(row):
    if row.get('cleanup_method') == 'outlined_sound_group':
        return 'aggressive'
    if row.get('removal_fallback',{}).get('status')=='removed_and_placed':
        return 'aggressive'
    if row.get('reconstruction',{}).get('status')=='composed_unverified':
        return 'learned' if row.get('cleanup_method') in ('learned_lama','learned_sdxl') else 'aggressive'
    return 'clean'


def key(value):
    return type(value).__name__, value


def _text_tile(text, font_data, coverage, width, height, style, work):
    """Use existing character-preserving layout; never its emergency small type."""
    n.budget_check(width*height <= MAX_LOCAL_PIXELS and min(width,height) > 2*PADDING,
                   'visible translation local layout budget')
    mask = Image.new('L',(width,height))
    ImageDraw.Draw(mask).rectangle((PADDING,PADDING,width-PADDING-1,height-PADDING-1),fill=255)
    layer, proof = n.layout(text,font_data,coverage,mask,{**style,'fill':[0,0,0],'outline':0},work=work)
    n.require(proof['original_text'] == text and proof['size'] >= style['min_size'], 'visible translation text/font drift')
    bounds = layer.getchannel('A').getbbox()
    n.require(bounds is not None,'empty visible translation')
    glyph = layer.crop(bounds)
    tile = Image.new('RGBA',(glyph.width+2*PADDING,glyph.height+2*PADDING))
    tile.paste(glyph,(PADDING,PADDING))
    proof.update(display_glyph_origin=[PADDING,PADDING], display_glyph_size=list(glyph.size),
                 original_layout_crop_bounds=list(bounds))
    return tile, proof


def _backing(size, caption):
    layer = Image.new('RGBA',size,(255,248,219,255) if caption else (255,255,255,255))
    ImageDraw.Draw(layer).rectangle((0,0,size[0]-1,size[1]-1),outline=(114,83,22,255),width=1)
    return layer


def _free(mask, blocked, x, y):
    h,w = mask.shape
    return (x >= 2 and y >= 2 and x+w <= blocked.shape[1]-2 and y+h <= blocked.shape[0]-2
            and not (mask & blocked[y:y+h,x:x+w]).any())


def place(rgb, source_bounds, own, blocked, text, font_data, coverage, style, *, protect_artwork=True):
    """Return local patch/text tiles and an explicit page offset/footprint."""
    left,top,right,bottom = source_bounds
    cx,cy = (left+right)//2,(top+bottom)//2
    if protect_artwork:
        # A source rectangle or a gap between OCR boxes does not imply blank art.
        # Limit the extra scan to the neighborhood reachable by these tiles.
        pad = 330
        ax, ay = max(0,left-pad), max(0,top-pad)
        ar, ab = min(rgb.width,right+pad), min(rgb.height,bottom+pad)
        from .aggressive import placement_obstacles
        blocked = blocked.copy()
        blocked[ay:ab,ax:ar] |= placement_obstacles(np.asarray(rgb.crop((ax,ay,ar,ab))),own[ay:ab,ax:ar])
    work = n.LayoutWork()
    attempts = []
    try:
        width,height = right-left+2*PADDING,bottom-top+2*PADDING
        if width*height > MAX_LOCAL_PIXELS or min(width,height) <= 2*PADDING:
            raise ValueError('source-local patch geometry requires a caption')
        text_tile, proof = _text_tile(text,font_data,coverage,width,height,style,work)
        w,h = text_tile.size
        mask = np.ones((h,w),bool)
        if w*h > MAX_FOOTPRINT_PIXELS:
            raise ValueError('source patch would be too large; try a local caption')
        positions = [(cx-w//2,cy-h//2),(cx-w//2,top-PADDING),(cx-w//2,bottom+PADDING-h)]
        for x,y in positions:
            if _free(mask,blocked,x,y):
                return 'patch',_backing((w,h),False),text_tile,(x,y),proof,attempts
        attempts.append({'route':'patch','reason':'known_neighbor_or_prior_edit_collision'})
    except n.ResourceLimit:
        raise
    except ValueError as error:
        attempts.append({'route':'patch','reason':str(error)})
    # Captions use the requested floor, not ever-smaller text. Bounded shapes are
    # cropped to actual glyph extents plus padding, not a whole balloon/panel.
    widths = sorted({min(rgb.width-4,w) for w in (128,192,256,320)})
    caption_blocked = blocked | own
    candidates = []
    for width in widths:
        try:
            tile,proof = _text_tile(text,font_data,coverage,width,min(256,rgb.height-4),
                {**style,'max_size':style['min_size']},work)
        except n.ResourceLimit:
            raise
        except ValueError as error:
            attempts.append({'route':'caption','width':width,'reason':str(error)})
            continue
        w,h = tile.size
        # No arbitrary page placement: captions stay beside their exact source,
        # with an explicit visible leader. Known source text is not covered by
        # caption backing (the small leader may touch its own source anchor).
        positions = [(right+8,cy-h//2),(left-w-8,cy-h//2),
            (cx-w//2,bottom+8),(cx-w//2,top-h-8),
            (right+8,top),(left-w-8,top),(right+8,bottom-h),(left-w-8,bottom-h)]
        for x,y in positions:
            body = np.ones((h,w),bool)
            if not _free(body,caption_blocked,x,y):
                continue
            ex,ey = max(x,min(cx,x+w-1)),max(y,min(cy,y+h-1))
            sx,sy = max(left,min(ex,right-1)),max(top,min(ey,bottom-1))
            distance = math.hypot(ex-sx,ey-sy)
            if not 0 < distance <= MAX_DISTANCE:
                continue
            bx,by = min(x,sx)-2,min(y,sy)-2
            br,bb = max(x+w,sx+3),max(y+h,sy+3)
            if (br-bx)*(bb-by) > MAX_LOCAL_PIXELS:
                continue
            patch = Image.new('RGBA',(br-bx,bb-by))
            draw = ImageDraw.Draw(patch)
            draw.line((sx-bx,sy-by,ex-bx,ey-by),fill=(114,83,22,255),width=2)
            draw.ellipse((sx-bx-1,sy-by-1,sx-bx+1,sy-by+1),fill=(114,83,22,255))
            patch.alpha_composite(_backing((w,h),True),dest=(x-bx,y-by))
            mask = np.asarray(patch.getchannel('A')) != 0
            if int(mask.sum()) > MAX_FOOTPRINT_PIXELS or not _free(mask,blocked,bx,by):
                continue
            lettering = Image.new('RGBA',patch.size)
            lettering.paste(tile,(x-bx,y-by))
            # Prefer a compact nearby caption; geometry is not narrative order.
            rank = (w*h+12*distance, distance, y, x, width)
            candidates.append((rank,patch,lettering,(bx,by),{**proof,
                'caption_bounds':[x,y,x+w,y+h], 'leader':[[sx,sy],[ex,ey]]}))
    n.require(candidates,'no bounded readable patch/caption avoids known text, edits and page boundaries')
    _,patch,lettering,xy,proof = min(candidates,key=lambda item:item[0])
    return 'caption',patch,lettering,xy,proof,attempts


def overlap_owners(bounds, members, polygons, own, blocked, reserved):
    """Resolve overlapping OCR boxes locally. A tighter centered box can own a glyph."""
    x,y,r,b = bounds
    if (r-x)*(b-y) > MAX_LOCAL_PIXELS:
        return blocked
    yy,xx = np.ogrid[y:b,x:r]
    nearest = np.full((b-y,r-x),np.inf)
    winner = np.zeros(nearest.shape,bool)
    for identity, polygon in polygons.items():
        xs,ys = zip(*polygon)
        l,t,right,bottom = min(xs),min(ys),max(xs),max(ys)
        if right<x or bottom<y or l>=r or t>=b:
            continue
        w,h = max(1,right-l),max(1,bottom-t)
        cost = ((2*xx-l-right)/w)**2 + ((2*yy-t-bottom)/h)**2 + .25*math.log(w*h)
        local = np.zeros(nearest.shape,np.uint8)
        cv2.fillPoly(local,[np.asarray(polygon,np.int32)-[x,y]],1)
        take = (local!=0) & (cost<nearest)
        nearest[take] = cost[take]
        winner[take] = identity in members
    result = blocked.copy()
    roi = np.s_[y:b,x:r]
    result[roi] &= ~(winner & own[roi] & ~reserved[roi])
    return result


def outlined_color_plate(crop, source, protected):
    """On dark art, infer behind white-rimmed colored strokes as whole components."""
    if np.median(crop[source].max(1)) > 150:
        return None
    peak = crop.max(2)
    bright = (peak > 150).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bright,8)
    chroma = np.ptp(crop,axis=2)
    white = (crop.min(2)>178) & (chroma<60)
    color = chroma>75
    white_counts = np.bincount(labels[white], minlength=len(stats))
    color_counts = np.bincount(labels[color], minlength=len(stats))
    x,y,w,h,area = stats.T
    candidates = ((area>=8) & (white_counts>.20*area) & (color_counts>.08*area) &
                  (x>0) & (y>0) & (x+w<crop.shape[1]) & (y+h<crop.shape[0]))
    candidates[0] = False
    touched = np.bincount(labels[source],minlength=len(stats))
    seeds = candidates & (touched>.18*area)
    if not seeds.any():
        return None
    # Match nearby detached marks to the ink color of a confirmed stroke.
    hue = cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)[:,:,0]
    colors = np.bincount(hue[color & seeds[labels]],minlength=180)
    seed_hue = colors.argmax()
    diff = abs(hue.astype(np.int16)-seed_hue)
    same = color & (np.minimum(diff,180-diff)<13)
    matched = np.bincount(labels[same],minlength=len(stats)) > .60*color_counts
    candidates &= matched
    if not (candidates & seeds).any():
        return None
    reach = min(28,max(6,max(int(w[seeds].max()),int(h[seeds].max()))))
    hits = np.bincount(labels[p.dilate(source,reach)],minlength=len(stats))
    commit = p.dilate((candidates & (hits>.25*area))[labels],2) & ~protected
    # A detached colored mark can touch a bright panel line. Grow from its
    # colored core through a narrow pale rim instead of absorbing the line.
    cores = same & (peak>150)
    _,core_ids,core_stats,_ = cv2.connectedComponentsWithStats(cores.astype(np.uint8),8)
    in_source = np.bincount(core_ids[p.dilate(source,3)],minlength=len(core_stats))
    attached = (in_source>.35*core_stats[:,4]) & (core_stats[:,4]>=3)
    attached[0] = False
    core = attached[core_ids]
    surround = p.dilate(core,4) & white
    rim = (p.dilate(surround,2) | p.dilate(core,2)) if surround.any() else np.zeros_like(core)
    commit |= rim & ~protected
    if not commit.any():
        return None
    # Neighboring bright strokes must not serve as samples, even where only the
    # current source is allowed to commit pixels to the page.
    inference = p.dilate(candidates[labels] | rim,4)
    plate = cv2.inpaint(crop,inference.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
    return plate,commit,'outlined_color_inpaint'


def preserve_crossing_rules(crop, plate, erase, paper):
    """Retain straight printed rules crossing a flat sign through the owned ink."""
    for original,out,mask in ((crop,plate,erase),(crop.swapaxes(0,1),plate.swapaxes(0,1),erase.T)):
        rows = np.flatnonzero(mask.any(axis=1))
        if not len(rows):
            continue
        spans = mask[rows]
        left = spans.argmax(1)
        right = spans.shape[1]-spans[:,::-1].argmax(1)
        valid = (left>=4) & (right+4<=spans.shape[1]) & (right-left>=8)
        rows,left,right = rows[valid],left[valid],right[valid]
        samples = np.concatenate((original[rows[:,None],left[:,None]+np.arange(-4,0)],
                                  original[rows[:,None],right[:,None]+np.arange(4)]),axis=1).astype(np.int16)
        tones = np.median(samples,axis=1)
        supported = (np.abs(samples-tones[:,None]).max(axis=(1,2))<12) & (np.abs(tones-paper).max(axis=1)>32)
        for y,tone in zip(rows[supported],tones[supported]):
            # Two unrelated line ends are insufficient. The original pixels
            # through the box must also contain the same continuous rule.
            inside = mask[y]
            if np.mean(np.abs(original[y,inside]-tone).max(axis=1)<18) >= .70:
                out[y,inside] = tone


def source_cleanplate(crop, source, protected, *, short_effect):
    from .aggressive import cheap

    erase = p.dilate(source, 2) & ~protected
    n.require(erase.any() and (~erase).any(), 'no local source or background available')
    outlined = outlined_color_plate(crop, source, protected) if short_effect else None
    if outlined is not None:
        return outlined
    # Carry a high-contrast stroke through a clipped box edge if the connected
    # component still fits the lettering scale and belongs mainly to this source.
    color = np.median(crop[erase],axis=0)
    distance = np.max(np.abs(crop.astype(np.int16)-color),axis=2)
    ring = p.dilate(erase,4) & ~erase & ~protected
    if ring.any() and (distance[erase]<35).mean()>.44 and (distance[ring]<35).mean()>.4:
        _,labels,stats,_ = cv2.connectedComponentsWithStats((distance>65).astype(np.uint8),8)
        hits = np.bincount(labels[erase],minlength=len(stats))
        _,_,w,h = cv2.boundingRect(erase.astype(np.uint8))
        accept = (hits>.24*stats[:,4]) & (stats[:,2]<=w+10) & (stats[:,3]<=h+10)
        accept[0] = False
        erase |= p.dilate(accept[labels],2) & ~protected
        if (distance[source]<4).mean()>.45 and (distance[ring]<4).mean()>.40:
            plate = crop.copy()
            plate[erase] = color
            preserve_crossing_rules(crop,plate,erase,color)
            return plate,erase,'dominant_source_surface'
    plate,method = cheap(crop,erase,protected)
    return plate,erase,method


def ruled_label_room(clean, source, blocked):
    """Use the connected paper inside a printed cell and exclude the rule itself."""
    x,y,w,h = cv2.boundingRect(source.astype(np.uint8))
    if w<8 or h<8:
        return None
    tone = np.median(clean[source],axis=0)
    paper = np.abs(clean.astype(np.int16)-tone).max(2)<10
    if paper[source].mean()<.8:
        return None
    # A long, contrasting rule must traverse the source. Restrict ordinary
    # balloons to their existing layout path even when their paper is flat.
    dark = (~paper).astype(np.uint8)
    rules = cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((1,w+6),np.uint8))
    rules |= cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((h+6,1),np.uint8))
    if not (source & (rules!=0)).any():
        return None
    free = cv2.erode((paper & ~blocked).astype(np.uint8),np.ones((5,5),np.uint8))
    _,labels = cv2.connectedComponents(free)
    support = np.bincount(labels[source],minlength=2)
    support[0] = 0
    if support.max()<.35*source.sum():
        return None
    return (labels==support.argmax())


def replace_at_source(rgb, bounds, own, blocked, text, font_data, coverage, style):
    """Last resort for replace-all: clean the owned polygons and letter the same region."""
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    # Tall Japanese columns often need a little extra horizontal room for English.
    margin = min(64, max(8, height // 5))
    context = max(24, min(56, max(width,height)//2))
    x, y = max(0, left - max(context,margin)), max(0, top - context)
    r, b = min(rgb.width, right + max(context,margin)), min(rgb.height, bottom + context)
    n.budget_check((r-x)*(b-y) <= 4 * MAX_LOCAL_PIXELS, 'source replacement crop budget')
    protected = blocked[y:b, x:r]
    source = own[y:b, x:r]
    crop = np.asarray(rgb.crop((x,y,r,b)))
    clean, erase, method = source_cleanplate(crop, source, protected, short_effect=len(text)<=20 and max(width,height)<=200)
    # Use the original box first. Relax the type floor only after trying a wider box.
    paper = np.median(clean[erase], axis=0).mean() >= 128
    lettering = {**style, 'fill': [0,0,0] if paper else [255,255,255],
                 'outline_fill': [255,255,255] if paper else [0,0,0], 'outline': 1}
    cell = ruled_label_room(clean,source,protected) if method=='dominant_source_surface' else None
    variants = [(0,style['min_size'],0), (margin,style['min_size'],0), (margin,8,0),
                (0,8,-height//4), (0,8,height//4)]
    if cell is not None:
        variants.insert(0,(context,10,0))
    for extra, minimum, shift in variants:
        room = np.zeros_like(erase)
        room[max(0,top-y+shift):max(0,bottom-y+shift), max(0,left-x-extra):right-x+extra] = True
        room &= ~protected
        if cell is not None:
            room &= cell
        try:
            layer, proof = n.layout(text, font_data, coverage, Image.fromarray(room.astype(np.uint8)*255),
                                    {**lettering,'min_size':minimum})
            break
        except n.ResourceLimit:
            raise
        except ValueError:
            continue
    else:
        raise ValueError('no exact source-local lettering fits clear of neighboring text')
    mask = erase | (np.asarray(layer.getchannel('A')) != 0)
    patch = Image.fromarray(clean).convert('RGBA')
    patch.putalpha(Image.fromarray(mask.astype(np.uint8)*255))
    return 'aggressive', patch, layer, (x,y), proof, [{'route':'source_local_replacement','cleanup':method}]


def source_window(rgb, polygons, members, reserved, native_clean, native_erase, lettering_taken, row, font, style, commit_mask, cleared_sources=None, earlier_patches=None):
    """Work in source coordinates. A saved cleanplate can receive its own lettering."""
    pts = [pt for k in members for pt in polygons[k]]
    xs,ys = zip(*pts)
    l,t = max(0,int(min(xs))),max(0,int(min(ys)))
    r,b = min(rgb.width,int(max(xs))+1),min(rgb.height,int(max(ys))+1)
    n.require(r>l and b>t,'empty clipped source region')
    window_limit = 512 if 'readability_floor' in style else 256
    pad = min(window_limit,max(64,int(1.8*style['min_size']*math.sqrt(len(row['binding']['text'])))))
    x,y = max(0,l-pad),max(0,t-pad)
    end,base = min(rgb.width,r+pad),min(rgb.height,b+pad)
    window = (x,y,end,base)
    bounds = (l-x,t-y,r-x,b-y)
    own = Image.new('1',(end-x,base-y))
    other = Image.new('1',own.size)
    nearby = {}
    for k,poly in polygons.items():
        px,py = zip(*poly)
        if max(px)<x or max(py)<y or min(px)>=end or min(py)>=base:
            continue
        points = [(u-x,v-y) for u,v in poly]
        nearby[k] = points
        ImageDraw.Draw(own if k in members else other).polygon(points,fill=1)
    own,other = np.asarray(own),np.asarray(other)
    roi = np.s_[y:base,x:end]
    protected = reserved[roi]
    removed = row.get('removal_fallback',{}).get('status') == 'removed'
    if removed:
        # Reuse a known cleanplate beneath new type while protecting every other edit.
        cleared = own & ~other
        if commit_mask is not None:
            cx,cy,cr,cb = row['removal_fallback']['bounds']
            n.require(commit_mask.size==(cr-cx,cb-cy),'cleanup mask geometry mismatch')
            ix,iy,ir,ib = max(x,cx),max(y,cy),min(end,cr),min(base,cb)
            if ix<ir and iy<ib:
                cleared[iy-y:ib-y,ix-x:ir-x] |= np.asarray(commit_mask.crop((ix-cx,iy-cy,ir-cx,ib-cy)))!=0
        protected = protected & ~(native_erase[roi] & cleared & ~lettering_taken[roi])
    blocked = other | protected
    if (own & other).any():
        blocked = overlap_owners(bounds,members,nearby,own,blocked,protected)
    label = row['binding']['text']
    if removed:
        clean = np.asarray(native_clean.crop(window))
        erase = np.zeros_like(own)
        method = 'existing_source_cleanplate'
    else:
        region = rgb.crop(window)
        try:
            placed = replace_at_source(region,bounds,own,blocked,label,*font,style)
        except ValueError:
            pass
        else:
            return placed,window,own,blocked,bounds
        source = own & ~blocked
        n.require(source.any(),'no owned source pixels to replace')
        crop = np.asarray(region)
        # A wider lettering search must not enlarge the inpainting mask.
        # Estimate the background next to the source, even on a large page.
        left,top,right,bottom = bounds
        area = (np.s_[:,:] if source.size <= 4*MAX_LOCAL_PIXELS else
                np.s_[max(0,top-32):bottom+32,max(0,left-32):right+32])
        clean = crop.copy()
        erase = np.zeros_like(source)
        if source[area].size <= 4*MAX_LOCAL_PIXELS and (~p.dilate(source[area],2)).any():
            clean[area],erase[area],method = source_cleanplate(crop[area],source[area],blocked[area],
                short_effect=len(label)<=20 and max(r-l,b-t)<=200)
        else:
            clean[source] = 255
            erase,method = source,'opaque_source_backing'
    if cleared_sources is not None:
        # Cleared sign rows can share paper. Keep their English and all
        # still-visible source lettering protected, even when boxes overlap.
        retained = Image.new('1',own.shape[::-1])
        for identity,points in nearby.items():
            if identity not in cleared_sources and identity not in members:
                ImageDraw.Draw(retained).polygon(points,fill=1)
        type_blocked = np.asarray(retained) | p.dilate(lettering_taken[roi],2)
        # Preserve every earlier cleanup under antialiased type that lands there.
        background = native_clean.crop(window).convert('RGBA')
        background.alpha_composite(earlier_patches.crop(window))
        plate = np.asarray(background.convert('RGB')).copy()
        plate[erase] = clean[erase]
        clean = plate
        blocked = type_blocked & ~erase
    patch,letters,proof = compact_lettering.paint(clean,erase,bounds,blocked,label,font[0],style)
    return ('aggressive',patch,letters,(0,0),proof,[{'route':'compact_source_lettering','cleanup':method}]),window,own,blocked,bounds


def display(rgb, raw, units, rows, native_clean, native_text, native_composite, font_data, coverage, style, *, max_displays=MAX_ATTEMPTS, native_erase=None, font_resolver=None, protect_artwork=True, replace_all=False, cleanup_artifacts=None):
    """Independent post-clean display pass. Failures never roll back native edits."""
    started = time.monotonic()
    shape = (rgb.height,rgb.width)
    patch_layer = Image.new('RGBA',rgb.size)
    fallback_text = Image.new('RGBA',rgb.size)
    footprint = np.zeros(shape,bool)
    native_alpha = np.asarray(native_text.getchannel('A')) != 0
    native_erase = np.any(np.asarray(native_clean) != np.asarray(rgb),axis=2) if native_erase is None else native_erase
    reserved = native_alpha | native_erase
    if not replace_all:
        reserved = p.dilate(reserved,2)
    lettering_taken = native_alpha.copy()
    reused_clean = np.zeros(shape,bool)
    counts = None if replace_all else np.zeros(shape,np.uint16)
    polygons = {}
    pixels = rgb.width*rgb.height
    work = 0 if replace_all else pixels*len(raw)
    n.budget_check(work <= MAX_WORK,'visible translation source geometry budget')
    for detection in raw:
        polygon = detection['polygon']
        if replace_all:
            n.validate_polygon(polygon,rgb.size,evidence=True)
        else:
            counts += np.asarray(n.polygon_mask(polygon,rgb.size,evidence=True)) != 0
        polygons[key(detection['source_id'])] = polygon
    cleared_sources = ({key(k) for row in rows if row['status']=='automatic_composed'
                        for k in row['binding']['member_ids']} if replace_all and 'readability_floor' in style else None)
    tiles = {}
    attempts = 0
    order = list(enumerate(zip(units,rows)))
    order.sort(key=lambda pair: (pair[1][1].get('reason_code') != 'replacement_layout_confidence',
        -(pair[1][1].get('replacement_confidence',{}).get('minimum_main_line_score') or 0), pair[0]))
    if cleared_sources is not None:
        # A short label can move into a gap that a paragraph cannot use.
        order.sort(key=lambda item: -len(item[1][1].get('binding',{}).get('text','')))
    for i,(unit,row) in order:
        if row['status'] == 'automatic_composed':
            row.update(display_status=composed_status(row),source_visibility='clean_replacement_unverified' if composed_status(row)=='clean' else 'aggressive_removal_unverified')
            continue
        if 'binding' not in row:
            if row.get('source_status') == 'abstained' and (row.get('original_label_retained') or row.get('original_artwork_retained')):
                row.update(display_status='unchanged',source_visibility='original_retained')
            else:
                row['display_status'] = 'missing'
            continue
        if row['binding']['text'] == unit.selected.text or row.get('reason_code') == 'replacement_unchanged_text':
            row.update(display_status='unchanged',source_visibility='original_retained')
            continue
        row['display_status'] = 'undisplayed'
        attempt = {'policy':POLICY,'status':'unresolved','native_reason':row.get('reason_code')}
        row['display_fallback'] = attempt
        if not replace_all and row.get('reason_code') == 'ruled_table_cell':
            attempt['reason'] = 'target_exceeds_printed_cell'
            continue
        if row.get('reason_code') in ('evaluation_cap','uncorroborated_source') or (not replace_all and row.get('reason_code')=='runtime_locality'):
            attempt['reason'] = 'source_validation_or_explicit_evaluation_limit'
            continue
        extra = pixels*(len(row['binding']['member_ids'])+16)+64_000_000
        if not replace_all and (attempts >= MAX_ATTEMPTS or len(tiles) >= max_displays or work+extra > MAX_WORK):
            attempt['reason'] = 'display_fallback_budget'
            continue
        attempts += 1
        if not replace_all:
            work += extra
        try:
            if font_resolver is not None:font_data,coverage=font_resolver(row)
            n.validate_text(row['binding']['text'],coverage)
            ox,oy = 0,0
            if replace_all:
                members = {key(k) for k in row['binding']['member_ids']}
                placement,window,own,blocked,bounds = source_window(rgb,polygons,members,reserved,native_clean,native_erase,
                    lettering_taken,row,(font_data,coverage),style,
                    (cleanup_artifacts or {}).get(f'recovery-final-{i:03d}-commit-mask.png'),cleared_sources,patch_layer)
                ox,oy,_,_ = window
                x,y,right,bottom = bounds
                x,y,w,h = x+ox,y+oy,right-x,bottom-y
                work += own.size
                kind,patch,lettering,xy,proof,routes = placement
            else:
                own_count = np.zeros(shape,np.uint16)
                for source_id in row['binding']['member_ids']:
                    own_count += np.asarray(n.polygon_mask(polygons[key(source_id)],rgb.size,evidence=True)) != 0
                own = own_count != 0
                x,y,w,h = cv2.boundingRect(own.astype(np.uint8))
                n.require(w > 0 and h > 0,'missing visible translation source geometry')
                blocked = p.dilate(counts > own_count,2) | reserved
                args = (rgb,(x,y,x+w,y+h),own,blocked,row['binding']['text'],font_data,coverage,style)
                kind,patch,lettering,xy,proof,routes = place(*args,protect_artwork=protect_artwork)
            px,py = xy;ph,pw = patch.height,patch.width
            ink = lettering.getchannel('A').getbbox()
            lx,ly = px,py
            px,py = px+ox,py+oy
            xy = (px,py)
            proof.update(page_tile_offset=list(xy), glyph_page_origin=[px+ink[0],py+ink[1]])
            mask = np.asarray(patch.getchannel('A')) != 0
            alpha = np.asarray(lettering.getchannel('A')) != 0
            n.require(not (alpha & ~mask).any() and not (mask & blocked[ly:ly+ph,lx:lx+pw]).any(),
                      'display fallback footprint/neighbor containment')
            patch_layer.alpha_composite(patch,dest=xy)
            fallback_text.alpha_composite(lettering,dest=xy)
            footprint[py:py+ph,px:px+pw] |= mask
            dest = np.s_[py:py+ph,px:px+pw]
            if kind=='aggressive':
                reserved[dest] |= mask
                lettering_taken[dest] |= alpha if cleared_sources is not None else mask
                if cleared_sources is not None:
                    cleared_sources.update(members)
                    reused_clean[dest] |= alpha & native_erase[dest]
                elif row.get('removal_fallback',{}).get('status')=='removed':
                    reused_clean[dest] |= alpha & native_erase[dest]
            else:
                occupied = np.zeros(shape,bool)
                occupied[dest] = mask
                reserved |= p.dilate(occupied,2)
            if len(tiles) < max_displays:
                tile_bounds = patch.getchannel('A').getbbox() if replace_all else None
                if tile_bounds:
                    tx,ty,_,_ = tile_bounds
                    tiles[i] = patch.crop(tile_bounds),lettering.crop(tile_bounds),(px+tx,py+ty)
                else:
                    tiles[i] = patch,lettering,xy
                attempt.update(patch_artifact=f'unit-{i:03d}-fallback-patch.png',
                    text_artifact=f'unit-{i:03d}-fallback-text.png',footprint_artifact=f'unit-{i:03d}-fallback-mask.png')
            row.update(display_status=kind,source_visibility='source_removed_unverified' if kind=='aggressive' or row.get('removal_fallback',{}).get('status')=='removed'
                else 'original_retained' if kind=='caption' else 'partly_covered_source_remnants_unverified')
            attempt.update(status='displayed',kind=kind,warning=WARNING,source_bounds=[x,y,x+w,y+h],
                bounds=[px,py,px+pw,py+ph],footprint_pixels=int(mask.sum()),
                layout=proof,
                source_polygon_covered_pixels=int((own[ly:ly+ph,lx:lx+pw]&mask).sum()),
                outside_source_polygon_footprint_pixels=int((~own[ly:ly+ph,lx:lx+pw]&mask).sum()),
                original_dark_pixels_covered=int((mask & (np.asarray(rgb.crop((px,py,px+pw,py+ph))).min(axis=2)<180)).sum()),
                occlusion_scope='original contrast only; not semantic artwork classification',
                source_polygon_pixels=int(own.sum()),source_text_removal_verified=False,
                artwork_reconstruction=kind=='aggressive',artwork_obscured_possible=True,routes=routes)
        except (ValueError,OSError,KeyError,TypeError) as error:
            attempt.update(reason=str(error),reason_code='display_placement_failed')
    text = Image.alpha_composite(native_text,fallback_text)
    composite = Image.alpha_composite(native_clean.convert('RGBA'),patch_layer)
    composite = Image.alpha_composite(composite,text).convert('RGB')
    before,after = np.asarray(native_composite),np.asarray(composite)
    outside = int(np.any(before[~footprint] != after[~footprint],axis=1).sum())
    native_changed = int(np.any(before[native_alpha] != after[native_alpha],axis=1).sum())
    clean_changed = int(np.any(before[native_erase & ~reused_clean] != after[native_erase & ~reused_clean],axis=1).sum())
    n.require(outside == 0 and native_changed == 0 and clean_changed == 0,'visible translation pixel locality/native preservation failure')
    summary = display_counts(rows)
    summary.update(displayed_translations=summary['clean_replacements']+summary['patch_backed_translations']+summary['nearby_captions']+summary.get('learned_reconstructions',0)+summary.get('aggressive_replacements',0),
        bound_available_translations=sum('binding'in row for row in rows),caption_sources_retained=sum(
            row.get('display_status')=='caption' and row.get('source_visibility')=='original_retained' for row in rows),
        policy=POLICY,warning=WARNING,attempted=attempts,maximum_attempts=None if replace_all else MAX_ATTEMPTS,
        raster_work_bound=work,maximum_work=MAX_WORK,maximum_displays=max_displays,seconds=time.monotonic()-started,
        fallback_footprint_pixels=int(footprint.sum()),outside_fallback_changed_pixels=outside,
        reused_cleanplate_ink_pixels=int(reused_clean.sum()),native_lettering_changed_pixels=native_changed,native_cleanup_changed_pixels=clean_changed,semantic_accuracy_verified=False)
    return composite,text,patch_layer,fallback_text,footprint,tiles,summary
