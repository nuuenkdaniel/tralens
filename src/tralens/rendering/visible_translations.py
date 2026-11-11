"""Explicit opaque translation displays, not cleanup or artwork reconstruction.

Runs after clean composition with its own budgets. It consumes bound exact targets
and original geometry only; no OCR/provider calls, model approval or source edits.
"""
import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import natural as n, preparation as p


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
    if any('reconstruction' in row for row in rows):
        counts.update(learned_reconstructions=sum(row.get('display_status')=='learned' for row in rows),
            aggressive_replacements=sum(row.get('display_status')=='aggressive' for row in rows))
    return counts


def composed_status(row):
    if row.get('removal_fallback',{}).get('status')=='removed_and_placed':
        return 'aggressive'
    if row.get('reconstruction',{}).get('status')=='composed_unverified':
        return 'learned' if row.get('cleanup_method')=='learned_lama' else 'aggressive'
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


def place(rgb, source_bounds, own, blocked, text, font_data, coverage, style):
    """Return local patch/text tiles and an explicit page offset/footprint."""
    left,top,right,bottom = source_bounds
    cx,cy = (left+right)//2,(top+bottom)//2
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


def display(rgb, raw, units, rows, native_clean, native_text, native_composite, font_data, coverage, style, *, max_displays=MAX_ATTEMPTS, native_erase=None):
    """Independent post-clean display pass. Failures never roll back native edits."""
    started = time.monotonic()
    shape = (rgb.height,rgb.width)
    patch_layer = Image.new('RGBA',rgb.size)
    fallback_text = Image.new('RGBA',rgb.size)
    footprint = np.zeros(shape,bool)
    native_alpha = np.asarray(native_text.getchannel('A')) != 0
    native_erase = np.any(np.asarray(native_clean) != np.asarray(rgb),axis=2) if native_erase is None else native_erase
    reserved = p.dilate(native_alpha | native_erase,2)
    counts = np.zeros(shape,np.uint16)
    polygons = {}
    pixels = rgb.width*rgb.height
    work = pixels*len(raw)
    n.budget_check(work <= MAX_WORK,'visible translation source geometry budget')
    for detection in raw:
        polygon = detection['polygon']
        mask = np.asarray(n.polygon_mask(polygon,rgb.size,evidence=True)) != 0
        counts += mask
        polygons[key(detection['source_id'])] = polygon
    tiles = {}
    attempts = 0
    order = list(enumerate(zip(units,rows)))
    order.sort(key=lambda pair: (pair[1][1].get('reason_code') != 'replacement_layout_confidence',
        -(pair[1][1].get('replacement_confidence',{}).get('minimum_main_line_score') or 0), pair[0]))
    for i,(unit,row) in order:
        if row['status'] == 'automatic_composed':
            row.update(display_status=composed_status(row),source_visibility='clean_replacement_unverified' if composed_status(row)=='clean' else 'aggressive_removal_unverified')
            continue
        if 'binding' not in row:
            row['display_status'] = 'missing'
            continue
        if row['binding']['text'] == unit.selected.text or row.get('reason_code') == 'replacement_unchanged_text':
            row.update(display_status='unchanged',source_visibility='original_retained')
            continue
        row['display_status'] = 'undisplayed'
        attempt = {'policy':POLICY,'status':'unresolved','native_reason':row.get('reason_code')}
        row['display_fallback'] = attempt
        if row.get('reason_code') in ('runtime_locality','evaluation_cap'):
            attempt['reason'] = 'source_validation_or_explicit_evaluation_limit'
            continue
        extra = pixels*(len(row['binding']['member_ids'])+16)+64_000_000
        if attempts >= MAX_ATTEMPTS or len(tiles) >= max_displays or work+extra > MAX_WORK:
            attempt['reason'] = 'display_fallback_budget'
            continue
        attempts += 1
        work += extra
        try:
            n.validate_text(row['binding']['text'],coverage)
            own_count = np.zeros(shape,np.uint16)
            for source_id in row['binding']['member_ids']:
                own_count += np.asarray(n.polygon_mask(polygons[key(source_id)],rgb.size,evidence=True)) != 0
            own = own_count != 0
            x,y,w,h = cv2.boundingRect(own.astype(np.uint8))
            n.require(w > 0 and h > 0,'missing visible translation source geometry')
            blocked = p.dilate(counts > own_count,2) | reserved
            kind,patch,lettering,xy,proof,routes = place(rgb,(x,y,x+w,y+h),own,blocked,
                row['binding']['text'],font_data,coverage,style)
            px,py = xy;ph,pw = patch.height,patch.width
            gx,gy = proof.get('caption_bounds',[px,py])[:2]
            proof.update(page_tile_offset=list(xy), glyph_page_origin=[gx+PADDING,gy+PADDING])
            mask = np.asarray(patch.getchannel('A')) != 0
            alpha = np.asarray(lettering.getchannel('A')) != 0
            n.require(not (alpha & ~mask).any() and not (mask & blocked[py:py+ph,px:px+pw]).any(),
                      'display fallback footprint/neighbor containment')
            patch_layer.alpha_composite(patch,dest=xy)
            fallback_text.alpha_composite(lettering,dest=xy)
            footprint[py:py+ph,px:px+pw] |= mask
            occupied = np.zeros(shape,bool);occupied[py:py+ph,px:px+pw] = mask
            reserved |= p.dilate(occupied,2)
            tiles[i] = patch,lettering,xy
            row.update(display_status=kind,source_visibility='source_removed_unverified' if row.get('removal_fallback',{}).get('status')=='removed'
                else 'original_retained' if kind=='caption' else 'partly_covered_source_remnants_unverified')
            attempt.update(status='displayed',kind=kind,warning=WARNING,source_bounds=[x,y,x+w,y+h],
                bounds=[px,py,px+pw,py+ph],footprint_pixels=int(mask.sum()),
                layout=proof,patch_artifact=f'unit-{i:03d}-fallback-patch.png',
                text_artifact=f'unit-{i:03d}-fallback-text.png',footprint_artifact=f'unit-{i:03d}-fallback-mask.png',
                source_polygon_covered_pixels=int((own[py:py+ph,px:px+pw]&mask).sum()),
                outside_source_polygon_footprint_pixels=int((~own[py:py+ph,px:px+pw]&mask).sum()),
                original_dark_pixels_covered=int((mask & (np.asarray(rgb)[py:py+ph,px:px+pw].min(axis=2)<180)).sum()),
                occlusion_scope='original contrast only; not semantic artwork classification',
                source_polygon_pixels=int(own.sum()),source_text_removal_verified=False,
                artwork_reconstruction=False,artwork_obscured_possible=True,routes=routes)
        except (ValueError,OSError,KeyError,TypeError) as error:
            attempt.update(reason=str(error),reason_code='display_placement_failed')
    text = Image.alpha_composite(native_text,fallback_text)
    composite = Image.alpha_composite(native_clean.convert('RGBA'),patch_layer)
    composite = Image.alpha_composite(composite,text).convert('RGB')
    before,after = np.asarray(native_composite),np.asarray(composite)
    outside = int(np.any(before[~footprint] != after[~footprint],axis=1).sum())
    native_changed = int(np.any(before[native_alpha] != after[native_alpha],axis=1).sum())
    clean_changed = int(np.any(before[native_erase] != after[native_erase],axis=1).sum())
    n.require(outside == 0 and native_changed == 0 and clean_changed == 0,'visible translation pixel locality/native preservation failure')
    summary = display_counts(rows)
    summary.update(displayed_translations=summary['clean_replacements']+summary['patch_backed_translations']+summary['nearby_captions']+summary.get('learned_reconstructions',0)+summary.get('aggressive_replacements',0),
        bound_available_translations=sum('binding'in row for row in rows),caption_sources_retained=sum(
            row.get('display_status')=='caption' and row.get('source_visibility')=='original_retained' for row in rows),
        policy=POLICY,warning=WARNING,attempted=attempts,maximum_attempts=MAX_ATTEMPTS,
        raster_work_bound=work,maximum_work=MAX_WORK,maximum_displays=max_displays,seconds=time.monotonic()-started,
        fallback_footprint_pixels=int(footprint.sum()),outside_fallback_changed_pixels=outside,
        native_lettering_changed_pixels=native_changed,native_cleanup_changed_pixels=clean_changed,semantic_accuracy_verified=False)
    return composite,text,patch_layer,fallback_text,footprint,tiles,summary
