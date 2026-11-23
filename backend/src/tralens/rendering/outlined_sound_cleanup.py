"""Reconstruct overlapping outlined sounds from the original painted panel."""
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from ..ocr import crop_readings, outlined_sounds, sound_readings
from . import reconstruction as rec


def joint_mask(rgb, owned, protected, zone, *, extend=False):
    x, y, right, bottom = zone
    roi = np.s_[y:bottom, x:right]
    pixels, support, other = rgb[roi], owned[roi], protected[roi]
    white = (pixels.min(2) > 157) & (np.ptp(pixels, axis=2) < 55)
    gray = pixels.max(2)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    items = []
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i, (xx, yy, w, h, area) in enumerate(stats[1:], 1):
        if (not 5 <= area <= 1800 or not 7 <= max(w,h) <= 90 or min(w,h) < 2
                or min(xx,yy) < 2 or xx+w+2 > pixels.shape[1] or yy+h+2 > pixels.shape[0]):
            continue
        local = np.s_[yy-2:yy+h+2, xx-2:xx+w+2]
        body = (labels[local] == i).astype(np.uint8)
        ring = (cv2.dilate(body, ellipse) != 0) & (body == 0)
        if np.mean(gray[local][ring] < 112) < .73 or (other[local] & (body != 0)).any():
            continue
        inside = np.mean(support[local][body != 0]) > .5
        glyph = body[2:-2, 2:-2]
        shape = cv2.resize(glyph.astype(np.float32), (18,18), interpolation=cv2.INTER_AREA)
        items.append((xx, yy, w, h, glyph, inside, shape))
        if len(items) > 768:
            return np.zeros_like(support)
    seeds = [item for item in items if item[5]]
    if len(seeds) < 8:
        return np.zeros_like(support)
    core = np.zeros(support.shape, np.uint8)
    matches = []
    for xx, yy, w, h, glyph, inside, shape in items:
        if inside:
            core[yy:yy+h, xx:xx+w] |= glyph
        elif any((extend or (abs(xx-sx)<150 and abs(yy-sy)<95)) and .52<h/sh<1.9
                 and .65<(w/h)/(sw/sh)<1.55
                 and np.minimum(shape, exemplar).sum() > .66*np.maximum(shape, exemplar).sum()
                 for sx, sy, sw, sh, _, _, exemplar in seeds):
            matches.append((xx, yy, w, h, glyph))
    # Without supplemental boxes, bridge adjacent copies only. Every shape
    # must still match an original seed, including on the second hop.
    for _ in range(2 if extend else 1):
        reach = cv2.dilate(core, np.ones((189, 299), np.uint8)) if extend else None
        for xx, yy, w, h, glyph in matches:
            if reach is None or reach[yy, xx]:
                core[yy:yy+h, xx:xx+w] |= glyph
    nearby = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))) != 0
    dark = (gray < 90).astype(np.uint8)
    _, components, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    nearby &= dark != 0
    hits = np.bincount(components[nearby], minlength=len(stats))
    keep = (hits > .70*stats[:,-1]) & (stats[:,-1] < 6000) & (hits >= 6)
    if extend:
        # Touching outlines can form one large dark component. Complete a
        # small leftover tip only when the detected strokes explain almost all
        # of that component; a long panel line stays outside this allowance.
        keep |= (hits >= .90 * stats[:, -1]) & (stats[:, -1] - hits <= 512) & (stats[:, 2:4].max(1) <= 220)
    keep[0] = False
    ink = (core != 0) | nearby | keep[components]
    return (cv2.dilate(ink.astype(np.uint8), np.ones((5,5),np.uint8)) != 0) & ~other


def repair_gutter(pixels, plate, mask, protected):
    """Remove small connected glyph tips hanging into a wide band of bare paper."""
    paper = (pixels.min(2)>=245) & (np.ptp(pixels,axis=2)<6)
    fraction = paper.mean(1)
    band = cv2.dilate((fraction>=.90).astype(np.uint8)[:,None],np.ones((51,1),np.uint8))[:,0]!=0
    band &= (fraction>.68) & ((plate.min(2)>235).mean(1)>.82)
    if not band.any():
        return np.zeros_like(mask)
    near = cv2.dilate(mask.astype(np.uint8),np.ones((57,57),np.uint8))!=0
    ink = (pixels.min(2)<198) & near & ~protected
    kernel = np.ones((3,3),np.uint8)
    connected = cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))
    for _ in range(28):
        connected = cv2.dilate(connected,kernel) & (ink|mask).astype(np.uint8)
    eligible = ink & band[:,None]
    _,labels,stats,_ = cv2.connectedComponentsWithStats(eligible.astype(np.uint8),8)
    touched = np.zeros(len(stats),bool)
    touched[np.unique(labels[eligible & (connected!=0)])] = True
    touched &= (stats[:,2]<32) & (stats[:,3]<45) & (stats[:,3]>=5)
    touched[0] = False
    tails = (cv2.dilate(touched[labels].astype(np.uint8),kernel)!=0) & band[:,None] & ~mask & ~protected
    for row in np.flatnonzero(tails.any(1)):
        plate[row,tails[row]] = np.median(pixels[row,paper[row]],axis=0)
    return tails


def classical_repair(pixels, mask):
    """Diffuse the available colors through a cluster without borrowing gutter white."""
    inference = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    known = inference == 0
    paper = (pixels.min(2) >= 247) & (np.ptp(pixels, axis=2) < 8)
    donors = known.sum(1)
    bands = (donors >= 24) & ((paper & known).sum(1) >= .93 * donors)
    bands = cv2.morphologyEx(bands.astype(np.uint8)[:, None], cv2.MORPH_OPEN,
                             np.ones((5, 1), np.uint8))[:, 0]
    edges = np.r_[0, np.flatnonzero(np.diff(bands)) + 1, len(bands)]
    restored = pixels.copy()
    for top, bottom in zip(edges, edges[1:]):
        commit = mask[top:bottom]
        if not commit.any():
            continue
        crop = pixels[top:bottom]
        available = known[top:bottom]
        if bands[top]:
            restored[top:bottom][commit] = np.median(crop[available & paper[top:bottom]], axis=0)
            continue
        if not available.any():
            # At least one sampled row exists in every detected paper band.
            # Preserve the previous cleanup if this isolated strip has no donor.
            raise ValueError('outlined sound strip has no visible background')
        weights = available.astype(np.float32)
        colors = crop * weights[:, :, None]
        # Coarse levels carry color across wide strokes. At every level
        # the original visible pixels constrain the estimate again.
        estimate = np.median(crop[available], axis=0).astype(np.float32).reshape(1, 1, 3)
        for factor in (16, 8, 4, 2, 1):
            size = max(1, crop.shape[1] // factor), max(1, crop.shape[0] // factor)
            fixed = cv2.resize(colors, size, interpolation=cv2.INTER_AREA)
            free = 1 - cv2.resize(weights, size, interpolation=cv2.INTER_AREA)[:, :, None]
            estimate = cv2.resize(estimate, size)
            for _ in range(20):
                estimate = fixed + free * cv2.blur(estimate, (3, 3))
        restored[top:bottom][commit] = estimate[commit].round().clip(0, 255).astype(np.uint8)
    return restored


def tile_repair(rgb, zone, mask, protected, source, binding, backend, deadline, budget, router):
    x, y, right, bottom = zone
    page_mask = np.zeros(rgb.shape[:2], np.uint8)
    page_mask[y:bottom,x:right] = mask
    infer = cv2.dilate(page_mask, np.ones((7,7),np.uint8)) != 0
    restored = rgb[y:bottom,x:right].copy()
    requests = []
    for yy in range(y, bottom, 320):
        for xx in range(x, right, 320):
            ex, ey = min(xx+320,right), min(yy+320,bottom)
            if not page_mask[yy:ey,xx:ex].any():
                continue
            l, t = max(0,xx-96), max(0,yy-96)
            r, b = min(rgb.shape[1],ex+96), min(rgb.shape[0],ey+96)
            tile = np.s_[t:b,l:r]
            center = np.s_[yy-t:ey-t,xx-l:ex-l]
            commit = np.zeros((b-t,r-l), bool)
            commit[center] = page_mask[yy:ey,xx:ex]
            crop, inference, obstacle = rgb[tile], infer[tile], protected[tile]
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not budget.available():
                raise TimeoutError('joint reconstruction budget exhausted')
            budget.calls += 1
            req = rec.request(source,binding,[l,t,r,b],crop,inference,commit,obstacle,
                              attempt=0,model=rec.model_identity(backend.config))
            answer = backend.reconstruct(req,crop,inference,commit,obstacle,timeout=min(req['limits']['seconds'],remaining))
            req,inference,answer = rec.review_outline(router,req,answer,crop,inference,commit,obstacle,budget,deadline)
            result = rec.compose(req,crop,inference,commit,obstacle,answer)
            restored[yy-y:ey-y,xx-x:ex-x][commit[center]] = result[center][commit[center]]
            requests.append(dict(request=req, output_pixels_sha256=rec.pixels_hash(result)))
            if 'outline_review' in answer:
                requests[-1]['outline_review'] = answer['outline_review']
    return restored, requests


def plan_lettering(original, text, layout, raw, active, rows, artifacts, zone, protected, font_resolver, style):
    """Pack the verified targets together, keeping every other existing glyph fixed."""
    from . import natural, replacement
    x,y,right,bottom = zone
    base = np.asarray(text).copy()
    occupied = layout.copy()
    backings = np.zeros(layout.shape,bool)
    for i, _, _, _ in active:
        row = rows[i]
        final = row.get('removal_fallback') or {}
        bounds = final.get('backing_bounds') or final.get('outlined_tile_bounds')
        if bounds:
            key = f'recovery-final-{i:03d}-text.png'
        else:
            key = f'recovery-{i:03d}-text.png'
            bounds = (row.get('reconstruction') or {}).get('bounds')
        layer = artifacts.get(key)
        if layer is None or not bounds:
            return None
        l,t,r,b = bounds
        alpha = np.asarray(layer.getchannel('A')) != 0
        if (alpha.shape != (b-t,r-l) or not np.array_equal(base[t:b,l:r][alpha],np.asarray(layer)[alpha])):
            return None
        base[t:b,l:r][alpha] = 0
        if final.get('backing_bounds') or final.get('outlined_tile_bounds'):
            occupied[t:b,l:r] = False
        else:
            occupied[t:b,l:r][alpha] = False
        if final.get('backing_bounds'):
            backings[t:b,l:r] = True
    fixed = occupied | (base[:,:,3]!=0)
    blocked = (fixed | protected)[y:bottom,x:right].copy()
    stage = Image.fromarray(base)
    placements = []
    # Give long lines first choice. Each subsequent label can still use any
    # free part of the verified source group inside its painted panel.
    for i,binding,_,ids in sorted(active,key=lambda entry:-len(entry[1]['text'])):
        polygons = np.vstack([d['polygon'] for d in raw if d['source_id'] in ids])
        l,t = np.floor(polygons.min(0)).astype(int)
        r,b = np.ceil(polygons.max(0)).astype(int)
        frame = (max(0,l-x),max(0,t-y),min(right-x,r-x),min(bottom-y,b-y))
        l,t,r,b = replacement.painted_panel_limits(original[y:bottom,x:right],frame)
        confined = np.ones_like(blocked)
        confined[t:b,l:r] = blocked[t:b,l:r]
        font, coverage = font_resolver(rows[i])
        try:
            layer,(px,py),proof = replacement.patch_layout(binding['text'],font,coverage,frame,confined,
                {**style,'max_size':style['min_size'],'fill':[0,0,0],'outline_fill':[255,255,255]},outline=2)
        except natural.ResourceLimit:
            raise
        except ValueError:
            return None
        h,w = layer.height,layer.width
        blocked[py:py+h,px:px+w] = True
        px,py = px+x,py+y
        occupied[py:py+h,px:px+w] = True
        stage.alpha_composite(layer,dest=(px,py))
        proof.update(page_tile_offset=[px,py],glyph_page_origin=[px+6,py+6])
        placements.append((i,proof,[px,py,px+w,py+h],layer))
    return stage,occupied,backings,placements


def finish(rgb, clean, text, layout, erase, raw, units, rows, source, backend, artifacts, font_resolver, style):
    extra = {d['source_id']: d for d in raw if d.get('reason') == 'supplemental_rotated_sound_region'
             and d.get('ocr_metadata',{}).get(outlined_sounds.POLICY)}
    candidates = []
    corroborated = set()
    for i, (unit, row) in enumerate(zip(units, rows)):
        if row['status'] != 'automatic_composed':
            continue
        binding = row['binding']
        selected = binding.get('reread_selected_revision') or unit.selected_revision
        reading = next((r for r in unit.readings if r.revision == selected), None)
        if reading is None:
            continue
        motif = sound_readings.repeat(reading.text) or sound_readings.clipped_motif(reading.text)
        if motif:
            ids = set(binding['member_ids'])
            centers = [np.mean(d['polygon'], axis=0) for d in raw if d['source_id'] in ids]
            if centers:
                candidates.append(((i, binding, motif, ids), centers))
                if reading.origin == crop_readings.ENGINE and len(ids) == 1:
                    corroborated.update(ids)
    if not extra and len(corroborated) < 2:
        return clean, text, []
    original = np.asarray(rgb)
    plate = np.asarray(clean).copy()
    zones = outlined_sounds.zones(original,[d['polygon'] for d in raw if d['source_id'] not in extra])
    reports = []
    deadline, budget = time.monotonic()+180, rec.Budget(maximum_calls=16)
    for index, bounds in enumerate(zones):
        active = [item for item, centers in candidates
                  if all(bounds[0]<=x<bounds[2] and bounds[1]<=y<bounds[3] for x,y in centers)]
        all_ids = set().union(*(item[3] for item in active))
        if not all_ids.difference(extra) or len({item[2] for item in active}) != 1:
            continue
        motif = active[0][2]
        added = all_ids.intersection(extra)
        if added:
            if not any(any(a[0] in all_ids.difference(extra) for a in extra[k]['ocr_metadata'][outlined_sounds.POLICY]['anchors'])
                       and sound_readings.repeat(outlined_sounds.confirmed_text(extra[k]['ocr_metadata'][outlined_sounds.POLICY]['readings']) or '') == motif
                       for k in added):
                continue
        elif len(all_ids & corroborated) < 2:
            continue
        owned = np.zeros(erase.shape,np.uint8)
        protected = np.zeros_like(owned)
        for d in raw:
            cv2.fillPoly(owned if d['source_id'] in all_ids else protected,[np.asarray(d['polygon'],np.int32)],1)
        obstacle = cv2.dilate(protected,np.ones((5,5),np.uint8)) != 0
        eligible = np.zeros_like(erase)
        for i, _, _, _ in active:
            for key, prefix in [('reconstruction',f'recovery-{i:03d}'),('removal_fallback',f'recovery-final-{i:03d}')]:
                detail = rows[i].get(key) or {}
                if key == 'reconstruction':
                    if detail.get('status') != 'composed_unverified':
                        continue
                    prefix += f"-{detail.get('selected_attempt')}"
                mask = artifacts.get(prefix+'-commit-mask.png')
                if mask is not None and detail.get('bounds'):
                    l,t,r,b = detail['bounds']
                    eligible[t:b,l:r] |= np.asarray(mask) != 0
        prior_other = obstacle
        l,t,r,b = bounds
        zone = [max(0,l-36),max(0,t-36),min(rgb.width,r+36),min(rgb.height,b+36)]
        x,y,right,bottom = zone
        roi = np.s_[y:bottom,x:right]
        lettering = None
        if len(active)<=18 and (right-x)*(bottom-y)<=900000:
            preferred = max(style['min_size'],round(.70*style['max_size']))
            for size in dict.fromkeys((preferred,style['min_size'])):
                lettering = plan_lettering(original,text,layout,raw,active,rows,artifacts,zone,prior_other,
                                           font_resolver,{**style,'min_size':size})
                if lettering is not None:
                    break
        backings = np.zeros_like(erase) if lettering is None else lettering[2]
        obstacle = prior_other | (erase & ~eligible & ~backings)
        for row in rows:
            backing = (row.get('removal_fallback') or {}).get('backing_bounds')
            if backing:
                l,t,r,b = backing
                obstacle[t:b,l:r] |= ~backings[t:b,l:r]
        mask = joint_mask(original,owned!=0,obstacle,zone,extend=not added)
        if not mask.any():
            continue
        wide = cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))) != 0
        old = eligible[roi] & erase[roi] & wide & ~obstacle[roi]
        pixels = original[roi]
        selected = rec.region_backend(backend,pixels,mask)
        if selected is None:
            # The coarse earlier pass often removed midtone wall and bird
            # detail between the characters. Keep their original paint.
            near = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
            black = pixels.max(2) < 65
            white = (pixels.min(2) > 200) & (np.ptp(pixels, axis=2) < 32)
            mask |= old & near & (black | white)
            known = eligible[roi] & erase[roi] & ~obstacle[roi] & ~(black | white) & ~mask
            known |= old & ~mask
        else:
            mask |= old & ((pixels.max(2)<110) | ((pixels.min(2)>155) & (np.ptp(pixels,axis=2)<55)))
            known = old & ~mask
        binding = dict(active[0][1], related_bindings=[item[1] for item in active[1:]])
        report = dict(bounds=zone,source_ids=sorted(all_ids))
        reports.append(report)
        try:
            if selected is None:
                restored = classical_repair(pixels, mask)
                requests = []
                report['method'] = 'classical_diffusion'
            else:
                restored, requests = tile_repair(original,zone,mask,obstacle,source,binding,selected,deadline,budget,backend)
        except rec.BindingError:
            raise
        except (ValueError,OSError,MemoryError,RuntimeError,subprocess.SubprocessError,cv2.error) as error:
            report.update(status='previous_cleanup_preserved',reason=str(error)[:500])
            continue
        if lettering is not None:
            text, new_layout, _, placements = lettering
            layout[:] = new_layout
            # The opaque patch originally covered pristine art and possibly
            # glyphs. Restore its source first, then compose the shared repair.
            plate[backings] = original[backings]
            for i,proof,tile,layer in placements:
                rows[i]['layout'] = proof
                rows[i]['outlined_sound_placement'] = tile
                artifacts[f'outlined-sound-{index}-{i}-text.png'] = layer
            report['placed_translations'] = len(placements)
            report['replaced_backing_pixels'] = int(backings.sum())
        plate[roi][mask] = restored[mask]
        plate[roi][known] = pixels[known]
        tails = repair_gutter(pixels,plate[roi],mask,obstacle[roi])
        if tails.any():
            artifacts[f'outlined-sound-{index}-gutter.png'] = Image.fromarray(tails.astype(np.uint8)*255)
            report['gutter_pixels'] = int(tails.sum())
        erase[roi] |= mask | tails
        report.update(status='composed',requests=requests,reconstructed_pixels=int(mask.sum()),restored_original_pixels=int(known.sum()))
        for label, support in [('commit',mask),('original',known)]:
            artifacts[f'outlined-sound-{index}-{label}.png'] = Image.fromarray(support.astype(np.uint8)*255)
    return Image.fromarray(plate), text, reports
