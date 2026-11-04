"""Experimental crop-corroborated whole strokes, not semantic text segmentation.

Raw source polygons/text stay immutable. Cropped model predictions and original
closed geometry are separate ownership hypotheses. No model/provider is run here.
"""
import cv2
import numpy as np

from . import automatic as a, natural as n, preparation as p


POLICY = 'cropped_text_closed_whole_stroke_hypothesis_v1'
MAX_LOCAL_PIXELS = 262144
MAX_RETRIES = 4
MAX_RETRY_WORK = 1_024_000_000


def saved_records(preview, rgb, detections, units):
    from ..ocr.source_localization import POLICY as evidence_policy, geometry_revision
    stage = preview.get('ocr',{}).get('source_localization')
    n.require(type(stage) is dict and stage.get('policy') == evidence_policy,
        'localized cleanup requires explicit saved crop-localization evidence')
    n.require(stage.get('source_sha256') == preview['input']['sha256']
        and stage.get('source_pixels_sha256') == n.digest(rgb.tobytes())
        and stage.get('geometry_revision') == geometry_revision(detections), 'localized page/geometry drift')
    identity = stage.get('model_identity',{})
    n.require(identity.get('profile') == 'v6_medium' and identity.get('local_weights') is True,
        'localized cleanup requires the explicit existing local medium OCR profile')
    n.require(type(stage.get('calls')) is int and 0 <= stage['calls'] <= 12,
        'localized inference call budget drift')
    rows = stage.get('records')
    n.require(type(rows) is list and len(rows) == len(units)
        and len({row['unit_id'] for row in rows}) == len(rows), 'localized evidence unit coverage drift')
    by_id = {unit.unit_id:unit for unit in units}
    result = {}
    for row in rows:
        unit = by_id.get(row['unit_id'])
        n.require(unit is not None and row.get('unit_revision') == unit.revision
            and n.canonical(row.get('member_ids')) == n.canonical(unit.member_ids), 'localized source unit drift')
        if row['status'] == 'predicted':
            n.require(row.get('prediction_revision') == n.revision(row['prediction']), 'localized prediction revision drift')
            result[unit.unit_id] = row
    reused = stage.get('reused_predictions', 0)
    n.require(type(reused) is int and 0 <= reused <= MAX_RETRIES, 'localized saved prediction budget drift')
    n.require(len(result) <= stage['calls']+reused, 'localized prediction/call count drift')
    if 'request_plan' in stage:
        n.require(set(result) <= set(stage['request_plan']['requests']), 'unrequested localized prediction')
    return result


def _single_ink(pixels, foreground, residual):
    """Apply the existing single-fill/transition limits, with no gray-art waiver."""
    core_samples = foreground & (residual > 32)
    a.check(int(core_samples.sum()) >= 8, 'localized_ink_core', 'insufficient original ink support')
    colors, counts = np.unique(pixels[core_samples], axis=0, return_counts=True)
    ink = colors[np.argmax(counts)]
    core = foreground & (np.abs(pixels.astype(np.int16)-ink.astype(np.int16)).max(axis=2) <= 32)
    unsupported = foreground & (residual > 16) & ~p.dilate(core, 2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(unsupported.astype(np.uint8), connectivity=8)
    n.budget_check(count <= 257, 'localized ink-interior budget')
    tips = []
    for label in range(1, count):
        x,y,w,h,area = map(int,stats[label])
        component = labels == label
        values = pixels[component]
        small_transition = area <= 4 and max(w,h) <= 3 and (area < 3 or int(np.ptp(values,axis=0).max()) > 2)
        a.check(small_transition, 'localized_gray_interior',
            'different-color interior exceeds the unchanged native single-fill/antialias policy',
            {'ink_rgb':ink.tolist(),'unsupported':n.offending_pixels(component)})
        tips.append({'bounds':[x,y,x+w,y+h],'pixels':area})
    return {'ink_rgb':ink.tolist(),'terminal_checks':tips,
        'interior_policy':'unchanged native single fill within 32 channel levels; 2px transition; only existing <=4px varying exceptions'}


def prerequisites(rgb, detections, member_ids):
    """Existing model-free enclosure checks, shared by planning and cleanup.

    Passing is not erasure permission. Prediction/whole-stroke/gray/art checks
    still run in derive. No prediction or translated wording is consulted here.
    """
    n.require(rgb.mode == 'RGB' and rgb.width*rgb.height <= p.MAX_PIXELS
        and len(detections)*rgb.width*rgb.height <= a.MAX_WORK, 'localized source budget/mode')
    n.require(type(member_ids) is list and 0 < len(member_ids) <= 8
        and all(type(i) in (str,int) and i != '' for i in member_ids)
        and len(set(member_ids)) == len(member_ids), 'localized typed unique members required')
    keys = [(type(d['source_id']).__name__,d['source_id']) for d in detections]
    selected = {(type(i).__name__,i) for i in member_ids}
    n.require(len(set(keys)) == len(keys) and selected <= set(keys), 'localized source identity mismatch')
    pixels = np.asarray(rgb)
    original = {key:d for key,d in zip(keys,detections)}
    masks = {key:np.asarray(n.polygon_mask(original[key]['polygon'],rgb.size,evidence=True)) != 0
             for key in selected}
    members = np.logical_or.reduce([masks[key] for key in selected])
    original_regions = p.OriginalRegions(rgb)
    envelopes, votes = [], []
    for labels, stats in original_regions.white:
        chosen=[]
        for key in selected:
            frequency=np.bincount(labels[masks[key]],minlength=len(stats)); frequency[0]=0
            label=int(frequency.argmax())
            a.check(label != 0 and int(frequency[label]) >= .9*int(frequency.sum()),
                'localized_enclosure_ambiguous','original source has no dominant closed light region')
            x,y,w,h,area=map(int,stats[label])
            a.check(x > 0 and y > 0 and x+w < rgb.width and y+h < rgb.height,
                'localized_open','source region reaches the page edge')
            chosen.append(label)
        a.check(len(set(chosen)) == 1,'localized_distinct_enclosures','members span different original enclosures')
        envelope=p.filled_outer(labels == chosen[0]); envelopes.append(envelope)
    cores=[p.erode(envelope,3) for envelope in envelopes]
    a.check(np.array_equal(*cores),'localized_connectivity','4/8-connected original usable cores disagree')
    envelope,core=envelopes[0],cores[0]
    a.check(64 <= int(envelope.sum()) <= min(MAX_LOCAL_PIXELS,rgb.width*rgb.height//3),
        'localized_enclosure_extent','original enclosure is too large for this pilot')
    # Closed light regions are supporting geometry, not semantic balloon labels.
    ring=p.dilate(envelope,1)&~envelope
    a.check(ring.any() and float((pixels[ring].min(axis=1)<180).mean()) >= .9,
        'localized_outline','original enclosure lacks dark boundary support')
    return keys, selected, original, masks, members, envelope, core


def derive(rgb, detections, member_ids, record, *, reserved=None):
    """Validate raw crop predictions and derive masks exclusively from original RGB."""
    from ..ocr.source_localization import crop_bounds
    bounds = record['bounds']
    n.require(bounds == crop_bounds(detections, member_ids, rgb.size)
        and (bounds[2]-bounds[0])*(bounds[3]-bounds[1]) <= MAX_LOCAL_PIXELS,
        'localized crop extent drift/budget')
    crop = rgb.crop(bounds)
    n.require(record['crop_pixels_sha256'] == n.digest(crop.tobytes()), 'localized original crop pixels drift')
    prediction = record['prediction']
    fields = ('rec_texts','rec_scores','rec_polys','text_word_region')
    n.require(all(type(prediction.get(k)) is list for k in fields)
        and 0 < len(prediction['rec_texts']) <= 64
        and len({len(prediction[k]) for k in fields}) == 1, 'localized prediction association contract')
    keys, selected, original, masks, members, envelope, core = prerequisites(rgb, detections, member_ids)
    masks.update({key:np.asarray(n.polygon_mask(original[key]['polygon'],rgb.size,evidence=True)) != 0
                  for key in keys if key not in selected})
    pixels = np.asarray(rgb)
    shape = pixels.shape[:2]
    n.require(reserved is None or (reserved.shape == shape and reserved.dtype == bool), 'localized reservation shape/type')
    reserved = np.zeros(shape,bool) if reserved is None else reserved
    refinements=[]; seeds=np.zeros(shape,bool); line_masks=np.zeros(shape,bool); used=set()
    for key in sorted(selected):
        candidates=[]
        for index,text in enumerate(prediction['rec_texts']):
            if type(text) is not str or text != original[key]['text']:
                continue
            score=prediction['rec_scores'][index]
            n.require(type(score) in (int,float) and np.isfinite(score) and 0 <= score <= 1,
                'localized recognition score contract')
            if score < .7:
                continue
            local=prediction['rec_polys'][index]
            n.polygon_mask(local,crop.size,evidence=True)
            poly=[[x+bounds[0],y+bounds[1]] for x,y in local]
            mask=np.asarray(n.polygon_mask(poly,rgb.size,evidence=True)) != 0
            candidates.append((int((mask&masks[key]).sum()),index,mask))
        candidates.sort(key=lambda item:(-item[0],item[1]))
        a.check(candidates and candidates[0][0] > 0 and
            (len(candidates)==1 or candidates[0][0]>candidates[1][0]),
            'localized_line_missing','no unique exact-text cropped line with original support')
        _,index,mask=candidates[0]
        a.check(index not in used,'localized_line_reuse','cropped line cannot corroborate two sources')
        used.add(index); line_masks |= mask
        regions=prediction['text_word_region'][index]
        n.require(type(regions) is list and 0 < len(regions) <= 128,'localized word-region budget')
        for poly in regions:
            n.polygon_mask(poly,crop.size,evidence=True)
            seed=np.asarray(n.polygon_mask([[x+bounds[0],y+bounds[1]] for x,y in poly],rgb.size,evidence=True)) != 0
            a.check(not (seed&~p.dilate(mask,2)).any(),'localized_word_extent','word seed leaves its detected line')
            seeds |= seed
        refinements.append({'source_id':key[1],'prediction_index':index,'text':original[key]['text'],
            'score':prediction['rec_scores'][index]})
    for index in set(range(len(prediction['rec_texts'])))-used:
        poly=prediction['rec_polys'][index]
        n.polygon_mask(poly,crop.size,evidence=True)
        mask=np.asarray(n.polygon_mask([[x+bounds[0],y+bounds[1]] for x,y in poly],rgb.size,evidence=True)) != 0
        a.check(not (mask&core&(pixels.min(axis=2)<180)).any(),'localized_extra_text',
            'unmatched cropped text has original ink inside the enclosure')
    values=pixels[core & (pixels.min(axis=2)>=180)]
    a.check(len(values)>=32,'localized_background','insufficient original flat background support')
    colors,frequencies=np.unique(values,axis=0,return_counts=True);color=colors[frequencies.argmax()]
    a.check(int(color.min())>=180 and int(np.ptp(color))<=16,'localized_background','light neutral background required')
    residual=np.abs(pixels.astype(np.int16)-color.astype(np.int16)).max(axis=2)
    contrast=residual>2
    count,labels,stats,_=cv2.connectedComponentsWithStats(contrast.astype(np.uint8),connectivity=8)
    n.budget_check(count<=8192,'localized whole-contrast component budget')
    near=p.dilate(members,2)&p.dilate(line_masks,2)
    candidate_ids=np.unique(labels[line_masks&core]); candidate_ids=candidate_ids[candidate_ids!=0]
    n.budget_check(len(candidate_ids)<=128,'localized candidate component budget')
    chosen=[]; ownership=[]
    for label in candidate_ids:
        component=labels==label
        if (component&~core).any():
            a.check(not (component&seeds).any(),'localized_seed_spill','word seed touches border/art-connected ink')
            continue  # Kept whole and protected as an original barrier, never clipped.
        x,y,w,h,area=map(int,stats[label])
        a.check(not (component&~near).any() and max(w,h)<=96 and area<=20000,
            'localized_component_extent','whole source component leaves corroborated line extent')
        own=int((component&members).sum())
        a.check(own>0,'localized_unowned_ink','component lacks original member support')
        competitors=[(key,int((component&mask).sum())) for key,mask in masks.items() if key not in selected]
        a.check(all(own>vote for key,vote in competitors),'localized_ownership_tie',
            'another original source owns as much of a whole component as the target')
        chosen.append(int(label))
        disputed=[{'source_id':key[1],'pixels':vote} for key,vote in competitors if vote]
        if disputed: ownership.append({'label':int(label),'member_pixels':own,'competing_votes':disputed})
    a.check(0<len(chosen)<=128,'localized_no_strokes','no complete localized source strokes')
    foreground=np.isin(labels,chosen)
    a.check(not ((seeds&contrast)&~foreground).any(),'localized_uncovered_seed','cropped text seed has unsupported native ink')
    # Any stable second-color interior remains a rejection, even inside a model box.
    ink_audit=_single_ink(pixels,foreground,residual)
    erase=p.dilate(foreground,1)
    a.check(not (erase&~core).any(),'localized_boundary_clearance','whole erasure leaves inset original region')
    collar=p.dilate(erase,2)&~erase
    a.check(int(collar.sum())>=32 and not (collar&contrast).any(),
        'localized_nonflat_collar','whole source strokes lack known flat original clearance')
    protection=np.zeros(shape,bool); resolved=[]
    for key,mask in masks.items():
        if key in selected: continue
        poly=original[key]['polygon']; cx=round(sum(pt[0] for pt in poly)/len(poly));cy=round(sum(pt[1] for pt in poly)/len(poly))
        a.check(not core[min(rgb.height-1,cy),min(rgb.width-1,cx)],'localized_enclosed_neighbor',
            'another source or annotation is centered inside the enclosure')
        if (p.dilate(foreground,3)&mask).any():
            # Unselected original ink is protected independently below. This
            # changes a coarse rectangle hypothesis, not its original record.
            protection |= mask&~core
            resolved.append(key[1])
        else: protection |= mask
    obstacles=(contrast&~foreground)|protection|reserved|~core
    a.check(not (erase&obstacles).any(),'localized_cleanup_collision','cleanup touches protected original/output pixels')
    layout=p.erode(core,2)&~p.dilate(obstacles,2)
    a.check(not (foreground&~layout).any(),'localized_layout_clearance','complete source lacks clear original lettering space')
    free_count,free_labels=cv2.connectedComponents(layout.astype(np.uint8),connectivity=4)
    n.budget_check(free_count<=8192,'localized layout component budget')
    free=np.unique(free_labels[foreground])
    a.check(len(free)==1 and free[0]!=0,'localized_layout_disconnected','source spans disconnected lettering pockets')
    layout=free_labels==free[0]
    evidence={'policy':POLICY,'cleanup_method':'localized_whole_stroke_flat_fill',
        'foreground_is_text_certified':False,'raw_geometry_unchanged':True,'source_pixels_sha256':n.digest(rgb.tobytes()),
        'prediction_revision':n.revision(record),'crop_bounds':bounds,'matched_lines':refinements,
        'connectivities':[4,8],'usable_core_disagreement_pixels':0,'enclosure_pixels':int(envelope.sum()),
        'selected_whole_components':len(chosen),'disputed_component_votes':ownership,
        'coarse_neighbor_rectangles_resolved':resolved,'all_unselected_original_ink_protected':True,
        'background_rgb':color.tolist(),'foreground_pixels':int(foreground.sum()),'erase_pixels':int(erase.sum()),
        'ink_audit':ink_audit,'warnings':['Crop/model agreement and connected geometry are uncertain ownership hypotheses, not semantic certification.']}
    return a.AutomaticPlan(foreground,erase,layout,obstacles,color.tolist(),evidence)
