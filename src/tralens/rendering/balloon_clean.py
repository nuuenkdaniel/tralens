"""Experimental original-interior glyph-cell clearing, not native-mask retry.

Background/topology are estimated first by LightInteriors. Independent printed
cell and blank-collar evidence bounds cleanup; no inpainting, gray-color waiver,
percentage component clipping or blanket balloon fill. This is uncertain machine
ownership, not semantic proof that arbitrary detector-box contents are text.
"""
import unicodedata

import cv2
import numpy as np

from ..balloons import key, erode, dilate, mask_revision, require


POLICY = 'plain_interior_regular_glyph_cells_v1'


def derive(context, interior, member_ids, *, reserved=None, coordinated_ids=None):
    from .automatic import AutomaticPlan
    pixels = context.pixels
    shape = pixels.shape[:2]
    reserved = np.zeros(shape,bool) if reserved is None else reserved
    require(reserved.shape == shape and reserved.dtype == bool, 'balloon reservation shape/type')
    require(interior.evidence['source_pixels_sha256'] == context.source_revision
        and interior.evidence['raw_detection_revision'] == context.geometry_revision
        and interior.evidence['core_mask_revision'] == mask_revision(interior.core)
        and interior.evidence['background_rgb'] == interior.background, 'stale balloon interior/background binding')
    selected = {key(i) for i in member_ids}
    require(0 < len(selected) == len(member_ids) <= 32 and selected <= context.raw.keys(), 'balloon cleanup membership')
    if coordinated_ids is not None and {key(i) for i in coordinated_ids} != selected:
        require(selected <= {key(i) for i in coordinated_ids}, 'coordinated cleanup must retain selected members')
        # Shared interior does not mean shared utterance. Validate EVERY known
        # cell in that original interior first; then authorize only the complete
        # cells belonging to this exact input. Other cells/outputs remain held
        # obstacles. No cropping a component or borrowing a prior native mask.
        complete = derive(context,interior,coordinated_ids)
        erase = np.zeros(shape,bool)
        cells = []
        for cell in complete.evidence['cells']:
            if key(cell['source_id']) in selected:
                x,y,r,b = cell['bounds'];erase[y:b,x:r] = True;cells.append(cell)
        others = complete.erase & ~erase
        require(not (erase & reserved).any(), 'coordinated source cleanup touches native output')
        obstacles = complete.obstacles | dilate(others,2) | reserved
        layout = complete.layout & ~dilate(others | reserved,2)
        require(not (erase & obstacles).any() and layout.any(), 'coordinated source cells/frame conflict')
        return AutomaticPlan(complete.foreground & erase,erase,layout,obstacles,complete.background,{
            **complete.evidence, 'cells':cells, 'coordinated_member_ids':list(coordinated_ids),
            'selected_member_ids':list(member_ids),'protected_other_cells':int(others.sum()),
            'cleanup_pixels':int(erase.sum()),'cleanup_mask_revision':mask_revision(erase),
            'flow_mask_revision':mask_revision(layout),'utterances_joined':False})
    require(sum(len(context.raw[k]['text']) for k in selected)*pixels.shape[0]*pixels.shape[1] <= 256_000_000,
        'balloon printed-cell raster work budget')
    core = interior.core
    color = np.array(interior.background,dtype=np.int16)
    residual = np.abs(pixels.astype(np.int16)-color).max(axis=2)
    contrast = residual > 2
    own = np.zeros(shape,bool)
    nonmembers = np.zeros(shape,bool)
    for k,mask in context.masks.items():
        if k in selected:
            own |= mask
        else:
            # Rectangle contact outside the independently measured boundary does
            # not transfer ownership. Any actual enclosed competing ink rejects.
            require(not (mask & core & contrast).any(), 'balloon shared/unselected interior text')
            nonmembers |= mask & core
    cells, erase, foreground = [], np.zeros(shape,bool), np.zeros(shape,bool)
    for k in sorted(selected):
        raw = context.raw[k]
        text = raw['text']
        count = sum(not c.isspace() and not unicodedata.category(c).startswith('M') for c in text)
        require(1 <= count <= 48, 'balloon printed character count unsupported')
        mask = context.masks[k]
        x,y,w,h = cv2.boundingRect(mask.astype(np.uint8))
        vertical = h >= 1.3*w
        require(vertical or w >= 1.3*h, 'balloon unknown glyph-flow orientation')
        ink = contrast & mask & core
        require(ink.any(), 'balloon source lacks interior contrast')
        bx,by,bw,bh = cv2.boundingRect(ink.astype(np.uint8))
        start,span,minor = (by,bh,bw) if vertical else (bx,bw,bh)
        pitch = span/count
        require(6 <= pitch <= 64 and .2*pitch <= minor <= 1.6*pitch,
            'balloon glyph pitch/scale unsupported')
        # Find blank seams near the predicted regular print pitch. No seam can
        # cut visible source pixels. Empty per-cell collars are checked below.
        projection = ink.any(axis=1 if vertical else 0)
        seams = [start]
        for j in range(1,count):
            target = start+j*pitch
            low,high = max(seams[-1]+3,round(target-.3*pitch)),round(target+.3*pitch)
            options = [v for v in range(low,high+1) if 1 <= v < len(projection)-1
                and not projection[v-1:v+1].any()]
            require(options, 'balloon no blank seam between predicted glyph cells')
            seams.append(min(options,key=lambda v:abs(v-target)))
        seams.append(start+span)
        for first,last in zip(seams,seams[1:]):
            glyph = ink.copy()
            if vertical:
                glyph[:first] = False;glyph[last:] = False
            else:
                glyph[:,:first] = False;glyph[:,last:] = False
            gx,gy,gw,gh = cv2.boundingRect(glyph.astype(np.uint8))
            require(glyph.any() and 2 <= max(gw,gh) <= 1.6*pitch, 'balloon unsupported glyph-cell shape')
            density = int(glyph.sum())/(gw*gh)
            require(.04 <= density <= .72, 'balloon solid/illustrated glyph-cell density')
            contours,_ = cv2.findContours(glyph[gy:gy+gh,gx:gx+gw].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour,True)
                require(not (area > .5*gw*gh and perimeter > 0 and 4*np.pi*area/perimeter**2 > .8),
                    'balloon smooth enclosed pictogram/illustration unsupported')
            # Filled local cell rectangle is a DIFFERENT cleanup hypothesis from
            # a connected glyph mask. Require measured original blank surround,
            # boundary inset and complete detector ownership before authorizing.
            cell = np.zeros(shape,bool)
            cell[gy:gy+gh,gx:gx+gw] = True
            collar = dilate(cell,1) & ~cell
            require(not (cell & ~core).any(), 'balloon glyph cell lacks boundary clearance')
            require(not (cell & contrast & ~ink).any(), 'balloon glyph cell contains unassigned contrast')
            require(not (collar & contrast).any(), 'balloon glyph cell lacks blank original collar')
            require(not (cell & (nonmembers | reserved)).any(), 'balloon glyph cell collides with protected output/text')
            # Color is not explained as single-ink antialias. A printed cell may
            # contain multiple neutral intensities; spatial ownership, not a
            # relaxed native ink-color cutoff, must support every erased pixel.
            colors = pixels[cell & contrast]
            require(int(np.ptp(colors,axis=1).max()) <= 8, 'balloon colored source/art')
            erase |= cell;foreground |= glyph
            cells.append({'source_id':k[1], 'bounds':[gx,gy,gx+gw,gy+gh],
                'direction':'vertical' if vertical else 'horizontal', 'pitch':pitch, 'density':density,
                'contrast_pixels':int(glyph.sum()), 'blank_collar_pixels':int(collar.sum())})
    # A detector/core boundary must never silently cut a stroke. Independently
    # veto any original contrast component touched by a selected cell whose
    # remainder escapes the cell union (including a stroke connected to a rim).
    # This is a protection test, not the old native-mask ownership prerequisite.
    component_count,labels = cv2.connectedComponents(contrast.astype(np.uint8),connectivity=8)
    require(component_count <= 8192, 'balloon whole-contrast protection budget')
    selected_components = np.unique(labels[foreground])
    selected_components = selected_components[selected_components != 0]
    require(not (np.isin(labels,selected_components) & ~erase).any(), 'balloon clipped/shared whole contrast stroke')
    # Never regard a plain field as empty by assumption. Unassigned dark art or
    # text invalidates this route. Small light marks stay explicitly protected
    # and are excluded from flow space rather than opportunistically erased.
    unexplained = core & contrast & ~erase
    require(not (unexplained & (pixels.min(axis=2) < 220)).any(), 'balloon unexplained dark interior/art')
    require(int(unexplained.sum()) <= 32, 'balloon unexplained texture/illustration')
    require(not (foreground & ~erase).any() and not (erase & reserved).any(), 'balloon cleanup containment')
    obstacles = ~core | dilate(unexplained | nonmembers,2) | reserved
    layout = erode(core,2) & ~dilate(unexplained | nonmembers | reserved,2)
    require(layout.any(), 'balloon no supported English flow space')
    return AutomaticPlan(foreground,erase,layout,obstacles,interior.background,{
        'policy':POLICY, 'interior':interior.evidence, 'cells':cells,
        'cleanup_mask_revision':mask_revision(erase), 'flow_mask_revision':mask_revision(layout),
        'cleanup_pixels':int(erase.sum()), 'envelope_pixels':int(interior.envelope.sum()),
        'protected_unassigned_light_pixels':int(unexplained.sum()),
        'pixel_ownership':'unverified bounded printed-cell hypothesis', 'prior_native_mask_required':False,
        'artwork_reconstruction':False, 'whole_balloon_fill':False})
