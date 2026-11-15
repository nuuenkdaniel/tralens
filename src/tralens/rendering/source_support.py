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
    thickness=min(cv2.minAreaRect(np.concatenate(contours))[1])+1
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
    return support,evidence


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
