"""Rejected presentation diagnostic.

Results: https://github.com/nuuenkdaniel/tralens/blob/b33a1775a1eac1e6e4feef966611969acdaf4755/docs/reconstruction-context-results-v2.md

Not imported by production. Presentation space never grants erase permission.
"""
import cv2
import numpy as np
from PIL import Image

from tralens.rendering import natural as n,preparation as p,typography

POLICY='closed_light_recovery_presentation_v1'


def presentation(crop,own,protected):
    """Intersect 4/8-connected closed light envelopes, excluding unremoved art."""
    light=crop.min(axis=2)>=220
    regions=[]
    for connectivity in (4,8):
        count,labels,stats,_=cv2.connectedComponentsWithStats(light.astype(np.uint8),connectivity=connectivity)
        n.budget_check(count<=8192,'presentation component budget')
        votes=np.bincount(labels[own],minlength=count);votes[0]=0
        positive=int(votes.sum());label=int(votes.argmax())
        n.require(positive>=.5*int(own.sum()) and votes[label]>=.9*positive,'ambiguous presentation support')
        x,y,w,h,area=map(int,stats[label])
        n.require(x>0 and y>0 and x+w<crop.shape[1] and y+h<crop.shape[0], 'open or cropped presentation boundary')
        envelope=p.filled_outer(labels==label)
        n.require(int(envelope.sum())<=min(262144,30*int(own.sum())),'presentation extent unsupported')
        regions.append(p.erode(envelope,4))
    # Different diagonal connections cannot create extra presentation permission.
    core=regions[0]&regions[1]
    mask=core & (light|p.dilate(own,5)) & ~p.dilate(protected,2)
    n.require(own.any() and mask.any(),'no free presentation space')
    return mask


def layout(text,font_data,coverage,crop,own,protected,style):
    baseline=p.dilate(own,24)&~protected
    candidates=[];failures=[]
    for name in ('source_local','closed_light'):
        try:
            allowed=baseline if name=='source_local' else presentation(crop,own,protected)
            layer,proof,content=typography.layout_dialogue(text,font_data,coverage,
                Image.fromarray(allowed.astype(np.uint8)*255),style)
            candidates.append((proof['typography']['probes'][proof['typography']['chosen_probe']]['score'],
                name,layer,proof,content))
        except n.ResourceLimit as error:
            if not candidates:raise
            failures.append({'route':name,'reason':str(error)})
        except ValueError as error:failures.append({'route':name,'reason':str(error)})
    n.require(candidates,'no readable recovery presentation: '+str(failures))
    # Ties retain the established frame. Exact wording/font constraints are imposed
    # by typography in both routes; source/layout/pixel ownership remain separate.
    selected=min(enumerate(candidates),key=lambda item:(item[1][0],item[0]))[1]
    _,name,layer,proof,content=selected
    proof['recovery_presentation']={'policy':POLICY,'selected':name,'failures':failures,
        'candidate_scores':{candidate[1]:candidate[0] for candidate in candidates},
        'erasure_permission':False,'semantic_balloon_claim':False}
    return layer,proof,content
