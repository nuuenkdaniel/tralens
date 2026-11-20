"""English flow frames over supported balloon space, independent of source boxes."""
from dataclasses import replace

import cv2
import numpy as np
from PIL import Image

from . import typography
from .natural import ResourceLimit
from .automatic import grow_rectangle
from ..balloons import mask_revision


def fit(target,plan,font_data,coverage,style):
    frames = [('supported_balloon_interior',plan.layout)]
    # The existing fitter centers in the mask's bounds. Shared/irregular lobes
    # can leave that center occupied by prior English even with ample free space
    # elsewhere. A separate source-anchored free frame may use the entire
    # supported interior, not a fixed expansion of the Japanese column.
    x,y,w,h = cv2.boundingRect(plan.layout.astype(np.uint8))
    try:
        frame,_,_ = grow_rectangle(plan.layout,plan.foreground,(x,y,x+w,y+h))
        if not np.array_equal(frame,plan.layout):
            frames.append(('source_anchored_supported_rectangle',frame))
    except ResourceLimit:
        raise
    except ValueError:
        pass
    sx,sy,sw,sh = cv2.boundingRect(plan.foreground.astype(np.uint8))
    cx,cy = sx+sw//2,sy+sh//2
    low = style['min_size']
    for center,height in ((cy,2*low),(cy-low,2*low),(cy+low,2*low),(cy,4*low)):
        if not (0 <= center < plan.layout.shape[0] and plan.layout[center,cx]):
            continue
        seed = np.zeros_like(plan.layout);seed[center,cx] = True
        try:
            frame,_,_ = grow_rectangle(plan.layout,seed,(x,max(y,center-height//2),x+w,min(y+h,center+height//2)))
            if not any(np.array_equal(frame,other) for _,other in frames):
                frames.append(('shallow_source_lobe_frame',frame))
        except ResourceLimit:
            raise
        except ValueError:
            pass
    failures = []
    for name,frame in frames:
        try:
            layer,proof,content = typography.layout_dialogue(target,font_data,coverage,
                Image.fromarray(frame.astype(np.uint8)*255),style)
            proof['balloon_flow'] = {'kind':name,'mask_revision':mask_revision(frame),
                'frame_pixels':int(frame.sum()),'failed_frames':failures,
                'membership_changed':False,'cleanup_changed':False}
            return replace(plan,layout=frame,evidence={**plan.evidence,'selected_flow_frame':proof['balloon_flow'],
                'flow_mask_revision':mask_revision(frame)}),layer,proof,content
        except ValueError as error:
            if isinstance(error,ResourceLimit):
                raise
            failures.append({'kind':name,'reason':str(error)})
    raise ValueError('no supported balloon flow frame fits exact target at requested floor: '+str(failures))
