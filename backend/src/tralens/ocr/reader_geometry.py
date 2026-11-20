"""Replayable quadrilateral presentations for readers, never cleanup geometry.

The original page and detector polygon stay in native coordinates. Rectification
interpolates a separate reader input; its inverse transform is evidence, not a
text-ownership or erase mask. No writing direction or reading is selected here.
"""
import math

import cv2
import numpy as np
from PIL import Image

from .recognition_units import image_fingerprint


POLICY = 'reader_only_quad_rectification_v1'
VIEWS = ('tight', 'axial', 'context')
MAX_SIDE = 1040
MAX_PIXELS = 300000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plan(size, polygon, *, view='tight', quarter_turns=0):
    """Size follows opposite-edge lengths; padding samples surrounding page pixels.

    Vertex order may start at any corner and run either way, but must describe a
    convex boundary. Quarter-turns are counterclockwise. None of these geometric
    choices claims that the polygon contains a complete passage or even text.
    """
    require(len(size)==2 and all(type(v) is int and v>0 for v in size), 'invalid reader page size')
    require(size[0]*size[1]<=4_000_000, 'reader page pixel budget')
    require(view in VIEWS and type(quarter_turns) is int and 0<=quarter_turns<=3,
        'invalid reader view/orientation')
    points=np.asarray(polygon)
    require(points.shape==(4,2) and points.dtype.kind in 'fiu', 'reader requires a numeric quadrilateral')
    points=points.astype(np.float64)
    require(np.isfinite(points).all() and (points>=0).all()
        and (points<=np.asarray(size)).all(), 'reader polygon outside source page')
    edges=np.roll(points,-1,axis=0)-points
    turns=edges[:,0]*np.roll(edges[:,1],-1)-edges[:,1]*np.roll(edges[:,0],-1)
    require((turns>1e-6).all() or (turns < -1e-6).all(), 'reader requires a convex nondegenerate boundary')
    ordered=points.copy() if turns[0]>0 else points[::-1].copy()
    first=min(range(4),key=lambda i:(sum(ordered[i]),ordered[i,1],ordered[i,0]))
    ordered=np.roll(ordered,-first,axis=0)
    lengths=np.linalg.norm(np.roll(ordered,-1,axis=0)-ordered,axis=1)
    width=math.ceil(max(lengths[0],lengths[2]))+1
    height=math.ceil(max(lengths[1],lengths[3]))+1
    require(min(width,height)>=2, 'reader quadrilateral too small')
    short=min(width,height)
    along=max(8,min(64,math.ceil(short*.75)))
    cross=max(6,min(16,math.ceil(short*.3)))
    if view=='tight':px=py=2
    elif width>=height:px=along;py=2 if view=='axial' else cross
    else:py=along;px=2 if view=='axial' else cross
    base_size=[width+2*px,height+2*py]
    require(max(base_size)<=MAX_SIDE and math.prod(base_size)<=MAX_PIXELS, 'reader presentation pixel budget')
    destination=np.float32([[px,py],[px+width-1,py],[px+width-1,py+height-1],[px,py+height-1]])
    matrix=cv2.getPerspectiveTransform(ordered.astype(np.float32),destination)
    inverse=np.linalg.inv(matrix)
    bw,bh=base_size
    # Reject padded views crossing the projective horizon rather than sampling
    # enormous/discontinuous areas from an otherwise valid seed quadrilateral.
    corners=np.array([[0,0,1],[bw-1,0,1],[bw-1,bh-1,1],[0,bh-1,1]],dtype=np.float64)
    denominators=(corners@inverse.T)[:,2]
    require(np.isfinite(matrix).all() and np.isfinite(inverse).all()
        and ((denominators>1e-8).all() or (denominators < -1e-8).all()), 'reader projective horizon')
    rotation=np.asarray(([[1,0,0],[0,1,0],[0,0,1]],
        [[0,1,0],[-1,0,bw-1],[0,0,1]],
        [[-1,0,bw-1],[0,-1,bh-1],[0,0,1]],
        [[0,-1,bh-1],[1,0,0],[0,0,1]])[quarter_turns],dtype=np.float64)
    forward=rotation@matrix
    return {'policy':POLICY,'source_size':list(size),'source_polygon':points.tolist(),
        'ordered_polygon':ordered.tolist(),'view':view,'quarter_turns':quarter_turns,
        'seed_size':[width,height],'padding':[px,py],'base_size':base_size,
        'size':base_size[::-1] if quarter_turns%2 else base_size.copy(),
        'source_to_base':matrix.tolist(),'source_to_view':forward.tolist(),
        'view_to_source':np.linalg.inv(forward).tolist(),'interpolation':'linear',
        'border_mode':'constant','border_rgb':[255,255,255],
        'axis_is_writing_direction':False,'erase_permission':False}


def pixels(image, mapping):
    """Replay only a canonical geometry plan over an original RGB page."""
    require(image.mode=='RGB' and list(image.size)==mapping['source_size'], 'reader source size/mode drift')
    expected=plan(image.size,mapping['source_polygon'],view=mapping['view'],quarter_turns=mapping['quarter_turns'])
    require(mapping==expected, 'reader geometry plan drift')
    rectified=cv2.warpPerspective(np.asarray(image),np.asarray(mapping['source_to_base']),
        tuple(mapping['base_size']),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(mapping['border_rgb']))
    return Image.fromarray(np.rot90(rectified,mapping['quarter_turns']).copy())


def source_coordinates(mapping):
    """Continuous source centers for each reader pixel; values may be off-page.

    Linear interpolation may also touch off-page neighbors. The constant border
    is declared in the plan. These coordinates must never become cleanup support.
    """
    expected=plan(mapping['source_size'],mapping['source_polygon'],view=mapping['view'],
        quarter_turns=mapping['quarter_turns'])
    require(mapping==expected, 'reader coordinate plan drift')
    w,h=mapping['size'];y,x=np.indices((h,w),dtype=np.float64)
    inverse=np.asarray(mapping['view_to_source'])
    divisor=inverse[2,0]*x+inverse[2,1]*y+inverse[2,2]
    return ((inverse[0,0]*x+inverse[0,1]*y+inverse[0,2])/divisor,
        (inverse[1,0]*x+inverse[1,1]*y+inverse[1,2])/divisor)


def present(image, polygon, *, view='tight', quarter_turns=0):
    mapping=plan(image.size,polygon,view=view,quarter_turns=quarter_turns)
    crop=pixels(image,mapping)
    return crop,{'geometry':mapping,'source_pixels_sha256':image_fingerprint(image),
        'view_pixels_sha256':image_fingerprint(crop),'opencv_version':cv2.__version__,
        'selected':False,'erase_permission':False}


def replay(image, evidence):
    require(evidence['selected'] is False and evidence['erase_permission'] is False
        and evidence['opencv_version']==cv2.__version__
        and evidence['source_pixels_sha256']==image_fingerprint(image), 'reader presentation binding drift')
    crop=pixels(image,evidence['geometry'])
    require(evidence['view_pixels_sha256']==image_fingerprint(crop), 'reader presentation pixels drift')
    return crop
