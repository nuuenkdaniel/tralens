"""Separate tightly stacked OCR rows when the scan shows repeated table rules."""
import numpy as np

from .types import Text_Group


def printed_grid(first, second, pixels):
    upper, lower = sorted((first, second), key=lambda box: sum(box.bbox[1::2]))
    ax, ay, ar, ab = upper.bbox
    bx, by, br, bb = lower.bbox
    left, right = max(ax, bx), min(ar, br)
    height = min(ab - ay, bb - by)
    pitch = ((by + bb) - (ay + ab)) / 2
    if height < 9 or right-left < max(25, 1.4*height) or not .60*height <= pitch <= 1.6*height:
        return False
    lo = max(0, int(left-2*height))
    hi = min(pixels.shape[1], int(right+height))
    if hi-lo < 55:
        return False
    x = np.unique(np.linspace(lo, hi-1, min(150,hi-lo)).astype(int))
    outside = x < left
    if outside.sum() < 9:
        return False
    slopes = []
    for box in (upper,lower):
        edges = np.roll(box.poly,-1,axis=0)-box.poly
        dx,dy = max(edges,key=lambda edge: abs(edge[0]))
        slopes.append(dy/dx)
    tilt = np.median(slopes)
    if abs(tilt) > .2:
        return False
    middle = (ay+ab+by+bb)/4
    top = max(0,int(middle-2*pitch-16))
    bottom = min(pixels.shape[0],int(middle+2*pitch+16))
    gray = pixels[top:bottom,lo:hi].max(2).astype(np.int16)
    xx = x-lo
    for slope in (tilt,tilt-.025,tilt+.025):
        drift = (x-(left+right)/2)*slope
        bands = []
        for k in range(-2,3):
            centers = np.arange(round(middle+(k-.30)*pitch),round(middle+(k+.30)*pitch)+1)
            ys = np.rint(centers[:,None]+drift-top).astype(int)
            ys = ys[(ys.min(1)>=3)&(ys.max(1)<len(gray)-3)]
            if not len(ys):
                bands.append(False)
                continue
            stroke = np.minimum(gray[ys,xx],gray[ys+1,xx])
            sides = np.minimum(gray[ys-2,xx],gray[ys+3,xx])
            ink = sides-stroke >= 15
            # Lines must extend beyond the text column into the adjacent cell.
            bands.append(np.any((ink.mean(1)>=.68)&(ink[:,outside].mean(1)>=.68)))
        if any(all(bands[i:i+3]) for i in range(3)):
            return True
    return False


def split(groups, pixels, excluded):
    result=[]
    for group in groups:
        boxes=group.group
        if (group.direction!='horizontal' or not 3<=len(boxes)<=12
                or any(box.box_id in excluded for box in boxes)):
            result.append(group)
            continue
        cuts=[i for i in range(1,len(boxes)) if printed_grid(boxes[i-1],boxes[i],pixels)]
        if len(cuts)<2:
            result.append(group)
            continue
        for start,end in zip([0,*cuts],[*cuts,len(boxes)]):
            result.append(Text_Group(boxes[start:end],method='separate_printed_table_rows',direction='horizontal'))
    return result
