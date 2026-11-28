"""Exact lettering for a source whose usual balloon layout could not fit."""
from io import BytesIO
import math
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def wrap(text, font, width):
    """Retain every character and prefer spaces. Split long words only as needed."""
    lines = []
    for paragraph in text.split('\n'):
        while paragraph and font.getlength(paragraph.rstrip()) > width:
            lo, hi = 0, len(paragraph)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if font.getlength(paragraph[:mid].rstrip()) <= width:
                    lo = mid
                else:
                    hi = mid - 1
            cut = max((i+1 for i,c in enumerate(paragraph[:lo]) if c.isspace()), default=lo)
            if cut == 0:
                return None
            # If a forced break lands just before closing punctuation, carry
            # the last letter across with it. Keep explicit newlines intact.
            if cut > 1 and paragraph[cut:cut + 1] in ('.', ',', '!', '?', ':', ';', ')', ']', '…'):
                cut -= 1
            while cut < len(paragraph) and unicodedata.combining(paragraph[cut]):
                cut += 1
            lines.append(paragraph[:cut])
            paragraph = paragraph[cut:]
        lines.append(paragraph)
    return lines


def fit(text, font_data, bounds, blocked, style, clean):
    """Find the closest unoccupied rectangle in this source's neighborhood."""
    left,top,right,bottom = bounds
    h,w = blocked.shape
    cx,cy = (left+right)/2,(top+bottom)/2
    integral = cv2.integral(blocked.astype(np.uint8))
    probe = ImageDraw.Draw(Image.new('L',(1,1)))
    paper = np.median(clean[top:bottom,left:right].reshape(-1,3),axis=0)
    distance = np.abs(clean.astype(np.int16)-paper.astype(np.int16)).max(2)
    flat = np.mean(distance[top:bottom,left:right] < 22) > .6
    edges = cv2.integral((distance > 40).astype(np.uint8)) if flat else None
    floor = style.get('readability_floor', 8)
    target = max(floor, 12, style['min_size'])
    best = None
    sizes = sorted({max(floor,round(target*f)) for f in (1,.85,.70,.56,.44,.34)} | {floor},reverse=True)
    for size in sizes:
        font = ImageFont.truetype(BytesIO(font_data),size)
        preferred = int(size * math.sqrt(max(1,len(text))) * 1.1)
        spans = [right-left, int(1.35*(right-left)), preferred, 2*preferred, w-2]
        if 'readability_floor' in style:
            words = text.split()
            if words:
                spans.append(math.ceil(max(font.getlength(word) for word in set(words))) + 2)
            spans.extend(range(max(size * 3, preferred // 2), min(w, 2 * preferred), size))
        widths = sorted({min(w-2,max(8,v)) for v in spans})
        for width in widths:
            lines = wrap(text,font,width-2)
            if not lines:
                continue
            shaped = '\n'.join(line.rstrip() for line in lines)
            spacing = max(1,size//8)
            outline = max(1, round(floor / 9))
            align = 'left' if 'readability_floor' in style and len(lines)>=4 and len(text)>=100 else 'center'
            options = dict(font=font,align=align,spacing=spacing,stroke_width=outline)
            x,y,r,b = probe.multiline_textbbox((0,0),shaped,**options)
            tw,th = math.ceil(r)-math.floor(x),math.ceil(b)-math.floor(y)
            if not (0 < tw <= w and 0 < th <= h):
                continue
            free = integral[th:,tw:] - integral[:-th,tw:] - integral[th:,:-tw] + integral[:-th,:-tw] == 0
            stride = 3
            ys,xs = np.nonzero(free[::stride,::stride])
            if len(xs):
                ys,xs = ys*stride,xs*stride
            else:
                ys,xs = np.nonzero(free)
                if not len(xs):
                    continue
            # All penalties are expressed in units of the requested type size.
            # Crossing a panel, separating a word or displacing the whole line
            # costs more than a modest reduction of the font.
            spill_x = np.maximum(0,left-xs) + np.maximum(0,xs+tw-right)
            spill_y = np.maximum(0,top-ys) + np.maximum(0,ys+th-bottom)
            costs = ((xs+tw/2-cx)**2 + (ys+th/2-cy)**2 + spill_x**2 + spill_y**2)/target**2
            if edges is not None:
                contrast = edges[ys+th,xs+tw]-edges[ys,xs+tw]-edges[ys+th,xs]+edges[ys,xs]
                costs += 60*contrast/(tw*th)
            idx = costs.argmin()
            breaks = sum(a and b and a[-1].isalnum() and b[0].isalnum()
                         for a,b in zip(lines,lines[1:]) if a and b)
            rank = float(costs[idx]) + 32*breaks + 2*(target/size-1)**2
            if 'readability_floor' in style:
                rank = (breaks, rank)
            if best is None or rank < best[0]:
                best = rank,(int(xs[idx]),int(ys[idx])),(tw,th),(-x,-y),shaped,options,lines,size
    if best:
        _,xy,dims,origin,shaped,options,lines,size = best
        return xy,dims,origin,shaped,options,dict(original_text=text,lines=lines,size=size)
    raise ValueError('no unoccupied local rectangle for compact lettering')


def letters(clean, bounds, blocked, text, font_data, style):
    left,top,right,bottom = bounds
    full_h,full_w = blocked.shape
    # Start locally. An especially dense panel can leave the only contiguous
    # space for a long paragraph just outside the first search window.
    limits = (768, 1280) if 'readability_floor' in style and max(full_w,full_h)>768 else (768,)
    for limit in limits:
        sw,sh = min(full_w,limit),min(full_h,limit)
        sx = min(full_w-sw,max(0,(left+right-sw)//2))
        sy = min(full_h-sh,max(0,(top+bottom-sh)//2))
        roi = np.s_[sy:sy+sh,sx:sx+sw]
        frame = max(0,left-sx),max(0,top-sy),min(sw,right-sx),min(sh,bottom-sy)
        try:
            xy,(w,h),origin,shaped,options,proof = fit(text,font_data,frame,blocked[roi],style,clean[roi])
            break
        except ValueError:
            if limit == limits[-1]:
                raise
    xy = (xy[0]+sx,xy[1]+sy)
    x,y = xy
    surface = clean[y:y+h,x:x+w]
    light = np.median(surface) >= 128
    ink,outline = ('black','white') if light else ('white','black')
    tile = Image.new('RGBA',(w,h))
    draw = ImageDraw.Draw(tile)
    if 'readability_floor' in style and len(proof['lines']) >= 3:
        # An outline separates a short word from line art. A paragraph over
        # detailed artwork needs contrast in the spaces between words as well.
        low,high = np.percentile(surface,(10,90))
        if high - low > 100:
            backing = (255,255,255,225) if light else (0,0,0,225)
            draw.rounded_rectangle((0,0,w-1,h-1),radius=options['stroke_width']*2,fill=backing)
            proof['contrast_backing'] = True
    draw.multiline_text(origin,shaped,fill=ink,stroke_fill=outline,**options)
    layer = Image.new('RGBA',(clean.shape[1],clean.shape[0]))
    layer.paste(tile,xy)
    proof.update(draw_text=shaped, ink_bounds=layer.getchannel('A').getbbox())
    return layer, proof


def paint(clean, erase, bounds, blocked, text, font_data, style):
    layer, proof = letters(clean, bounds, blocked, text, font_data, style)
    patch = Image.fromarray(clean).convert('RGBA')
    mask = erase | (np.asarray(layer.getchannel('A')) != 0)
    patch.putalpha(Image.fromarray(mask))
    return patch,layer,proof
