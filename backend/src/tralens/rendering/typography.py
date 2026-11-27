"""Bounded automatic dialogue typesetting; no wording changes or erase decisions.

Comfort margins are a subset of the supplied safe layout. A few size probes share
the existing RAQM work limits; scores are presentation heuristics, not linguistic
judgments, source-font matching or proof of optimal typography.
"""
import re

import cv2
import numpy as np
from PIL import Image

from . import natural as n


POLICY = 'comfortable_dialogue_v1'
MAX_PROBES = 4


def line_quality(original, shaped, advances):
    """Prefer balanced lines without isolated short words or a tiny final line."""
    lines = shaped.split('\n')
    n.require(len(lines) == len(advances) and len(advances) > 0, 'line metric count mismatch')
    width = max(advances)
    orphan_count = 0
    widow_count = 0
    # Do not fight intentionally supplied line breaks or single-word/name inputs.
    if '\n' not in original and len(original.split()) > 2:
        orphan_count = sum(len(line.split()) == 1 and
            0 < sum(char.isalnum() for char in line) <= 2 for line in lines)
        widow_count = int(len(lines) > 1 and len(lines[-1].split()) == 1
            and sum(char.isalnum() for char in lines[-1]) > 2 and advances[-1] < .45*width)
    mean = sum(advances)/len(advances)
    imbalance = sum((value-mean)**2 for value in advances)/(len(advances)*width*width) if width else 0
    # Keep independent spoken beats on separate lines when width permits.
    mixed_beats = sum(bool(re.search(r'[.!?…]\s+\S', line.strip())) for line in lines)
    cost = .23*mixed_beats + .6*orphan_count + .2*widow_count + .18*imbalance + .025*(len(lines)-1)
    return {'cost': cost, 'short_word_orphans': orphan_count, 'tiny_final_line': widow_count,
        'normalized_line_variance': imbalance, 'line_count': len(lines)}


def content_mask(mask, max_size):
    n.budget_check(mask.mode == 'L' and mask.width*mask.height <= n.MAX_PAGE_PIXELS, 'layout page pixel budget')
    bounds = mask.getbbox()
    n.require(bounds is not None, 'empty layout region')
    n.integer(max_size, 8, 144)
    width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
    margin = max(2, min(8, round(min(width, height)*.06), round(max_size*.2)))
    pixels = (np.asarray(mask) != 0).astype(np.uint8)*255
    inset = cv2.erode(pixels, np.ones((2*margin+1, 2*margin+1), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0)
    result = Image.fromarray(inset)
    n.require(result.getbbox() is not None, 'no layout remains inside comfortable text margins')
    return result, margin


def layout_dialogue(text, font_data, coverage, mask, style):
    """Choose a readable contained fit without always maximizing font size.

    At most four descending size caps: supplied max, 90%, 80%, 65%. The explicit
    min/max bounds remain authoritative. All probes share one measurement, DP and
    raster budget; resource/coverage/shaping failures never become a weaker fit.
    """
    low, high = style['min_size'], style['max_size']
    n.integer(low, 8, 144)
    n.integer(high, low, 144)
    content, margin = content_mask(mask, high)
    preferred = max(low, round(high*.9))
    caps = sorted({high, preferred, max(low, round(high*.8)), max(low, round(high*.65))}, reverse=True)
    n.budget_check(len(caps) <= MAX_PROBES, 'typography probe budget')
    work = n.LayoutWork()
    if style.get('preset')=='thought' and '\n' not in text:
        offsets = [m.end() for m in re.finditer(r'[!?…]|(?<!\.)\.(?!\.)|\.{3}',text)
                   if text[m.end():].startswith(' ')]
        offsets = [i+1 for i in offsets if i+1<len(text) and text[i+1]!=' ']
        if 1<=len(offsets)<=4:
            try:
                layer,proof = n.layout(text,font_data,coverage,content,
                    {**style,'min_size':max(low,round(high*.70))},work=work,fixed_breaks=offsets)
            except n.ResourceLimit:
                raise
            except ValueError as error:
                if not str(error).startswith('overflow:'):
                    raise
            else:
                proof['line_break_policy'] = 'thought_sentence_breaks'
                return layer,proof,content
    probes = []
    best = None
    last_size = high+1
    for cap in caps:
        if cap >= last_size:
            continue  # A larger cap already found the same largest fitting size.
        try:
            layer, provenance = n.layout(text, font_data, coverage, content, {**style, 'max_size': cap}, work=work,
                line_rank=lambda shaped, advances: line_quality(text, shaped, advances)['cost'])
        except n.ResourceLimit:
            raise
        except ValueError as error:
            if str(error) != 'overflow: no contained layout within supplied size bounds':
                raise
            probes.append({'max_size': cap, 'status': 'overflow'})
            # A smaller cap cannot fit if the entire remaining range was tried.
            break
        last_size = provenance['size']
        quality = line_quality(text, provenance['draw_text'], provenance['line_fit_advances'])
        size_cost = 5*((last_size-preferred)/preferred)**2
        score = round(quality['cost']+size_cost, 8)
        probes.append({'max_size': cap, 'status': 'fit', 'size': last_size, 'score': score,
            'line_quality': quality})
        key = (score, -last_size, tuple(provenance['inserted_break_offsets']))
        if best is None or key < best[0]:
            box = layer.getchannel('A').getbbox()
            best = key, layer.crop(box), box[:2], provenance, len(probes)-1
    n.require(best is not None, 'overflow: no contained layout within comfortable margins and supplied size bounds')
    _, tile, xy, provenance, index = best
    # Keep a short hyphenated name or compound together when one line still
    # fits at a readable size. Preserve the two-line fit for narrower balloons.
    if (provenance['inserted_break_offsets'] and
            re.fullmatch(r"[A-Za-z]{2,12}[-‐][A-Za-z]{2,9}[.!?…]*",text)):
        try:
            single,metrics = n.layout(text,font_data,coverage,content,
                {**style,'min_size':max(low,round(provenance['size']*.75)),
                 'max_size':provenance['size']},work=work,fixed_breaks=())
        except n.ResourceLimit:
            raise
        except ValueError as error:
            if not str(error).startswith('overflow:'):
                raise
        else:
            box = single.getchannel('A').getbbox()
            tile,xy,provenance = single.crop(box),box[:2],metrics
            probes[index]['unbroken_size'] = metrics['size']
    layer = Image.new('RGBA', mask.size)
    layer.paste(tile, xy)
    provenance.update(rasterized_pixels=work.rasterized_pixels, search_work=dict(work.search),
        line_break_policy='comfortable_legal_breaks_v1',
        typography={'policy': POLICY, 'margin_pixels': margin, 'content_bounds': list(content.getbbox()),
            'content_pixels': int(np.count_nonzero(np.asarray(content))), 'preferred_size': preferred,
            'chosen_probe': index, 'probes': probes, 'score_scope': 'bounded presentation heuristic; not an optimality or language claim'})
    return layer, provenance, content


def interior_box(paper):
    """Find a contained text rectangle in a small irregular balloon or caption."""
    ys,xs = np.nonzero(paper)
    if not len(xs):
        return Image.fromarray(np.uint8(paper)*255)
    x,y,w,h = cv2.boundingRect(np.uint8(paper))
    tile = paper[y:y+h,x:x+w]
    sums = cv2.integral(np.uint8(~tile))
    # A balloon tail or a thin connected gutter should not dictate the aspect
    # ratio or center of the English text.
    aspect = np.clip((np.percentile(xs,90)-np.percentile(xs,10)+1) /
                     (np.percentile(ys,90)-np.percentile(ys,10)+1),.5,5.)
    cx,cy = np.median(xs)-x,np.median(ys)-y
    result = np.zeros_like(paper,np.uint8)
    lo,hi = 1,h
    while lo<=hi:
        rh = (lo+hi)//2
        rw = max(1,round(rh*aspect))
        if rw>w:
            hi = rh-1
            continue
        outside = sums[rh:,rw:]-sums[:-rh,rw:]-sums[rh:,:-rw]+sums[:-rh,:-rw]
        rows,cols = np.nonzero(outside==0)
        if not len(rows):
            hi = rh-1
            continue
        i = ((rows+(rh-1)/2-cy)**2+(cols+(rw-1)/2-cx)**2).argmin()
        left,top = x+cols[i],y+rows[i]
        result.fill(0)
        result[top:top+rh,left:left+rw] = 255
        lo = rh+1
    return Image.fromarray(result)
