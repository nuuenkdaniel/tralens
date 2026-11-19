"""Read an upright Latin abbreviation in a vertical Japanese column."""
import re
import unicodedata

import numpy as np
from PIL import Image


POLICY = 'upright_latin_reflow_v1'
BAR = re.compile(r'^[A-Z]{2,8}([-ー一])(?=[ぁ-ん])')


def normalized(text):
    return unicodedata.normalize('NFKC',text)


def presentations(image,record,columns):
    if record['direction']!='vertical' or not 1<=len(columns)<=3:
        return None
    match=BAR.match(normalized(columns[0]))
    if not match:
        return None
    parts=[]
    for i,(column,polygon) in enumerate(zip(columns,record['polygons'])):
        length=len(normalized(column))
        if not 1<=length<=12:
            return None
        xs,ys=zip(*polygon)
        left,top,right,bottom=min(xs),min(ys),max(xs)+1,max(ys)+1
        w,h=right-left,bottom-top
        if not (0<=left<right<=image.width and 0<=top<bottom<=image.height
                and min(w,h)>=6 and 1.6*w<=h<=512 and w<=128):
            return None
        pixels=np.asarray(image.crop((left,top,right,bottom)))
        if np.mean(pixels.min(2)>238)<.65:
            return None
        ink=pixels.max(2)<130
        occupied=ink.any(1)
        rows=np.flatnonzero(occupied)
        if not len(rows):
            return None
        top,bottom=rows[0],rows[-1]+1
        pitch=(bottom-top)/length
        if pitch<7:
            return None
        cuts=[top]
        for j in range(1,length):
            center=top+j*pitch
            gaps=np.flatnonzero(~occupied & (np.abs(np.arange(h)-center)<.26*pitch))
            if not len(gaps):
                return None
            cuts.append(gaps[np.argmin(abs(gaps-center))])
        cuts.append(bottom)
        for start,end in zip(cuts,cuts[1:]):
            ys,xs=np.nonzero(ink[start:end])
            if not len(xs):
                return None
            parts.append(Image.fromarray(pixels[start+ys.min():start+ys.max()+1,xs.min():xs.max()+1]))
        if i==0:
            bar=parts[match.start(1)]
            if bar.width>.32*bar.height or any(not .75<=part.height/bar.height<=1.35
                    for part in parts[:match.start(1)]):
                return None
    height=max(part.height for part in parts)
    views=[]
    for gap in (2,5):
        canvas=Image.new('RGB',(12+sum(part.width+gap for part in parts),height+12),'white')
        x=6
        for part in parts:
            canvas.paste(part,(x,6+(height-part.height)//2))
            x+=part.width+gap
        views.append(canvas)
    return views


def matches(record):
    source=normalized(record['source'])
    match=BAR.match(source)
    if not match or len(record['readings'])!=3:
        return False
    idx=match.start(1)
    fixed=source[:idx]+'I'+source[idx+1:]
    first,second,native=map(normalized,record['readings'])
    return (first==second==fixed and len(native)==len(source) and native[idx] in '-ー一I'
        and native[:idx]==source[:idx] and native[idx+1:]==source[idx+1:])
