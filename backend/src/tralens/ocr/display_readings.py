"""Bounded reader presentations for oversized, horizontal painted lettering."""
import cv2
import numpy as np
from PIL import Image,ImageDraw


POLICY='oversized_color_ink_views_v1'


def clipped_repetitions(image, boxes, reader):
    """Two detected kanji can be the opening of a repeated hand-lettered line."""
    import re
    from collections import Counter
    from .masked_lines import bind, corrected, source_views
    counts = Counter(b.text for b in boxes if re.fullmatch(r'[一-鿿]{2}[ぁ-ゖァ-ヺー]{1,5}',b.text))
    repeats = {word[:2]:word for word in counts if counts[word]>=2
               and sum(other.startswith(word[:2]) and counts[other]>=2 for other in counts)==1}
    if not repeats:
        return boxes
    result = list(boxes)
    for i,first in enumerate(boxes):
        if not any(word.startswith(first.text) for word in repeats) or len(first.text)!=1:
            continue
        x,y,r,b = first.bbox
        for j,second in enumerate(boxes):
            word = repeats.get(first.text+second.text)
            if not word:
                continue
            l,t,right,bottom = second.bbox
            height = bottom-t
            if (not 12<=height<=70 or not -.45*height<=l-r<=.3*height
                    or abs(b-bottom)>.35*height or not .65*height<b-y<1.35*height):
                continue
            end = min(image.width-1,right+round(.72*height*(len(word)-2)))
            others = [(k,v) for k,v in enumerate(boxes) if k not in (i,j) and
                      v.bbox[0]<end and v.bbox[2]>right and v.bbox[1]<bottom and v.bbox[3]>t]
            if len(others)>1:
                continue
            tail = np.int32([[l,t],[end,t],[end,bottom],[l,bottom]])
            parts = [(i,first,first.poly,first.text),(j,second,tail,word[1:])]
            if others:
                k,extra = others[0]
                a,top,z,base = extra.bbox
                if (extra.reason!='frontier_discovery' or not extra.text or not word[2:].startswith(extra.text)
                        or not right<=a<z<=end or abs(t-top)+abs(bottom-base)>height*.45):
                    continue
                suffix = np.int32([[a,t],[end,t],[end,bottom],[a,bottom]])
                parts[1] = (j,second,second.poly,second.text)
                parts.append((k,extra,suffix,word[2:]))
            try:
                crops = source_views(image,[poly for _,_,poly,_ in parts])
                if (any(reader(v)!=word for v in crops)
                        or any(reader(v)!=word[1:] for v in source_views(image,[tail]))
                        or (others and any(reader(v)!=word[2:] for v in source_views(image,[suffix])))
                        or reader(source_views(image,[first.poly])[1])!=first.text):
                    continue
            except (ValueError,RuntimeError,OSError,cv2.error):
                continue
            members = [corrected(v.box_id,v.confidence,poly,text,v.paddle_text,v) for _,v,poly,text in parts]
            bind(members,word,crops,'horizontal')
            for (index,_,_,_),member in zip(parts,members):
                result[index] = member
    return result


def split_interjections(image, boxes):
    """A two-kana reaction can be timed across two separately framed close-ups."""
    from copy import copy
    words={'あら','あれ','おや','まあ','へえ','ほう'}
    singles=[(i,b) for i,b in enumerate(boxes) if len(b.text)==1 and b.confidence>=.85]
    gray=None
    matches=[]
    for i,first in singles:
        x,y,r,b=first.bbox
        for j,second in singles:
            word=first.text+second.text
            if word not in words:
                continue
            l,t,right,bottom=second.bbox
            size=min(b-y,bottom-t)
            if (size<8 or not 4*size<x-right<14*size or abs(y-t)>.4*size or
                    max(b-y,bottom-t)>1.3*size or abs((r-x)-(right-l))>.4*size):
                continue
            if any(k not in (i,j) and right<(v.bbox[0]+v.bbox[2])/2<x
                   and v.bbox[3]>y-.4*size and v.bbox[1]<b+.4*size for k,v in enumerate(boxes)):
                continue
            if gray is None:
                gray=np.asarray(image.convert('L'))
            strip=gray[max(0,y-size):min(image.height,int(b+.65*size)),right:x]
            edges=np.flatnonzero((strip<85).mean(0)>.94)
            bars=[part for part in np.split(edges,np.flatnonzero(np.diff(edges)>1)+1)
                  if 0<len(part)<=.3*size]
            if any(.12*size<nxt[0]-prev[-1]-1<1.7*size and
                   (strip[:,prev[-1]+1:nxt[0]]>220).mean()>.88 for prev,nxt in zip(bars,bars[1:])):
                matches.append((i,j,word))
    result=list(boxes)
    for i,j,word in matches:
        if any((a,b)!=(i,j) and (i in (a,b) or j in (a,b)) for a,b,_ in matches):
            continue
        for index,beat in [(i,0),(j,1)]:
            member=copy(boxes[index])
            member.ocr_metadata={**member.ocr_metadata,'split_interjection':{'word':word,'beat':beat}}
            result[index]=member
    return result


def broad_pen_ink(gray,radius):
    """Release broad ink strokes from thin adjacent illustration and frame lines."""
    ink=np.uint8(gray<150)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    _,labels,stats,_=cv2.connectedComponentsWithStats(cv2.erode(ink,kernel))
    keep=[i for i,(x,y,w,h,area) in enumerate(stats[1:],1) if area>=15 and
          min(x,y)>2 and x+w<gray.shape[1]-2 and y+h<gray.shape[0]-2 and
          not (max(w,h)>4*min(w,h) and min(w,h)<=radius)]
    if not 2<=len(keep)<=10:
        return
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+5,2*radius+5))
    return cv2.dilate(np.uint8(np.isin(labels,keep)),kernel)&np.uint8(gray<160)


def voiced_motion(image,box,reader):
    """Restore a large kana's voicing mark and its separate small tsu."""
    import re,unicodedata
    from .masked_lines import bind,corrected,source_views
    from .recognition_units import image_fingerprint
    x,y,r,b=box.bbox
    if (not re.fullmatch('[ぁ-ゖァ-ヺ]',box.text) or box.confidence>=.75
            or not 70<=min(r-x,b-y)<=max(r-x,b-y)<=400 or min(r-x,b-y)<.57*max(r-x,b-y)):
        return
    crops=source_views(image,[box.poly])
    word=reader(crops[0])
    if not re.fullmatch(r'[ぁ-ゖァ-ヺ][っッ]',word or ''):
        return
    plain=unicodedata.normalize('NFD',word[0])
    if len(plain)!=2 or plain[0]!=box.text or any(reader(c)!=word for c in crops[1:]):
        return
    pad=max(6,round(.055*min(r-x,b-y)))
    bounds=[max(0,x-pad),max(0,y-pad),min(image.width,r+1+pad),min(image.height,b+1+pad)]
    native=image.crop(bounds)
    gray=np.asarray(native.convert('L'))
    radius=max(2,round(min(gray.shape)*.015))
    first=broad_pen_ink(gray,radius)
    second=broad_pen_ink(gray,max(1,radius-1))
    if first is None or second is None or not first.any() or not second.any():
        return
    for mask in (first,second):
        l,t,w,h=cv2.boundingRect(mask)
        chip=Image.fromarray(np.where(mask!=0,gray,255)).convert('RGB').crop((l,t,l+w,t+h))
        if reader(chip)!=word:
            return
    member=corrected(box.box_id,box.confidence,box.poly,word,box.paddle_text,box)
    bind([member],word,crops,'vertical')
    member.ocr_metadata['heavy_print']={'bounds':bounds,'digest':image_fingerprint(native),'motion_radius':radius}
    return member


def heavy_print_strokes(gray):
    """Isolate wide pen strokes disconnected from the drawing at the crop edge."""
    count,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(gray<150))
    if count>4500:
        return
    ink=cv2.distanceTransform(np.uint8(gray<150),cv2.DIST_L2,5)
    valid=[]
    for i,(l,t,w,h,area) in enumerate(stats[1:],1):
        if min(l,t)<2 or l+w>=gray.shape[1]-1 or t+h>=gray.shape[0]-1 or area<24:
            continue
        tile=np.s_[t:t+h,l:l+w]
        stroke=ink[tile][labels[tile]==i]
        if np.percentile(stroke,80)>=max(2.,.012*min(gray.shape)):
            valid.append(i)
    if not 4<=len(valid)<=32:
        return
    return np.uint8(np.isin(labels,valid)),stats[valid]


def clipped_exclamation(image, box, reader):
    """A weak vertical reading ending in small tsu may have dropped the final kana and punctuation."""
    import re
    from .masked_lines import bind,corrected,source_views
    x,y,r,b=box.bbox
    if (not re.fullmatch('[ぁ-ゖァ-ヺー一-鿿]{1,6}[っッ]',box.text) or box.confidence>=.80
            or not 65<=r-x<=300 or not 1.1*(r-x)<b-y<=420):
        return
    gray=np.asarray(image.crop((x,y,r+1,b+1)).convert('L'))
    if (gray>235).mean()<.62:
        return
    crops=source_views(image,[box.poly])
    word=reader(crops[0])
    if not re.fullmatch(re.escape(box.text)+r'[ぁ-ゖァ-ヺ][!！]{1,3}',word or ''):
        return
    # Thick lettering can share a detector rectangle with thin speed lines and a balloon edge.
    shapes=heavy_print_strokes(gray)
    if shapes is None:
        return
    body,letters=shapes
    l,t,w,h=cv2.boundingRect(body)
    isolated=Image.fromarray(np.where(body!=0,gray,255)).convert('RGB')
    ending=word[len(word.rstrip('!！')):]
    stems=[v for v in letters if v[1]>t+.54*h and v[3]>=2.1*v[2]]
    if len(stems)!=len(ending):
        return
    left=min(v[0] for v in stems); right=max(v[0]+v[2] for v in stems)
    top=min(v[1] for v in stems);pad=max(3,round(.04*w))
    punctuation=isolated.crop((left-pad,top-pad,right+pad,t+h+pad))
    if (reader(punctuation)!=ending or reader(isolated.crop((l,t,l+w,t+h)))!=word
            or any(reader(crop)!=word for crop in crops[1:])):
        return
    poly=np.float32([(x+l-2,y+t-2),(x+l+w+1,y+t-2),(x+l+w+1,y+t+h+1),(x+l-2,y+t+h+1)])
    new_views=source_views(image,[poly])
    if any(reader(chip)!=word for chip in new_views):
        return
    member=corrected(box.box_id,box.confidence,poly,word,box.paddle_text,box)
    bind([member],word,new_views,'vertical')
    from .recognition_units import image_fingerprint
    member.ocr_metadata['heavy_print']={'bounds':[x,y,r+1,b+1],'digest':image_fingerprint(image.crop((x,y,r+1,b+1)))}
    return member


def diagonal_effects(image, boxes, reader):
    """A descending kana effect sometimes looks like a single digit to page OCR."""
    import re
    from .masked_lines import bind,corrected,source_views
    result=list(boxes)
    for index,box in enumerate(boxes):
        full=clipped_exclamation(image,box,reader) or voiced_motion(image,box,reader)
        if full is not None:
            result[index]=full
            continue
        x,y,r,b=box.bbox
        if (not re.fullmatch(r'[0-9０-９]',box.text) or box.confidence>=.75 or min(r-x,b-y)<80
                or max(r-x,b-y)>512 or not .7<(r-x)/(b-y)<2):
            continue
        try:
            chips=source_views(image,[box.poly])
            word=reader(chips[0])
            if (not re.fullmatch('[ァ-ヺ][ァィゥェォャュョー]{1,5}ッ',word or '')
                    or reader(chips[1])!=word or reader(chips[2])!=word):
                continue
            # Read the large initial glyph and the small terminal glyph on their
            # own. A genuine numeral or a face is unlikely to support both.
            native=chips[0]
            w,h=native.size
            initial=native.crop((0,0,round(.63*w),round(.86*h)))
            final=native.crop((round(.8*w),round(.74*h),w,h))
            if reader(initial)!=word[0] or reader(final)!=word[-1]:
                continue
            member=corrected(box.box_id,box.confidence,box.poly,word,box.paddle_text,box)
            bind([member],word,chips,'horizontal')
            member.ocr_metadata['diagonal_effect']={'original':box.text,'ends':[word[0],word[-1]]}
            result[result.index(box)]=member
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
    return result


def candidate(bounds,size,direction):
    if direction!='horizontal' or len(bounds)!=4 or any(type(v) is not int for v in bounds):
        return False
    x,y,r,b=bounds
    w,h=r-x,b-y
    return (0<=x<r<=size[0] and 0<=y<b<=size[1] and w>=1.7*h
        and w<=1536 and h<=768 and 262144<w*h<=786432)


def presentations(image,record):
    x,y,r,b=record['bounds']
    if not candidate(record['bounds'],image.size,record['direction']):
        return None
    source=image.crop((x,y,r,b))
    own=Image.new('L',source.size)
    for poly in record['polygons']:
        ImageDraw.Draw(own).polygon([(a-x,c-y) for a,c in poly],fill=255)
    own=np.asarray(own)!=0
    hue,saturation,value=cv2.split(cv2.cvtColor(np.asarray(source),cv2.COLOR_RGB2HSV))
    saturated=(saturation>=150)&(value>=120)&own
    if np.count_nonzero(saturated)<.08*own.sum() or not saturated.any():
        return None
    hist=np.bincount(hue[saturated],minlength=180)
    center=int(np.argmax(sum(np.roll(hist,j) for j in range(-4,5))))
    delta=np.abs(hue.astype(np.int16)-center)
    band=saturated&(np.minimum(delta,180-delta)<=6)
    if band.sum()<.75*saturated.sum():
        return None
    count,labels,stats,_=cv2.connectedComponentsWithStats(band.astype(np.uint8))
    if count>8192:
        return None
    keep=(stats[:,cv2.CC_STAT_AREA]>=55)&(stats[:,cv2.CC_STAT_WIDTH]>3)&(stats[:,cv2.CC_STAT_HEIGHT]>3)
    keep[0]=False
    if keep.sum()<2:
        return None
    ink=keep[labels].astype(np.uint8)*255
    images=[source,Image.fromarray(255-ink).convert('RGB'),
        Image.fromarray(255-cv2.dilate(ink,np.ones((5,5),np.uint8))).convert('RGB')]
    for view in images:
        view.thumbnail((720,320))
    return images


def colored_word(image, boxes, reader):
    """Identify one large word split among overlapping, low-confidence OCR boxes."""
    import re
    from .masked_lines import corrected
    from .recognition_units import image_fingerprint
    width,height = image.size
    large = [b for b in boxes if b.confidence<.73 and .075*height<b.bbox[3]-b.bbox[1]<.33*height
             and b.bbox[2]-b.bbox[0]>.075*width]
    if not 2<=len(large)<=12:
        return
    for first in large:
        if not re.fullmatch('[ぁ-ゖァ-ヺ]',first.text):
            continue
        x,y,r,b = first.bbox
        members = [first]
        for other in sorted(large,key=lambda v:v.bbox[0]):
            l,t,right,bottom = other.bbox
            if other not in members and x<l<r and right>r and abs(t-y)<.35*(b-y):
                members.append(other)
                r,b = right,max(b,bottom)
        if len(members)<2 or not 1.9*(b-y)<r-x<5*(b-y) or (r-x)*(b-y)>700000:
            continue
        pad = round((b-y)*.17)
        left,top=max(0,x-pad),max(0,y-pad)
        roi = np.asarray(image.crop((left,top,min(width,r+pad),min(height,b+pad//2))))
        hue,sat,val = cv2.split(cv2.cvtColor(roi,cv2.COLOR_RGB2HSV))
        bright = (sat>=170)&(val>=150)
        seed = bright[y-top:first.bbox[3]-top,x-left:first.bbox[2]-left]
        if seed.mean()<.1:
            continue
        colors = hue[y-top:first.bbox[3]-top,x-left:first.bbox[2]-left][seed]
        center = int(np.bincount(colors,minlength=180).argmax())
        delta = abs(hue.astype(np.int16)-center)
        band = bright & (np.minimum(delta,180-delta)<5)
        count,labels,stats,_ = cv2.connectedComponentsWithStats(band.astype(np.uint8))
        if count>4096:
            continue
        keep = (stats[:,4]>max(55,.0007*band.size)) & (stats[:,2]>6) & (stats[:,3]>8)
        keep[0]=False
        if keep.sum()<4:
            continue
        mask=keep[labels].astype(np.uint8)
        l,t,w,h = cv2.boundingRect(mask)
        if not (w>1.7*h and w>.9*(r-x) and h<1.25*(b-y)):
            continue
        # The isolated color discards the subtitle, a neighboring character and
        # publisher marks. Only accept a word that also reads on the actual cover.
        views = [Image.fromarray(roi),Image.fromarray(255-255*mask).convert('RGB'),
                 Image.fromarray(255-255*cv2.dilate(mask,np.ones((3,3),np.uint8))).convert('RGB')]
        for view in views:
            view.thumbnail((900,320),Image.Resampling.LANCZOS)
        try:
            reads = [reader(v) for v in views]
        except (ValueError,RuntimeError,OSError,cv2.error):
            continue
        if (not isinstance(reads[0],str) or reads!=[reads[0]]*3 or not reads[0].startswith(first.text)
                or not re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿々]{4,18}[！!？?]?',reads[0])):
            continue
        pad=8
        l,t,r,b=max(0,left+l-pad),max(0,top+t-pad),min(width-1,left+l+w+pad),min(height-1,top+t+h+pad)
        poly=np.float32([[l,t],[r,t],[r,b],[l,b]])
        merged=corrected(first.box_id,first.confidence,poly,reads[0],first.paddle_text,first)
        merged.ocr_metadata['colored_word']={'hue':center,'reading':reads[0],
                'source_bounds':[int(left),int(top),int(left+roi.shape[1]),int(top+roi.shape[0])],
                'source_digest':image_fingerprint(Image.fromarray(roi)),
                'reads':reads, 'views':[image_fingerprint(v) for v in views],
                'fragments':[{'id':m.box_id,'polygon':m.poly.tolist(),'text':m.text} for m in members]}
        return [merged if v is first else v for v in boxes if v is first or v not in members]


def nameplates(image, boxes, reader):
    """Read a role and a name when two vertical name parts each carry printed kana."""
    import re
    from .masked_lines import corrected

    result = list(boxes)
    for role in boxes:
        if not (role.confidence>=.9 and re.fullmatch(r'[ぁ-ゖァ-ヺ一-鿿々]{2,16}(?:長|員|主任|監督|代表|顧問|主将|担当)',role.text)):
            continue
        x,y,r,b = role.bbox
        w = r-x
        if not (12<=w<=110 and 2.3*w<b-y<12*w):
            continue
        names = sorted((v for v in result if v is not role and v.confidence>=.85
                        and re.fullmatch(r'[一-鿿々]{1,4}',v.text) and .6*w<=v.bbox[2]-v.bbox[0]<=1.4*w
                        and .55*w<x-v.bbox[0]<2.2*w and y-.4*w<v.bbox[1]<v.bbox[3]<=b+.3*w),
                       key=lambda v:v.bbox[1])
        if (len(names)!=2 or abs(names[0].bbox[0]-names[1].bbox[0])>.25*w or
                not 0<names[1].bbox[1]-names[0].bbox[3]<w or abs(names[0].bbox[1]-y)>.45*w):
            continue
        annotations = []
        for name in names:
            l,t,right,end = name.bbox
            furigana = [v for v in result if re.fullmatch(r'[ぁ-ゖァ-ヺー]{1,10}',v.text) and
                        .18*w<=v.bbox[2]-v.bbox[0]<=.8*w and right-.45*w<=v.bbox[0]<right+.25*w and
                        abs(v.bbox[1]-t)<.4*w and abs(v.bbox[3]-end)<.4*w]
            if len(furigana)==1:
                annotations.append(furigana[0])
        if len(annotations)!=2 or annotations[0] is annotations[1]:
            continue
        parts = [role,*names,*annotations]
        left,top = min(v.bbox[0] for v in parts),min(v.bbox[1] for v in parts)
        right,bottom = max(v.bbox[2] for v in parts),max(v.bbox[3] for v in parts)
        rect=(left,top,right+1,bottom+1)
        if any(v not in parts and v.bbox[0]<right and v.bbox[2]>left and v.bbox[1]<bottom
               and v.bbox[3]>top for v in result):
            continue
        expected = role.text + ''.join(v.text for v in names)
        def padded(bounds, pad):
            a,t,z,b=bounds
            return image.crop((max(0,a-pad),max(0,t-pad),min(image.width,z+pad),min(image.height,b+pad)))
        try:
            if any(reader(padded(rect,k))!=expected for k in (0,3)):
                continue
            readings = []
            for ruby in annotations:
                a,t,z,b = ruby.bbox
                pair = [reader(padded((a,t,z+1,b+1),k)) for k in (0,2)]
                if pair[0]==pair[1] and isinstance(pair[0],str) and re.fullmatch(r'[ぁ-ゖァ-ヺー]{1,10}',pair[0]):
                    readings.append(pair[0])
        except (OSError,ValueError,RuntimeError,cv2.error):
            continue
        if len(readings)!=2:
            continue
        poly=np.float32([[left,top],[right,top],[right,bottom],[left,bottom]])
        card=corrected(role.box_id,min(v.confidence for v in [role,*names]),poly,expected,role.text,role)
        card.ocr_metadata['nameplate']={'role':role.text,'names':[v.text for v in names], 'kana':readings}
        result=[card if v is role else v for v in result if v is role or v not in parts]
    return result


def quoted_footnotes(engine, image, reader):
    """Recover inline Latin spelling from a margin note split across OCR boxes."""
    import re
    from PIL import ImageOps
    from .masked_lines import corrected, source_views
    from .automatic_units import propose_page_from_boxes

    boxes = engine.raw_detections[0]
    by_id = {v.box_id:v for v in boxes}
    result = list(boxes)
    for group in engine.automatic_unit_proposals[0].proposals:
        if group.direction != 'horizontal' or not 1 <= len(group.main_ids) <= 3 or group.member_ids != group.main_ids:
            continue
        parts = [by_id[i] for i in group.main_ids]
        source = ''.join(v.text for v in parts)
        l,t,r,b = group.crop_bounds
        h = b-t+1
        if (not 15 <= h <= 80 or not 6*h < r-l <= 840 or min(v.confidence for v in parts) >= .94
                or not re.search(r'[“"「][^”"」 ]{0,5}[A-Za-z]{3}', source)):
            continue
        bounds=(max(0,l-round(.33*h)),max(0,t-1),min(image.width,r+round(.16*h)),min(image.height,b-1))
        native=image.crop(bounds)
        # The recognizer needs a quiet margin after the final character.
        # At the page edge this border is outside the source footprint.
        if bounds[2] == image.width:
            native = ImageOps.expand(native, border=(0,0,max(1,round(.12*h)),0), fill='white')
        zoom=native.resize((round(native.width*1.18),round(native.height*1.18)),Image.Resampling.BICUBIC)
        try:
            reading=reader(native)
            if (not isinstance(reading,str) or not reading.startswith('※') or not 10<=len(reading)<=96
                    or reader(zoom)!=reading):
                continue
            quoted = re.findall(r'["“「]([A-Za-z][A-Za-z0-9 -]{2,24})["”」]',reading)
            if len(quoted)!=1 or quoted[0] not in reader(source_views(image,[v.poly for v in parts])[1]):
                continue
        except (OSError,ValueError,RuntimeError,cv2.error):
            continue
        x,y,right,bottom=bounds
        markers=[v for v in boxes if v not in parts and v.confidence<.7
                 and x-h*.4 <= v.bbox[0] <= l and abs(v.bbox[2]-l)<.45*h
                 and v.bbox[1]>=y and v.bbox[3]<=bottom]
        if len(markers)!=1:
            continue
        marker=markers[0]
        mark=image.crop((marker.bbox[0],marker.bbox[1],marker.bbox[2]+1,marker.bbox[3]+1))
        if reader(mark)!='※' or reader(mark.resize((2*mark.width,2*mark.height),Image.Resampling.BICUBIC))!='※':
            continue
        # The bounding rectangle above is only a reading view. Keep the
        # actual lettering footprint tight when another drawing abuts it.
        points=np.vstack([v.poly for v in [marker,*parts]]).astype('float32')
        poly=cv2.boxPoints(cv2.minAreaRect(points))
        head=parts[0]
        merged=corrected(head.box_id,min(v.confidence for v in parts),poly,reading,head.text,head)
        merged.ocr_metadata['quoted_footnote']={'term':quoted[0]}
        result=[merged if v is head else v for v in result if v is head or (v not in parts and v is not marker)]
    if result!=boxes:
        engine.raw_detections=[result]
        engine.automatic_unit_proposals=[propose_page_from_boxes(result,engine.image_path,complementary_ownership=True,regroup_companions=True)]


def printed_alias(image, rows, reader, next_id):
    """Verify a centered alias header, the name below it and optional honorific."""
    from .masked_lines import READ_ERRORS, bind, corrected, source_views
    header=rows[0]
    if (any(z.bbox[2]-z.bbox[0]<=1.6*(z.bbox[3]-z.bbox[1]) for z in rows)
            or any(a.poly[:,1].mean()>=b.poly[:,1].mean() for a,b in zip(rows,rows[1:]))
            or any(not z.bbox[0]<header.poly[:,0].mean()<z.bbox[2] for z in rows[1:])):
        return
    try:
        members=[]
        for z in rows:
            reads=[reader(v) for v in source_views(image,[z.poly])]
            if (not reads[0] or reads!=[reads[0]]*3 or
                    (reads[0]!=z.text and (z is header or not set(z.text)&set(reads[0])))):
                return
            members.append(corrected(z.box_id,z.confidence,z.poly,reads[0],z.paddle_text,z))
        views=source_views(image,[z.poly for z in rows])
        text=''.join(z.text for z in members)
        if all(reader(v)==text for v in views):
            # Keep a physically separate ornament on the original artwork.
            # A face chosen for English lettering may lack the heart or star.
            tail=members[-1]
            ornament=None
            if len(tail.text)>1 and tail.text[-1] in '♡♥☆★':
                l,t,r,b=tail.bbox
                ink=np.asarray(image.crop((l,t,r+1,b+1)).convert('L'))<140
                columns=np.flatnonzero(ink.any(axis=0))
                gaps=np.flatnonzero(np.diff(columns)>2)
                if len(gaps):
                    cut=l+int((columns[gaps[-1]]+columns[gaps[-1]+1])//2)
                    prefix=np.float32([[l,t],[cut,t],[cut,b],[l,b]])
                    suffix=np.float32([[cut+1,t],[r,t],[r,b],[cut+1,b]])
                    if (all(reader(v)==tail.text[-1] for v in source_views(image,[suffix]))
                            and all(reader(v)==tail.text[:-1] for v in source_views(image,[prefix]))):
                        members[-1]=corrected(tail.box_id,tail.confidence,prefix,tail.text[:-1],tail.paddle_text,rows[-1])
                        ornament=corrected(next_id,tail.confidence,suffix,tail.text[-1],'')
                        text=text[:-1]
                        views=source_views(image,[z.poly for z in members])
            bind(members,text,views,'horizontal')
            return members+([ornament] if ornament else [])
    except READ_ERRORS:
        return


def annotated_passages(engine, image, reader):
    """Read a complete printed passage and its much smaller alternate reading."""
    import re
    from .masked_lines import corrected
    from .automatic_units import propose_page_from_boxes

    boxes=engine.raw_detections[0]
    by_id={v.box_id:v for v in boxes}
    result=list(boxes)
    for proposal in engine.automatic_unit_proposals[0].proposals:
        if len(proposal.annotation_ids)==1 and 1<=len(proposal.main_ids)<=3:
            header=by_id[proposal.annotation_ids[0]]
            if re.fullmatch(r'A\.?K\.?A\.?',header.text,re.I):
                rows=[header,*sorted((by_id[i] for i in proposal.main_ids),key=lambda z:z.bbox[1])]
                members=printed_alias(image,rows,reader,max((v.box_id for v in result if type(v.box_id) is int),default=-1)+1)
                if members:
                    for old,new in zip(rows,members):
                        result[result.index(old)]=new
                    result.extend(members[len(rows):])
                continue
        if (proposal.direction!='vertical' or not 2<=len(proposal.main_ids)<=5
                or len(proposal.annotation_parents)!=1 or len(proposal.annotation_ids)!=1):
            continue
        child,parent=proposal.annotation_parents[0]
        main=[by_id[i] for i in proposal.main_ids]
        side,base=by_id[child],by_id[parent]
        if (min(v.confidence for v in main)<.92 or not all(re.fullmatch(r'[ぁ-ゖァ-ヺ一-鿿々ー]{2,20}',v.text) for v in main)):
            continue
        l,t,r,b=side.bbox
        x,y,right,bottom=base.bbox
        width=right-x
        if not (3<=r-l<=.68*width and l>=right-.28*width and r<=right+.72*width and y<=t<b<=bottom):
            continue
        rect=proposal.crop_bounds
        neighbors=[v for v in boxes if v.box_id not in proposal.member_ids and v.bbox[0]<rect[2]
                   and v.bbox[2]>rect[0] and v.bbox[1]<rect[3] and v.bbox[3]>rect[1]]
        # A detector may split one narrow pronunciation into several tiny
        # fragments. Only absorb weak marks confined to the same verified strip.
        if any(v.confidence>=.65 or v.bbox[0]<l-2 or v.bbox[2]>r+2
               or v.bbox[1]<y or v.bbox[3]>bottom for v in neighbors):
            continue
        text=''.join(v.text for v in main)
        def view(bounds, pad):
            x,y,r,b=bounds
            return image.crop((max(0,x-pad),max(0,y-pad),min(image.width,r+pad+1),min(image.height,b+pad+1)))
        try:
            if any(reader(view(rect,k))!=text for k in (0,3)):
                continue
            readings=[]
            for pad in (0,1):
                strip=view((l,y,r,bottom),pad)
                strip=strip.resize((strip.width*3,strip.height*3),Image.Resampling.BICUBIC)
                readings.append(reader(strip))
            annotation=readings[0]
            if (not isinstance(annotation,str) or annotation!=readings[1] or side.text not in annotation
                    or not re.fullmatch(r'[ぁ-ゖァ-ヺー]{1,24}|※',annotation)):
                continue
        except (OSError,ValueError,RuntimeError,cv2.error):
            continue
        a,t,z,b=rect
        poly=np.float32([[a,t],[z,t],[z,b],[a,b]])
        anchor=main[0]
        merged=corrected(anchor.box_id,min(v.confidence for v in main),poly,text,anchor.text,anchor)
        merged.ocr_metadata['printed_annotation']={'main':base.text,'reading':annotation}
        absorbed=set(proposal.member_ids) | {v.box_id for v in neighbors}
        result=[merged if v is anchor else v for v in result if v is anchor or v.box_id not in absorbed]
    if result!=boxes:
        engine.raw_detections=[result]
        engine.automatic_unit_proposals=[propose_page_from_boxes(result,engine.image_path,complementary_ownership=True,regroup_companions=True)]
