"""Opt-in three-presentation crop readings over immutable detector membership.

Native, polygon-isolated and padded views must agree. This is a consistency
hypothesis, not independent-reader verification. Raw evidence, existing
alternatives and all failed/disagreeing trials remain available.
"""
from dataclasses import asdict,is_dataclass
from difflib import SequenceMatcher
from io import BytesIO
import hashlib
import json
import re
from pathlib import Path
import unicodedata

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps

from . import display_readings,outlined_sounds,sideways_sound,sound_readings,upright_latin
from .automatic_recognition import AutomaticRecognitionResult
from .manga_dialogue import valid_bounds
from .recognition_units import image_fingerprint
from .types import menu_shortcuts
from ..translation.contracts import canonical,strict_json


POLICY='native_isolated_padded_crop_consistency_v1'
ENGINE='consistent_crop_reading_unverified'
MAX_REGIONS=128


def require(value,message):
    if not value:raise ValueError(message)


def normalized(text):return ''.join(c for c in unicodedata.normalize('NFKC',text) if not c.isspace())


def weak_member(lines, readings):
    """Locate one weak line corroborated in a paragraph whose only dispute is another kanji."""
    if not 2<=len(lines)<=8 or len(readings)!=3:
        return None
    weak=[i for i,v in enumerate(lines) if v['confidence']<.70]
    if len(weak)!=1 or any(v['confidence']<.95 for i,v in enumerate(lines) if i!=weak[0]):
        return None
    words=[normalized(v['text']) for v in lines]
    i=weak[0]
    if lines[i]['confidence']<.35 or japanese_count(words[i])<2:
        return None
    source=''.join(words)
    if len(source)<8 or any(not isinstance(s,str) or len(normalized(s))!=len(source) for s in readings):
        return None
    alternatives=[normalized(s) for s in readings]
    differences=[pos for pos,char in enumerate(source) if any(s[pos]!=char for s in alternatives)]
    if len(differences)!=1:
        return None
    pos=differences[0]
    start=sum(map(len,words[:i]))
    if (start<=pos<start+len(words[i]) or len({s[pos] for s in alternatives})<2
            or not all('一'<=s[pos]<='鿿' for s in [source,*alternatives])):
        return None
    return i


def japanese_count(text):
    return sum(unicodedata.name(c,'').startswith(('HIRAGANA LETTER ','KATAKANA LETTER ',
        'CJK UNIFIED IDEOGRAPH-','CJK COMPATIBILITY IDEOGRAPH-')) for c in text)


def corrected_compound(source, reading):
    """Three matching views may restore a dictionary noun on a possessive label."""
    prefix, sep, raw = source.rpartition('の')
    if (not prefix or not sep or not re.fullmatch(r'[一-鿿]{2,4}',raw)
            or not reading.startswith(prefix+sep) or len(source)!=len(reading)):
        return False
    proposed = reading[len(prefix)+1:]
    if not re.fullmatch(r'[一-鿿]+',proposed) or sum(a!=b for a,b in zip(raw,proposed))!=1:
        return False
    try:
        from fugashi import Tagger
        import unidic_lite
        tagger = Tagger(f'-d "{unidic_lite.DICDIR}"')
    except (ImportError, RuntimeError):
        return False
    old, new = list(tagger(raw)), list(tagger(proposed))
    return (len(old)==2 and old[0].feature.pos2=='普通名詞' and old[1].feature.pos1=='接尾辞'
            and len(new)==1 and not new[0].is_unk and new[0].feature.pos2=='普通名詞')


def selection(source,readings,scores,direction):
    if (direction not in ('horizontal','vertical') or len(readings)!=3
            or not all(type(s) is str and 2<=len(s)<=512 and s.strip() for s in readings)
            or not scores or not all(type(s) in (int,float) and 0<=s<=1 for s in scores)):
        return False
    if any(unicodedata.category(c).startswith('C') and c!='\n' for text in readings for c in text):return False
    raw=normalized(source);first=normalized(readings[0])
    agreed = all(normalized(s)==first for s in readings[1:])
    if (not agreed and len(scores)==2 and max(scores)>=.90 and normalized(readings[1])==first
            and re.fullmatch(r'([ぁ-ゖァ-ヺー]{2,4})\1',first)
            and raw!=first and first.translate(str.maketrans('ゃゅょャュョ','やゆよヤユヨ'))==raw):
        # Two printed beats support the same small kana. A padded view may
        # pick up the first stroke again just outside their source polygons.
        agreed = normalized(readings[2])==first+first[0]
    if not agreed and normalized(readings[1]) == first and len(first) >= 8:
        # An interjection can contain an extra small tsu. All lexical content
        # must agree, including the presence of at least one small tsu there.
        agreed = re.sub(r'([っッ])\1+',r'\1',first) == re.sub(r'([っッ])\1+',r'\1',normalized(readings[2]))
    if not agreed and direction=='horizontal' and normalized(readings[1])==first:
        # Native and isolated menus agree on both labels. Padding the crop
        # can destroy the visible shortcut boundaries in the third reading.
        keys = menu_shortcuts(raw)
        agreed = len(keys)>=2 and keys==menu_shortcuts(first) and keys!=menu_shortcuts(normalized(readings[2]))
    if (raw==first or not agreed or japanese_count(first)<1
            or sum(c.isalpha() for c in first)<2):
        return False
    if first in ('「'+raw+'」','『'+raw+'』') and japanese_count(raw)>=2:
        return True
    # A stable newer reader still misreads Latin names. Do not replace only an
    # ASCII spelling on that model's self-consistency; retain it as an alternative.
    without_latin=lambda s:''.join(c for c in s if not ('a'<=c<='z' or 'A'<=c<='Z'))
    if without_latin(raw)==without_latin(first):return False
    # A small ッ immediately before the prolonged-vowel mark is easily
    # confused with full-size ツ. Keep the correction inside a phrase and
    # require identical readings of all the other characters in every view.
    if (len(raw)>=5 and raw.count('ッー')==1 and raw.replace('ッー','ツー')==first
            and all(normalized(s)==first for s in readings)):
        return True
    # Strong existing readings need independent support. A wholly different
    # detected script can trigger an opt-in hypothesis even at a high raw score.
    return (min(scores)<.92 or (japanese_count(raw)==0 and japanese_count(first)>=1)
            or (direction=='horizontal' and len(scores)==2 and all(normalized(s)==first for s in readings)
                and corrected_compound(raw, first)))


def views(image,bounds,polygons):
    require(image.mode=='RGB','crop reading requires original RGB pixels')
    require(valid_bounds(bounds,image.size),'crop reading bounds/budget')
    require(1<=len(polygons)<=8,'crop reading member budget')
    x,y,r,b=bounds;native=image.crop(bounds)
    mask=Image.new('L',native.size);draw=ImageDraw.Draw(mask)
    for polygon in polygons:
        points=np.asarray(polygon)
        require(points.ndim==2 and points.shape[1]==2 and 3<=len(points)<=128 and np.isfinite(points).all(),
            'invalid crop reading polygon')
        draw.polygon([(px-x,py-y) for px,py in polygon],fill=255)
    support=cv2.dilate(np.asarray(mask),np.ones((3,3),np.uint8))!=0
    require(support.any(),'empty crop reading membership')
    pixels=np.asarray(native);edge=np.concatenate((pixels[0],pixels[-1],pixels[:,0],pixels[:,-1]))
    color=tuple(int(v) for v in np.median(edge,axis=0))
    isolated=np.full_like(pixels,color);isolated[support]=pixels[support]
    isolated=Image.fromarray(isolated)
    return native,isolated,ImageOps.expand(isolated,border=8,fill=color)


EXACT_TILTED_VIEWS = 'rectified_and_two_native_isolations_v1'
TILTED_VIEWS = 'rectified_and_two_native_isolations_v2'


def tilted_mapping(size, record):
    from . import reader_geometry
    if record['direction'] != 'unknown' or len(record['polygons']) != 1:
        return None
    try:
        mapping = reader_geometry.plan(size, record['polygons'][0])
    except ValueError:
        return None
    width,height = mapping['seed_size']
    left,right = mapping['ordered_polygon'][:2]
    dx,dy = right[0]-left[0],abs(right[1]-left[1])
    return mapping if width >= 1.45*height and dx > 0 and .2*dx <= dy <= dx else None


def record_views(image, record):
    if record.get('view_policy')==sideways_sound.POLICY:
        result=sideways_sound.presentations(image,record)
        require(result is not None,'sideways sound geometry drift')
        return result
    if record.get('view_policy')==outlined_sounds.POLICY:
        return outlined_sounds.views(image,record['polygons'][0])
    if record.get('view_policy')==display_readings.POLICY:
        result=display_readings.presentations(image,record)
        require(result is not None,'display crop ink or geometry drift')
        return result
    native, isolated, padded = views(image,record['bounds'],record['polygons'])
    if record.get('view_policy')==upright_latin.POLICY:
        alternate=upright_latin.presentations(image,record,record['column_texts'])
        require(alternate is not None,'upright Latin crop geometry drift')
        return (*alternate,native)
    if 'view_policy' in record:
        from . import reader_geometry
        mapping = tilted_mapping(image.size,record)
        require(record['view_policy'] in (EXACT_TILTED_VIEWS,TILTED_VIEWS) and mapping is not None,'tilted crop geometry drift')
        native = reader_geometry.pixels(image,mapping)
    return native, isolated, padded


def clipped_kana_candidate(record):
    return (not record.get('view_policy') and len(record['polygons']) == 1
            and record['direction'] in ('vertical', 'unknown') and min(record['scores']) < .92
            and re.fullmatch(r'[ぁ-ゖァ-ヺー]{2,6}', record['source'])
            and record['readings'].count(record['source']) >= 2)


def clipped_kana_views(image, record):
    """Read a single stroke crossing the foot of an otherwise agreed kana column."""
    from . import reader_geometry
    if not clipped_kana_candidate(record):
        return None
    try:
        mapping = reader_geometry.plan(image.size, record['polygons'][0], view='axial')
    except ValueError:
        return None
    width, height = mapping['seed_size']
    if height < 1.35 * width:
        return None
    px, py = mapping['padding']
    edge = py + height
    pixels = np.asarray(reader_geometry.pixels(image, mapping))
    _, labels, stats, _ = cv2.connectedComponentsWithStats((pixels.max(2) < 145).astype(np.uint8), 8)
    strokes = [(i, x, y, w, h) for i, (x, y, w, h, area) in enumerate(stats[1:], 1)
               if px < x and x+w < px+width and y <= edge-2 and edge+4 <= y+h < len(pixels)-2
               and 5 <= w <= width and 8 <= h <= width and area >= 16]
    if len(strokes) != 1:
        return None
    i, x, y, w, h = strokes[0]
    support = cv2.dilate((labels == i).astype(np.uint8), np.ones((5, 5), np.uint8)) != 0
    isolated = np.full_like(pixels, 255)
    isolated[support] = pixels[support]
    glyph = Image.fromarray(isolated[max(0,y-3):y+h+3, max(0,x-3):x+w+3].copy())
    isolated[:edge] = pixels[:edge]
    column = Image.fromarray(isolated[py:y+h+4, px:px+width])
    return column, glyph


def clipped_kana_text(record):
    trial = record.get('clipped_kana', {})
    readings = trial.get('readings', [])
    if len(readings) == 2 and len(trial.get('view_hashes', [])) == 2 and clipped_kana_candidate(record):
        column, glyph = readings
        if isinstance(glyph, str) and re.fullmatch(r'[ぁ-ゖァ-ヺ]', glyph) and column == record['source'] + glyph:
            return column
    return None


def clipped_column_pair(record):
    """Padding can drop the shorter left-hand column of a handwritten reply."""
    reads=record['readings']
    if (record.get('view_policy') or record['direction']!='vertical' or len(record['polygons'])!=2
            or len(reads)!=3 or not all(isinstance(v,str) for v in reads) or reads[0]!=reads[1]
            or not re.fullmatch(r'[ぁ-ゖァ-ヺ一-鿿々ー]{3,14}',reads[0]) or not 1<=len(reads[2])<len(reads[0])
            or not reads[0].startswith(reads[2]) or reads[0]==record['source'] or max(record['scores'])>=.92):
        return False
    bounds=[]
    for poly in record['polygons']:
        x,y=zip(*poly)
        bounds.append((min(x),min(y),max(x),max(y)))
    (x,t,r,b),(left,top,right,bottom)=bounds
    return (left<x and abs(t-top)<=.45*(r-x) and bottom<b and -.25*(r-x)<=x-right<.9*(r-x))


def column_pair_views(image, record):
    crops=[]
    for poly in record['polygons']:
        x,y=zip(*poly)
        crops.append(image.crop((int(min(x)),int(min(y)),int(max(x))+1,int(max(y))+1)))
    return crops


def isolated_passage_text(record):
    """Repeated ink outside the isolated polygons can append part of a nearby line."""
    reads = record['readings']
    if (record.get('view_policy') or len(record.get('polygons',()))<2 or len(reads)!=3
            or not 5<=len(record['source'])<=96 or not record['scores'] or min(record['scores'])>=.92):
        return None
    native,isolated,padded = map(normalized,reads)
    raw = normalized(record['source'])
    if (isolated!=padded or len(raw)!=len(isolated) or sum(a!=b for a,b in zip(raw,isolated))>1
            or japanese_count(raw)<5 or not native.startswith(isolated)):
        return None
    extra = native[len(isolated):]
    # A long vowel or small tsu may extend beyond the last polygon of
    # hand lettering. All three views must still agree on the correction.
    trailing_emphasis=(len(record['polygons'])>=3 and extra in ('ー','っ','ッ')
                       and raw[-2:]==isolated[-2:])
    if 1<=len(extra)<=5 and (extra in isolated or trailing_emphasis) and reads[1]!=record['source']:
        return reads[1]
    return None


def shared_line_edits(record):
    """Keep only edits seen in all three presentations of the same multiline sign."""
    source, views = record['source'], record['readings']
    if (record.get('view_policy') or record['direction'] != 'horizontal'
            or len(record['polygons']) < 3 or not 15 <= len(source) <= 128
            or len(views) != 3 or len(set(views)) != 2 or views[0] != views[1]
            or min(record['scores']) >= .92 or re.search('[A-Za-z0-9]', source)):
        return None
    edits = None
    for view in (views[0], views[2]):
        match = SequenceMatcher(None, source, view, autojunk=False)
        if match.ratio() < .72:
            return None
        spans = {(a,b,view[c:d]) for tag,a,b,c,d in match.get_opcodes() if tag != 'equal'}
        edits = spans if edits is None else edits & spans
    # At least two independent locations must agree; preserve disputed source ink.
    if len(edits) < 2:
        return None
    for a,b,text in sorted(edits, reverse=True):
        if max(b-a, len(text)) > 3 or not text or re.search(r'[^ぁ-ゖァ-ヺー一-鿿々!！?？。、]',text):
            return None
        source = source[:a]+text+source[b:]
    return source


def anchored_columns(record, columns):
    """Allow a padded crop to disagree only inside independently legible columns."""
    reads = record['readings']
    if (record.get('view_policy') or len(reads)!=3 or not 2<=len(columns)<=8
            or len(columns)!=len(record['scores']) or ''.join(columns)!=record['source']):
        return False
    raw = ''.join(map(normalized,columns))
    first,second,third = map(normalized,reads)
    if (first!=second or first==third or raw==first or not 5<=len(first)<=80
            or min(record['scores'])>.70 or not all(japanese_count(s)>=2 for s in (first,third))):
        return False
    confirmed = set()
    for column,score in zip(columns,record['scores']):
        word = normalized(column)
        if score>=.94 and japanese_count(word)>=2 and first.count(word)==1 and raw.count(word)==1:
            pos = first.index(word)
            confirmed.update(range(pos,pos+len(word)))
        if score>.70 and (not word or first.count(word)!=1):
            return False
    changes = [(b,c,d,e) for kind,b,c,d,e in SequenceMatcher(None,first,third,autojunk=False).get_opcodes()
               if kind!='equal']
    return (len(changes)==1 and 0<changes[0][1]-changes[0][0]<=2
            and changes[0][1]-changes[0][0]==changes[0][3]-changes[0][2]
            and set(range(*changes[0][:2]))<=confirmed
            and selection(record['source'],[first]*3,record['scores'],record['direction']))


def column_passage(detections):
    if not detections:
        return None
    anchor = strict_json(detections[0]['metadata_json']).get('corroborated_columns')
    if isinstance(anchor,dict) and anchor.get('ids')==[v['source_id'] for v in detections]:
        return anchor.get('reading')


def quoted_literal_tail(record):
    """Keep the original spelling of a quoted Latin literal and reread its Japanese suffix."""
    if (record.get('view_policy') or record.get('direction')!='horizontal'
            or len(record.get('scores',[]))!=1 or not .55<=record['scores'][0]<.95
            or len(record.get('readings',[]))!=3):
        return
    pattern = r'["“]([A-Za-z0-9][A-Za-z0-9 ._+/:-]{2,64})["”](.+)'
    raw=re.fullmatch(pattern,record['source'])
    if not raw or not raw[2].isascii() or not 2<=len(raw[2])<=24:
        return
    tails=[]
    for value in record['readings']:
        other=re.fullmatch(pattern,value) if isinstance(value,str) else None
        if not other or normalized(other[1])!=normalized(raw[1]) or not re.fullmatch(r'[ぁ-ゖァ-ヺー一-鿿々]{3,20}[。！!？?]{0,2}',other[2]):
            return
        tails.append(other[2])
    if all(normalized(t)==normalized(tails[0]) for t in tails):
        return record['source'][:raw.start(2)]+tails[0]


def candidate_text(record):
    if record.get('quoted_literal_tail'):
        return quoted_literal_tail(record)
    if record.get('isolated_passage'):
        return isolated_passage_text(record)
    if record.get('shared_line_edits'):
        return shared_line_edits(record)
    if text := clipped_kana_text(record):
        return text
    if record.get('view_policy')==outlined_sounds.POLICY:
        return outlined_sounds.confirmed_text(record['readings']) or record['readings'][0]
    trial=record.get('sound_windows')
    if trial and sound_readings.matches(record):
        return trial['readings'][0]
    return record['readings'][0]


def select_record(record):
    if record.get('quoted_literal_tail'):
        return quoted_literal_tail(record) is not None
    if pair := record.get('clipped_column_pair'):
        return (clipped_column_pair(record) and len(pair['readings'])==2 and len(pair['view_hashes'])==2
                and pair['readings'][0]==record['readings'][2] and ''.join(pair['readings'])==record['readings'][0])
    if record.get('column_passage'):
        reads = record['readings']
        return (len(reads)==3 and reads[0]==reads[1]==record['column_passage']
                and reads[2] in (reads[0],record['source'])
                and selection(record['source'],[reads[0]]*3,record['scores'],record['direction']))
    if record.get('anchored_columns'):
        return anchored_columns(record,record['column_texts'])
    if record.get('isolated_passage'):
        return isolated_passage_text(record) is not None
    if record.get('shared_line_edits'):
        return shared_line_edits(record) is not None
    if 'clipped_kana' in record:
        return clipped_kana_text(record) is not None
    if record.get('view_policy')==sideways_sound.POLICY:
        return sideways_sound.matches(record)
    if record.get('view_policy')==outlined_sounds.POLICY:
        text=outlined_sounds.confirmed_text(record['readings'])
        return text is not None and record['source']!=text
    if record.get('sound_windows') and sound_readings.matches(record):
        return selection(record['source'],[candidate_text(record)]*3,record['scores'],'horizontal')
    direction = record['direction']
    if record.get('view_policy')==display_readings.POLICY:
        return direction=='horizontal' and selection(record['source'],record['readings'],record['scores'],direction)
    if record.get('view_policy')==upright_latin.POLICY:
        return (direction=='vertical' and ''.join(record['column_texts'])==record['source']
            and upright_latin.matches(record))
    if 'view_policy' in record:
        require(record['view_policy'] in (EXACT_TILTED_VIEWS,TILTED_VIEWS) and direction=='unknown','tilted crop policy drift')
        readings = record['readings']
        if len(readings) != 3 or not all(isinstance(v,str) for v in readings):
            return False
        texts = [normalized(v) for v in readings]
        if any(not 4 <= len(t) <= 12 or not all('ァ' <= c <= 'ヺ' or c=='ー' for c in t)
                for t in texts):
            return False
        if record['view_policy'] == EXACT_TILTED_VIEWS:
            text=texts[0];half=len(text)//2
            return (len(text)%2==0 and text[:half]==text[half:]
                and selection(record['source'],readings,record['scores'],'horizontal'))
        # An edge can clip one repetition of the same sound in a rotated
        # presentation. Keep the exact rectified response and all three counts.
        for length in range(2,5):
            motif = texts[0][:length]
            counts = [len(t)//length for t in texts]
            if (len(set(motif)) > 1 and min(counts)>=2 and max(counts)-min(counts)<=1
                    and all(t == motif*count for t,count in zip(texts,counts))):
                # Check each original response independently against the raw
                # reading (including control-character and confidence checks).
                return all(selection(record['source'],[v]*3,record['scores'],'horizontal')
                    for v in readings)
        return False
    # A stacked, repeated sound can form a square box with no dominant axis.
    if (direction == 'unknown' and len(record['polygons']) == 1 and len(record['scores']) == 1
            and record['scores'][0] < .6 and re.fullmatch(r'[ぁ-ゖァ-ヺ]', record['source'])
            and record['readings'] and re.fullmatch(r'([ぁ-ゖァ-ヺー]{2,4})\1', record['readings'][0])
            and record['source'] in record['readings'][0]):
        return selection(record['source'],record['readings'],record['scores'],'horizontal')
    return selection(record['source'],record['readings'],record['scores'],direction)


def refine(engine,image,original,*,reader,model_identity,output,max_regions=MAX_REGIONS):
    require(type(max_regions) is int and 0<=max_regions<=MAX_REGIONS,'crop reading region budget')
    require(type(model_identity) is str and bool(model_identity.strip()),'crop reader identity required')
    batch=engine.automatic_unit_proposals[0]
    if batch is None:return
    batch=strict_json(canonical(asdict(batch) if is_dataclass(batch) else batch))
    require(hashlib.sha256(original).hexdigest()==batch['page']['image_sha256']
        and image_fingerprint(image)==batch['page']['pixels_sha256'],'crop reader source binding')
    from ..translation.contracts import automatic_units
    from .types import is_credit_list
    prior=tuple(getattr(engine,'automatic_recognition_results',()) or ())
    before=prior[0] if prior else [AutomaticRecognitionResult(p['proposal_id']) for p in batch['proposals']]
    results=[AutomaticRecognitionResult(**strict_json(canonical(asdict(r) if is_dataclass(r) else r))) for r in before]
    original_units=automatic_units(batch,results)
    require([r.proposal_id for r in results]==[p['proposal_id'] for p in batch['proposals']],
        'crop reading proposal/result coverage')
    raw={d['source_id']:d for d in batch['detections']}
    stage={'policy':POLICY,'model_identity':model_identity,'calls':0,'selected':0,'records':[],
        'maximum_regions':max_regions,'artifacts':{}}
    engine.crop_readings=stage;attempted=0
    engine.automatic_recognition_results=(tuple(results),)
    for index,(proposal,result) in enumerate(zip(batch['proposals'],results)):
        require(not any(c['engine']==ENGINE for c in result.candidates),'crop reader already attempted; use fresh evidence')
        ids=proposal['main_ids'];bounds=proposal['crop_bounds']
        oversized=display_readings.candidate(bounds,image.size,proposal['direction'])
        if (proposal['annotation_ids'] or ids!=proposal['member_ids'] or not 1<=len(ids)<=8
                or not (valid_bounds(bounds,image.size) or oversized) or attempted>=max_regions):continue
        detections=[raw[i] for i in ids];source=''.join(d['text'] for d in detections)
        if len(detections)==1 and detections[0]['reason']=='corroborated_native_line':
            # The full printed menu and its account field were already
            # independently read inside their own rows. A larger crop also
            # includes the neighboring screen rule.
            metadata=strict_json(detections[0]['metadata_json'])
            if any(k in metadata for k in ('screen_field','outlined_ellipsis','hollow_print')) or metadata.get('forum_post',{}).get('heading'):
                continue
        if (not 0<len(source)<=512 or is_credit_list(source) or
                any(d['status'] not in ('accepted','review') for d in detections)):continue
        attempted+=1
        record={'proposal_id':proposal['proposal_id'],'source':source,'bounds':bounds,'main_ids':ids,
            'polygons':[d['polygon'] for d in detections],'scores':[d['confidence'] for d in detections],
            'direction':proposal['direction'],'readings':[],'view_hashes':[],'selected':False,
            'automatic_accuracy_verified':False}
        stage['records'].append(record)
        try:
            columns=[d['text'] for d in detections]
            if oversized:
                display_views=display_readings.presentations(image,record)
                if display_views is None:
                    record['reason']='no dominant display ink'
                    continue
                record['view_policy']=display_readings.POLICY
            elif len(detections)==1 and detections[0]['reason']=='supplemental_rotated_sound_region':
                record['view_policy']=outlined_sounds.POLICY
            elif sideways_sound.presentations(image,record) is not None:
                record['view_policy']=sideways_sound.POLICY
            elif upright_latin.presentations(image,record,columns) is not None:
                record.update(view_policy=upright_latin.POLICY,column_texts=columns)
            elif tilted_mapping(image.size,record) is not None:
                record['view_policy']=TILTED_VIEWS
            def read_views(target,presentations,suffix=''):
                for j,view in enumerate(presentations):
                    stream=BytesIO();view.save(stream,format='PNG');data=stream.getvalue()
                    name=f'crop-reading-{index:03d}{suffix}-{j}.png'
                    with (Path(output)/name).open('xb') as f:f.write(data)
                    stage['artifacts'][name]=hashlib.sha256(data).hexdigest()
                    target['view_hashes'].append(image_fingerprint(view));stage['calls']+=1
                    value=reader(view)
                    require(type(value) is str and len(value)<=512,'crop reader output bound/type')
                    target['readings'].append(value)
            read_views(record,display_views if oversized else record_views(image,record))
            if proposal.get('crop_safe') and (weak:=weak_member(detections,record['readings'])) is not None:
                from .masked_lines import source_views
                member=record['weak_member']={'readings':[],'view_hashes':[]}
                read_views(member,source_views(image,[np.asarray(detections[weak]['polygon'])]),'-weak')
            passage = column_passage(detections)
            if passage and not record.get('view_policy') and record['readings'][:2]==[passage,passage]:
                record['column_passage'] = passage
            elif quoted_literal_tail(record):
                record['quoted_literal_tail'] = True
            elif anchored_columns(record,columns):
                record.update(anchored_columns=True,column_texts=columns)
            elif clipped_column_pair(record):
                pair=record['clipped_column_pair']={'readings':[], 'view_hashes':[]}
                read_views(pair,column_pair_views(image,record),'-short-columns')
            elif isolated_passage_text(record):
                record['isolated_passage'] = True
            elif shared_line_edits(record):
                record['shared_line_edits'] = True
            if not select_record(record):
                windows=sound_readings.presentations(image,record)
                if windows is not None:
                    record['sound_windows']={'readings':[],'view_hashes':[]}
                    read_views(record['sound_windows'],windows,'-windows')
                    if not sound_readings.matches(record) and sound_readings.shifted_target(record):
                        record['shifted_windows']={'readings':[],'view_hashes':[]}
                        read_views(record['shifted_windows'],sound_readings.shifted_views(windows[0]),'-shifted')
            if not select_record(record):
                extra = clipped_kana_views(image, record)
                if extra is not None:
                    record['clipped_kana'] = {'readings': [], 'view_hashes': []}
                    read_views(record['clipped_kana'], extra, '-clipped-kana')
            record['selected']=select_record(record)
            if record['selected']:
                from . import manga_dialogue,quality_reread
                previous=quality_reread.preferred_reading(original_units[index]) or manga_dialogue.preferred_reading(original_units[index])
                if previous is not None:
                    record['selected']=False
                    record['deferred_to_prior_reading_revision']=previous.revision
            proof={'policy':POLICY,'record':record,'source_pixels_sha256':batch['page']['pixels_sha256'],
                'model_identity':model_identity,'erase_permission':False}
            candidate={'text':candidate_text(record),'reference_main_text':source,'differing_spans':[],
                'crop':{'proposal_id':proposal['proposal_id'],'page':batch['page'],'member_ids':proposal['member_ids'],
                    'main_ids':ids,'bounds':bounds,'checked_neighbor_ids':[i for i in raw if i not in ids],
                    'batch_evidence_sha256':batch['evidence_sha256'],
                    'proposal_sha256':hashlib.sha256(json.dumps(proposal,sort_keys=True,ensure_ascii=False).encode()).hexdigest()},
                'engine':ENGINE,'model_identity':model_identity,'model_settings_json':canonical(proof),
                'status':'automated_unverified','reviewed':False}
            # Disagreement/unchanged alternatives stay in this stage's records;
            # only a selected changed input changes the utterance revision.
            if record['selected']:
                result.candidates=(*result.candidates,candidate);result.attempts=1
                result.status='automated_unverified';stage['selected']+=1
        except Exception as error:record['error']=type(error).__name__+': '+str(error)[:512]
    engine.automatic_recognition_results=(tuple(results),)


def validate_candidate(candidate,batch):
    proof=strict_json(candidate['model_settings_json']);record=proof['record'];crop=candidate['crop']
    proposal=next(p for p in batch['proposals'] if p['proposal_id']==crop['proposal_id'])
    raw={d['source_id']:d for d in batch['detections']};main=[raw[i] for i in proposal['main_ids']]
    require(proof['policy']==POLICY and proof['erase_permission'] is False
        and proof['model_identity']==candidate['model_identity']
        and proof['source_pixels_sha256']==batch['page']['pixels_sha256']
        and candidate['status']=='automated_unverified' and candidate['engine']==ENGINE
        and type(candidate['model_identity']) is str and bool(candidate['model_identity'].strip())
        and len(proposal['main_ids']) in range(1,9)
        and all(d['status'] in ('accepted','review') for d in main)
        and proposal['main_ids']==proposal['member_ids'] and not proposal['annotation_ids']
        and crop['checked_neighbor_ids']==[d['source_id'] for d in batch['detections'] if d['source_id'] not in proposal['main_ids']]
        and record['proposal_id']==proposal['proposal_id'] and record['main_ids']==proposal['main_ids']
        and record['polygons']==[d['polygon'] for d in main] and record['scores']==[d['confidence'] for d in main]
        and (not (record.get('view_policy')==upright_latin.POLICY or record.get('anchored_columns')) or record['column_texts']==[d['text'] for d in main])
        and (not record.get('column_passage') or record['column_passage']==column_passage(main))
        and record['direction']==proposal['direction'] and record['bounds']==crop['bounds']
        and record['source']==candidate['reference_main_text']==''.join(d['text'] for d in main)
        and candidate['text']==candidate_text(record) and candidate['reviewed'] is False
        and record['automatic_accuracy_verified'] is False and len(record['view_hashes'])=={outlined_sounds.POLICY:5,sideways_sound.POLICY:4}.get(record.get('view_policy'),3)
        and record['selected'] is True
        and record['selected']==select_record(record),
        'crop reading candidate evidence drift')


def confirms_raw(record, unit, proposal, image, detections=()):
    """Corroborate bound native lettering from the source pixels."""
    evidence = strict_json(unit.selected.evidence_json)
    if (isinstance(evidence,list) and evidence and all(r['reason'] in ('corroborated_mask_line','corroborated_native_line') for r in evidence)):
        metadata = strict_json(evidence[0]['metadata_json'])
        peer = metadata.get('repeated_kana')
        if peer:
            box=peer.get('peer_bounds',[])
            if (len(box)!=4 or not all(type(v) is int for v in box) or not valid_bounds(box,image.size)
                    or image_fingerprint(image.crop(tuple(box)))!=peer.get('peer_digest')):
                return False
        color = metadata.get('colored_word') or metadata.get('painted_effect') or peer
        if len(evidence)==1 and isinstance(color,dict) and color.get('reads')==[unit.selected.text]*3:
            bounds = color.get('source_bounds',color.get('bounds',[]))
            if (len(bounds)==4 and all(type(v) is int for v in bounds) and
                    0<=bounds[0]<bounds[2]<=image.width and 0<=bounds[1]<bounds[3]<=image.height and
                    (bounds[2]-bounds[0])*(bounds[3]-bounds[1])<=1500000):
                return color.get('source_digest')==image_fingerprint(image.crop(tuple(bounds)))
        passage = metadata.get('masked_passage',{})
        if passage.get('ids')==list(unit.order) and passage.get('reading')==unit.selected.text:
            if len(evidence)==1 and (note:=metadata.get('outlined_ellipsis')):
                bounds=note.get('bounds',[])
                if not valid_bounds(bounds,image.size) or max(bounds[2]-bounds[0],bounds[3]-bounds[1])>640:
                    return False
                native=image.crop(tuple(bounds))
                if image_fingerprint(native)!=note.get('digest'):
                    return False
                found=outlined_sounds.outlined_ellipsis(np.asarray(native.convert('L')))
                return found is not None and passage.get('view_hashes')==[
                    image_fingerprint(v) for v in outlined_sounds.ellipsis_views(*found)]
            polygons = [v['polygon'] for v in evidence]
            from .masked_lines import source_views
            return passage.get('view_hashes')==[image_fingerprint(v) for v in source_views(image,polygons)]
    if isinstance(evidence,list) and evidence and all(r['reason']=='corroborated_repeated_effect' for r in evidence):
        from .repeated_effects import bound_pair
        return bound_pair(evidence,image,detections)
    if (not record or record.get('error') or record.get('proposal_id') != unit.unit_id
            or record.get('main_ids') != list(unit.order) or record.get('bounds') != proposal['crop_bounds']
            or record.get('source') != unit.selected.text
            or len(record.get('readings',[])) != 3):
        return False
    if record.get('polygons') != [line['polygon'] for line in evidence]:
        return False
    # Padding sometimes admits one final full stop from outside the native crop.
    # The independent column join must also identify this exact member sequence.
    text=unit.selected.text
    punctuation_only=(record['readings']==[text,text,text+'。'] and column_passage(evidence)==text)
    # A terminal small tsu clips the sound. Three views agreeing on that
    # extra beat still corroborate every kana that the detector read.
    ending_only = (len(unit.order)==1 and re.fullmatch(r'[ぁ-ゖァ-ヺー]{2,12}',text)
                   and record['readings'] in ([text+'っ']*3,[text+'ッ']*3))
    if not punctuation_only and not ending_only and not all(isinstance(s,str) and normalized(s)==normalized(text) for s in record['readings']):
        from .masked_lines import source_views
        weak=weak_member(evidence,record['readings']) if proposal.get('crop_safe') else None
        member=record.get('weak_member',{})
        if weak is None or member.get('readings')!=[evidence[weak]['text']]*3:
            return False
        chips=source_views(image,[np.asarray(evidence[weak]['polygon'])])
        if member.get('view_hashes')!=[image_fingerprint(view) for view in chips]:
            return False
    return record.get('view_hashes') == [image_fingerprint(view) for view in record_views(image, record)]


def preferred_reading(unit):
    choices=[]
    for reading in unit.readings:
        if reading.origin!=ENGINE:continue
        candidate=strict_json(reading.evidence_json);proof=strict_json(candidate['model_settings_json']);record=proof['record']
        raw=strict_json(unit.selected.evidence_json)
        require(proof['policy']==POLICY and record['proposal_id']==unit.unit_id
            and proof['erase_permission'] is False and record['automatic_accuracy_verified'] is False
            and candidate['reviewed'] is False and candidate['status']=='automated_unverified'
            and candidate['engine']==reading.origin and proof['model_identity']==candidate['model_identity']
            and proof['source_pixels_sha256']==strict_json(unit.page_json)['pixels_sha256']
            and candidate['crop']['page']==strict_json(unit.page_json)
            and candidate['crop']['proposal_id']==unit.unit_id
            and candidate['crop']['member_ids']==list(unit.member_ids)
            and candidate['crop']['main_ids']==list(unit.order)
            and candidate['crop']['bounds']==record['bounds']
            and candidate['reference_main_text']==unit.selected.text and record['selected'] is True
            and candidate['text']==reading.text==candidate_text(record)
            and record['source']==unit.selected.text and record['main_ids']==list(unit.order)
            and (not (record.get('view_policy')==upright_latin.POLICY or record.get('anchored_columns')) or record['column_texts']==[d['text'] for d in raw])
            and (not record.get('column_passage') or record['column_passage']==column_passage(raw))
            and (record.get('view_policy')!=outlined_sounds.POLICY or (len(raw)==1 and raw[0]['reason']=='supplemental_rotated_sound_region'
                and strict_json(raw[0]['metadata_json'])[outlined_sounds.POLICY]['readings']==record['readings']))
            and record['scores']==[d['confidence'] for d in raw]
            and record['polygons']==[d['polygon'] for d in raw]
            and record['selected']==select_record(record),
            'crop reading selection drift')
        if record['selected']:choices.append(reading)
    require(len(choices)<=1,'ambiguous crop reading candidates')
    return choices[0] if choices else None


def validate_pixels(candidate,image):
    proof=strict_json(candidate['model_settings_json']);record=proof['record']
    require(record['bounds']==candidate['crop']['bounds'] and proof['source_pixels_sha256']==image_fingerprint(image)
        and record['view_hashes']==[image_fingerprint(v) for v in record_views(image,record)],
        'crop reading pixels changed')
    if 'clipped_column_pair' in record:
        require(record['clipped_column_pair']['view_hashes']==[image_fingerprint(v) for v in column_pair_views(image,record)],
                'separate column pixels changed')
    if 'clipped_kana' in record:
        extra = clipped_kana_views(image, record)
        require(extra is not None and record['clipped_kana']['view_hashes']==[image_fingerprint(v) for v in extra],
                'clipped kana crop pixels changed')
    if 'sound_windows' in record:
        windows=sound_readings.presentations(image,record)
        require(windows is not None and record['sound_windows']['view_hashes']==[image_fingerprint(v) for v in windows],
            'sound window crop pixels changed')

    if 'shifted_windows' in record:
        require('sound_windows' in record and sound_readings.shifted_target(record) is not None
            and record['shifted_windows']['view_hashes']==[image_fingerprint(v) for v in sound_readings.shifted_views(windows[0])],
            'shifted sound crop pixels changed')
