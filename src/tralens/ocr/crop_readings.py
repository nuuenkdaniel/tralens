"""Opt-in three-presentation crop readings over immutable detector membership.

Native, polygon-isolated and padded views must agree. This is a consistency
hypothesis, not independent-reader verification. Raw evidence, existing
alternatives and all failed/disagreeing trials remain available.
"""
from dataclasses import asdict,is_dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import unicodedata

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps

from . import display_readings,outlined_sounds,sideways_sound,sound_readings,upright_latin
from .automatic_recognition import AutomaticRecognitionResult
from .manga_dialogue import valid_bounds
from .recognition_units import image_fingerprint
from ..translation.contracts import canonical,strict_json


POLICY='native_isolated_padded_crop_consistency_v1'
ENGINE='consistent_crop_reading_unverified'
MAX_REGIONS=128


def require(value,message):
    if not value:raise ValueError(message)


def normalized(text):return ''.join(c for c in unicodedata.normalize('NFKC',text) if not c.isspace())


def japanese_count(text):
    return sum(unicodedata.name(c,'').startswith(('HIRAGANA LETTER ','KATAKANA LETTER ',
        'CJK UNIFIED IDEOGRAPH-','CJK COMPATIBILITY IDEOGRAPH-')) for c in text)


def selection(source,readings,scores,direction):
    if (direction not in ('horizontal','vertical') or len(readings)!=3
            or not all(type(s) is str and 2<=len(s)<=512 and s.strip() for s in readings)
            or not scores or not all(type(s) in (int,float) and 0<=s<=1 for s in scores)):
        return False
    if any(unicodedata.category(c).startswith('C') and c!='\n' for text in readings for c in text):return False
    raw=normalized(source);first=normalized(readings[0])
    if (raw==first or any(normalized(s)!=first for s in readings[1:]) or japanese_count(first)<1
            or sum(c.isalpha() for c in first)<2):
        return False
    # A stable newer reader still misreads Latin names. Do not replace only an
    # ASCII spelling on that model's self-consistency; retain it as an alternative.
    without_latin=lambda s:''.join(c for c in s if not ('a'<=c<='z' or 'A'<=c<='Z'))
    if without_latin(raw)==without_latin(first):return False
    # Strong existing readings need independent support. A wholly different
    # detected script can trigger an opt-in hypothesis even at a high raw score.
    return min(scores)<.92 or (japanese_count(raw)==0 and japanese_count(first)>=1)


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


def candidate_text(record):
    if record.get('view_policy')==outlined_sounds.POLICY:
        return outlined_sounds.confirmed_text(record['readings']) or record['readings'][0]
    trial=record.get('sound_windows')
    if trial and sound_readings.matches(record):
        return trial['readings'][0]
    return record['readings'][0]


def select_record(record):
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
        if not 0<len(source)<=512 or any(d['status'] not in ('accepted','review') for d in detections):continue
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
            if not select_record(record):
                windows=sound_readings.presentations(image,record)
                if windows is not None:
                    record['sound_windows']={'readings':[],'view_hashes':[]}
                    read_views(record['sound_windows'],windows,'-windows')
                    if not sound_readings.matches(record) and sound_readings.shifted_target(record):
                        record['shifted_windows']={'readings':[],'view_hashes':[]}
                        read_views(record['shifted_windows'],sound_readings.shifted_views(windows[0]),'-shifted')
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
        and (record.get('view_policy')!=upright_latin.POLICY or record['column_texts']==[d['text'] for d in main])
        and record['direction']==proposal['direction'] and record['bounds']==crop['bounds']
        and record['source']==candidate['reference_main_text']==''.join(d['text'] for d in main)
        and candidate['text']==candidate_text(record) and candidate['reviewed'] is False
        and record['automatic_accuracy_verified'] is False and len(record['view_hashes'])=={outlined_sounds.POLICY:5,sideways_sound.POLICY:4}.get(record.get('view_policy'),3)
        and record['selected'] is True
        and record['selected']==select_record(record),
        'crop reading candidate evidence drift')


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
            and (record.get('view_policy')!=upright_latin.POLICY or record['column_texts']==[d['text'] for d in raw])
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
    if 'sound_windows' in record:
        windows=sound_readings.presentations(image,record)
        require(windows is not None and record['sound_windows']['view_hashes']==[image_fingerprint(v) for v in windows],
            'sound window crop pixels changed')

    if 'shifted_windows' in record:
        require('sound_windows' in record and sound_readings.shifted_target(record) is not None
            and record['shifted_windows']['view_hashes']==[image_fingerprint(v) for v in sound_readings.shifted_views(windows[0])],
            'shifted sound crop pixels changed')
