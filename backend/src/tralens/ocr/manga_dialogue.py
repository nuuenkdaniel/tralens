"""Optional local dialogue-reader ensemble over immutable detector evidence.

Low-score, isolated Japanese runs receive two crop presentations. Agreement is
an automatic hypothesis, not independent-reader proof or a transcription claim.
Original Paddle readings and every attempted alternative remain available.
"""
from dataclasses import asdict
from difflib import SequenceMatcher
from io import BytesIO
import hashlib
import json
import math
from types import SimpleNamespace
import unicodedata

import numpy as np
from PIL import Image,ImageOps

from .automatic_recognition import AutomaticRecognitionResult
from .automatic_units import crop_conflicts
from .recognition_units import image_fingerprint
from ..translation.contracts import canonical,strict_json


POLICY='isolated_low_score_consistent_manga_reading_v1'
ENGINE='consistent_manga_dialogue_unverified'
MAX_REGIONS=48


def normalized(text):
    # A reader-added comma changes phrasing, not the character reading. Question
    # marks, repetitions, long vowels and small kana still must agree exactly.
    return ''.join(c for c in unicodedata.normalize('NFKC',text) if not c.isspace() and c not in ',、')


def japanese(text):
    return any('\u3041'<=c<='\u3096' or '\u30a1'<=c<='\u30fa' for c in text)


def selection(raw,first,second):
    if not all(type(s) is str and 2<=len(s)<=256 and japanese(s) for s in (raw,first,second)):
        return False
    a,b,c=map(normalized,(raw,first,second))
    return (a!=b and b==c and .65<=len(b)/len(a)<=1.5
        and SequenceMatcher(None,a,b,autojunk=False).ratio()>=.6)


def presentations(image,bounds):
    if not valid_bounds(bounds,image.size):raise ValueError('dialogue crop geometry bound')
    crop=image.crop(bounds)
    # Constant padding changes reader framing, not source extent or membership.
    edge=np.concatenate((np.asarray(crop)[0],np.asarray(crop)[-1],np.asarray(crop)[:,0],np.asarray(crop)[:,-1]))
    color=tuple(int(v) for v in np.median(edge,axis=0))
    return crop,ImageOps.expand(crop,border=8,fill=color)


def valid_bounds(bounds,size):
    return (type(bounds) in (list,tuple) and len(bounds)==4 and all(type(v) is int for v in bounds)
        and 0<=bounds[0]<bounds[2]<=size[0] and 0<=bounds[1]<bounds[3]<=size[1]
        and max(bounds[2]-bounds[0],bounds[3]-bounds[1])<=1024
        and (bounds[2]-bounds[0])*(bounds[3]-bounds[1])<=262144)


def low_score(scores):
    return bool(scores) and all(type(s) in (int,float) and math.isfinite(s) and 0<=s<=1 for s in scores) and min(scores)<.92


def refine(engine,image,original,*,reader,model_identity,output):
    from pathlib import Path
    batch=engine.automatic_unit_proposals[0]
    if batch is None:return
    if hashlib.sha256(original).hexdigest()!=batch.page.image_sha256 or image_fingerprint(image)!=batch.page.pixels_sha256:
        raise ValueError('dialogue reader source mismatch')
    raw={d.source_id:d for d in batch.detections}
    prior=tuple(getattr(engine,'automatic_recognition_results',()) or ())
    results=list(prior[0]) if prior else [AutomaticRecognitionResult(p.proposal_id) for p in batch.proposals]
    if len(results)!=len(batch.proposals):raise ValueError('dialogue reader proposal/result mismatch')
    stage={'policy':POLICY,'model_identity':model_identity,'calls':0,'selected':0,'records':[],'artifacts':{}}
    engine.dialogue_readings=stage
    for index,(proposal,result) in enumerate(zip(batch.proposals,results)):
        if result.proposal_id!=proposal.proposal_id:raise ValueError('dialogue result/proposal drift')
        source=''.join(raw[i].text for i in proposal.main_ids)
        scores=[raw[i].confidence for i in proposal.main_ids]
        if (result.candidates or not proposal.crop_safe or proposal.crop_conflicts or proposal.annotation_ids
                or proposal.direction not in ('vertical','horizontal') or not 2<=len(source)<=256
                or not japanese(source) or not low_score(scores) or stage['calls']>=2*MAX_REGIONS):continue
        if any(raw[i].status not in ('accepted','review') for i in proposal.main_ids):continue
        if (not valid_bounds(proposal.crop_bounds,image.size)
                or crop_conflicts(proposal.crop_bounds,proposal.member_ids,batch.detections)):continue
        crops=presentations(image,proposal.crop_bounds)
        record={'proposal_id':proposal.proposal_id,'source':source,'bounds':list(proposal.crop_bounds),
            'main_ids':list(proposal.main_ids),'minimum_raw_score':min(scores),'readings':[],'crop_hashes':[],
            'selected':False,'automatic_accuracy_verified':False}
        stage['records'].append(record)
        try:
            for view,crop in enumerate(crops):
                stream=BytesIO();crop.save(stream,format='PNG');content=stream.getvalue()
                name=f'dialogue-{index:03d}-{view}.png'
                with (Path(output)/name).open('xb') as f:f.write(content)
                stage['artifacts'][name]=hashlib.sha256(content).hexdigest()
                record['crop_hashes'].append(image_fingerprint(crop))
                stage['calls']+=1
                value=reader(crop)
                if type(value) is not str or not value.strip() or len(value)>512:raise ValueError('dialogue reader output bound')
                record['readings'].append(value)
            record['selected']=selection(source,*record['readings'])
            proof={'policy':POLICY,'record':record,'source_pixels_sha256':batch.page.pixels_sha256,
                'crop_presentation':'native rectangle plus constant 8px border; no resizing',
                'model_identity':model_identity}
            candidate={'text':record['readings'][0],'reference_main_text':source,'differing_spans':[],
                'crop':{'proposal_id':proposal.proposal_id,'page':asdict(batch.page),
                    'member_ids':list(proposal.member_ids),'main_ids':list(proposal.main_ids),'bounds':list(proposal.crop_bounds),
                    'checked_neighbor_ids':[d.source_id for d in batch.detections if d.source_id not in proposal.member_ids],
                    'batch_evidence_sha256':batch.evidence_sha256,
                    'proposal_sha256':hashlib.sha256(json.dumps(asdict(proposal),sort_keys=True,ensure_ascii=False).encode()).hexdigest()},
                'engine':ENGINE,'model_identity':model_identity,'model_settings_json':canonical(proof),
                'status':'automated_unverified','reviewed':False}
            result.attempts=1;result.candidates=(candidate,)
            result.status='automated_unverified';result.reason='crop_consistency_hypothesis_not_truth'
            stage['selected']+=int(record['selected'])
        except Exception as error:record['error']=type(error).__name__+': '+str(error)[:512]
    engine.automatic_recognition_results=(tuple(results),)


def preferred_reading(unit):
    options=[]
    for reading in unit.readings:
        if reading.origin!=ENGINE:continue
        candidate=strict_json(reading.evidence_json)
        proof=strict_json(candidate['model_settings_json']);record=proof['record']
        raw=strict_json(unit.selected.evidence_json)
        scores=[d['confidence'] for d in raw]
        if (proof['policy']!=POLICY or record['proposal_id']!=unit.unit_id
                or record['main_ids']!=list(unit.order) or candidate['text']!=reading.text
                or record['source']!=unit.selected.text or record['bounds']!=candidate['crop']['bounds']
                or len(record['readings'])!=2 or record['readings'][0]!=reading.text
                or type(record['selected']) is not bool or not low_score(scores)
                or record['minimum_raw_score']!=min(scores)
                or proof['model_identity']!=candidate['model_identity']
                or record['selected']!=selection(unit.selected.text,*record['readings'])
                or candidate['reference_main_text']!=unit.selected.text or candidate['reviewed'] is not False):
            raise ValueError('dialogue candidate identity drift')
        if record['selected'] and selection(unit.selected.text,*record['readings']):options.append(reading)
    if len(options)>1:raise ValueError('ambiguous dialogue reader selection')
    return options[0] if options else None


def validate_candidate(candidate,batch):
    """Recheck saved crop isolation/eligibility, not just its claimed safety flag."""
    crop=candidate['crop']
    proposal=next(p for p in batch['proposals'] if p['proposal_id']==crop['proposal_id'])
    detections=[SimpleNamespace(**d) for d in batch['detections']]
    raw={d.source_id:d for d in detections}
    neighbors=[d.source_id for d in detections if d.source_id not in proposal['member_ids']]
    if (proposal['crop_safe'] is not True or proposal['crop_conflicts'] or proposal['annotation_ids']
            or proposal['direction'] not in ('vertical','horizontal')
            or proposal['main_ids']!=proposal['member_ids']
            or crop['checked_neighbor_ids']!=neighbors
            or crop_conflicts(crop['bounds'],proposal['member_ids'],detections)
            or any(raw[i].status not in ('accepted','review') for i in proposal['main_ids'])
            or not low_score([raw[i].confidence for i in proposal['main_ids']])):
        raise ValueError('dialogue candidate crop eligibility drift')


def validate_pixels(candidate,image):
    proof=strict_json(candidate['model_settings_json']);record=proof['record']
    if (proof['source_pixels_sha256']!=image_fingerprint(image)
            or record['crop_hashes']!=[image_fingerprint(crop) for crop in presentations(image,record['bounds'])]
            or record['bounds']!=candidate['crop']['bounds']):
        raise ValueError('dialogue reader pixels changed')
