"""Bound changed source proposals to NEW TG27 requests; retain every old result."""
from dataclasses import asdict,replace

from .contracts import Reading, TranslationRequest, canonical, revision
from ..language_review import require
from ..ocr import source_review as sr
from ..ocr.source_review import normalized


POLICY = 'corroborated_source_candidate_translation_v1'


def request_for(original,queue_row,candidate,corroboration,*,preview,plan,rgb):
    request,*_ = sr.prepare(preview,plan,original.unit_id,rgb)
    require(queue_row == next(r for r in plan['records'] if r['unit_id'] == original.unit_id),
        'recovered source queue row drift')
    sr.assert_current(candidate,rgb,request)
    require(candidate['clear'] and candidate['request_revision'] == revision(candidate['binding'])
        and candidate['binding']['unit_id'] == original.unit_id
        and candidate['binding']['unit_revision'] == original.revision
        and queue_row['unit_revision'] == original.revision
        and candidate['binding']['member_ids'] == list(original.member_ids)
        and candidate['binding']['main_order'] == list(original.order),'stale recovered-source parent binding')
    require(len(corroboration) == len(candidate['lines']) and all(e['matches'] for e in corroboration)
        and all(normalized(e['reading']['text']) == normalized(line['text']) for e,line in zip(corroboration,candidate['lines']))
        and [e['source_id'] for e in corroboration] == list(original.order),'complete independently cropped reader support required')
    for line,evidence in zip(candidate['lines'],corroboration):
        x,y,r,b = line['proposed_page_extent']
        box = [max(0,x-3),max(0,y-3),min(rgb.width,r+3),min(rgb.height,b+3)]
        require(evidence['bounds'] == box
            and evidence['pixels_sha256'] == sr.hashlib.sha256(rgb.crop(box).tobytes()).hexdigest(),
            'independent reader original crop binding drift')
    require(candidate['text'].strip() and len(candidate['text']) <= 1000
        and normalized(candidate['text']) != normalized(queue_row['source_text']), 'unchanged source must reuse primary, not retranslate')
    require(set(original.blocking_reasons) <= {'unknown_region_ownership','mixed_region_ownership',
        'unknown_reading_orientation','unsafe_crop','crop_isolation_unresolved'},
        'source candidate cannot waive ambiguous annotation or unreadable original structure')
    proof = {'policy':POLICY,'parent':asdict(original),'queue_row_revision':revision(queue_row),
        'candidate':candidate,'corroboration':corroboration,'annotation_roles_changed':False,
        'speaker':'unknown','human_verified':False,'corroboration_is_truth':False}
    reading = Reading(candidate['text'],'corroborated_source_recovery_candidate',canonical(proof))
    unit = replace(original,unit_id='source-candidate-'+revision(proof)[:24],
        readings=original.readings+(reading,),selected_revision=reading.revision,
        evidence_revision=revision(proof),blocked_reasons=(),
        lexical_reasons=tuple(dict.fromkeys(original.lexical_reasons+('source_recovery_unverified',))),
        completeness='unverified_recovered_source_not_utterance_or_pixel_ownership_proof')
    return TranslationRequest((unit,),(unit.unit_id,))
