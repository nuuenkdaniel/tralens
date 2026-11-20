"""Derived dialogue translation requests; originals are immutable historical input."""
from dataclasses import asdict

from .contracts import Reading, Utterance, TranslationRequest, canonical, revision
from ..balloons import require


POLICY = 'source_bound_derived_balloon_dialogue_v1'


def derive(candidate, proposal_record, group, originals, batch):
    require(proposal_record['status'] == 'proposed' and proposal_record['candidate_id'] == candidate['candidate_id']
        and proposal_record['evidence'] == candidate['evidence']
        and group in proposal_record['proposal']['groups'] and len(group['order']) >= 2
        and proposal_record['proposal']['surface'] == 'plain_dialogue', 'current bounded grouping proposal required')
    by_id = {u.unit_id:u for u in originals}
    by_label = {r['label']:r for r in candidate['evidence']['units']}
    parents = [by_id[by_label[label]['unit_id']] for label in group['order']]
    require(all(p.revision == by_label[label]['unit_revision'] for label,p in zip(group['order'],parents)),
        'stale grouping parent unit')
    require(len({p.page_json for p in parents}) == 1, 'foreign grouping parent page')
    require(all(p.selected.origin == 'raw_main_reading' for p in parents),
        'derived grouping does not guess an order for corrected/alternative source readings')
    require(all(p.selected.text.strip() and set(p.blocking_reasons) <= {
        'unknown_region_ownership','unsafe_crop','unknown_reading_orientation','crop_isolation_unresolved'} for p in parents),
        'grouping cannot waive unreadable source or ambiguous annotation roles')
    roles = []
    proposals = {p['proposal_id']:p for p in batch['proposals']}
    for parent in parents:
        raw = proposals[parent.unit_id]
        require(tuple(raw['member_ids']) == parent.member_ids and tuple(raw['main_ids']) == parent.order,
            'grouping parent raw roles/order drift')
        annotations = dict(raw['annotation_parents'])
        roles.extend({'source_id':i, 'role':'main' if i in parent.order else 'annotation',
            'annotation_parent':annotations.get(i), 'parent_unit_id':parent.unit_id} for i in parent.member_ids)
    members = tuple(i for p in parents for i in p.member_ids)
    order = tuple(i for p in parents for i in p.order)
    require(len(set(members)) == len(members) and set(order) <= set(members), 'overlapping derived members')
    evidence = {'policy':POLICY, 'candidate_id':candidate['candidate_id'],
        'proposal_revision':revision(proposal_record), 'relation':group['relation'],
        'parent_units':[asdict(p) for p in parents], 'member_roles':roles,
        'parent_blocking_reasons':[{'unit_id':p.unit_id,'reasons':list(p.blocking_reasons)} for p in parents],
        'new_spatial_support':'original interior hypothesis; not human verification',
        'source_join':'newline between exact parent main readings; no new source words',
        'speaker':'unknown','grouping_verified':False,'pixel_ownership_inferred':False}
    reading = Reading('\n'.join(p.selected.text for p in parents), 'derived_raw_main_dialogue', canonical(evidence))
    unit = Utterance('balloon-'+revision(evidence)[:24],parents[0].page_json,members,order,(reading,),reading.revision,
        revision(evidence),('automatic_not_human_reviewed','uncertain_balloon_dialogue_grouping'),
        tuple(dict.fromkeys(reason for p in parents for reason in p.lexical_reasons))+('uncertain_balloon_grouping',),
        (), 'proposed_balloon_dialogue_not_verified_full_utterance')
    return TranslationRequest((unit,),(unit.unit_id,)), evidence


def execute(request, translator):
    events = []
    result = translator.translate(request,observe=lambda event:events.append(event))
    result.assert_current(request)
    return {'request':asdict(request), 'request_revision':request.revision, 'result':asdict(result),
        'events':events, 'source_grouping_verified':False}
