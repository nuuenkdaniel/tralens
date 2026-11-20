"""Opt-in adjacent vertical-column hypotheses, never balloon/utterance truth."""
from dataclasses import asdict, replace
import hashlib
import json
import math

import numpy as np

from .automatic_units import crop_conflicts


POLICY = 'enclosed_aligned_vertical_columns_v1'
REASON = 'automatic_vertical_continuation_hypothesis'


def vertical_groups(batch, owners, image):
    """Join one unambiguous pair with a clear ORIGINAL seam; never chain pairs.

    Same white connectivity alone is insufficient: touching balloons, paragraphs,
    annotations, intervening detections and ambiguous neighbors must stay apart.
    No source strings, confidence thresholds, page IDs or model output are used.
    """
    raw = {d.source_id: d for d in batch.detections}
    pixels = np.asarray(image)

    def bounds(ids):
        points = [point for i in ids for point in raw[i].polygon]
        return (math.floor(min(x for x, y in points)), math.floor(min(y for x, y in points)),
                math.ceil(max(x for x, y in points)), math.ceil(max(y for x, y in points)))

    eligible = [p for p in batch.proposals if not p.annotation_ids and
        (p.direction == 'vertical' or (p.direction == 'unknown' and len(p.main_ids) == 1))]
    options = []
    for index, first in enumerate(eligible):
        for second in eligible[index+1:]:
            if max(len(first.main_ids), len(second.main_ids)) < 2:
                continue
            ids = first.main_ids+second.main_ids
            if (len(ids) > 8 or not all(i in owners for i in ids)
                    or len({owners[i][0] for i in ids}) != 1):
                continue
            columns = sorted(((i, bounds((i,))) for i in ids), key=lambda item: -item[1][0])
            widths = [b[2]-b[0] for _, b in columns]
            scale = min(widths)
            if (scale < 4 or max(widths) > 1.5*scale
                    or max(b[1] for _, b in columns)-min(b[1] for _, b in columns) > .35*scale
                    or max(b[3]-b[1] for _, b in columns) < 2.5*scale):
                continue
            # Only a single seam between two existing groups, never interleaving.
            membership = [i in first.main_ids for i, _ in columns]
            if sum(a != b for a, b in zip(membership, membership[1:])) != 1:
                continue
            seams = []
            for (right_id, right), (left_id, left) in zip(columns, columns[1:]):
                if (right_id in first.main_ids) == (left_id in first.main_ids):
                    continue
                gap = right[0]-left[2]
                top = math.ceil(max(right[1], left[1])+.15*scale)
                bottom = math.floor(min(right[3], left[3])-.15*scale)
                if not 2 <= gap <= .9*scale or bottom-top < .5*scale:
                    break
                corridor = pixels[top:bottom, left[2]:right[0]]
                white = corridor.min(axis=2) >= 245
                # A connecting neck elsewhere in the balloon cannot substitute
                # for clear facing space along these actual source columns.
                if float(white.mean()) < .98 or float(white.all(axis=0).mean()) < .5:
                    break
                seams.append({'source_ids': [right_id, left_id],
                    'bounds': [left[2], top, right[0], bottom], 'white_fraction': float(white.mean()),
                    'fully_white_column_fraction': float(white.all(axis=0).mean()),
                    'original_rgb_sha256': hashlib.sha256(corridor.tobytes()).hexdigest()})
            if len(seams) != 1:
                continue
            order = tuple(i for i, _ in columns)
            union = bounds(order)
            conflicts = crop_conflicts(union, order, batch.detections)
            previous_conflicts = {c.source_id for p in (first, second) for c in p.crop_conflicts}
            if (any(c.source_id not in previous_conflicts for c in conflicts)
                    or crop_conflicts(seams[0]['bounds'], order, batch.detections)):
                continue
            options.append((first, second, order, union, seams, conflicts))
    proposals, records = list(batch.proposals), []
    for first, second, ids, union, seams, conflicts in options:
        if any(sum(p.proposal_id in (a.proposal_id, b.proposal_id) for a, b, *_ in options) != 1
               for p in (first, second)):
            continue
        identity = {'policy': batch.policy, 'page': asdict(batch.page), 'main_ids': ids, 'members': ids,
            'member_polygons': [(i, raw[i].polygon) for i in ids], 'annotation_parents': (), 'direction': 'vertical'}
        if batch.ownership_policy != 'production8':
            identity['ownership_policy'] = batch.ownership_policy
        if batch.companion_policy != 'disabled':
            identity['companion_policy'] = batch.companion_policy
        proposal_id = 'proposal-'+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        merged = replace(first, proposal_id=proposal_id, main_ids=ids, member_ids=ids, direction='vertical',
            grouping_method=POLICY, crop_bounds=union, crop_safe=not conflicts, crop_conflicts=conflicts,
            structural_reasons=tuple(dict.fromkeys(first.structural_reasons+second.structural_reasons+(REASON,))),
            word_reasons=tuple(dict.fromkeys(first.word_reasons+second.word_reasons)),
            crop_reasons=tuple(dict.fromkeys(first.crop_reasons+second.crop_reasons)),
            rejected_merge_evidence=first.rejected_merge_evidence+second.rejected_merge_evidence,
            region_ids=tuple(sorted(set(first.region_ids+second.region_ids))))
        proposals[proposals.index(first)] = merged
        proposals.remove(second)
        records.append({'policy': POLICY, 'previous_proposal_ids': [first.proposal_id, second.proposal_id],
            'proposal_id': proposal_id, 'main_ids': list(ids), 'bounds': list(union), 'seams': seams,
            'enclosure_evidence': [(i, *owners[i]) for i in ids],
            'crop_conflicts': [asdict(c) for c in conflicts],
            'membership': 'unverified geometric continuation; not certified ownership or a complete utterance',
            'source_strings_modified': False, 'erase_permission': False})
    return replace(batch, proposals=tuple(proposals)), records
