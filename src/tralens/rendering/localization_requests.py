"""Model-free, source-bound crop demand after native rendering.

This plans evidence, never erasure. Only the coordinator can dispatch OCR. The
existing localization prerequisites and retry order/limits remain authoritative.
"""
import time

from . import natural as n, localized_strokes as strokes
from ..ocr.source_localization import crop_bounds, geometry_revision, MAX_CROP_PIXELS


POLICY = 'unresolved_cleanup_crop_demand_v1'
MAX_PREREQUISITE_WORK = 1_024_000_000


def preview_revision(preview):
    return n.revision({**preview, 'ocr': {k: v for k, v in preview['ocr'].items()
                                        if k != 'source_localization'}})


def plan(preview, rgb, raw, units, report):
    started = time.monotonic()
    proposals = {p['proposal_id']: p for p in preview['ocr']['batches'][0]['proposals']}
    sources = {(type(d['source_id']), d['source_id']): d for d in raw}
    pairs = list(zip(units, report['units']))
    n.require(len(units) == len(report['units']) and all(u.unit_id == r['unit_id'] for u, r in pairs),
              'localization demand native unit coverage drift')
    pairs.sort(key=lambda pair: (proposals[pair[0].unit_id]['direction'] != 'vertical',
        len(pair[0].order) < 2, -len(pair[0].selected.text), pair[0].unit_id))
    rows, requests = [], []
    work = check_work = crop_pixels = 0
    pixels = rgb.width*rgb.height
    for unit, row in pairs:
        item = {'unit_id': unit.unit_id, 'unit_revision': unit.revision,
                'member_ids': list(unit.member_ids), 'status': 'not_requested', 'reason': 'not_unresolved_cleanup'}
        rows.append(item)
        if row['status'] in ('ready', 'automatic_composed'):
            item['reason'] = 'already_placed'
            continue
        if not (row['status'] == 'skipped' and row.get('attempted')
                and row['reason_code'] == 'replacement_cleanup_confidence'):
            continue
        proposal = proposals[unit.unit_id]
        main = [sources[(type(i), i)] for i in proposal['main_ids']]
        if (proposal['annotation_ids'] or not 0 < len(main) <= 8
                or n.canonical(proposal['main_ids']) != n.canonical(unit.member_ids)
                or any(d['confidence'] < .7 or not d['text'].strip() for d in main)):
            item['reason'] = 'source_not_eligible_for_localization'
            continue
        bounds = crop_bounds(raw, unit.member_ids, rgb.size)
        area = (bounds[2]-bounds[0])*(bounds[3]-bounds[1])
        if area > MAX_CROP_PIXELS or max(bounds[2]-bounds[0], bounds[3]-bounds[1]) > 1024:
            item['reason'] = 'localization_crop_budget'
            continue
        extra_check = pixels*(len(unit.member_ids)+8)
        if check_work+extra_check > MAX_PREREQUISITE_WORK:
            item['reason'] = 'localization_prerequisite_budget'
            continue
        check_work += extra_check
        try:
            strokes.prerequisites(rgb, raw, list(unit.member_ids))
        except (ValueError, KeyError, TypeError) as error:
            item.update(reason=getattr(error, 'code', 'validation'), detail=str(error))
            continue
        extra = pixels*(len(raw)+128)
        if (len(requests) >= strokes.MAX_RETRIES or work+extra > strokes.MAX_RETRY_WORK
                or crop_pixels+area > 4_000_000):
            item['reason'] = 'localization_retry_budget'
            continue
        work += extra
        crop_pixels += area
        item.update(status='requested', reason='unresolved_cleanup', bounds=bounds,
            crop_pixels_sha256=n.digest(rgb.crop(bounds).tobytes()), retry_work=extra)
        requests.append(unit.unit_id)
    result = {'policy': POLICY, 'source': report['source'], 'fonts_sha256': report['fonts_sha256'],
        'font_id': report['font_id'], 'style': report['style'], 'preview_revision': preview_revision(preview),
        'geometry_revision': geometry_revision(raw), 'render_policy': report['render_policy'],
        'background_method': report['background_method'], 'evaluation_cap': report['evaluation_cap'],
        'layout_policy': report.get('layout_policy', 'source-local'),
        'replacement_policy': report.get('replacement_policy'),
        'requests': requests, 'records': rows, 'retry_work': work,
        'prerequisite_work': check_work, 'crop_pixels': crop_pixels,
        'maximum_retries': strokes.MAX_RETRIES, 'maximum_retry_work': strokes.MAX_RETRY_WORK,
        'maximum_prerequisite_work': MAX_PREREQUISITE_WORK}
    return result, time.monotonic()-started


def validate(saved, expected):
    n.require(n.canonical(saved) == n.canonical(expected), 'localization demand source/font/options/native plan drift')
