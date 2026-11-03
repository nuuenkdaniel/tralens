"""Offline threshold comparison against explicitly provisional model visual labels.

Labels are scoring-only and are never imported by production. One exposed page
cannot establish accuracy calibration or held-out quality. Grid decisions use
saved cleanup/layout evidence; optional candidate decisions use a real rerender.
No recognizer, translator, network, model, font or response sidecar is opened.
"""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from tools import manga_quality_cases as frozen
from tralens.rendering import natural as n, replacement_confidence as rc


THRESHOLDS = (0, .5, .6, .65, .7, .75, .8, .85, .9, .95, .99, 1)


def observations(data, preview, report, preview_bytes, manifest_bytes):
    n.require(type(data) is dict and type(data['schema_version']) is int and data['schema_version'] == 1
              and data['purpose'] == 'model_visual_replacement_observations'
              and data['independent_reference'] is False and data['split'] == 'development'
              and type(data['reviewer']) is str and bool(data['reviewer'].strip())
              and type(data['limitations']) is str and bool(data['limitations'].strip()),
              'explicit provisional development-only model visual observations required')
    n.require(data['source_sha256'] == preview['input']['sha256']
              and data['preview_sha256'] == n.digest(preview_bytes)
              and data['baseline_manifest_sha256'] == n.digest(manifest_bytes), 'visual observation source/baseline drift')
    records = data['records']
    n.require(type(records) is list and len(records) == len(report['units']), 'all saved units need labels, including uncertain')
    for index, (record, source) in enumerate(zip(records, preview['units'])):
        n.require(type(record) is dict and set(record) == {'index', 'member_ids', 'ocr_match', 'replacement', 'notes'}
                  and type(record['index']) is int and record['index'] == index
                  and n.canonical(record['member_ids']) == n.canonical(source['unit']['member_ids'])
                  and record['ocr_match'] in ('match', 'mismatch', 'uncertain')
                  and record['replacement'] in ('acceptable', 'unacceptable', 'uncertain')
                  and type(record['notes']) is str and 0 < len(record['notes'].strip()) <= 2048,
                  'foreign/duplicate/missing/invalid visual observation')
    return records


def decision_counts(decisions, labels):
    n.require(len(decisions) == len(labels) and all(type(value) is bool for value in decisions), 'exact all-case decisions required')
    result = Counter()
    for accepted, label in zip(decisions, labels):
        result[('replace_' if accepted else 'preserve_') + label['replacement']] += 1
    return {key: result[key] for key in ('replace_acceptable', 'replace_unacceptable', 'replace_uncertain',
                                        'preserve_acceptable', 'preserve_unacceptable', 'preserve_uncertain')}


def threshold_row(units, report, labels, threshold):
    selected, simulated, statuses = [], [], []
    for unit, row in zip(units, report['units']):
        target = row.get('binding', {}).get('text')
        if target is None:
            selected.append(False)
            simulated.append(False)
            statuses.append('no_bound_translation')
            continue
        source = rc.assess_source(unit, target, threshold)
        # OCR-only metrics intentionally do not confuse identity no-ops with errors.
        selected.append(source['minimum_main_line_score'] is not None and source['minimum_main_line_score'] >= threshold)
        native = row.get('cleanup_method', '').startswith(('flat_fill', 'inpaint_')) or row.get('cleanup_method') == 'best_effort_native_fill'
        layout_ok = rc.layout_rejection(row.get('layout', {}), report['style']['min_size']) is None
        accept = bool(source['allowed'] and native and layout_ok and row['status'] == 'automatic_composed')
        simulated.append(accept)
        statuses.append('native_candidate' if accept else source['reason_code'] or 'cleanup_or_layout_not_supported')
    accepted_labels = Counter(label['ocr_match'] for accepted, label in zip(selected, labels) if accepted)
    assessed = accepted_labels['match'] + accepted_labels['mismatch']
    return {'threshold': threshold, 'ocr_only': {'selected': sum(selected), 'all_units': len(units),
        'visual_matches': accepted_labels['match'], 'visual_mismatches': accepted_labels['mismatch'],
        'unassessed': accepted_labels['uncertain'],
        'assessed_visual_precision': accepted_labels['match'] / assessed if assessed else None,
        'matching_units_deferred': sum(not accept and label['ocr_match'] == 'match' for accept, label in zip(selected, labels))},
        'saved_evidence_simulation': decision_counts(simulated, labels), 'simulated_statuses': statuses,
        'simulation_is_rerender': False}


def run(*, preview, image, baseline, labels, output, candidate=None):
    output = Path(output).absolute()
    n.require(not output.exists() and output.parent.is_dir()
              and not any(p.is_symlink() for p in (output, *output.parents)), 'fresh local output required')
    preview_bytes = n.read_explicit(preview)
    original = n.read_explicit(image, 32 * 1024 * 1024)
    label_bytes = n.read_explicit(labels, 1024 * 1024)
    saved, report, units, _, blobs = frozen.load_baseline(preview_bytes, original, Path(image).absolute(), baseline)
    document = n.document(label_bytes)
    records = observations(document, saved, report, preview_bytes, blobs['manifest.json'])
    grid = [threshold_row(units, report, records, threshold) for threshold in THRESHOLDS]
    viable = [row for row in grid if row['ocr_only']['assessed_visual_precision'] is not None
              and row['ocr_only']['assessed_visual_precision'] >= .9]
    recommended = viable[0]['threshold'] if viable else None
    result = {'schema_version': 1, 'purpose': 'development_model_visual_confidence_comparison',
        'source_sha256': saved['input']['sha256'], 'preview_sha256': n.digest(preview_bytes),
        'baseline_manifest_sha256': n.digest(blobs['manifest.json']), 'observations_sha256': n.digest(label_bytes),
        'reviewer': document['reviewer'], 'independent_reference': False, 'held_out_cases': 0,
        'recommendation_rule': 'lowest grid threshold with >=0.90 OCR-only precision against assessed model visual labels; uncertain labels excluded and reported',
        'recommended_minimum': recommended, 'grid': grid, 'all_unit_denominator': len(units),
        'baseline_decisions': decision_counts([row['status'] == 'automatic_composed' for row in report['units']], records),
        'actual_candidate': None, 'new_inference_calls': 0, 'accuracy_probability_calibrated': False,
        'decision_label_scope': 'Labels assess baseline operations; candidate counts measure selection, not automatic visual approval of rerendered pixels.',
        'limitations': [document['limitations'], 'The same exposed cases inform threshold choice and assessment; this is development tuning, not independent validation.',
            'OCR-only precision is agreement with model visual labels, not translation accuracy or calibrated confidence.',
            'Saved-evidence simulations may differ from rerenders because accepted placements affect later layout/reservations.',
            'Fewer replacements can reduce damage while leaving source text visible; report both tradeoffs.']}
    artifacts = {'observations.json': label_bytes, 'source.png': blobs['source.png'],
                 'baseline-composite.png': blobs['composite.png']}
    if candidate is not None:
        _, rendered, _, _, candidate_blobs = frozen.load_baseline(preview_bytes, original, Path(image).absolute(), candidate)
        n.require('replacement_policy' in rendered and 'replacement_confidence' in rendered, 'candidate must use an explicit confidence gate')
        result['actual_candidate'] = {'manifest_sha256': n.digest(candidate_blobs['manifest.json']),
            'settings': rendered['replacement_policy'], 'coverage': rendered['coverage'],
            'decisions': decision_counts([row['status'] == 'automatic_composed' for row in rendered['units']], records),
            'summary': rendered['replacement_confidence'],
            'units': [{'unit_id': row['unit_id'], 'member_ids': list(unit.member_ids), 'selected_text': unit.selected.text,
                      'status': row['status'], 'reason_code': row['reason_code'], 'visual_observation': label}
                     for row, unit, label in zip(rendered['units'], units, records)]}
        artifacts['candidate-composite.png'] = candidate_blobs['composite.png']
    output.mkdir(mode=0o700)
    for name, content in artifacts.items():
        with (output / name).open('xb') as stream:
            stream.write(content)
    with (output / 'report.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; '
        'style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        '<title>Development replacement confidence</title><style>body{font:16px sans-serif;margin:2rem}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;height:auto}</style>'
        '<h1>Provisional model visual calibration — not accuracy validation</h1>'
        '<p>Preserved source lettering is not a translation replacement. All saved translations remain available.</p>'
        '<pre>' + escape(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre>'
        + ''.join('<h2>' + escape(name) + '</h2><img alt="' + escape(name) + '" src="' + name + '">'
                  for name in artifacts if name.endswith('.png')) + '</html>')
    with (output / 'review.html').open('x') as stream:
        stream.write(html)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preview', 'image', 'baseline', 'labels', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    args = vars(parser.parse_args(argv))
    try:
        result = run(**args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, 'confidence comparison failed: ' + str(error) + '\n')
    compact = {key: result[key] for key in ('recommended_minimum', 'all_unit_denominator', 'baseline_decisions', 'new_inference_calls')}
    actual = result['actual_candidate']
    compact['actual_candidate'] = ({key: actual[key] for key in ('manifest_sha256', 'settings', 'decisions', 'summary')}
                                   if actual is not None else None)
    print(json.dumps(compact, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
