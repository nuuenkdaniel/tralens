"""Freeze native quality failures from saved automatic outputs; no inference.

Case windows/anchors are evaluation metadata, never production masks or source
corrections. This tool measures integrity and exposure, NOT OCR/artistic accuracy.
It opens only explicit files and basename-only artifacts, never model sidecars,
font paths, saved upload paths or URLs found in JSON. Historical implementations
are allowed, provided their saved bindings and artifacts are internally current.
"""
import argparse
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tralens.rendering import natural as n
from tralens.translation import coverage


MAX_BYTES = 160 * 1024 * 1024
LAYERS = {'source': 'RGB', 'clean': 'RGB', 'composite': 'RGB', 'text': 'RGBA',
          'erase-mask': 'L', 'protected-mask': 'L'}
UNASSESSED = ['transcription_errors', 'grouping_errors', 'source_text_remnants',
              'hidden_art_reconstruction', 'readability']


def same(actual, expected, message):
    n.require(n.canonical(actual) == n.canonical(expected), message)


def safe_name(name):
    n.require(type(name) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name)
              and name not in ('.', '..'), 'unsafe artifact/case name')
    return name


def png(content, mode, size=None):
    with Image.open(BytesIO(content)) as opened:
        n.require(opened.format == 'PNG' and opened.mode == mode and getattr(opened, 'n_frames', 1) == 1,
                  'unexpected saved PNG format/mode/frame count')
        n.require(opened.width * opened.height <= 4_000_000 and max(opened.size) <= 4096,
                  'saved PNG pixel budget')
        n.require(size is None or opened.size == size, 'saved layer size drift')
        return opened.copy()


def load_baseline(preview_bytes, original_bytes, image_path, rendering):
    """Return a bounded, hash-checked snapshot, not a new rendering or review."""
    preview = n.document(preview_bytes)
    with Image.open(BytesIO(original_bytes)) as opened:
        n.require(getattr(opened, 'n_frames', 1) == 1 and opened.width * opened.height <= 4_000_000
                  and max(opened.size) <= 4096, 'original frame/pixel budget')
        rgb = opened.convert('RGB')
    root = Path(rendering)
    manifest_bytes = n.read_explicit(root / 'manifest.json', 8 * 1024 * 1024)
    report = n.document(manifest_bytes)
    n.require(report['purpose'] == 'automatic_candidate_output' and report['approved_export'] is False,
              'saved automatic candidate required')
    same(report['job_revision'], n.revision({k: v for k, v in report.items()
         if k not in ('job_revision', 'artifacts')}), 'render job revision drift')
    n.require(type(report['artifacts']) is dict and 1 <= len(report['artifacts']) <= 512,
              'saved artifact count budget')
    blobs = {'manifest.json': manifest_bytes}
    total = len(preview_bytes) + len(original_bytes) + len(manifest_bytes)
    for name, digest in report['artifacts'].items():
        safe_name(name)
        n.require(name != 'manifest.json', 'manifest cannot hash itself')
        content = n.read_explicit(root / name, min(64 * 1024 * 1024, max(0, MAX_BYTES - total)))
        total += len(content)
        n.require(total <= MAX_BYTES and n.digest(content) == digest, 'saved artifact hash/byte drift: ' + name)
        blobs[name] = content
    n.require(blobs['evidence-preview.json'] == preview_bytes, 'render/preview bytes differ')
    n.require(blobs['original-file.bin'] == original_bytes, 'render/original bytes differ')
    n.require(report['preview_sha256'] == n.digest(preview_bytes)
              and report['fonts_sha256'] == n.digest(blobs['evidence-fonts.json']), 'preview/font hash drift')
    contract = {'schema_version': 1, 'purpose': 'automatic_candidate_only', 'source': report['source'],
                'preview_sha256': report['preview_sha256'], 'fonts_sha256': report['fonts_sha256'], 'units': []}
    n.bind(preview, image_path, rgb, original_bytes, preview_bytes, contract,
           blobs['evidence-fonts.json'], selection_kind='automatic_candidate_only')
    candidates = preview['ocr'].get('automatic_recognition_results', [])
    units = n.automatic_units(preview['ocr']['batches'][0], candidates[0] if candidates else ())
    same([row['unit_id'] for row in report['units']], [unit.unit_id for unit in units],
         'render/source unit order drift')
    for record, unit, row in zip(preview['units'], units, report['units']):
        n.require(row['source_record_sha256'] == n.revision(record)
                  and row['source_status'] == record['status'], 'render/source record drift')
        same(row['input_kind'], record.get('input_kind', 'strict_utterance'), 'render input-kind drift')
        expected = None
        if record['status'] == 'preview':
            expected = (coverage.bound_result(record, unit) if record.get('input_kind') == coverage.KIND
                        else n.strict_result_binding(record, unit))
        if 'binding' in row:
            n.require(expected is not None, 'render binding without available translation')
            same(row['binding'], expected, 'render/translation binding drift')
        n.require(row['status'] != 'automatic_composed' or 'binding' in row,
                  'composed output lacks source-bound translation')
    layers = {name: png(blobs[name + '.png'], mode, rgb.size) for name, mode in LAYERS.items()}
    n.require(np.array_equal(np.asarray(layers['source']), np.asarray(rgb)), 'saved source raster drift')
    # Recompute locality rather than treating old audit counters as quality proof.
    alpha = np.asarray(layers['text'].getchannel('A')) != 0
    erase = np.asarray(layers['erase-mask']) != 0
    protected = np.asarray(layers['protected-mask']) != 0
    audit = n.pixel_audit(rgb, layers['clean'], layers['composite'], erase, alpha, protected)
    n.verify_pixel_audit(audit)
    same(audit, report['pixel_audit'], 'saved pixel audit drift')
    same(report['coverage']['saved_units'], len(units), 'saved unit count drift')
    same(report['coverage']['rendered'], sum(r['status'] == 'automatic_composed' for r in report['units']),
         'saved replacement count drift')
    return preview, report, units, layers, blobs


def validate_spec(spec, preview, size):
    n.require(type(spec) is dict and set(spec) == {'schema_version', 'purpose', 'source_sha256',
              'split', 'exposure', 'missing_strata', 'cases'}, 'exact quality-case spec fields required')
    n.require(type(spec['schema_version']) is int and spec['schema_version'] == 1
              and spec['purpose'] == 'native_manga_quality_development'
              and spec['split'] == 'development', 'exposed saved failures are development only')
    n.require(spec['source_sha256'] == preview['input']['sha256'], 'case spec/source drift')
    n.require(type(spec['exposure']) is str and 0 < len(spec['exposure'].strip()) <= 2048,
              'explicit prior-exposure note required')
    n.require(type(spec['missing_strata']) is list and len(spec['missing_strata']) <= 32
              and all(type(s) is str and 0 < len(s) <= 256 for s in spec['missing_strata']), 'invalid missing strata')
    n.require(type(spec['cases']) is list and 1 <= len(spec['cases']) <= 32, 'quality case count budget')
    raw = preview['ocr']['raw_detections'][0]
    known = {(type(d['source_id']), d['source_id']): d for d in raw}
    seen = set()
    for case in spec['cases']:
        n.require(type(case) is dict and set(case) == {'id', 'kind', 'anchor_ids', 'bounds', 'tags', 'observation'},
                  'exact case fields required; no references/masks/corrections')
        key = safe_name(case['id'])
        n.require(key not in seen, 'duplicate case ID')
        seen.add(key)
        n.require(case['kind'] in ('detected_text', 'undetected_text_observation', 'negative_art', 'negative_neighbor'),
                  'invalid native case kind')
        ids, bounds = case['anchor_ids'], case['bounds']
        n.require(type(ids) is list and len(ids) <= 64
                  and all(type(i) in (int, str) and (type(i), i) in known for i in ids)
                  and len(set(ids)) == len(ids), 'foreign/duplicate/invalid anchor ID')
        n.require(bool(ids) == (case['kind'] == 'detected_text'), 'only detected cases have source anchors')
        n.require(type(bounds) is list and len(bounds) == 4 and all(type(v) is int for v in bounds)
                  and 0 <= bounds[0] < bounds[2] <= size[0] and 0 <= bounds[1] < bounds[3] <= size[1],
                  'invalid native case bounds')
        for source_id in ids:
            points = known[(type(source_id), source_id)]['polygon']
            n.require(all(bounds[0] <= x < bounds[2] and bounds[1] <= y < bounds[3] for x, y in points),
                      'case must contain each entire anchor polygon')
        n.require(type(case['tags']) is list and 1 <= len(case['tags']) <= 16
                  and all(type(t) is str and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', t) for t in case['tags']),
                  'invalid case tags')
        n.require(type(case['observation']) is str and 0 < len(case['observation'].strip()) <= 2048,
                  'provisional visual observation required')
    return spec


def case_rows(spec, preview, report, units, layers):
    rows, crops = [], {}
    proposals = {p['proposal_id']: p for p in preview['ocr']['batches'][0]['proposals']}
    for case in spec['cases']:
        anchors = {(type(i), i) for i in case['anchor_ids']}
        selected = []
        for record, unit, rendered in zip(preview['units'], units, report['units']):
            if anchors & {(type(i), i) for i in unit.member_ids}:
                selected.append({'unit_id': unit.unit_id, 'member_ids': list(unit.member_ids),
                    'order': list(unit.order), 'selected_text': unit.selected.text if unit.selected else None,
                    'readings': record['unit']['readings'], 'selected_revision': unit.selected_revision,
                    'record_sha256': n.revision(record), 'proposal': proposals[unit.unit_id],
                    'translation_status': record['status'], 'input_kind': record.get('input_kind', 'strict_utterance'),
                    'translation': next((text for uid, text in (record.get('result') or {}).get('translations', [])
                                         if uid == unit.unit_id), None) if record['status'] == 'preview' else None,
                    'render_status': rendered['status'], 'render_quality': rendered.get('render_quality'),
                    'cleanup_method': rendered.get('cleanup_method'), 'layout': rendered.get('layout'),
                    'render_record_sha256': n.revision(rendered)})
        local = {name: layer.crop(case['bounds']) for name, layer in layers.items()}
        for name, image in local.items():
            stream = BytesIO()
            image.save(stream, format='PNG')
            crops['cases/' + case['id'] + '/' + name + '.png'] = stream.getvalue()
        source, clean = np.asarray(local['source']), np.asarray(local['clean'])
        composite = np.asarray(local['composite'])
        changed = np.any(source != clean, axis=2)
        erase = np.asarray(local['erase-mask']) != 0
        protected = np.asarray(local['protected-mask']) != 0
        rows.append({**case, 'split': 'development', 'baseline_units': selected,
            'measurements': {'pixels': int(changed.size), 'page_cleanup_mask_pixels': int(erase.sum()),
                'page_cleanup_changed_pixels': int(changed.sum()),
                'page_composite_source_changed_pixels': int(np.any(source != composite, axis=2).sum()),
                'outside_page_cleanup_changed_pixels': int((changed & ~erase).sum()),
                'protected_clean_changed_pixels': int((changed & protected).sum())},
            'quality_assessment': {key: None for key in UNASSESSED},
            'measurement_scope': 'page result within diagnostic crop, including edits by neighboring units; not accuracy'})
    return rows, crops


def review_html(rows):
    cards = []
    for row in rows:
        key = row['id']
        cards.append('<section><h2>' + escape(key) + '</h2><p>' + escape(row['observation']) + '</p>'
            + '<pre>' + escape(json.dumps({'units': row['baseline_units'], 'measurements': row['measurements']},
                                         ensure_ascii=False, indent=2)) + '</pre>'
            + ''.join('<figure><figcaption>' + escape(name) + '</figcaption><img alt="' + escape(name)
                      + ' — ' + escape(key) + '" src="cases/' + key + '/' + name + '.png"></figure>'
                      for name in LAYERS) + '</section>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; '
        'style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        '<title>Frozen manga quality cases</title><style>body{font-family:system-ui;margin:2rem}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:20rem;overflow:auto}'
        'figure{display:inline-block;vertical-align:top;margin:.5rem}img{max-width:100%;height:auto}'
        'section{border-top:2px solid #aaa;margin-top:2rem}</style></head><body>'
        '<h1>Development evidence — not quality success</h1><p>Native-pixel crops; no resampling. '
        'OCR correctness, grouping, source remnants, hidden art and readability remain unassessed. '
        'These case windows are not production erase masks. Negative observations are provisional.</p>'
        + ''.join(cards) + '</body></html>')


def summarize(saved, report, units, rows):
    return {'native_cases': len(rows), 'development_cases': len(rows), 'held_out_cases': 0,
        'saved_units': len(units), 'translation_results': sum(r['status'] == 'preview' for r in saved['units']),
        'replacements': sum(r['status'] == 'automatic_composed' for r in report['units']),
        'best_effort_replacements': sum(r['status'] == 'automatic_composed' and r.get('render_quality') == 'best_effort'
                                        for r in report['units']),
        'unassessed_dimensions': UNASSESSED, 'new_inference_calls': 0, 'quality_improvement_claimed': False}


def freeze(*, preview, image, rendering, cases, output):
    output = Path(output).absolute()
    n.require(not output.exists() and output.parent.is_dir()
              and not any(p.is_symlink() for p in (output, *output.parents)), 'fresh nonsymlink output required')
    preview_bytes = n.read_explicit(preview)
    original_bytes = n.read_explicit(image, 32 * 1024 * 1024)
    spec_bytes = n.read_explicit(cases, 1024 * 1024)
    saved, report, units, layers, blobs = load_baseline(preview_bytes, original_bytes, Path(image).absolute(), rendering)
    spec = validate_spec(n.document(spec_bytes), saved, layers['source'].size)
    rows, crops = case_rows(spec, saved, report, units, layers)
    artifacts = {'baseline/' + key: value for key, value in blobs.items()}
    artifacts.update(crops)
    artifacts['case-spec.json'] = spec_bytes
    artifacts['freeze-source.py'] = Path(__file__).read_bytes()
    artifacts['review.html'] = review_html(rows).encode('utf-8')
    n.require(len(artifacts) <= 1024 and sum(map(len, artifacts.values())) <= MAX_BYTES, 'freeze artifact budget')
    summary = summarize(saved, report, units, rows)
    manifest = {'schema_version': 1, 'purpose': 'frozen_native_manga_quality_development', 'source': saved['input'],
        'preview_sha256': n.digest(preview_bytes), 'baseline_manifest_sha256': n.digest(blobs['manifest.json']),
        'case_spec_sha256': n.digest(spec_bytes), 'exposure': spec['exposure'], 'missing_strata': spec['missing_strata'],
        'cases': rows, 'summary': summary, 'artifacts': {key: n.digest(value) for key, value in artifacts.items()},
        'limitations': ['Native hidden backgrounds and checked transcriptions are unavailable; quality scores stay null.',
            'Case anchors/windows and visual notes are evaluation observations, not truth or production user inputs.',
            'All crops from this exposed page are development; no native held-out claim.',
            'Retained hashes check local consistency, not publisher authenticity or rights to publish.',
            'Raw response sidecars remain at their original local paths; they are not opened or certified.']}
    # Recheck explicit inputs before publishing; never mutate old evidence.
    for path, content in ((preview, preview_bytes), (image, original_bytes), (cases, spec_bytes)):
        n.require(n.read_explicit(path) == content, 'explicit input changed during freeze')
    output.mkdir(mode=0o700)
    for name, content in artifacts.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
    content = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with (output / 'manifest.json').open('xb') as stream:
        stream.write(content)
    with (output / 'manifest.sha256').open('x') as stream:
        stream.write(n.digest(content) + '\n')
    return {**summary, 'manifest_sha256': n.digest(content)}


def verify(frozen, *, expected_sha256=None):
    """Verify a self-contained snapshot; no access to its original absolute paths."""
    root = Path(frozen)
    content = n.read_explicit(root / 'manifest.json', 16 * 1024 * 1024)
    pin = n.read_explicit(root / 'manifest.sha256', 128).decode().strip()
    n.require(n.digest(content) == pin and (expected_sha256 is None or pin == expected_sha256), 'frozen manifest drift')
    manifest = n.document(content)
    n.require(manifest['schema_version'] == 1 and manifest['purpose'] == 'frozen_native_manga_quality_development'
              and type(manifest['artifacts']) is dict and 1 <= len(manifest['artifacts']) <= 1024,
              'invalid frozen quality manifest')
    total = 0
    for name, digest in manifest['artifacts'].items():
        n.require(type(name) is str, 'frozen artifact path type')
        parts = name.split('/')
        n.require(1 <= len(parts) <= 3, 'frozen artifact path depth')
        for part in parts:
            safe_name(part)
        data = n.read_explicit(root / name, min(64 * 1024 * 1024, max(0, MAX_BYTES - total)))
        total += len(data)
        n.require(total <= MAX_BYTES and n.digest(data) == digest, 'frozen artifact drift: ' + name)
    baseline = root / 'baseline'
    saved, report, units, layers, blobs = load_baseline(n.read_explicit(baseline / 'evidence-preview.json'),
        n.read_explicit(baseline / 'original-file.bin'), manifest['source']['path'], baseline)
    spec_bytes = n.read_explicit(root / 'case-spec.json', 1024 * 1024)
    spec = validate_spec(n.document(spec_bytes), saved, layers['source'].size)
    rows, crops = case_rows(spec, saved, report, units, layers)
    same(manifest['source'], saved['input'], 'frozen source drift')
    same(manifest['cases'], rows, 'frozen case/baseline drift')
    same(manifest['summary'], summarize(saved, report, units, rows), 'frozen summary drift')
    same(manifest['exposure'], spec['exposure'], 'frozen exposure drift')
    same(manifest['missing_strata'], spec['missing_strata'], 'frozen missing strata drift')
    n.require(manifest['preview_sha256'] == n.digest(blobs['evidence-preview.json'])
              and manifest['baseline_manifest_sha256'] == n.digest(blobs['manifest.json'])
              and manifest['case_spec_sha256'] == n.digest(spec_bytes), 'frozen input digest drift')
    required = {'baseline/' + key: n.digest(value) for key, value in blobs.items()}
    required.update({key: n.digest(value) for key, value in crops.items()})
    required.update({'case-spec.json': n.digest(spec_bytes),
                     'freeze-source.py': n.digest(n.read_explicit(root / 'freeze-source.py')),
                     'review.html': n.digest(review_html(rows).encode('utf-8'))})
    same(manifest['artifacts'], required, 'frozen artifact inventory/derivation drift')
    return {'verified': True, 'manifest_sha256': pin, 'native_cases': len(rows),
            'held_out_cases': 0, 'new_inference_calls': 0, 'quality_improvement_claimed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('freeze')
    for name in ('preview', 'image', 'rendering', 'cases', 'output'):
        create.add_argument('--' + name, type=Path, required=True)
    check = sub.add_parser('verify')
    check.add_argument('--frozen', type=Path, required=True)
    check.add_argument('--expected-sha256')
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    try:
        result = freeze(**args) if command == 'freeze' else verify(**args)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, 'quality cases failed: ' + str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
