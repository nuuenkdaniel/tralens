"""Static candidate provenance and explicit, unsigned aesthetic review records.

No review grants translation truth, human identity certification or final export.
Only CLI/API-supplied paths are opened; artifact names have a closed whitelist.
"""
from io import BytesIO
import html
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from . import natural as n


POLICY = 'phase3_candidate_styles_review_v1'
STATES = ('not_assessed', 'accepted', 'rejected', 'rework')
CSP = "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; object-src 'none'"
BASE_FILES = {'evidence-preview.json', 'evidence-selection.json', 'evidence-fonts.json', 'original-file.bin',
              'source.png', 'clean.png', 'composite.png', 'cleanup-diff.png', 'composite-diff.png'}


def implementation_hash():
    root = Path(__file__).parent
    return n.revision({name: n.digest(n.read_explicit(root/name, 256*1024))
                       for name in ('natural.py', 'styles.py', 'review.py')})


def job_revision(report):
    return n.revision({k: report[k] for k in ('policy', 'binding', 'mode', 'engine',
                       'implementation_sha256', 'selection_sha256', 'parent_candidate', 'comparison_candidates')})


def artifact_name(name):
    return name in BASE_FILES or name == 'review.html' or bool(re.fullmatch(
        r'unit-[0-9]{3}-(?:interior|layout|permission|protected|text|glyph-cleanup|specimen-(?:source|clean|text|composite|interior|layout|permission|protected|glyph-cleanup))\.png', name)) or bool(re.fullmatch(r'comparison-[0-3]\.png', name))


def read_bundle(candidate):
    root = Path(candidate).absolute()
    raw = n.read_explicit(root/'manifest.json', 8*1024*1024)
    report = n.document(raw)
    n.require(report['schema_version'] == 1 and report['purpose'] == 'candidate_only'
              and report['approved_export'] is False and report['mode'] in ('plan', 'compose'), 'foreign candidate manifest')
    artifacts = report['artifacts']
    n.require(type(report['units']) is list and 1 <= len(report['units']) <= 64, 'candidate unit budget')
    n.require(type(artifacts) is dict and BASE_FILES <= set(artifacts) and len(artifacts) <= 1200, 'artifact inventory budget/missing evidence')
    total = 0
    for name, expected in artifacts.items():
        n.require(type(name) is str and artifact_name(name), 'unapproved artifact path/name')
        if name.startswith('unit-'):
            n.require(int(name[5:8]) < len(report['units']), 'foreign artifact unit index')
        n.require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected), 'invalid artifact hash')
        total += (root/name).stat().st_size
        n.budget_check(total <= 128*1024*1024, 'candidate artifact byte budget')
        n.require(n.digest(n.read_explicit(root/name)) == expected, 'candidate artifact hash drift: '+name)
    binding = report['binding']
    for name, key in (('evidence-preview.json', 'preview_sha256'), ('original-file.bin', 'image_sha256'),
                      ('evidence-fonts.json', 'fonts_sha256')):
        n.require(artifacts[name] == binding[key], 'candidate evidence binding drift')
    selected = n.document(n.read_explicit(root/'evidence-selection.json', 4*1024*1024))
    n.require(n.canonical(selected) == binding['selection_json'], 'candidate selection binding drift')
    if 'job_revision' in report:
        n.require('review.html' in artifacts, 'missing hashed static report')
        n.require(report['policy'] == POLICY and report['job_revision'] == job_revision(report), 'job revision drift')
    return report, n.digest(raw)


def specimen_html(report):
    escape = html.escape
    head = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="'
            +escape(CSP, quote=True)+'"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Candidate aesthetic specimens</title><style>*{box-sizing:border-box} '
            'body{font:16px sans-serif;margin:clamp(.75rem,3vw,2rem);overflow-wrap:anywhere} '
            '.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1em} .grid>*{min-width:0} '
            'figure{margin:0;min-width:0} img{display:block;max-width:100%;height:auto;border:1px solid #999} '
            'pre{white-space:pre-wrap;overflow-wrap:anywhere} .text{background:#eee} section{border-top:2px solid #888;margin-top:2em}'
            '@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}} '
            '@media(max-width:420px){.grid{grid-template-columns:minmax(0,1fr)}} '
            '</style><h1>Candidate aesthetic specimens</h1><p>No linguistic approval, identity verification or final export. '
            'Aesthetic state starts not_assessed; explicit external JSON review records are separate.</p><p>Job revision: '
            +escape(report['job_revision'])+'</p><p><a href="manifest.json">Full provenance</a></p>')
    sections = []
    for index, row in enumerate(report['units']):
        prefix = f'unit-{index:03d}-specimen-'
        figures = ''.join('<figure><figcaption>'+name+'</figcaption><img class="text" alt="'+name+'" src="'
                          +prefix+name+'.png"></figure>' for name in ('source', 'clean', 'text', 'composite'))
        masks = ''.join('<figure><figcaption>'+('glyph-cleanup (applied)' if name == 'glyph-cleanup' else name)
                        +'</figcaption><img alt="'+name+'" src="'+prefix+name+'.png"></figure>'
                        for name in ('interior', 'layout', 'permission', 'protected', 'glyph-cleanup'))
        diagnostic = row.get('cleanup_diagnostics')
        rejection = ''
        if diagnostic is not None:
            rejection = ('<aside><h3>Cleanup rejection: '+escape(diagnostic['code'])+'</h3><p>'
                +escape(diagnostic['message'])+'</p><p>Background estimate RGB: '
                +escape(str(diagnostic['background_estimate_rgb']))+'; sample pixels: '
                +escape(str(diagnostic['sample_count']))+'. Offending pixels: '
                +escape(str(diagnostic['offending_count']))+'; source bounds (half-open): '
                +escape(str(diagnostic['offending_bounds']))+'.</p><p>Pixel evidence only; semantic cause is not classified. '
                'No cleanup or translated layer is applied for this blocked unit.</p><p><strong>Next safe action:</strong> '
                +escape(diagnostic['next_safe_action'])+'</p></aside>')
        sections.append('<section><h2>'+escape(row['unit_id'])+' — '+escape(row['status'])+'</h2><p>'
                        +escape(row.get('reason', 'Candidate only'))+'</p>'+rejection+'<div class="grid">'+figures+'</div><div class="grid">'
                        +masks+'</div><details><summary>Exact text, settings, approximation and locality counters</summary><pre>'
                        +escape(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False))+'</pre></details></section>')
    comparisons = ''.join('<section><h2>Explicit source-bound comparison '+str(i)+'</h2><pre>'
                          +escape(json.dumps(item, indent=2))+'</pre><img alt="comparison candidate" src="comparison-'
                          +str(i)+'.png"></section>' for i, item in enumerate(report['comparison_candidates']))
    return head+''.join(sections)+comparisons+'</html>'


def parent_link(candidate, image_sha256):
    report, sha = read_bundle(candidate)
    n.require(report['binding']['image_sha256'] == image_sha256, 'foreign parent source')
    return {'manifest_sha256': sha, 'job_revision': report.get('job_revision'),
            'source_sha256': image_sha256, 'scope': 'hash-checked candidate snapshot; no review inherited'}


def read_png(root, name, size):
    data = n.read_explicit(Path(root)/name)
    with Image.open(BytesIO(data)) as opened:
        n.require(opened.format == 'PNG' and opened.size == size and getattr(opened, 'n_frames', 1) == 1,
                  'candidate raster size/frame drift')
        n.require(opened.mode in ('RGB', 'RGBA', 'L'), 'candidate raster mode drift')
        return opened.copy()


def verify_candidate(*, candidate, preview, image, selection, fonts):
    """Re-read current inputs, re-layout accepted units and reconstruct all pixels.

    This is deterministic consistency validation, not identity or quality proof.
    """
    report, manifest_sha = read_bundle(candidate)
    n.require(report.get('policy') == POLICY and report.get('implementation_sha256') == implementation_hash(),
              'candidate implementation/policy changed; render a fresh job')
    contents = [n.read_explicit(p, limit) for p, limit in ((preview, 64*1024*1024), (image, 64*1024*1024),
                                                        (selection, 4*1024*1024), (fonts, 256*1024))]
    p, s, f = (n.document(contents[i]) for i in (0, 2, 3))
    with Image.open(BytesIO(contents[1])) as opened:
        n.require(opened.width*opened.height <= n.MAX_PAGE_PIXELS and opened.mode in ('L', 'RGB')
                  and getattr(opened, 'n_frames', 1) == 1 and not opened.info.get('icc_profile')
                  and opened.getexif().get(274, 1) == 1, 'unsupported current source')
        rgb = opened.convert('RGB')
    binding = n.bind(p, image, rgb, contents[1], contents[0], s, contents[3])
    n.require(n.digest(contents[2]) == report['artifacts']['evidence-selection.json'] == report['selection_sha256'],
              'exact selection file changed')
    n.require(n.canonical(report['binding']) == n.canonical(n.asdict(binding)), 'current input/selection binding changed')
    n.budget_check(len(s['units'])*rgb.width*rgb.height <= n.MAX_UNIT_PIXELS, 'verification mask budget')
    font_map = n.load_fonts(f)  # Revalidate actual local font and license bytes.
    current_engine = {'pillow': n.pillow_version, 'opencv': n.cv2.__version__,
        **{key: n.features.version_feature(key) for key in ('raqm', 'harfbuzz', 'fribidi')},
        'freetype': n.features.version_module('freetype2')}
    n.require(report['engine'] == current_engine, 'candidate shaping engine changed; render a fresh job')
    n.require([r['unit_id'] for r in report['units']] == [u['unit_id'] for u in s['units']], 'candidate unit inventory drift')
    n.require(np.array_equal(np.asarray(read_png(candidate, 'source.png', rgb.size)), np.asarray(rgb)), 'source raster drift')
    clean = np.asarray(rgb).copy()
    tiles = []
    specimen_inputs = []
    protected_union = np.zeros((rgb.height, rgb.width), dtype=bool)
    cleanup_union = np.zeros_like(protected_union)
    alpha_union = np.zeros_like(protected_union)
    occupied = np.zeros_like(protected_union)
    raw = p['ocr']['raw_detections'][0]
    for index, (selected, row) in enumerate(zip(s['units'], report['units'])):
        n.require(row.get('aesthetic_state') == 'not_assessed' and row.get('linguistic_approval') == 'not_assessed',
                  'candidate cannot manufacture review/linguistic approval')
        n.require('cleanup_diagnostics' in row, 'cleanup diagnostics drift: missing field')
        masks = {}
        for key, field in (('interior', 'interior'), ('layout', 'layout'), ('permission', 'cleanup_permission'), ('protected', 'protected')):
            mask = n.union_mask(selected[field], rgb.size)
            actual = np.asarray(read_png(candidate, f'unit-{index:03d}-{key}.png', rgb.size))
            n.require(actual.ndim == 2 and np.array_equal(actual, mask.astype(np.uint8)*255), 'candidate mask drift')
            masks[key] = mask
        n.require(masks['layout'].any() and not (masks['layout'] & (~masks['interior'] | masks['protected'])).any(),
                  'verification layout outside interior/protection')
        n.require(masks['permission'].any() and not (masks['permission'] & (~masks['interior'] | masks['protected'])).any(),
                  'verification cleanup permission outside interior/protection')
        n.require(not ((masks['interior'] | masks['protected']) & occupied).any(), 'verification region collision')
        occupied |= masks['interior'] | masks['protected']
        member_keys = {(type(value).__name__, value) for value in selected['member_ids']}
        for detection in raw:
            key = (type(detection['source_id']).__name__, detection['source_id'])
            if key not in member_keys:
                neighbor = np.asarray(n.polygon_mask(detection['polygon'], rgb.size, evidence=True)) != 0
                n.require(not (neighbor & (masks['interior'] | masks['layout'] | masks['permission'])).any(),
                          'verification neighbor/annotation contamination')
        protected_union |= masks['protected']
        masks['glyph-cleanup'] = np.zeros_like(cleanup_union)
        specimen_inputs.append((masks, None))
        if row['status'] == 'blocked':
            # Blocked layouts may have a text diagnostic tile; rederive it when
            # present rather than trusting the specimen's visual content.
            text_name = f'unit-{index:03d}-text.png'
            expected_diagnostics = None
            if text_name in report['artifacts']:
                font = font_map[selected['font_id']]
                n.require(not isinstance(font, n.ResourceLimit), 'blocked diagnostic font resource drift')
                diagnostic, provenance = n.layout(selected['text'], *font,
                    Image.fromarray(masks['layout'].astype(np.uint8)*255), selected['style'])
                n.require(n.canonical(provenance) == n.canonical(row['layout']), 'blocked diagnostic layout drift')
                n.require(np.array_equal(np.asarray(read_png(candidate, text_name, rgb.size)), np.asarray(diagnostic)), 'blocked diagnostic text drift')
                box = diagnostic.getchannel('A').getbbox()
                specimen_inputs[-1] = (masks, (diagnostic.crop(box), box[:2]))
                if report['mode'] == 'compose':
                    try:
                        n.cleanup(rgb, masks['interior'], masks['permission'], masks['protected'])
                    except n.CleanupBlocked as error:
                        expected_diagnostics = error.diagnostics
            n.require(n.canonical(row['cleanup_diagnostics']) == n.canonical(expected_diagnostics), 'cleanup diagnostics drift')
            if expected_diagnostics is not None:
                n.require(row['reason'] == expected_diagnostics['message'] and row['failure_kind'] == 'validation',
                          'cleanup diagnostic reason drift')
            continue
        n.require(row['status'] == ('candidate_composed' if report['mode'] == 'compose' else 'candidate_layout_no_erase'), 'candidate status drift')
        n.require(row['cleanup_diagnostics'] is None, 'cleanup diagnostics drift: successful unit has rejection evidence')
        font = font_map[selected['font_id']]
        n.require(not isinstance(font, n.ResourceLimit), 'font resource failure in accepted candidate')
        layer, provenance = n.layout(selected['text'], *font, Image.fromarray(masks['layout'].astype(np.uint8)*255), selected['style'])
        n.require(n.canonical(provenance) == n.canonical(row['layout']) and n.canonical(row['resolved_style']) == n.canonical(provenance['resolved_style'])
                  and row['style_approximation'] == provenance['style_approximation']
                  and row['text'] == selected['text'] and row['font_id'] == selected['font_id'], 'resolved style/layout/text drift')
        actual = read_png(candidate, f'unit-{index:03d}-text.png', rgb.size)
        n.require(actual.mode == 'RGBA' and np.array_equal(np.asarray(actual), np.asarray(layer)), 'text layer drift')
        alpha = np.asarray(layer.getchannel('A')) != 0
        n.require(not (alpha & (~masks['layout'] | masks['protected'])).any(), 'transformed alpha outside layout/protection')
        plate = rgb.copy()
        glyph = np.zeros_like(cleanup_union)
        if report['mode'] == 'compose':
            plate, glyph, color = n.cleanup(rgb, masks['interior'], masks['permission'], masks['protected'])
            actual = np.asarray(read_png(candidate, f'unit-{index:03d}-glyph-cleanup.png', rgb.size))
            n.require(actual.ndim == 2 and np.array_equal(actual, glyph.astype(np.uint8)*255), 'cleanup mask drift')
            clean[glyph] = np.asarray(plate)[glyph]
            cleanup_union |= glyph
            masks['glyph-cleanup'] = glyph
            n.require(row['background'] == color and row['cleanup_pixels'] == int(glyph.sum()), 'cleanup provenance counters drift')
        unit_composite = Image.alpha_composite(plate.convert('RGBA'), layer).convert('RGB')
        unit_audit = n.pixel_audit(rgb, plate, unit_composite, glyph, alpha, masks['protected'])
        n.verify_pixel_audit(unit_audit)
        n.require(unit_audit == row['pixel_audit'] and row['outside_layout_alpha_pixels'] == 0
                  and row['outside_cleanup_permission_pixels'] == 0, 'unit locality counters drift')
        alpha_union |= alpha
        box = layer.getchannel('A').getbbox()
        tiles.append((layer.crop(box), box[:2]))
        specimen_inputs[-1] = (masks, tiles[-1])
    expected_clean = Image.fromarray(clean, 'RGB')
    expected_composite = expected_clean.convert('RGBA')
    for tile, xy in tiles:
        expected_composite.alpha_composite(tile, dest=xy)
    expected_composite = expected_composite.convert('RGB')
    actual_clean = read_png(candidate, 'clean.png', rgb.size)
    actual_composite = read_png(candidate, 'composite.png', rgb.size)
    n.require(np.array_equal(np.asarray(expected_clean), np.asarray(actual_clean)) and
              np.array_equal(np.asarray(expected_composite), np.asarray(actual_composite)), 'candidate output pixels drift')
    audit = n.pixel_audit(rgb, actual_clean, actual_composite, cleanup_union, alpha_union, protected_union)
    n.verify_pixel_audit(audit)
    n.require(audit == report['pixel_audit'], 'candidate locality counters drift')
    for index, (masks, tile) in enumerate(specimen_inputs):
        metadata, rasters = n.specimen_rasters(rgb, actual_clean, actual_composite, masks, tile)
        n.require(report['units'][index]['specimen'] == metadata, 'specimen geometry drift')
        for name, expected in rasters.items():
            actual = read_png(candidate, f'unit-{index:03d}-specimen-{name}.png', expected.size)
            n.require(actual.mode == expected.mode and np.array_equal(np.asarray(actual), np.asarray(expected)), 'specimen raster drift')
    n.require(n.read_explicit(Path(candidate)/'review.html', 4*1024*1024) == specimen_html(report).encode(),
              'static specimen report drift')
    return report, manifest_sha


def review_binding(report, manifest_sha, unit_id):
    rows = [r for r in report['units'] if r['unit_id'] == unit_id]
    n.require(len(rows) == 1 and rows[0]['status'] == 'candidate_composed', 'foreign/blocked/non-composed review target')
    return {'manifest_sha256': manifest_sha, 'job_revision': report['job_revision'], 'unit_id': unit_id,
            'source_sha256': report['binding']['image_sha256'], 'preview_sha256': report['binding']['preview_sha256'],
            'selection_sha256': report['artifacts']['evidence-selection.json'], 'fonts_sha256': report['binding']['fonts_sha256'],
            'unit_sha256': n.revision(rows[0]), 'artifacts_sha256': n.revision(report['artifacts'])}


def fresh_file(path, data):
    path = Path(path).absolute()
    n.require(not any(p.is_symlink() for p in (path, *path.parents)) and not path.exists() and path.parent.is_dir(),
              'review output must be fresh, without symlinks, with existing parent')
    with path.open('xb') as stream:
        stream.write(data)


def record_review(*, candidate, unit_id, state, operator, note, output, **inputs):
    n.require(state in STATES, 'explicit aesthetic state required')
    n.require(type(operator) is str and 0 < len(operator.strip()) <= 120 and type(note) is str
              and 0 < len(note.strip()) <= 2000, 'explicit operator and review note required')
    report, sha = verify_candidate(candidate=candidate, **inputs)
    record = {'schema_version': 1, 'scope': 'aesthetic_only', 'state': state, 'operator': operator, 'note': note,
              'operator_action': 'explicit_cli_or_api_submission', 'identity_verified': False,
              'linguistic_approval': 'not_assessed', 'candidate_only': True,
              'binding': review_binding(report, sha, unit_id)}
    record['record_revision'] = n.revision(record)
    fresh_file(output, (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
    return record


def verify_review(*, candidate, review, **inputs):
    record = n.document(n.read_explicit(review, 16*1024))
    expected_keys = {'schema_version', 'scope', 'state', 'operator', 'note', 'operator_action', 'identity_verified',
                     'linguistic_approval', 'candidate_only', 'binding', 'record_revision'}
    n.require(set(record) == expected_keys and type(record['schema_version']) is int and record['schema_version'] == 1
              and record['scope'] == 'aesthetic_only' and record['state'] in STATES and record['identity_verified'] is False
              and record['candidate_only'] is True and record['linguistic_approval'] == 'not_assessed'
              and record['operator_action'] == 'explicit_cli_or_api_submission', 'foreign/fabricated review contract')
    core = {k: v for k, v in record.items() if k != 'record_revision'}
    n.require(record['record_revision'] == n.revision(core), 'review record drift')
    n.require(type(record['operator']) is str and 0 < len(record['operator'].strip()) <= 120
              and type(record['note']) is str and 0 < len(record['note'].strip()) <= 2000, 'review action evidence missing')
    report, sha = verify_candidate(candidate=candidate, **inputs)
    n.require(record['binding'] == review_binding(report, sha, record['binding']['unit_id']), 'stale/foreign review binding')
    return record
