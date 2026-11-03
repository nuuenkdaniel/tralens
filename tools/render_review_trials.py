"""Freeze/run local rendering trials, never OCR or translation accuracy tests.

Native cases use fixed evaluation-only source boxes and placeholder target text.
Controlled overlays alone have clean-background/target-alpha ground truth. No
reference enters planning. No image/provider/model downloads or production hooks.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tralens.rendering import automatic as a, best_effort as b, natural as n, restoration as r, typography as t


# Native coordinates, selected before any rendering trial, not erase masks.
BOXES = {
    'ubunchu04_0002.png': [(995, 145, 1108, 255), (145, 142, 390, 309),
                         (970, 400, 1118, 610), (130, 581, 275, 821)],
    'ubunchu04_0005.png': [(955, 165, 1155, 365), (140, 395, 272, 585),
                         (630, 895, 752, 1070), (1060, 895, 1155, 1030)],
    'ubunchu06_0002.png': [(1035, 358, 1115, 500), (545, 350, 582, 453),
                         (92, 355, 355, 525), (117, 820, 230, 998)],
    'ubunchu06_0005.png': [(930, 130, 1150, 298), (130, 125, 400, 300),
                         (155, 435, 310, 595), (945, 395, 1145, 578)],
    'nemo-1905-10-15.jpg': [(1130, 328, 1277, 376), (714, 420, 848, 582),
                            (1195, 424, 1318, 493), (1023, 825, 1106, 878)],
    'nemo-1905-10-22.jpg': [(557, 158, 707, 263), (1430, 211, 1526, 294),
                            (232, 447, 324, 513), (824, 500, 907, 568)],
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def rectangle(box):
    x, y, right, bottom = box
    return [[x, y], [right, y], [right, bottom], [x, bottom]]


def freeze(examples, fonts, output):
    ledger_bytes = (examples/'acquisition.json').read_bytes()
    ledger = json.loads(ledger_bytes)
    manifest = json.loads(fonts.read_bytes())
    font_id = manifest['fonts'][0]['id']
    data, _ = n.load_fonts(manifest)[font_id]
    face = ImageFont.truetype(BytesIO(data), 28, layout_engine=ImageFont.Layout.RAQM)
    output.mkdir(mode=0o700)
    cases = []

    def save_case(case, source, reference=None, alpha=None):
        folder = output/case['id']
        folder.mkdir()
        source.save(folder/'source.png')
        if reference is not None:
            reference.save(folder/'reference.png')
            alpha.save(folder/'target-alpha.png')
        case['artifacts'] = {p.name: sha(p.read_bytes()) for p in folder.iterdir()}
        cases.append(case)

    for name, boxes in BOXES.items():
        record = next(row for row in ledger['records'] if row['file'] == name)
        original = (examples/name).read_bytes()
        n.require(sha(original) == record['sha256'], 'acquired example drift')
        with Image.open(BytesIO(original)) as opened:
            image = opened.convert('RGB')
        for index, box in enumerate(boxes):
            left, top, right, bottom = box
            crop = (max(0, left-80), max(0, top-70), min(image.width, right+80), min(image.height, bottom+70))
            local = [left-crop[0], top-crop[1], right-crop[0], bottom-crop[1]]
            save_case(dict(id=f'{Path(name).stem}-native-{index}', kind='native_real_text',
                original_file=name, original_sha256=record['sha256'], native_crop=list(crop),
                detections=[dict(source_id='main', polygon=rectangle(local))], member_ids=['main'],
                text=('Please wait! We need to discuss this carefully.' if index % 2 else 'Let us go!'),
                target_kind='English rendering fixture, NOT a translation', clean_ground_truth=False), image.crop(crop))
        for variant in ('short', 'neighbor', 'longword', 'overfull'):
            crop = [350, 500, 734, 756]
            reference = image.crop(crop)
            alpha = Image.new('L', reference.size)
            pen = ImageDraw.Draw(alpha)
            pen.text((75, 100), 'SOURCE?!', font=face, fill=255, stroke_width=0)
            bounds = alpha.getbbox()
            detections = [dict(source_id='main', polygon=rectangle([bounds[0]-2, bounds[1]-2, bounds[2]+1, bounds[3]+1]))]
            if variant == 'neighbor':
                ImageDraw.Draw(reference).text((270, 105), 'KEEP', font=face, fill='black')
                detections.append(dict(source_id='other', polygon=rectangle([268, 105, 349, 138])))
            source = Image.composite(Image.new('RGB', reference.size, 'black'), reference, alpha)
            text = {'short': 'Wait, really?!', 'neighbor': 'We should go now.',
                'longword': 'Supercalifragilisticexpialidocious'*2+'!',
                'overfull': 'Every supplied character must remain. '*20}[variant]
            save_case(dict(id=f'{Path(name).stem}-overlay-{variant}', kind='controlled_overlay',
                original_file=name, original_sha256=record['sha256'], native_crop=crop,
                detections=detections, member_ids=['main'], text=text,
                target_kind='English rendering fixture, NOT a translation', clean_ground_truth=True), source, reference, alpha)
    doc = dict(purpose='frozen_native_and_controlled_render_review', schema_version=1,
        acquisition_sha256=sha(ledger_bytes), fonts_path=str(fonts.absolute()), fonts_sha256=sha(fonts.read_bytes()),
        freeze_source_sha256=sha(Path(__file__).read_bytes()), cases=cases,
        limitations=['Native boxes are evaluation annotations, not OCR results or production user inputs.',
                    'Placeholder target text does not measure translation quality.',
                    'All crops retain native pixels; no resizing. Real hidden backgrounds are unknown.',
                    'Reference images/alpha are scoring-only, not passed to the rendering algorithms.'])
    write_json(output/'manifest.json', doc)
    (output/'manifest.sha256').write_text(sha((output/'manifest.json').read_bytes())+'\n')
    (output/'freeze-source.py').write_bytes(Path(__file__).read_bytes())
    return doc


def run(frozen, output):
    content = (frozen/'manifest.json').read_bytes()
    n.require(sha(content) == (frozen/'manifest.sha256').read_text().strip(), 'frozen manifest drift')
    manifest = json.loads(content)
    fonts = Path(manifest['fonts_path'])
    n.require(sha(fonts.read_bytes()) == manifest['fonts_sha256'], 'font manifest drift')
    data, coverage = next(iter(n.load_fonts(json.loads(fonts.read_bytes())).values()))
    output.mkdir(mode=0o700)
    rows = []
    for case in manifest['cases']:
        folder = frozen/case['id']
        for name, digest in case['artifacts'].items():
            n.require(sha((folder/name).read_bytes()) == digest, 'frozen raster drift')
        image = Image.open(folder/'source.png').convert('RGB')
        raw, members = case['detections'], case['member_ids']
        for policy in ('conservative', 'best-effort'):
            started = time.perf_counter()
            row = dict(id=case['id'], kind=case['kind'], policy=policy, status='skipped', attempts=[])
            plan = None
            routes = [('flat', lambda: a.derive_plan(image, raw, members)),
                      ('classical', lambda: r.derive_plan(image, raw, members))]
            if policy == 'best-effort':
                routes += [('native_hint', lambda: b.native_plan(image, raw, members)),
                           ('footprint', lambda: b.derive_plan(image, raw, members))]
            work = n.LayoutWork()
            for route, derive in routes:
                try:
                    plan = derive()
                    color = plan.evidence.get('background_rgb_at_center', plan.background)
                    fill = [255]*3 if np.dot(color, [.2126, .7152, .0722]) < 128 else [0]*3
                    style = dict(min_size=16, max_size=28, outline=0 if route == 'flat' else 1,
                        fill=fill, outline_fill=[255-v for v in fill])
                    mask = Image.fromarray(plan.layout.astype(np.uint8)*255)
                    # Preserve successful conservative lettering, just like the
                    # production router. Relax only after a conservative failure.
                    fit = t.layout_dialogue if route in ('flat', 'classical') else b.fit
                    try:
                        layer, meta, _ = fit(case['text'], data, coverage, mask, style,
                            **({'work': work} if fit is b.fit else {}))
                    except ValueError as error:
                        if policy == 'conservative' or isinstance(error, n.ResourceLimit) or not str(error).startswith('overflow:'):
                            raise
                        if route == 'footprint':
                            plan = b.overlay_plan(plan)
                        style = {**style, 'outline': 1}
                        layer, meta, _ = b.fit(case['text'], data, coverage,
                            Image.fromarray(plan.layout.astype(np.uint8)*255), style, work=work)
                    alpha = np.asarray(layer.getchannel('A')) != 0
                    n.require(not (alpha & (~plan.layout | plan.obstacles)).any(), 'lettering containment')
                    clean = np.array(image)
                    a.apply_cleanup(clean, plan)
                    plate = Image.fromarray(clean)
                    composite = Image.alpha_composite(plate.convert('RGBA'), layer).convert('RGB')
                    audit = n.pixel_audit(image, plate, composite, plan.erase, alpha, plan.obstacles)
                    n.verify_pixel_audit(audit)
                    n.require(b._insert(case['text'], meta['inserted_break_offsets']) == meta['draw_text'], 'exact wording')
                    destination = output/(case['id']+'-'+policy)
                    destination.mkdir()
                    for name, value in [('source', image), ('clean', plate), ('composite', composite),
                        ('erase', Image.fromarray(plan.erase.astype(np.uint8)*255))]:
                        value.save(destination/(name+'.png'))
                    row.update(status='composed', route=route, size=meta['size'], layout=meta,
                        evidence=plan.evidence, pixel_audit=audit, exact_target_characters=True)
                    # Ground truth becomes available ONLY AFTER planning/compositing.
                    if case['clean_ground_truth']:
                        reference = np.array(Image.open(folder/'reference.png').convert('RGB'))
                        target = np.array(Image.open(folder/'target-alpha.png')) != 0
                        error = np.abs(clean.astype(float)-reference.astype(float))
                        row['reference_scores'] = dict(target_mae=float(error[target].mean()),
                            target_support_coverage=float((target & plan.erase).sum()/target.sum()),
                            changed_unoccluded_pixels=int((np.any(clean != reference, axis=2) & ~target).sum()),
                            exact_clean=bool(np.array_equal(clean, reference)))
                    break
                except ValueError as error:
                    row['attempts'].append(dict(route=route, reason=str(error),
                        code='resource_budget' if isinstance(error, n.ResourceLimit) else getattr(error, 'code', 'validation')))
            row['seconds'] = time.perf_counter()-started
            rows.append(row)
            print(row['id'], policy, row['status'], row.get('route'), row.get('size'), flush=True)
    report = dict(purpose='lower_level_render_quality_trials_NOT_pipeline_coverage', frozen_sha256=sha(content),
        implementation_sha256=a.implementation_hash(), runner_sha256=sha(Path(__file__).read_bytes()),
        cases=rows, limitations=manifest['limitations'], no_models_or_providers=True)
    write_json(output/'report.json', report)
    (output/'runner-source.py').write_bytes(Path(__file__).read_bytes())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--examples', type=Path, required=True)
    f.add_argument('--fonts', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    rparser = sub.add_parser('run')
    rparser.add_argument('--frozen', type=Path, required=True)
    rparser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    mode = args.pop('mode')
    globals()[mode](**args)
