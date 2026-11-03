"""Freeze local mask/restoration fixtures BEFORE learned inference. No downloads.

Evaluation support is not production erase permission. Synthetic reference masks
come from known overlays; real pages have neither text-mask nor cleanplate truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)+'\n')


def polygon(bounds):
    x1, y1, x2, y2 = bounds
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def freeze(*, root, output, fonts):
    root, output = Path(root).resolve(), Path(output).absolute()
    font_manifest = json.loads(Path(fonts).read_text())
    font = font_manifest['fonts'][0]
    font_path = Path(font['path'])
    assert digest(font_path.read_bytes()) == font['sha256'], 'font drift'
    source = root/'test-results/automatic-unit-diagnosis-parent-v3/automatic-proposals.json'
    proposals = json.loads(source.read_text())
    output.mkdir(parents=True, exist_ok=False)
    cases, pages = [], []
    for page in proposals['pages']:
        name = Path(page['page']['image_path']).name
        if not name.startswith('ubunchu'):
            continue
        path = root/'images/ubunchu-ja-test'/name
        assert digest(path.read_bytes()) == page['page']['image_sha256']
        split = 'held_out' if name.startswith('ubunchu03') else 'development'
        page_id = path.stem
        raw = [{k: d[k] for k in ('source_id', 'polygon')} for d in page['batch']['detections']]
        pages.append(dict(id=page_id, path=str(path.relative_to(root)), sha256=digest(path.read_bytes()),
                          split=split, detections=raw))
        units = page['batch']['proposals']
        # Selection is positional and fixed, not selected using model outcomes.
        for i in sorted({0, 1, 2, 3, len(units)-2, len(units)-1}):
            cases.append(dict(id=f'{page_id}-unit-{i:02}', page_id=page_id, split=split,
                              kind='real_no_ground_truth', member_ids=units[i]['member_ids']))
    # Keep exact production membership for the two known unresolved translations.
    path = root/'test-results/workflow-fragments-priority-parent-v1/preview.json'
    preview = json.loads(path.read_text())
    raw = [{k: d[k] for k in ('source_id', 'polygon')} for d in preview['ocr']['raw_detections'][0]]
    pages.append(dict(id='saved_translation_membership', path='images/ubunchu-ja-test/ubunchu01_04.png',
        sha256=preview['input']['sha256'], split='development', detections=raw))
    for ids in ([37, 36], [40, 39, 38]):
        cases.append(dict(id='saved-unresolved-'+'-'.join(map(str, ids)),
            page_id='saved_translation_membership', split='development',
            kind='real_no_ground_truth', member_ids=ids))

    categories = ('white', 'offwhite', 'outline_touch', 'punctuation', 'neighbor',
                  'gradient', 'periodic_tone', 'line_crossing')
    for split, seed in (('development', 71), ('held_out', 193)):
        for index, category in enumerate(categories):
            case_id = f'synthetic-{split}-{category}'
            folder = output/case_id
            folder.mkdir()
            yy, xx = np.mgrid[:384, :384]
            background = np.full((384, 384, 3), 255, np.uint8)
            if category == 'offwhite':
                background[:] = [233, 224, 210]
            elif category == 'gradient':
                base = 135 if split == 'development' else 120
                gray = base+xx*.12+yy*.08
                background = np.stack((gray+15, gray+5, gray), axis=2).astype(np.uint8)
            elif category == 'periodic_tone':
                period = 8 if split == 'development' else 11
                dots = ((xx+seed) % period-period//2)**2+((yy+seed) % period-period//2)**2 <= 3
                background[dots] = 75
            elif category == 'line_crossing':
                plate = Image.fromarray(background)
                draw = ImageDraw.Draw(plate)
                draw.line([(20, 290), (360, 110)], fill=(10, 10, 10), width=3)
                draw.arc((35, 60, 340, 325), 210, 350, fill=(40, 40, 40), width=3)
                background = np.array(plate)
            alpha = Image.new('L', (768, 768))
            draw = ImageDraw.Draw(alpha)
            size = 36 if split == 'development' else 41
            face = ImageFont.truetype(str(font_path), size*2)
            text = 'Wait?!' if category == 'punctuation' else 'TEXT'
            vertical = category == 'offwhite'
            if vertical:
                for j, char in enumerate(text):
                    draw.text((330, 145+j*95), char, font=face, fill=255)
            else:
                draw.text((190, 300), text, font=face, fill=255,
                          stroke_width=3 if category == 'outline_touch' else 0)
            if category == 'line_crossing':
                alpha = alpha.rotate(17 if split == 'development' else -13, Image.Resampling.BICUBIC)
            alpha = alpha.resize((384, 384), Image.Resampling.LANCZOS)
            support = np.asarray(alpha) != 0
            x, y, w, h = cv2.boundingRect(support.astype(np.uint8))
            main_polygon = polygon((x-3, y-3, x+w+2, y+h+2))
            image = Image.composite(Image.new('RGB', (384, 384), 'black'), Image.fromarray(background), alpha)
            detections = [dict(source_id='main', polygon=main_polygon)]
            protected = Image.new('L', image.size)
            if category == 'neighbor':
                ImageDraw.Draw(protected).text((x+w+8, y+6), 'no',
                    font=ImageFont.truetype(str(font_path), 16), fill=255)
                image = Image.composite(Image.new('RGB', image.size, 'black'), image, protected)
                px, py, pw, ph = cv2.boundingRect(np.asarray(protected))
                detections.append(dict(source_id='neighbor', polygon=polygon((px-2, py-2, px+pw+1, py+ph+1))))
            if category == 'outline_touch':
                # A known art line crosses a target stroke, not just a clean collar.
                bx = x+w//2
                art = Image.fromarray(background)
                ImageDraw.Draw(art).line((bx, y-45, bx, y+h+45), fill='black', width=2)
                background = np.asarray(art).copy()
                image = Image.composite(Image.new('RGB', image.size, 'black'), art, alpha)
            clean = Image.fromarray(background)
            if category == 'neighbor':
                clean = Image.composite(Image.new('RGB', image.size, 'black'), clean, protected)
            image.save(folder/'source.png')
            clean.save(folder/'reference.png')
            alpha.save(folder/'target-alpha.png')
            protected.save(folder/'protected.png')
            erase = cv2.dilate(support.astype(np.uint8), np.ones((3, 3), np.uint8))*255
            Image.fromarray(erase).save(folder/'fixed-mask.png')
            artifacts = {p.name: digest(p.read_bytes()) for p in sorted(folder.glob('*.png'))}
            pages.append(dict(id=case_id, path=str((folder/'source.png').relative_to(output)),
                path_base='manifest', sha256=artifacts['source.png'], split=split, detections=detections))
            cases.append(dict(id=case_id, page_id=case_id, split=split, category=category,
                kind='synthetic_known_overlay', member_ids=['main'], orientation='vertical' if vertical else
                'rotated' if category == 'line_crossing' else 'horizontal', artifacts=artifacts))

    manifest = dict(schema_version=1, purpose='frozen_learned_mask_and_fixed_mask_restoration_v1',
        approved_export=False, noncommercial_local_evaluation=True,
        no_translation_or_production_changes=True, source_proposals_sha256=digest(source.read_bytes()),
        saved_preview_sha256=digest(path.read_bytes()), font=font,
        implementation_sha256=digest(Path(__file__).read_bytes()), pages=pages, cases=cases,
        policy=dict(ctd_input_side=1024, ctd_binary_threshold=0.3, ctd_context='full page or synthetic canvas',
            ctd_refinement='pinned upstream INPAINT refinement; supplied source-unit bounding box, not CTD regrouping',
            ctd_no_undetected_recovery=True, mask_polarity='nonzero selects target text / removal',
            restoration_context_margins=[16, 64], model_side=512, restoration_no_resizing=True,
            fixed_mask='synthetic alpha > 0 dilated once with 3x3; real only accepted current automatic plan',
            real_selection='first four and last two saved proposal memberships per page, plus two saved unresolved units',
            held_out='chapter 3; synthetic separate sizes/phase/rotation, no threshold tuning after inference',
            acceptance='no production adoption from this feasibility pass; require separately reviewed held-out art damage and coverage',
            limitations=['one manga title', 'single Latin synthetic font', 'real pages have no pixel ground truth',
                        'chapter 3 was used by older geometry tests; held out only from this learned-model experiment',
                        'real case categories not exhaustively annotated; synthetic categories supply controlled coverage']))
    save_json(output/'manifest.json', manifest)
    frozen_hash = digest((output/'manifest.json').read_bytes())
    (output/'manifest.sha256').write_text(frozen_hash+'\n')
    return manifest, frozen_hash


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fonts', type=Path, required=True)
    args = parser.parse_args()
    manifest, sha = freeze(root=args.root, output=args.output, fonts=args.fonts)
    print(f"Frozen {len(manifest['cases'])} cases; manifest SHA256 {sha}")
