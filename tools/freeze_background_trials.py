"""Prospective confirmation fixtures for native-mask/periodic-surface trials.

Only local fonts/pages; no acquisition, inference, or production changes. The
new confirmation overlays are not used for parameter selection. Original manga
pages have already been seen: this is not an unseen-title/page benchmark.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.freeze_learned_evaluation import digest, polygon, save_json


def freeze(root, output, fonts):
    output.mkdir(parents=True, exist_ok=False)
    font = json.loads(fonts.read_text())['fonts'][0]
    assert digest(Path(font['path']).read_bytes()) == font['sha256']
    cases, pages = [], []
    for split, size, period, phase in (('development', 35, 8, 3), ('confirmation', 43, 13, 7)):
        categories = ('white', 'color', 'dark', 'vertical', 'rotated', 'neighbor',
                      'dots', 'stripes', 'rectangular_dots', 'noisy_dots',
                      'crossing_art_negative', 'contained_art_negative')
        for category in categories:
            case_id = split+'-'+category
            folder = output/case_id
            folder.mkdir()
            yy, xx = np.mgrid[:384, :384]
            background = np.full((384, 384, 3), 255, np.uint8)
            if category == 'color':
                background[:] = [215, 185, 145]
            elif category == 'dark':
                background[:] = [30, 50, 75]
            elif category in ('dots', 'rectangular_dots', 'noisy_dots'):
                px, py = period, period+3 if category == 'rectangular_dots' else period
                selected = (((xx+phase) % px-px//2)**2 + ((yy-phase) % py-py//2)**2) <= (3 if split == 'development' else 5)
                background[selected] = 65 if split == 'development' else 90
                if category == 'noisy_dots':
                    noise = np.random.default_rng(833 if split == 'development' else 1979).integers(-1, 2, background.shape[:2])
                    background = np.clip(background.astype(np.int16)+noise[..., None], 0, 255).astype(np.uint8)
            elif category == 'stripes':
                background[(xx+phase) % period < 2] = 105
            alpha_hi = Image.new('L', (768, 768))
            draw = ImageDraw.Draw(alpha_hi)
            face = ImageFont.truetype(font['path'], size*2)
            if category == 'vertical':
                for i, char in enumerate('WAIT'):
                    draw.text((330, 150+95*i), char, font=face, fill=255)
            else:
                draw.text((185, 300), 'Wait?!' if category == 'neighbor' else 'TEXT', font=face, fill=255)
            if category == 'rotated':
                alpha_hi = alpha_hi.rotate(-11 if split == 'development' else 19, Image.Resampling.BICUBIC)
            alpha = alpha_hi.resize((384, 384), Image.Resampling.LANCZOS)
            selected = np.array(alpha) != 0
            x, y, w, h = cv2.boundingRect(selected.astype(np.uint8))
            protected = Image.new('L', (384, 384))
            if category in ('crossing_art_negative', 'contained_art_negative'):
                # Draw an independently known art stroke AFTER locating the text,
                # ensuring the negative truly crosses/occupies the source area.
                pen = ImageDraw.Draw(protected)
                if category == 'crossing_art_negative':
                    pen.line((x+w//2, y-40, x+w//2, y+h+40), fill=255, width=3)
                else:
                    pen.ellipse((x+w//2-5, y+4, x+w//2+5, y+h-4), outline=255, width=2)
                background[np.array(protected) != 0] = 80
            clean = Image.fromarray(background)
            foreground = Image.new('RGB', (384, 384), 'white' if category == 'dark' else 'black')
            image = Image.composite(foreground, clean, alpha)
            detections = [dict(source_id='main', polygon=polygon((x-3, y-3, x+w+2, y+h+2)))]
            if category == 'neighbor':
                neighbor = Image.new('L', (384, 384))
                ImageDraw.Draw(neighbor).text((x+w+8, y+4), 'no', font=ImageFont.truetype(font['path'], 17), fill=255)
                clean = Image.composite(Image.new('RGB', clean.size, 'black'), clean, neighbor)
                image = Image.composite(Image.new('RGB', image.size, 'black'), image, neighbor)
                protected = neighbor
                nx, ny, nw, nh = cv2.boundingRect(np.array(neighbor))
                detections.append(dict(source_id='other', polygon=polygon((nx-2, ny-2, nx+nw+1, ny+nh+1))))
            for name, value in (('source.png', image), ('reference.png', clean), ('target-alpha.png', alpha),
                                 ('protected.png', protected)):
                value.save(folder/name)
            artifacts = {p.name: digest(p.read_bytes()) for p in folder.glob('*.png')}
            pages.append(dict(id=case_id, path=case_id+'/source.png', path_base='manifest',
                sha256=artifacts['source.png'], detections=detections, split=split))
            cases.append(dict(id=case_id, page_id=case_id, member_ids=['main'], category=category,
                split=split, kind='synthetic_known_overlay', artifacts=artifacts,
                expected='abstain_or_preserve_art' if category.endswith('_negative') else 'replace_if_supported'))
    manifest = dict(schema_version=1, pages=pages, cases=cases, font=font,
        purpose='prospective_native_association_and_periodic_trials', implementation_sha256=digest(Path(__file__).read_bytes()),
        policy=dict(ctd_variants=['fit_1024', 'native_pad_1024'], ctd_threshold=.3,
            source_fraction=.75, semantic_fraction=.5, faint_area=64, faint_delta=16, faint_distance=3,
            period_limit=32, periodic_max_known_error=2, periodic_known_mae=.25,
            min_phase_samples=8, context_margin_flat=32, context_margin_periodic=64,
            comparison='current production mask vs automatic native association; periodic surface vs LaMa on identical selected masks',
            confirmation='new overlays/surfaces with parameters fixed using development only; not unseen manga pages',
            no_reference_input=True, no_production_adoption_without_review=True))
    save_json(output/'manifest.json', manifest)
    (output/'manifest.sha256').write_text(digest((output/'manifest.json').read_bytes())+'\n')
    (output/'freeze-source.py').write_bytes(Path(__file__).read_bytes())
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fonts', type=Path, required=True)
    args = parser.parse_args()
    m = freeze(args.root, args.output, args.fonts)
    print(f"Frozen {len(m['cases'])} prospective cases in {args.output}")
