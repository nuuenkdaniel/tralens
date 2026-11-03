"""Evaluation-only morphological stroke repair versus the frozen coarse fallback.

This is deliberately not a production route. Thresholds are fixed here before
running; known reference images are accessed only for scoring after restoration.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tralens.rendering import automatic as a, best_effort as b, natural as n


def trial(image, raw, members):
    plan = b.derive_plan(image, raw, members, background_method='flat')
    left, top, right, bottom = plan.evidence['local_bounds']
    pixels = np.array(image)[top:bottom, left:right]
    allowed = plan.erase[top:bottom, left:right]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    dark = float(np.median(gray[allowed])) < 128
    kernel = np.ones((21, 21), np.uint8)
    residual = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT if dark else cv2.MORPH_BLACKHAT, kernel)
    foreground = (residual > 24) & allowed
    erase = cv2.dilate(foreground.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & allowed
    n.require(int(foreground.sum()) >= 8, 'no stroke contrast')
    color = np.median(pixels[allowed & ~erase], axis=0) if (allowed & ~erase).any() else np.median(pixels[allowed], axis=0)
    candidates = []
    for name, method in [('navier_stokes', cv2.INPAINT_NS), ('telea', cv2.INPAINT_TELEA)]:
        result = cv2.inpaint(pixels, erase.astype(np.uint8)*255, 3, method)
        n.require(np.array_equal(result[~erase], pixels[~erase]), 'backend locality')
        boundary = erase & ~cv2.erode(erase.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        score = float(np.abs(result[boundary].astype(float)-color).mean())
        candidates.append((score, name, result[erase].copy()))
    _, method, restored = min(candidates, key=lambda row: (row[0], row[1]))
    full = np.zeros_like(plan.erase)
    full[top:bottom, left:right] = erase
    fg = np.zeros_like(full)
    fg[top:bottom, left:right] = foreground
    return replace(plan, erase=full, foreground=fg, restored_pixels=restored,
        evidence={**plan.evidence, 'experimental_method': 'morphological_top_black_hat_21_threshold_24',
                  'background_rgb_at_center': color.tolist(), 'backend': method})


def run(frozen, output):
    from tools.render_review_trials import sha, write_json
    content = (frozen/'manifest.json').read_bytes()
    n.require(sha(content) == (frozen/'manifest.sha256').read_text().strip(), 'frozen drift')
    manifest = json.loads(content)
    output.mkdir(mode=0o700)
    rows = []
    for case in manifest['cases']:
        if case['id'].endswith(('-longword', '-overfull')):
            continue  # Same source/clean ground truth as short; do not double count.
        folder = frozen/case['id']
        for name, digest in case['artifacts'].items():
            n.require(sha((folder/name).read_bytes()) == digest, 'raster drift')
        image = Image.open(folder/'source.png').convert('RGB')
        row = dict(id=case['id'], kind=case['kind'], methods={})
        views = [image]
        for method, derive in [('footprint', b.derive_plan), ('strokes', trial)]:
            try:
                plan = derive(image, case['detections'], case['member_ids'])
                pixels = np.array(image)
                a.apply_cleanup(pixels, plan)
                n.require(np.array_equal(pixels[~plan.erase], np.array(image)[~plan.erase]), 'locality')
                details = dict(status='planned', erase_pixels=int(plan.erase.sum()), evidence=plan.evidence)
                if case['clean_ground_truth']:
                    reference = np.array(Image.open(folder/'reference.png').convert('RGB'))
                    target = np.array(Image.open(folder/'target-alpha.png')) != 0
                    visible_target = np.any(np.array(image) != reference, axis=2)
                    error = np.abs(pixels.astype(float)-reference.astype(float))
                    details.update(target_mae=float(error[target].mean()),
                        target_support_coverage=float((plan.erase & visible_target).sum()/max(1, visible_target.sum())),
                        changed_unoccluded_pixels=int((np.any(pixels != reference, axis=2) & ~target).sum()),
                        total_absolute_error=float(error.sum()))
                row['methods'][method] = details
                views.extend([Image.fromarray(plan.erase.astype(np.uint8)*255).convert('RGB'), Image.fromarray(pixels)])
            except ValueError as error:
                row['methods'][method] = dict(status='skipped', reason=str(error))
                views.extend([Image.new('RGB', image.size, '#888888')]*2)
        sheet = Image.new('RGB', (image.width*len(views), image.height))
        for index, view in enumerate(views):
            sheet.paste(view, (index*image.width, 0))
        sheet.save(output/(case['id']+'.png'))
        rows.append(row)
    report = dict(purpose='experimental_morphology_NOT_production', frozen_sha256=sha(content), cases=rows,
        runner_sha256=sha(Path(__file__).read_bytes()), implementation_sha256=a.implementation_hash(),
        columns=['source', 'footprint erase', 'footprint clean', 'stroke erase', 'stroke clean'])
    write_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(**vars(parser.parse_args()))
