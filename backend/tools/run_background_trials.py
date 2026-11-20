"""Bounded local trials; source hashes, automatic decisions, no reference inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from tools import background_trials as trial
from tools.learned_restoration_experiment import CTD_SHA, document, load_inputs, mask_copy, overlay, sha
from tralens.rendering import automatic as a, restoration as r

RUNNER_SOURCE = Path(__file__).read_bytes()
TRIAL_SOURCE = Path(trial.derive.__code__.co_filename).read_bytes()


def predict(net, source, variant):
    h, w = source.shape[:2]
    if variant == 'fit_1024':
        scale = min(1024/h, 1024/w)
        rw, rh = round(w*scale), round(h*scale)
        resized = cv2.resize(source, (rw, rh), interpolation=cv2.INTER_LINEAR)
    else:
        if max(w, h) > 1024:
            raise ValueError('native-pad CTD context exceeds 1024px')
        rw, rh = w, h
        resized = source
    padded = cv2.copyMakeBorder(resized, 0, 1024-rh, 0, 1024-rw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    net.setInput(cv2.dnn.blobFromImage(padded, scalefactor=1/255., size=(1024, 1024)))
    values = net.forward(net.getUnconnectedOutLayersNames())
    masks = [v for v in values if v.ndim == 4 and v.shape[1] == 1]
    if len(masks) != 1:
        raise ValueError('ambiguous CTD mask output')
    probability = masks[0].squeeze()
    if probability.shape != (1024, 1024) or not np.isfinite(probability).all():
        raise ValueError('invalid CTD probability shape/range')
    raw = (probability[:rh, :rw].clip(0, 1)*255).astype(np.uint8)
    if (rw, rh) != (w, h):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
    return raw, dict(variant=variant, native_size=[w, h], resized_size=[rw, rh],
                     padding_right_bottom=[1024-rw, 1024-rh], input_color='RGB', input_range=[0, 1])


def run(frozen, runtime, output, split, variants):
    manifest = load_inputs(ROOT.parent, frozen)
    if sha(runtime/'models/comictextdetector.pt.onnx') != CTD_SHA:
        raise ValueError('CTD model drift')
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    net = cv2.dnn.readNetFromONNX(str(runtime/'models/comictextdetector.pt.onnx'))
    rows = []
    for case in manifest['cases']:
        if case['split'] != split:
            continue
        page = next(p for p in manifest['pages'] if p['id'] == case['page_id'])
        image = Image.open(page['_path']).convert('RGB')
        source = np.array(image)
        # References are used ONLY AFTER each automatic plan has been derived.
        baseline = None
        for method, derive in (('flat', a.derive_plan), ('classical', r.derive_plan)):
            try:
                baseline = derive(image, page['detections'], case['member_ids'])
                break
            except ValueError:
                continue
        for variant in variants:
            folder = output/(case['id']+'-'+variant)
            folder.mkdir()
            start = time.perf_counter()
            raw, transform = predict(net, source, variant)
            inference_seconds = time.perf_counter()-start
            Image.fromarray(raw).save(folder/'raw.png')
            row = dict(id=case['id'], split=split, category=case.get('category'), expected=case.get('expected'),
                variant=variant, current_plan=baseline is not None, candidates={}, transform=transform,
                inference_seconds=inference_seconds)
            views = [image]
            for surface in ('flat', 'periodic'):
                try:
                    start = time.perf_counter()
                    plan = trial.derive(image, page['detections'], case['member_ids'], raw, surface=surface)
                    elapsed = time.perf_counter()-start
                    clean = source.copy()
                    a.apply_cleanup(clean, plan)
                    Image.fromarray(clean).save(folder/(surface+'-clean.png'))
                    Image.fromarray(plan.erase.astype(np.uint8)*255).save(folder/(surface+'-erase.png'))
                    Image.fromarray(plan.layout.astype(np.uint8)*255).save(folder/(surface+'-layout.png'))
                    details = dict(status='planned', seconds=elapsed, evidence=plan.evidence,
                        changed_pixels=int(np.any(clean != source, axis=2).sum()),
                        outside_mask_changed_pixels=int(np.any(clean[~plan.erase] != source[~plan.erase], axis=1).sum()))
                    if case['kind'] == 'synthetic_known_overlay':
                        reference = np.array(Image.open(frozen/case['id']/'reference.png').convert('RGB'))
                        protected = np.array(Image.open(frozen/case['id']/'protected.png')) != 0
                        target = np.array(Image.open(frozen/case['id']/'target-alpha.png')) != 0
                        error = np.abs(clean.astype(float)-reference.astype(float))
                        changed_reference = np.any(source != reference, axis=2)
                        details.update(reference_changed_pixels_before=int(changed_reference.sum()),
                            reference_changed_pixels_after=int(np.any(clean != reference, axis=2).sum()),
                            target_mae=float(error[target].mean()),
                            protected_reference_error_pixels=int((np.any(clean != reference, axis=2) & protected).sum()),
                            protected_unoccluded_changed_pixels=int((np.any(clean != source, axis=2) & protected & ~target).sum()),
                            exact_reconstruction=bool(np.array_equal(clean, reference)))
                    row['candidates'][surface] = details
                    views.extend([overlay(source, plan.erase), Image.fromarray(clean)])
                except ValueError as error:
                    row['candidates'][surface] = dict(status='skipped', reason_code=getattr(error, 'code', 'validation'),
                        reason=str(error), evidence=getattr(error, 'evidence', {}))
                    views.extend([Image.new('RGB', image.size, '#bbbbbb')]*2)
            if case['kind'] == 'synthetic_known_overlay':
                views.append(Image.open(frozen/case['id']/'reference.png').convert('RGB'))
            sheet = Image.new('RGB', (image.width*len(views), image.height))
            for i, view in enumerate(views):
                sheet.paste(view, (i*image.width, 0))
            sheet.save(folder/'source-flat-periodic-reference.png')
            row['artifacts'] = {p.name: sha(p) for p in folder.glob('*.png')}
            rows.append(row)
            print(case['id'], variant, {k: (v['status'], v.get('reason_code'), v.get('target_mae')) for k, v in row['candidates'].items()}, flush=True)
    report = dict(purpose='native_source_association_and_phase_locked_trials', frozen_manifest_sha256=sha(frozen/'manifest.json'),
        runner_sha256=hashlib.sha256(RUNNER_SOURCE).hexdigest(),
        trial_sha256=hashlib.sha256(TRIAL_SOURCE).hexdigest(), ctd_sha256=CTD_SHA,
        split=split, variants=variants, cases=rows, production_routing_changed=False,
        contact_sheet_columns=['source', 'flat erase', 'flat clean', 'periodic erase', 'periodic clean', 'reference'])
    document(output/'report.json', report)
    (output/'runner-source.py').write_bytes(RUNNER_SOURCE)
    (output/'trial-source.py').write_bytes(TRIAL_SOURCE)
    return report


def render_saved(preview, image, fonts, raw_path, output, expected_raw_sha, expected_source_sha):
    """Use the genuine automatic binding/layout/audit path, but trial masks only.

    The temporary Python hook is confined to this experiment process; the
    production function/file is not edited. Its provenance is explicitly added
    to the candidate report and review, never presented as the default policy.
    """
    if sha(raw_path) != expected_raw_sha:
        raise ValueError('cached CTD output hash drift')
    if sha(image) != expected_source_sha:
        raise ValueError('cached CTD source image binding drift')
    raw = np.array(Image.open(raw_path).convert('L'))
    original = a.derive_plan
    decisions = []

    def derive(rgb, detections, member_ids, *, context=None):
        try:
            return original(rgb, detections, member_ids, context=context)
        except ValueError as error:
            plan = trial.derive(rgb, detections, member_ids, raw, surface='flat')
            decisions.append(dict(member_ids=member_ids, original_reason_code=getattr(error, 'code', 'validation'),
                                  evidence=plan.evidence))
            return plan

    with patch.object(a, 'derive_plan', derive):
        report = a.run_automatic(preview=preview, image=image, fonts=fonts, output=output, render_policy='conservative')
    report['experimental_mask_trial'] = dict(policy=trial.POLICY, trial_sha256=hashlib.sha256(TRIAL_SOURCE).hexdigest(),
        runner_sha256=hashlib.sha256(RUNNER_SOURCE).hexdigest(), ctd_sha256=CTD_SHA, raw_path=str(raw_path), raw_sha256=sha(raw_path),
        raw_source_sha256=expected_source_sha,
        decisions=decisions, no_default_routing_change=True,
        warning='local trial candidate; semantic masks and artistic/linguistic quality remain unverified')
    # Keep report/image provenance honest rather than silently changing the
    # original source hashes, strict bindings, typography or collision checks.
    report['policy'] = 'experimental_flat_native_semantic_association_v1'
    document(output/'manifest.json', report)
    (output/'review.html').write_bytes(a._report_html(report))
    # review.html is a reported artifact; refresh its recorded hash.
    report['artifacts']['review.html'] = sha(output/'review.html')
    document(output/'manifest.json', report)
    (output/'trial-source.py').write_bytes(TRIAL_SOURCE)
    (output/'runner-source.py').write_bytes(RUNNER_SOURCE)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'confirmation'), required=True)
    parser.add_argument('--variant', choices=('both', 'fit_1024', 'native_pad_1024'), default='both')
    args = parser.parse_args()
    variants = ['fit_1024', 'native_pad_1024'] if args.variant == 'both' else [args.variant]
    run(args.frozen, args.runtime, args.output, args.split, variants)
