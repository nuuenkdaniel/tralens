"""Local-only CTD / fixed-mask LaMa feasibility experiment, never production routing.

Requires an already frozen manifest and explicitly acquired ONNX files/source.
No downloads, translation, source edits, auto-installation, or model conversion.
The upstream CTD refinement runs with frozen source membership, NOT newly grouped
detector boxes. Candidate masks do not constitute validated erasure permission.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from tralens.rendering import automatic as a, natural as n, restoration as r


CTD_SHA = '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'
LAMA_SHA = '1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6'
SOURCE_SHA = {
    'utils/textmask.py': 'f233adecc2f478b19ef6aa374ec45b63e9fe534b40086687346b1c2279c21088',
    'utils/textblock.py': 'a15041d2803aa654dcf1ba1dae3b0c91c902f39676d3a3b31b7e28b94dc511f3',
    'utils/imgproc_utils.py': 'a6940ea705d38830cd24114b924146d2cb1790663fdd42c2af26b2d6e8e95ae0',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False)+'\n')


def binary_metrics(candidate, reference):
    candidate, reference = np.asarray(candidate, bool), np.asarray(reference, bool)
    assert candidate.shape == reference.shape
    tp = int((candidate & reference).sum())
    fp = int((candidate & ~reference).sum())
    fn = int((~candidate & reference).sum())
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                iou=tp/(tp+fp+fn) if tp+fp+fn else None)


def native_raw(probability, native_size, resized_size):
    """Pinned CTD order: quantize, remove right/bottom pad, linear native resize."""
    width, height = native_size
    rw, rh = resized_size
    assert probability.shape == (1024, 1024)
    assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
    assert 0 < rw <= 1024 and 0 < rh <= 1024 and width > 0 and height > 0
    return cv2.resize((probability*255).astype(np.uint8)[:rh, :rw],
                      (width, height), interpolation=cv2.INTER_LINEAR)


def mask_copy(original, candidate, mask):
    assert original.shape == candidate.shape and original.dtype == candidate.dtype == np.uint8
    assert original.shape[:2] == mask.shape and mask.dtype == bool
    result = original.copy()
    result[mask] = candidate[mask]
    assert np.array_equal(result[~mask], original[~mask])
    return result


def pad_lama_context(crop, selected):
    """Reflect missing support with the image; never expose mirrored target ink.

    With large 512px padding, reflecting the image but zero-padding its mask
    creates known copies of the target letters. Padding both identically commutes
    with masking, matching the converter's image-and-mask padding semantics.
    """
    assert crop.dtype == np.uint8 and selected.dtype == bool
    assert crop.shape[:2] == selected.shape and crop.shape[2] == 3
    height, width = selected.shape
    assert 0 < min(height, width) and max(height, width) <= 512
    padded = np.pad(crop, ((0, 512-height), (0, 512-width), (0, 0)), mode='symmetric')
    mask = np.pad(selected, ((0, 512-height), (0, 512-width)), mode='symmetric')
    masked_crop = crop.copy()
    masked_crop[selected] = 0
    expected = np.pad(masked_crop, ((0, 512-height), (0, 512-width), (0, 0)), mode='symmetric')
    actual = padded.copy()
    actual[mask] = 0
    assert np.array_equal(actual, expected), 'padding exposed hidden target pixels'
    return padded, mask


def unit_bounds(page, case):
    points = np.array([d['polygon'] for d in page['detections']
                       if d['source_id'] in case['member_ids']]).reshape(-1, 2)
    return [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()]


def crop_bounds(bounds, size, margin):
    left, top, right, bottom = bounds
    width, height = size
    return [max(0, left-margin), max(0, top-margin),
            min(width, right+margin+1), min(height, bottom+margin+1)]


def metrics_against_reference(result, reference, selected):
    error = result.astype(float)[selected]-reference.astype(float)[selected]
    mse = float(np.mean(error**2))
    dark = (reference.mean(axis=2) < 96) & selected
    return dict(masked_mae=float(np.abs(error).mean()), masked_rmse=float(np.sqrt(mse)),
        masked_max_error=float(np.abs(error).max()),
        masked_psnr_db=10*np.log10(255**2/mse) if mse else None,
        exact_reconstruction=not bool(error.any()),
        dark_reference_pixels=int(dark.sum()),
        dark_reference_recall_at_96=float((result.mean(axis=2)[dark] < 96).mean()) if dark.any() else None)


def overlay(source, mask):
    result = source.copy()
    result[mask] = (result[mask].astype(float)*.4+[153, 0, 0]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)


def load_inputs(root, frozen):
    raw = (frozen/'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (frozen/'manifest.sha256').read_text().strip(), 'manifest drift'
    manifest = json.loads(raw)
    for page in manifest['pages']:
        base = frozen if page.get('path_base') == 'manifest' else root
        path = base/page['path']
        assert sha(path) == page['sha256'], 'source image drift'
        page['_path'] = path
    for case in manifest['cases']:
        for name, expected in case.get('artifacts', {}).items():
            assert sha(frozen/case['id']/name) == expected, 'synthetic fixture drift'
    return manifest


def run_masks(root, frozen, runtime, output):
    manifest = load_inputs(root, frozen)
    assert sha(runtime/'models/comictextdetector.pt.onnx') == CTD_SHA
    for name, expected in SOURCE_SHA.items():
        assert sha(runtime/'ctd-source'/name) == expected, 'upstream source drift'
    sys.path.insert(0, str(runtime/'ctd-source'))
    from utils.imgproc_utils import letterbox, expand_textwindow
    from utils.textblock import TextBlock
    from utils.textmask import refine_mask, REFINEMASK_INPAINT

    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    started = time.perf_counter()
    net = cv2.dnn.readNetFromONNX(str(runtime/'models/comictextdetector.pt.onnx'))
    load_seconds = time.perf_counter()-started
    page_reports, rows, cache = [], [], {}
    for page in manifest['pages']:
        with Image.open(page['_path']) as opened:
            image = opened.convert('RGB')
        source = np.array(image)
        h, w = source.shape[:2]
        if page['sha256'] not in cache:
            resized, ratio, (dw, dh) = letterbox(source, new_shape=(1024, 1024), auto=False, stride=64)
            net.setInput(cv2.dnn.blobFromImage(resized, scalefactor=1/255., size=(1024, 1024)))
            start = time.perf_counter()
            values = net.forward(net.getUnconnectedOutLayersNames())
            elapsed = time.perf_counter()-start
            segmentation = [value for value in values if value.ndim == 4 and value.shape[1] == 1]
            assert len(segmentation) == 1, 'ambiguous model output'
            raw = native_raw(segmentation[0].squeeze(), (w, h), (1024-dw, 1024-dh))
            # No warmup is hidden: first unique page is process-cold inference.
            cache[page['sha256']] = (raw, dict(native_size=[w, h], resized_size=[1024-dw, 1024-dh],
                padding_right_bottom=[dw, dh], forward_seconds=elapsed,
                output_shapes=[list(v.shape) for v in values]))
        raw, timing = cache[page['sha256']]
        page_dir = output/page['id']
        page_dir.mkdir()
        Image.fromarray(raw).save(page_dir/'raw-grayscale.png')
        page_reports.append(dict(id=page['id'], source_sha256=page['sha256'], **timing))
        flat_context, restore_context = None, None
        flat_context_error, restore_context_error = None, None
        try:
            flat_context = a.OriginalComponents(image)
            flat_context.bind_geometry(page['detections'])
        except (ValueError, a.AutomaticSkip) as error:
            flat_context_error = str(error)
        try:
            restore_context = r.OriginalGeometry(image, page['detections'], geometry=flat_context)
        except ValueError as error:
            restore_context_error = str(error)
        for case in [c for c in manifest['cases'] if c['page_id'] == page['id']]:
            folder = output/case['id']
            folder.mkdir(exist_ok=True)
            bounds = unit_bounds(page, case)
            x1, y1, x2, y2 = expand_textwindow(source.shape, bounds, expand_r=16)
            selected_raw = np.zeros((h, w), bool)
            selected_raw[y1:y2, x1:x2] = raw[y1:y2, x1:x2]/255. > manifest['policy']['ctd_binary_threshold']
            start = time.perf_counter()
            refined = refine_mask(cv2.cvtColor(source, cv2.COLOR_RGB2BGR), raw.copy(),
                                  [TextBlock(bounds)], refine_mode=REFINEMASK_INPAINT) != 0
            row = dict(id=case['id'], kind=case['kind'], split=case['split'], category=case.get('category'),
                       bounds=bounds, refinement_seconds=time.perf_counter()-start, current_attempts=[], candidates={})
            plan = None
            for method, context, context_error, derive in (
                    ('flat', flat_context, flat_context_error, a.derive_plan),
                    ('classical', restore_context, restore_context_error, r.derive_plan)):
                try:
                    if context_error:
                        raise ValueError(context_error)
                    plan = derive(image, page['detections'], case['member_ids'], context=context)
                    row['current_attempts'].append(dict(method=method, status='planned'))
                    break
                except (ValueError, a.AutomaticSkip, r.RestorationSkip) as error:
                    row['current_attempts'].append(dict(method=method, status='skipped',
                        reason_code=getattr(error, 'code', 'validation'), reason=str(error)))
            masks = {'ctd_raw': selected_raw, 'ctd_refined_source_bound': refined}
            if plan is not None:
                masks['current_foreground'] = plan.foreground
                masks['current_erase'] = plan.erase
                row['current_plan'] = plan.evidence
            nonmember = Image.new('L', image.size)
            draw = ImageDraw.Draw(nonmember)
            for detection in page['detections']:
                if detection['source_id'] not in case['member_ids']:
                    draw.polygon([tuple(p) for p in detection['polygon']], fill=255)
            nonmember = np.asarray(nonmember) != 0
            truth, support, protected = None, None, None
            if case['kind'] == 'synthetic_known_overlay':
                truth = np.asarray(Image.open(frozen/case['id']/'target-alpha.png')) != 0
                support = np.asarray(Image.open(frozen/case['id']/'fixed-mask.png')) != 0
                protected = np.asarray(Image.open(frozen/case['id']/'protected.png')) != 0
            for name, mask in masks.items():
                Image.fromarray(mask.astype(np.uint8)*255).save(folder/(name+'.png'))
                details = dict(pixels=int(mask.sum()), nonmember_polygon_pixels=int((mask & nonmember).sum()),
                               mask_semantics='ink' if name == 'current_foreground' else 'removal candidate')
                if truth is not None:
                    details.update(ink_metrics=binary_metrics(mask, truth),
                        support_metrics=binary_metrics(mask, support),
                        protected_neighbor_ink_pixels=int((mask & protected).sum()))
                row['candidates'][name] = details
            crop = crop_bounds(bounds, image.size, 28)
            views = [image.crop(crop)] + [overlay(source, masks[name]).crop(crop) if name in masks
                else Image.new('RGB', (crop[2]-crop[0], crop[3]-crop[1]), '#bbbbbb')
                for name in ('current_erase', 'ctd_raw', 'ctd_refined_source_bound')]
            sheet = Image.new('RGB', (views[0].width*4, views[0].height))
            for i, view in enumerate(views):
                sheet.paste(view, (i*view.width, 0))
            sheet.save(folder/'source-current-raw-refined.png')
            row['artifacts'] = {p.name: sha(p) for p in sorted(folder.glob('*.png'))}
            rows.append(row)
            print('mask', case['id'], {k: v['nonmember_polygon_pixels'] for k, v in row['candidates'].items()}, flush=True)
    report = dict(purpose='mask_only_feasibility_not_erase_authorization',
        frozen_manifest_sha256=sha(frozen/'manifest.json'), implementation_sha256=sha(__file__),
        current_implementation_sha256=a.implementation_hash(),
        ctd_model_sha256=CTD_SHA, ctd_expected_digest_trust='downloaded-byte identity, not publisher authentication',
        ctd_source_revision='440b978563c71b758e31aaa315d100faba1efa2f',
        load_seconds=load_seconds, opencv=cv2.__version__, pages=page_reports, cases=rows,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        summary=dict(cases=len(rows), current_plans=sum('current_erase' in row['candidates'] for row in rows),
            real_cases=sum(row['kind'] == 'real_no_ground_truth' for row in rows)),
        limitations=manifest['policy']['limitations'],
        comparison_columns=['source', 'current erase (gray if skipped)', 'CTD raw threshold 0.3', 'CTD source-bound INPAINT refinement'],
        raw_preprocessing='RGB NCHW /255; INTER_LINEAR letterbox with black bottom/right padding; quantize then unpad and linear native resize',
        refinement='upstream refinement uses one frozen unit union bounding box; no CTD box regrouping or undetected recovery',
        no_production_adoption=True)
    document(output/'report.json', report)
    return report


def run_restoration(root, frozen, runtime, masks, output):
    import onnxruntime as ort
    manifest = load_inputs(root, frozen)
    mask_report = json.loads((masks/'report.json').read_text())
    assert mask_report['frozen_manifest_sha256'] == sha(frozen/'manifest.json')
    assert sha(runtime/'models/lama_fp32.onnx') == LAMA_SHA
    output.mkdir(parents=True, exist_ok=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    cv2.setNumThreads(4)
    start = time.perf_counter()
    session = ort.InferenceSession(str(runtime/'models/lama_fp32.onnx'), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    load_seconds = time.perf_counter()-start
    inputs = {value.name: dict(shape=value.shape, type=value.type) for value in session.get_inputs()}
    assert set(inputs) == {'image', 'mask'}
    assert inputs['image']['shape'][1:] == [3, 512, 512]
    assert inputs['mask']['shape'][1:] == [1, 512, 512]
    rows, skipped = [], []
    for case in manifest['cases']:
        page = next(p for p in manifest['pages'] if p['id'] == case['page_id'])
        image = Image.open(page['_path']).convert('RGB')
        source = np.array(image)
        reference = None
        if case['kind'] == 'synthetic_known_overlay':
            fixed_path = frozen/case['id']/'fixed-mask.png'
            reference = np.array(Image.open(frozen/case['id']/'reference.png').convert('RGB'))
            mask_origin = 'known synthetic overlay; not automatic learned-mask acceptance'
        else:
            fixed_path = masks/case['id']/'current_erase.png'
            if not fixed_path.exists():
                skipped.append(dict(id=case['id'], reason='no accepted current mask; CTD candidates do not authorize erasure'))
                continue
            expected = next(row for row in mask_report['cases'] if row['id'] == case['id'])['artifacts']['current_erase.png']
            assert sha(fixed_path) == expected
            mask_origin = 'current automatic plan only; real mask remains a hypothesis'
        fixed = np.array(Image.open(fixed_path)) != 0
        assert fixed.any()
        x, y, width, height = cv2.boundingRect(fixed.astype(np.uint8))
        for margin in manifest['policy']['restoration_context_margins']:
            bounds = crop_bounds([x, y, x+width-1, y+height-1], image.size, margin)
            left, top, right, bottom = bounds
            crop, selected = source[top:bottom, left:right], fixed[top:bottom, left:right]
            ch, cw = selected.shape
            if max(ch, cw) > 512 or selected[0].any() or selected[-1].any() or selected[:, 0].any() or selected[:, -1].any():
                skipped.append(dict(id=case['id'], margin=margin, reason='native context does not fit 512px or lacks clearance'))
                continue
            folder = output/(case['id']+f'-context-{margin}')
            folder.mkdir()
            Image.fromarray(crop).save(folder/'source.png')
            Image.fromarray(selected.astype(np.uint8)*255).save(folder/'fixed-mask.png')
            row = dict(id=case['id'], split=case['split'], category=case.get('category'), kind=case['kind'],
                       margin=margin, bounds=bounds, fixed_mask_sha256=sha(fixed_path), mask_origin=mask_origin,
                       transform=dict(native_crop_size=[cw, ch], pad_right_bottom=[512-cw, 512-ch],
                                      image_padding='numpy symmetric', mask_padding='numpy symmetric, including reflected target holes', scale=1), candidates={})
            results = {}
            for method in ('flat_ring_median', 'telea', 'navier_stokes', 'lama'):
                start = time.perf_counter()
                try:
                    if method == 'lama':
                        padded, mask = pad_lama_context(crop, selected)
                        values = session.run(None, {'image': np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)/255.,
                                                   'mask': mask[None, None].astype(np.float32)})[0]
                        assert values.shape == (1, 3, 512, 512) and np.isfinite(values).all()
                        # Match converter example: output is 0..255, truncate uint8.
                        candidate = np.clip(values[0, :, :ch, :cw].transpose(1, 2, 0), 0, 255).astype(np.uint8)
                    elif method == 'flat_ring_median':
                        collar = cv2.dilate(selected.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
                        collar &= ~selected
                        assert collar.any()
                        candidate = np.full_like(crop, np.median(crop[collar], axis=0).astype(np.uint8))
                    else:
                        candidate = np.array(r.inpaint(Image.fromarray(crop), Image.fromarray(selected.astype(np.uint8)*255), method))
                    elapsed = time.perf_counter()-start
                    raw_outside_changes = int(np.any(candidate[~selected] != crop[~selected], axis=1).sum())
                    result = mask_copy(crop, candidate, selected)
                    Image.fromarray(result).save(folder/(method+'.png'))
                    details = dict(status='completed', seconds=elapsed, raw_outside_changed_pixels=raw_outside_changes,
                        composited_outside_changed_pixels=int(np.any(result[~selected] != crop[~selected], axis=1).sum()))
                    if reference is not None:
                        details.update(metrics_against_reference(result, reference[top:bottom, left:right], selected))
                    row['candidates'][method] = details
                    results[method] = Image.fromarray(result)
                except (ValueError, r.RestorationSkip) as error:
                    row['candidates'][method] = dict(status='skipped', seconds=time.perf_counter()-start,
                        reason_code=getattr(error, 'code', 'validation'), reason=str(error))
            views = [Image.fromarray(crop)] + [results.get(method, Image.new('RGB', (cw, ch), '#bbbbbb'))
                     for method in ('flat_ring_median', 'telea', 'navier_stokes', 'lama')]
            if reference is not None:
                views.append(Image.fromarray(reference[top:bottom, left:right]))
            sheet = Image.new('RGB', (cw*len(views), ch))
            for i, view in enumerate(views):
                sheet.paste(view, (i*cw, 0))
            sheet.save(folder/'comparison.png')
            row['artifacts'] = {p.name: sha(p) for p in sorted(folder.glob('*.png'))}
            rows.append(row)
            document(output/'progress.json', dict(cases=rows, skipped=skipped))
            print('restoration', row['id'], margin,
                {k: round(v['masked_mae'], 3) for k, v in row['candidates'].items() if 'masked_mae' in v}, flush=True)
    report = dict(purpose='fixed_mask_restoration_feasibility', frozen_manifest_sha256=sha(frozen/'manifest.json'),
        mask_report_sha256=sha(masks/'report.json'), implementation_sha256=sha(__file__),
        model_sha256=LAMA_SHA, model_card_revision='c3c0c9e468934d62e79c329e35d82dd09ff8c444',
        converter_revision='f5fb39a18022c34a71bf9a47a6ec393c804b49ca',
        providers=session.get_providers(), inputs=inputs, load_seconds=load_seconds,
        packages={name: importlib.metadata.version(name) for name in ('onnxruntime', 'numpy', 'Pillow', 'opencv-python-headless')},
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cases=rows, skipped=skipped, no_production_adoption=True,
        preprocessing='float32 RGB NCHW 0..1; mask float32 N1HW 1=remove; 512px matched symmetric image and mask padding; no resize; output 0..255 truncated uint8',
        correction='v1/v2 zero-mask padding exposed mirrored source letters; invalidated, not included in quality conclusions',
        comparison_columns=['source', 'ring-median fill', 'Telea', 'Navier-Stokes', 'LaMa', 'known reference (synthetic only)'],
        limitations=manifest['policy']['limitations']+['fixed-mask comparison is not end-to-end automatic replacement coverage',
            'export already blends unmasked source; raw difference is diagnostic, masked compositor is the equality invariant'])
    document(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('masks', 'restoration'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--masks', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'masks':
        report = run_masks(args.root, args.frozen, args.runtime, args.output)
    else:
        if args.masks is None:
            parser.error('restoration requires --masks')
        report = run_restoration(args.root, args.frozen, args.runtime, args.masks, args.output)
    print(f"Saved {len(report['cases'])} cases to {args.output/'report.json'}")
