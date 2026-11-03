"""Conservative classical restoration for smooth/nonwhite and mildly textured areas.

Source polygons seed contrast components; they are never painted rectangles.
This is not learned segmentation or recovery of the true hidden artwork. Unknown
detail, dense texture and connected text/art stay unsupported rather than erased.
"""
import cv2
import numpy as np
from PIL import Image

from . import natural as n, preparation as p


POLICY = 'source_bound_classical_inpaint_v1'
MAX_LOCAL_PIXELS = 262144
MAX_LOCAL_SIDE = 1024
MAX_SAMPLES = 65536
MAX_WORK = 512_000_000
RADIUS = 3


class RestorationSkip(ValueError):
    def __init__(self, code, message, evidence=None):
        self.code = code
        self.evidence = {} if evidence is None else evidence
        super().__init__(message)


def check(condition, code, message, evidence=None):
    if not condition:
        raise RestorationSkip(code, message, evidence)


class OriginalGeometry:
    def __init__(self, rgb, detections, geometry=None):
        n.budget_check(rgb.mode == 'RGB' and rgb.width*rgb.height <= p.MAX_PIXELS
            and max(rgb.size) <= p.MAX_SIDE, 'restoration source pixel/side budget')
        n.require(type(detections) is list and 1 <= len(detections) <= 512, 'restoration detection budget')
        keys = [(type(d['source_id']).__name__, d['source_id']) for d in detections]
        n.require(all(type(d['source_id']) in (int, str) and d['source_id'] != '' for d in detections)
            and len(set(keys)) == len(keys), 'typed unique detection IDs required')
        self.revision = n.revision([(key, d['polygon']) for key, d in zip(keys, detections)])
        self.polygons = {key: d['polygon'] for key, d in zip(keys, detections)}
        if geometry is not None:
            geometry.bind_geometry(detections)
            n.require(np.array_equal(np.asarray(rgb), geometry.pixels), 'restoration source drift')
            self.pixels, self.counts = geometry.pixels, geometry.source_counts
        else:
            n.budget_check(len(detections)*rgb.width*rgb.height <= MAX_WORK, 'restoration geometry raster budget')
            self.pixels = np.asarray(rgb).copy()
            self.counts = np.zeros((rgb.height, rgb.width), np.uint16)
            for detection in detections:
                self.counts += np.asarray(n.polygon_mask(detection['polygon'], rgb.size, evidence=True)) != 0


def inpaint(rgb, mask, method):
    """Bounded mask-only primitive. No file paths, model calls or implicit masks."""
    n.require(rgb.mode == 'RGB' and mask.mode == 'L' and rgb.size == mask.size, 'inpaint mode/shape mismatch')
    n.budget_check(rgb.width*rgb.height <= MAX_LOCAL_PIXELS and max(rgb.size) <= MAX_LOCAL_SIDE,
                   'inpaint local pixel/side budget')
    original = np.asarray(rgb)
    values = np.asarray(mask)
    n.require(np.isin(values, (0, 255)).all(), 'inpaint mask must be binary')
    selected = values != 0
    n.require(selected.any() and not selected.all(), 'inpaint needs masked and known pixels')
    n.require(not selected[0].any() and not selected[-1].any()
        and not selected[:, 0].any() and not selected[:, -1].any(), 'inpaint mask touches local frame')
    n.budget_check(int(selected.sum()) <= MAX_LOCAL_PIXELS//3, 'inpaint masked pixel budget')
    n.require(method in ('navier_stokes', 'telea'), 'unknown classical inpaint method')
    flag = cv2.INPAINT_NS if method == 'navier_stokes' else cv2.INPAINT_TELEA
    try:
        restored = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), values, RADIUS, flag), cv2.COLOR_BGR2RGB)
    except cv2.error as error:
        raise RestorationSkip('inpaint_failure', 'OpenCV inpainting failed') from error
    check(restored.shape == original.shape and restored.dtype == np.uint8
        and np.array_equal(restored[~selected], original[~selected]),
        'inpaint_locality', 'inpainting changed pixels outside its mask')
    return Image.fromarray(restored)


def derive_plan(rgb, detections, member_ids, *, context=None):
    from .automatic import AutomaticPlan, source_local_layout

    context = OriginalGeometry(rgb, detections) if context is None else context
    n.require(np.array_equal(np.asarray(rgb), context.pixels), 'restoration source pixels changed')
    n.require(context.revision == n.revision([((type(d['source_id']).__name__, d['source_id']), d['polygon'])
        for d in detections]), 'restoration geometry changed')
    n.require(type(member_ids) is list and 1 <= len(member_ids) <= 32
        and all(type(i) in (int, str) and i != '' for i in member_ids)
        and len(set(member_ids)) == len(member_ids), 'typed unique members required')
    n.budget_check(len(member_ids)*rgb.width*rgb.height <= n.MAX_UNIT_PIXELS, 'restoration member raster budget')
    selected = {(type(i).__name__, i) for i in member_ids}
    n.require(selected <= set(context.polygons), 'foreign restoration member')
    members = np.zeros((rgb.height, rgb.width), bool)
    own_counts = np.zeros_like(context.counts)
    for key in selected:
        mask = np.asarray(n.polygon_mask(context.polygons[key], rgb.size, evidence=True)) != 0
        members |= mask
        own_counts += mask
    nonmembers = context.counts > own_counts
    x, y, w, h = cv2.boundingRect(members.astype(np.uint8))
    check(x >= 8 and y >= 8 and x+w+8 < rgb.width and y+h+8 < rgb.height,
          'restoration_frame', 'background support is clipped by the source frame')
    left, top, right, bottom = max(0, x-40), max(0, y-40), min(rgb.width, x+w+40), min(rgb.height, y+h+40)
    cw, ch = right-left, bottom-top
    n.budget_check(cw*ch <= MAX_LOCAL_PIXELS and max(cw, ch) <= MAX_LOCAL_SIDE, 'restoration local pixel/side budget')
    crop = context.pixels[top:bottom, left:right]
    local_members = members[top:bottom, left:right]
    local_nonmembers = nonmembers[top:bottom, left:right]
    sample_mask = p.dilate(local_members, 6) & ~p.dilate(local_members, 2)
    check(not (sample_mask & local_nonmembers).any(), 'restoration_sample_neighbor', 'background support intersects a nonmember')
    sample_count = int(sample_mask.sum())
    n.budget_check(sample_count <= MAX_SAMPLES, 'restoration sample budget')
    check(sample_count >= 64, 'restoration_samples', 'fewer than 64 original background samples')
    yy, xx = np.mgrid[-1:1:complex(ch), -1:1:complex(cw)]
    basis = np.stack((np.ones_like(xx), xx, yy), axis=2)
    coefficients, _, rank, _ = np.linalg.lstsq(basis[sample_mask], crop[sample_mask].astype(np.float64), rcond=None)
    check(rank == 3, 'restoration_model_rank', 'background samples do not support a two-dimensional model')
    model = basis @ coefficients
    delta = np.max(np.abs(crop.astype(np.float64)-model), axis=2)
    maximum = float(delta[sample_mask].max())
    rmse = float(np.sqrt(np.mean((crop[sample_mask].astype(np.float64)-model[sample_mask])**2)))
    evidence = {'policy': POLICY, 'background_samples': sample_count,
        'model': 'affine RGB from original 2..6px member collar; no sample outliers discarded',
        'model_coefficients': coefficients.tolist(), 'sample_max_deviation': maximum, 'sample_rmse': rmse,
        'local_bounds': [left, top, right, bottom], 'inpaint_radius': RADIUS,
        'restoration_quality': 'unverified; model consistency is not true hidden-artwork recovery',
        'ownership': 'machine source-local hypothesis, not text/art semantic certification'}
    check(maximum <= 12 and rmse <= 6 and float(model.min()) >= -2 and float(model.max()) <= 257
        and float(np.abs(coefficients[1]*2/max(1, cw-1)).max()) <= 2
        and float(np.abs(coefficients[2]*2/max(1, ch-1)).max()) <= 2,
        'restoration_background_complex', 'background is not supported smooth color/gradient or mild texture', evidence)
    floor = max(2, int(np.ceil(max(0, maximum-1e-6)))+2)
    strong = delta > max(48, 4*floor)
    contrast = delta > floor
    count, labels, stats, _ = cv2.connectedComponentsWithStats(contrast.astype(np.uint8), connectivity=8)
    n.budget_check(count <= 8192, 'restoration contrast component budget')
    ids = np.unique(labels[strong & local_members])
    ids = ids[ids != 0]
    check(0 < len(ids) <= 64, 'restoration_foreground', 'no bounded high-contrast foreground support', evidence)
    lookup = np.zeros(count, bool)
    lookup[ids] = True
    foreground = lookup[labels]
    neighborhood = p.dilate(local_members, 2)
    check(not (foreground & ~neighborhood).any(), 'restoration_component_spill', 'whole contrast component leaves source neighborhood', evidence)
    for label in ids:
        fx, fy, fw, fh, area = (int(value) for value in stats[label])
        # Joined letters/counters can have a large mostly-empty bounding box.
        # Audit actual foreground occupancy, not that empty rectangle; still
        # retain the whole component and the same side/absolute-area limits.
        check(fx > 0 and fy > 0 and fx+fw < cw and fy+fh < ch and max(fw, fh) <= 96
            and area <= 20000 and area <= max(64, int(local_members.sum())//2),
            'restoration_component_extent', 'contrast component may be border/art or clipped', evidence)
    check(not (neighborhood & contrast & ~foreground).any(), 'restoration_unassigned_contrast',
          'unassigned original contrast in source neighborhood', evidence)
    erase = p.dilate(foreground, 1)
    collar = p.dilate(erase, 2) & ~erase
    check(not (p.dilate(erase, 2) & local_nonmembers).any(), 'restoration_nonmember',
          'foreground/clearance intersects a nonmember polygon', evidence)
    check(int(collar.sum()) >= 32 and not (collar & contrast).any(), 'restoration_clearance',
          'source foreground has no clear background collar', evidence)
    predicted = np.clip(model, 0, 255)
    attempts = []
    restored = None
    order = ('navier_stokes', 'telea') if rmse <= 2 else ('telea', 'navier_stokes')
    for method in order:
        candidate = inpaint(Image.fromarray(crop), Image.fromarray(erase.astype(np.uint8)*255), method)
        error = np.abs(np.asarray(candidate).astype(np.float64)[erase]-predicted[erase])
        error_max, error_rmse = float(error.max()), float(np.sqrt(np.mean(error**2)))
        accepted = error_max <= max(32, 4*floor) and error_rmse <= max(12, 2*floor)
        attempts.append({'method': method, 'mask_model_max_deviation': error_max,
            'mask_model_rmse': error_rmse, 'model_consistent': accepted})
        if accepted:
            restored = np.asarray(candidate)
            break
    check(restored is not None, 'restoration_model_mismatch', 'classical methods did not produce model-consistent restoration',
          {**evidence, 'attempts': attempts})
    local_obstacles = (contrast & ~foreground) | local_nonmembers
    obstacles = np.ones_like(members)
    obstacles[top:bottom, left:right] = local_obstacles
    full_foreground = np.zeros_like(members)
    full_foreground[top:bottom, left:right] = foreground
    full_erase = np.zeros_like(members)
    full_erase[top:bottom, left:right] = erase
    safe = ~p.dilate(obstacles, 2)
    safe[:2] = safe[-2:] = False
    safe[:, :2] = safe[:, -2:] = False
    mx, my = max(4, min(32, (w+1)//2)), max(4, min(24, (h+9)//10))
    window = [max(2, x-mx), max(2, y-my), min(rgb.width-2, x+w+mx), min(rgb.height-2, y+h+my)]
    layout, bounds, steps, derivation = source_local_layout(safe, p.dilate(full_foreground, 2), window)
    center = np.median(model[local_members], axis=0).clip(0, 255).tolist()
    evidence.update(method=method, attempts=attempts, foreground_contrast_floor=floor,
        foreground_pixels=int(foreground.sum()), erase_pixels=int(erase.sum()),
        component_occupancy_policy='whole foreground pixels <= half member-mask area, with 64px floor',
        background_rgb_at_center=center, layout_bounds=bounds, source_local_window=window,
        layout_expansion_tests=steps, layout_derivation=derivation)
    return AutomaticPlan(full_foreground, full_erase, layout, obstacles, None, evidence,
                         restored_pixels=restored[erase].copy())
