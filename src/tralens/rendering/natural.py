"""Deterministic, candidate-only flat-balloon rendering. No engine/provider imports.

All file inputs are explicit. JSON geometry is inline, never a sidecar path.
Coordinates address source pixels (polygon vertices inclusive); no color management
is performed. The RGB fingerprint is distinct from the original file hash.
"""
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import html
import json
import math
from pathlib import Path
import re
import struct
import unicodedata

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, features, __version__ as pillow_version

from ..translation.contracts import (TranslationRequest, TranslationResult, UnitOutcome,
                                     automatic_units, canonical, revision, strict_json)
from .styles import PRESETS, face_metadata, resolve_style


MAX_PAGE_PIXELS = 16_000_000
MAX_LOCAL_PIXELS = 4_000_000
MAX_LOCAL_SIDE = 8192
MAX_COVERAGE_POINTS = 65536
MAX_FONT_BYTES = 64 * 1024 * 1024
MAX_UNIT_PIXELS = 32_000_000


class ResourceLimit(ValueError):
    """A bounded operation was refused before allocating its large result."""


class CleanupBlocked(ValueError):
    """Observed cleanup constraint failure, with bounded source-pixel evidence."""
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(diagnostics['message'])


def budget_check(condition, message):
    if not condition:
        raise ResourceLimit(message)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_explicit(path, limit=64 * 1024 * 1024):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink input path")
    require(path.is_file(), "input must be a regular file")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    require(len(content) <= limit, "input byte budget exceeded")
    return content


def bounded(value, depth=0):
    require(depth <= 32, "JSON nesting budget")
    if isinstance(value, dict):
        require(len(value) <= 4096, "JSON object budget")
        for v in value.values():
            bounded(v, depth + 1)
    elif isinstance(value, list):
        require(len(value) <= 8192, "JSON list budget")
        for v in value:
            bounded(v, depth + 1)
    elif isinstance(value, str):
        require(len(value) <= 1024 * 1024, "JSON string budget")


def document(content):
    value = strict_json(content.decode("utf-8"))
    bounded(value)
    return value


def integer(value, lo, hi):
    require(type(value) is int and lo <= value <= hi, "integer out of bounds")
    return value


def polygon_mask(points, size, *, evidence=False):
    require(type(points) is list and 3 <= len(points) <= 128, "polygon vertex budget")
    for p in points:
        require(type(p) is list and len(p) == 2, "invalid polygon vertex")
        if evidence:
            require(all(type(v) in (int, float) and math.isfinite(v) for v in p)
                    and 0 <= p[0] < size[0] and 0 <= p[1] < size[1], "invalid source geometry")
        else:
            integer(p[0], 0, size[0] - 1)
            integer(p[1], 0, size[1] - 1)
    # Simple polygons only: reject crossing/touching nonadjacent edges.
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    n = len(points)
    require(len({tuple(p) for p in points}) == n, "duplicate polygon vertex")
    for i in range(n):
        a, b = points[i], points[(i+1) % n]
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            c, d = points[j], points[(j+1) % n]
            if (max(a[0], b[0]) >= min(c[0], d[0]) and max(c[0], d[0]) >= min(a[0], b[0])
                    and max(a[1], b[1]) >= min(c[1], d[1]) and max(c[1], d[1]) >= min(a[1], b[1])):
                require(not (orient(a,b,c)*orient(a,b,d) <= 0 and orient(c,d,a)*orient(c,d,b) <= 0),
                        "self-intersecting polygon")
    area = sum(points[i][0]*points[(i+1)%n][1]-points[(i+1)%n][0]*points[i][1] for i in range(n))
    require(abs(area) >= 2, "degenerate polygon")
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in points], fill=255)
    return mask


def union_mask(polygons, size):
    require(type(polygons) is list and len(polygons) <= 128, "mask polygon budget")
    result = np.zeros((size[1], size[0]), dtype=bool)
    for points in polygons:
        result |= np.asarray(polygon_mask(points, size)) != 0
    return result


def font_coverage(data, *, max_points=None):
    """Read Unicode cmap 4/12 of a single sfnt font; unknown formats fail closed.

    This verifies mapped nonzero glyph IDs, not linguistic shaping quality.
    Collections/variable fonts are deliberately unsupported in this baseline.
    """
    try:
        point_limit = MAX_COVERAGE_POINTS if max_points is None else min(MAX_COVERAGE_POINTS, max_points)
        budget_check(len(data) <= 32*1024*1024, "font byte budget")
        require(data[:4] in (b"\x00\x01\x00\x00", b"OTTO"), "unsupported font container")
        tables = {}
        count = struct.unpack_from(">H", data, 4)[0]
        budget_check(count <= 128, "font table budget")
        for i in range(count):
            tag, _, offset, length = struct.unpack_from(">4sIII", data, 12+16*i)
            require(offset+length <= len(data) and tag not in tables, "invalid/duplicate font table")
            tables[tag] = memoryview(data)[offset:offset+length]
        require(b"fvar" not in tables, "variable fonts unsupported")
        cmap = tables[b"cmap"]
        glyphs = struct.unpack_from(">H", tables[b"maxp"], 4)[0]
        require(struct.unpack_from(">H", cmap, 2)[0] <= 64, "font encoding table budget")
        coverage = None
        supported = False
        for i in range(struct.unpack_from(">H", cmap, 2)[0]):
            platform, encoding, offset = struct.unpack_from(">HHI", cmap, 4+8*i)
            if platform != 0 and (platform, encoding) not in ((3, 1), (3, 10)):
                continue
            fmt = struct.unpack_from(">H", cmap, offset)[0]
            require(fmt in (4, 12, 14), "unknown Unicode font coverage format")
            # Intersect the supported Unicode charmaps rather than certifying a
            # character present only in a map the drawing engine may not choose.
            mapped = set()
            expanded = 0
            if fmt == 12:
                supported = True
                count = struct.unpack_from(">I", cmap, offset+12)[0]
                require(count <= 200000, "font cmap budget")
                previous = -1
                for j in range(count):
                    start, end, gid = struct.unpack_from(">III", cmap, offset+16+12*j)
                    require(previous < start <= end <= 0x10ffff and gid+end-start < glyphs, "invalid font cmap")
                    previous = end
                    expanded += end-start+1
                    budget_check(expanded <= point_limit, "font coverage expansion budget")
                    mapped.update(range(start + (gid == 0), end+1))
            elif fmt == 4:
                supported = True
                count = struct.unpack_from(">H", cmap, offset+6)[0]//2
                require(0 < count <= 4096, "font cmap segment budget")
                ends = offset+14
                starts = ends+2*count+2
                deltas = starts+2*count
                ranges = deltas+2*count
                previous = -1
                for j in range(count):
                    end = struct.unpack_from(">H", cmap, ends+2*j)[0]
                    start = struct.unpack_from(">H", cmap, starts+2*j)[0]
                    delta = struct.unpack_from(">h", cmap, deltas+2*j)[0]
                    ro = struct.unpack_from(">H", cmap, ranges+2*j)[0]
                    require(previous < start <= end, "invalid font cmap segment")
                    previous = end
                    expanded += end-start+1
                    budget_check(expanded <= point_limit, "font coverage expansion budget")
                    for cp in range(start, min(end, 65534)+1):
                        gid = (cp+delta) & 65535
                        if ro:
                            gid = struct.unpack_from(">H", cmap, ranges+2*j+ro+2*(cp-start))[0]
                            gid = ((gid+delta) & 65535) if gid else 0
                        if 0 < gid < glyphs:
                            mapped.add(cp)
            if fmt in (4, 12):
                coverage = mapped if coverage is None else coverage.intersection(mapped)
        require(supported and coverage, "unknown font coverage")
        return frozenset(coverage)
    except (KeyError, struct.error, IndexError) as error:
        raise ValueError("unknown or malformed font coverage") from error


def load_fonts(manifest):
    require(manifest["schema_version"] == 1 and 1 <= len(manifest["fonts"]) <= 16, "font manifest schema/budget")
    result = {}
    consumed = 0
    points = 0
    for f in manifest["fonts"]:
        require(type(f["id"]) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", f["id"]), "font ID")
        require(f["id"] not in result and f["role"] in PRESETS and f["weight"] in ("regular", "bold"), "font role/weight/duplicate")
        # Explicit file sizes are checked before reading/expanding font bytes.
        # Resource failures are retained per font for candidate blocked reports.
        try:
            size = Path(f["path"]).stat().st_size
            budget_check(size <= 32*1024*1024 and consumed+size <= MAX_FONT_BYTES, "aggregate font byte budget")
            consumed += size
            data = read_explicit(f["path"], 32*1024*1024)
        except ResourceLimit as error:
            result[f["id"]] = error
            continue
        license_data = read_explicit(f["license_path"], 1024*1024)
        require(digest(data) == f["sha256"] and digest(license_data) == f["license_sha256"], "font/license hash mismatch")
        require(type(f["license"]) is str and bool(f["license"].strip()) and bool(license_data), "font license required")
        try:
            coverage = font_coverage(data, max_points=2*MAX_COVERAGE_POINTS-points)
            budget_check(points+len(coverage) <= 2*MAX_COVERAGE_POINTS, "aggregate font coverage budget")
            points += len(coverage)
            result[f["id"]] = (data, coverage)
        except ResourceLimit as error:
            result[f["id"]] = error
    return result


def break_positions(text):
    """Legal insertion offsets; preserve all original whitespace/punctuation.

    Break after ordinary spaces or a single existing hyphen between letters/digits.
    U+2011, dash runs, leading hyphens and punctuation-only suffixes are not sites.
    """
    positions = [0]
    for i, char in enumerate(text):
        if char == " " and i+1 < len(text) and text[i+1] != " " and text[:i+1].strip():
            positions.append(i+1)
        elif char in ("-", "\u2010") and 0 < i < len(text)-1 and text[i-1].isalnum() and text[i+1].isalnum():
            positions.append(i+1)
    if text:
        positions.append(len(text))
    return positions


def line_candidates(text, measure, width, *, budget=None):
    """Bounded balanced dynamic programming plus six legal greedy alternatives.

    For each paragraph and line count retain its minimum squared-advance layout.
    Return at most 32 whole-text candidates, with exact source insertion offsets.
    """
    budget = {"measures": 0, "transitions": 0} if budget is None else budget
    paragraphs = text.split("\n")
    sites = [break_positions(p) for p in paragraphs]
    budget_check(len(paragraphs) <= 32 and sum(len(p) for p in sites) <= 128, "line break site budget")
    combined = [(0, (), 0)]  # inserted line count, offsets, balance cost
    base = 0
    for paragraph, positions in zip(paragraphs, sites):
        if not paragraph.strip():
            options = [(0, (), 0)]
        else:
            cache = {}
            def advance(i, j):
                key = (i, j)
                if key not in cache:
                    budget["measures"] += 1
                    budget_check(budget["measures"] <= 65536, "line measurement budget")
                    cache[key] = measure(paragraph[positions[i]:positions[j]])
                return cache[key]
            previous = {0: (0, ())}
            options = []
            for lines in range(1, min(16, len(positions)-1)+1):
                current = {}
                for j in range(1, len(positions)):
                    choices = []
                    for i, (cost, offsets) in previous.items():
                        if i >= j:
                            continue
                        budget["transitions"] += 1
                        budget_check(budget["transitions"] <= 1_000_000, "balanced line search budget")
                        length = advance(i, j)
                        if length <= width and paragraph[positions[i]:positions[j]].strip():
                            choices.append((cost+length*length, offsets+((positions[i],) if i else ())))
                    if choices:
                        current[j] = min(choices)
                if len(positions)-1 in current:
                    cost, offsets = current[len(positions)-1]
                    options.append((lines-1, offsets, cost))
                previous = current
                if not previous:
                    break
            # Alternate legal partitions help irregular masks; no character is
            # discarded when the next line begins, including original spaces.
            for fraction in (1.0, .9, .8, .7, .6, .5):
                start, offsets = 0, []
                cost = 0
                while start < len(positions)-1:
                    ends = [j for j in range(start+1, len(positions)) if advance(start, j) <= width*fraction]
                    if not ends:
                        break
                    end = max(ends)
                    cost += advance(start, end)**2
                    if end < len(positions)-1:
                        offsets.append(positions[end])
                    start = end
                if start == len(positions)-1 and len(offsets) < 16:
                    options.append((len(offsets), tuple(offsets), cost))
            options = sorted(set(options))
        combined = sorted({(n+count, old+tuple(base+p for p in offsets), score+cost)
                           for n, old, score in combined for count, offsets, cost in options
                           if n+count+len(paragraphs) <= 32}, key=lambda x: (x[0], x[2], x[1]))[:32]
        base += len(paragraph)+1
    results = []
    seen = set()
    for _, offsets, _ in combined:
        if offsets in seen:
            continue
        seen.add(offsets)
        shaped = text
        for offset in reversed(offsets):
            shaped = shaped[:offset]+"\n"+shaped[offset:]
        results.append((shaped, offsets))
    return results


def layout(text, font_data, coverage, mask, style):
    budget_check(mask.mode == "L" and mask.width*mask.height <= MAX_PAGE_PIXELS, "layout page pixel budget")
    require(type(text) is str and 0 < len(text) <= 1000 and text.strip(), "text budget/blank")
    require(all(c == "\n" or (32 <= ord(c) < 127) or
                (unicodedata.category(c)[0] in ("L", "M") and
                 ("LATIN" in unicodedata.name(c, "") or unicodedata.category(c)[0] == "M")) or
                unicodedata.category(c)[0] == "P" for c in text), "baseline English horizontal text only")
    require(all(ord(c) in coverage for c in text if c != "\n"), "missing glyph coverage")
    require(features.check_feature("raqm"), "RAQM unavailable")
    resolved, approximation = resolve_style(style, face_metadata(font_data))
    lo, hi, stroke = (resolved[k] for k in ("min_size", "max_size", "outline"))
    allowed = np.asarray(mask) != 0
    bounds = mask.getbbox()
    require(bounds is not None, "empty layout region")
    left, top, right, bottom = bounds
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    search_budget = {"measures": 0, "transitions": 0}
    raster_pixels = 0
    for size in range(hi, lo-1, -1):
        font = ImageFont.truetype(BytesIO(font_data), size, layout_engine=ImageFont.Layout.RAQM)
        measure = lambda s: probe.textlength(s, font=font, direction="ltr", language="en")
        # A retained original space before an inserted newline has advance but
        # no ink. Ignore that blank advance for screening, never in drawn text.
        fit_measure = lambda s: measure(s.rstrip(' '))
        for shaped, offsets in line_candidates(text, fit_measure, right-left-2*stroke, budget=search_budget):
            spacing = int(size*resolved["leading"])
            box = probe.multiline_textbbox((0, 0), shaped, font=font, spacing=spacing,
                                          align=resolved["align"], stroke_width=stroke, direction="ltr", language="en")
            if box[3]-box[1] > bottom-top:
                continue
            # Rasterize independently of page edges. Guard padding covers normal
            # bearing/effect discrepancies; touching it is an explicit block.
            padding = 2*size+2*stroke+16
            local_size = (math.ceil(box[2]-box[0])+2*padding, math.ceil(box[3]-box[1])+2*padding)
            local_pixels = local_size[0]*local_size[1]
            budget_check(0 < min(local_size) and max(local_size) <= MAX_LOCAL_SIDE
                         and local_pixels <= MAX_LOCAL_PIXELS, "local raster budget")
            raster_pixels += local_pixels
            budget_check(raster_pixels <= 64_000_000, "cumulative local raster budget")
            local = Image.new("RGBA", local_size)
            xy = (padding-math.floor(box[0]), padding-math.floor(box[1]))
            ImageDraw.Draw(local).multiline_text(xy, shaped, font=font, fill=tuple(resolved["fill"]), spacing=spacing,
                align=resolved["align"], stroke_width=stroke, stroke_fill=tuple(resolved["outline_fill"]), direction="ltr", language="en")
            unrotated_ink = local.getchannel("A").getbbox()
            require(unrotated_ink is not None and unrotated_ink[0] > 0 and unrotated_ink[1] > 0
                    and unrotated_ink[2] < local.width and unrotated_ink[3] < local.height,
                    "local raster touches padding guard; complete ink unverified")
            pretransform_count = int(np.count_nonzero(np.asarray(local.getchannel("A"))))
            if resolved["rotation"]:
                angle = math.radians(resolved["rotation"])
                rotated_size = (math.ceil(abs(math.cos(angle))*local.width+abs(math.sin(angle))*local.height)+4,
                                math.ceil(abs(math.sin(angle))*local.width+abs(math.cos(angle))*local.height)+4)
                rotated_pixels = rotated_size[0]*rotated_size[1]
                budget_check(max(rotated_size) <= MAX_LOCAL_SIDE and rotated_pixels <= MAX_LOCAL_PIXELS,
                             "rotated local raster budget")
                raster_pixels += rotated_pixels
                budget_check(raster_pixels <= 64_000_000, "cumulative local raster budget")
                local = local.rotate(resolved["rotation"], resample=Image.Resampling.BICUBIC, expand=True)
            ink = local.getchannel("A").getbbox()
            if not ink:
                continue
            # Independently confirm the post-transform crop against every
            # nonzero alpha pixel using the installed OpenCV raster primitive.
            cx, cy, cw, ch = cv2.boundingRect(np.asarray(local.getchannel('A')))
            require((cx, cy, cx+cw, cy+ch) == ink, 'post-transform crop bounds disagree')
            require(ink[0] > 0 and ink[1] > 0 and ink[2] < local.width and ink[3] < local.height,
                    "local raster touches padding guard; complete ink unverified")
            tile = local.crop(ink)
            x = (left+right-tile.width)//2
            y = (top+bottom-tile.height)//2
            if x < 0 or y < 0 or x+tile.width > mask.width or y+tile.height > mask.height:
                continue
            alpha = np.asarray(tile.getchannel("A")) != 0
            if not (alpha & ~allowed[y:y+tile.height, x:x+tile.width]).any():
                layer = Image.new("RGBA", mask.size)
                layer.paste(tile, (x, y))
                local_count = int(alpha.sum())
                require(int(np.count_nonzero(np.asarray(layer.getchannel("A")))) == local_count, "ink clipped during placement")
                return layer, {"size": size, "outline": stroke, "draw_text": shaped,
                    "original_text": text, "inserted_break_offsets": list(offsets),
                    "resolved_style": resolved, "style_approximation": approximation,
                    "pretransform_alpha_pixels": pretransform_count,
                    "ink_bounds": layer.getchannel("A").getbbox(), "local_canvas_size": local_size,
                    "local_alpha_pixels": local_count, "rasterized_pixels": raster_pixels,
                    "search_work": dict(search_budget), "line_break_policy": "balanced_legal_hyphen_visible_space_v2",
                    "origin": None if resolved["rotation"] else [xy[0]+x-ink[0], xy[1]+y-ink[1]],
                    "local_draw_origin": list(xy), "posttransform_canvas_size": list(local.size), "spacing": spacing,
                    "line_advances": [measure(s) for s in shaped.split("\n")],
                    "line_fit_advances": [fit_measure(s) for s in shaped.split("\n")],
                    "engine": "RAQM", "direction": "ltr", "language": "en"}
    raise ValueError("overflow: no contained layout within supplied size bounds")


def offending_pixels(mask, limit=32):
    """Exact count/bounds and a row-major prefix, without full coordinate arrays."""
    integer(limit, 1, 32)
    count = int(np.count_nonzero(mask)) if mask is not None else 0
    coordinates, bounds = [], None
    if count:
        x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
        bounds = [x, y, x+width, y+height]
        for row in range(y, y+height):
            for start in range(x, x+width, 1024):
                values = np.flatnonzero(mask[row, start:min(start+1024, x+width)])
                coordinates.extend([[int(start+col), row] for col in values[:limit-len(coordinates)]])
                if len(coordinates) == limit:
                    break
            if len(coordinates) == limit:
                break
    return {'offending_count': count, 'offending_bounds': bounds, 'offending_coordinates': coordinates,
            'coordinate_limit': limit, 'coordinates_truncated': count > len(coordinates),
            'bounds_convention': 'source pixels, half-open [left, top, right, bottom]'}


def cleanup(rgb, interior, permission, protected):
    """Derive dark glyph pixels only inside explicitly permitted flat interior.

    All nonbackground components in the interior must be covered by permission;
    this intentionally rejects unexplained neighboring marks. Connected dark ink
    touching the interior boundary is unsupported. No rectangle whitening.
    """
    require(interior.any() and permission.any(), "empty cleanup region")
    require(not (permission & ~interior).any(), "cleanup permission outside interior")
    require(not (permission & protected).any(), "cleanup permission touches protected pixels")
    pixels = np.asarray(rgb)
    sample_mask = interior & ~permission & ~protected
    sample = pixels[sample_mask]
    color = np.median(sample, axis=0).astype(np.uint8) if len(sample) else None
    evidence = {'schema_version': 1, 'stage': 'cleanup', 'sample_count': len(sample),
                'sample_region': 'interior outside cleanup permission and protection',
                'background_estimate_rgb': color.tolist() if color is not None else None,
                'minimum_sample_count': 32, 'maximum_channel_delta': 2, 'minimum_background_channel': 180,
                'observed_sample_max_channel_delta': None, 'glyph_count': None,
                'interior_count': int(np.count_nonzero(interior)), 'semantic_classification': 'not_inferred'}
    def block(code, message, offending=None):
        actions = {
            'insufficient_sample': 'Inspect the original source and provide enough explicitly reviewed flat interior outside cleanup permission/protection, or leave blocked.',
            'nonflat_sample': 'Inspect the original source at the reported coordinates. Revise only explicitly reviewed interior/permission/protection geometry in a fresh selection. If a flat sample cannot be established, leave blocked; do not relax the guard or enlarge erasure to conceal marks.',
            'unsupported_dark_background': 'Inspect the original source. This method supports light flat interiors only; leave unsupported backgrounds blocked.',
            'unpermitted_ink': 'Inspect the original source and protected/non-permitted marks. Do not erase neighbors or art; revise geometry only with explicit source inspection, or leave blocked.',
            'no_cleanup_ink': 'Inspect the original source and selection. No differing cleanup ink was found; do not invent erase permission to force composition.',
            'excessive_ink_area': 'Inspect the original source and cleanup permission. The differing-pixel area exceeds this flat-interior method; leave unsupported content blocked.',
            'boundary_touch': 'Inspect the original source at the reported boundary pixels. Keep border/art protected; revise only explicitly reviewed geometry or leave blocked.',
        }
        raise CleanupBlocked({**evidence, 'code': code, 'message': message, **offending_pixels(offending),
                              'next_safe_action': actions[code]})
    if len(sample) < 32:
        block('insufficient_sample', f'insufficient background sample: {len(sample)} pixels; need at least 32')
    if int(color.min()) < 180:
        block('unsupported_dark_background', 'unsupported dark interior estimate', sample_mask & (pixels.min(axis=2) < 180))
    delta = np.max(np.abs(pixels.astype(np.int16)-color.astype(np.int16)), axis=2)
    evidence['observed_sample_max_channel_delta'] = int(delta[sample_mask].max())
    nonflat = sample_mask & (delta > 2)
    if nonflat.any():
        block('nonflat_sample', f'background sample not flat: {int(nonflat.sum())} of {len(sample)} pixels exceed 2 channel levels', nonflat)
    glyph = interior & (delta > 2)
    evidence['glyph_count'] = int(glyph.sum())
    if not glyph.any():
        block('no_cleanup_ink', 'no differing cleanup ink found')
    if (glyph & ~permission).any():
        block('unpermitted_ink', 'unpermitted ink contamination', glyph & ~permission)
    if int(glyph.sum()) > int(interior.sum())//3:
        block('excessive_ink_area', 'excessive differing-pixel area for flat-interior cleanup', glyph)
    # One-pixel collar must be background; no text crossing permission boundary.
    expanded = glyph.copy()
    expanded[1:] |= glyph[:-1]
    expanded[:-1] |= glyph[1:]
    expanded[:, 1:] |= glyph[:, :-1]
    expanded[:, :-1] |= glyph[:, 1:]
    boundary = expanded & (~permission | ~interior | protected)
    if boundary.any():
        block('boundary_touch', 'ink collar crosses cleanup permission/interior/protection boundary', boundary)
    clean = pixels.copy()
    clean[glyph] = color
    return Image.fromarray(clean, "RGB"), glyph, color.tolist()


def pixel_audit(source, clean, composite, glyph, alpha, protected):
    """Count actual canonical RGB changes; masks alone are not a locality proof."""
    original = np.asarray(source)
    plate = np.asarray(clean)
    result = np.asarray(composite)
    require(source.mode == clean.mode == composite.mode == "RGB" and
            source.size == clean.size == composite.size and
            glyph.shape == alpha.shape == protected.shape == original.shape[:2], "pixel audit shape/mode mismatch")
    clean_diff = np.any(plate != original, axis=2)
    compose_diff = np.any(result != plate, axis=2)
    source_diff = np.any(result != original, axis=2)
    count = lambda pixels: int(np.count_nonzero(pixels))
    return {"source_pixels": int(glyph.size), "cleanup_mask_pixels": count(glyph),
        "clean_changed_pixels": count(clean_diff), "clean_unchanged_pixels": count(~clean_diff),
        "outside_cleanup_pixels": count(~glyph), "outside_cleanup_changed_pixels": count(clean_diff & ~glyph),
        "outside_cleanup_unchanged_pixels": count(~clean_diff & ~glyph),
        "protected_pixels": count(protected), "protected_clean_changed_pixels": count(clean_diff & protected),
        "protected_composite_changed_pixels": count(source_diff & protected),
        "protected_unchanged_pixels": count(~(clean_diff | source_diff) & protected),
        "translated_alpha_pixels": count(alpha), "composite_changed_pixels": count(compose_diff),
        "outside_alpha_pixels": count(~alpha), "outside_alpha_changed_pixels": count(compose_diff & ~alpha),
        "outside_alpha_unchanged_pixels": count(~compose_diff & ~alpha),
        "composite_source_changed_pixels": count(source_diff), "untouched_source_pixels": count(~source_diff)}


def verify_pixel_audit(audit):
    require(all(audit[key] == 0 for key in ("outside_cleanup_changed_pixels", "outside_alpha_changed_pixels",
            "protected_clean_changed_pixels", "protected_composite_changed_pixels")), "runtime pixel locality/protection violation")


def specimen_rasters(source, clean, composite, masks, text_tile):
    """Bounded thumbnails; the full-resolution evidence is never resized."""
    bounds = Image.fromarray(masks['interior'].astype(np.uint8)*255).getbbox()
    require(bounds is not None, 'empty specimen interior')
    bounds = (max(0, bounds[0]-8), max(0, bounds[1]-8), min(source.width, bounds[2]+8), min(source.height, bounds[3]+8))
    width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
    scale = min(1, 1000/width, 1000/height)
    size = (max(1, int(width*scale)), max(1, int(height*scale)))
    rasters = {name: raster.resize(size, box=bounds) for name, raster in
               (('source', source), ('clean', clean), ('composite', composite))}
    text = Image.new('RGBA', size)
    if text_tile is not None:
        tile, xy = text_tile
        if scale != 1:
            tile = tile.resize((max(1, int(tile.width*scale)), max(1, int(tile.height*scale))))
        text.paste(tile, (round((xy[0]-bounds[0])*scale), round((xy[1]-bounds[1])*scale)))
    rasters['text'] = text
    for name, mask in masks.items():
        rasters[name] = Image.fromarray(mask.astype(np.uint8)*255).resize(size, resample=Image.Resampling.NEAREST, box=bounds)
    return {'source_bounds': list(bounds), 'size': list(size), 'scale': scale}, rasters


@dataclass(frozen=True)
class BoundSelection:
    """Immutable canonical snapshot; includes all selected geometry/style evidence."""
    selection_json: str
    preview_sha256: str
    image_sha256: str
    fonts_sha256: str


def strict_result_binding(record, unit):
    """Current strict candidate data only; no geometry or review authorization."""
    require(unit.disposition == 'preview' and record['status'] == 'preview'
            and record.get('input_kind', 'strict_utterance') == 'strict_utterance'
            and record['binding_current'] is True, 'blocked/fragment/alternate selection unsupported')
    request = TranslationRequest((unit,), (unit.unit_id,))
    require(canonical(record['request']) == canonical(asdict(request)) and record['request_revision'] == request.revision
            and record['unit_revision'] == unit.revision, 'stale request/unit')
    data = dict(record['result'])
    data['translations'] = tuple(tuple(p) for p in data['translations'])
    data['outcomes'] = tuple(UnitOutcome(**{**o, 'reasons': tuple(o['reasons'])}) for o in data['outcomes'])
    result = TranslationResult(**data)
    result.assert_current(request)
    require(result.status == 'preview' and len(result.translations) == 1 and result.translations[0][0] == unit.unit_id,
            'result not a strict translated candidate')
    return {'unit_id': unit.unit_id, 'member_ids': list(unit.member_ids), 'order': list(unit.order),
            'unit_revision': unit.revision, 'request_revision': request.revision,
            'result_sha256': revision(record['result']), 'text': result.translations[0][1]}


def bind(preview, image_path, rgb, image_bytes, preview_bytes, selection, fonts_bytes, *, selection_kind='candidate_only'):
    require(type(preview["schema_version"]) is int and preview["schema_version"] in (1, 2, 3)
            and preview["preview_only"] is True, "unsupported saved preview")
    source = {"path": str(Path(image_path).resolve()), "sha256": digest(image_bytes),
              "pixels_sha256": digest(f"RGB:{rgb.width}:{rgb.height}:".encode()+rgb.tobytes()), "size": list(rgb.size)}
    require(canonical(preview["input"]) == canonical(source) and canonical(selection["source"]) == canonical(source),
            "stale source image/file/pixel binding")
    require(selection_kind in ('candidate_only', 'automatic_candidate_only'), 'unknown rendering binding policy')
    require(type(selection["schema_version"]) is int and selection["schema_version"] == 1
            and selection["purpose"] == selection_kind, "candidate-only selection required")
    require(selection["preview_sha256"] == digest(preview_bytes) and selection["fonts_sha256"] == digest(fonts_bytes), "stale preview/font manifest binding")
    require(type(selection["units"]) is list and (0 if selection_kind == 'automatic_candidate_only' else 1)
            <= len(selection["units"]) <= 64, "selection unit budget")
    batch = preview["ocr"]["batches"][0]
    require(batch["page"]["image_sha256"] == source["sha256"] and batch["page"]["pixels_sha256"] == source["pixels_sha256"]
            and batch["page"]["image_path"] == source["path"], "foreign batch page")
    candidates = preview["ocr"].get("automatic_recognition_results", [])
    units = automatic_units(batch, candidates[0] if candidates else ())
    records = preview["units"]
    require([r["unit"] for r in records] == [json.loads(canonical(asdict(u))) for u in units], "saved unit/source evidence drift")
    by_id = {r["unit"]["unit_id"]: (r, u) for r, u in zip(records, units)}
    require(len(by_id) == len(records), "duplicate unit ID")
    raw = preview["ocr"]["raw_detections"][0]
    require(len(raw) <= 2048, "raw detection budget")
    identities = [(type(d["source_id"]).__name__, d["source_id"]) for d in raw]
    require(all(type(d["source_id"]) in (str, int) for d in raw) and len(set(identities)) == len(raw), "invalid typed source IDs")
    for d in raw:
        polygon_mask(d["polygon"], rgb.size, evidence=True)
    raw_map = {(type(d["source_id"]).__name__, d["source_id"]): d for d in raw}
    require(len(raw_map) == len(batch["detections"]), "raw detection coverage drift")
    for d in batch["detections"]:
        original = raw_map.get((type(d["source_id"]).__name__, d["source_id"]))
        require(original is not None and all(canonical(original[k]) == canonical(d[k])
                for k in ("polygon", "text", "confidence", "selected_text", "status", "reason")), "raw geometry/text evidence drift")
    for p in batch["proposals"]:
        identity = {"policy": batch["policy"], "page": batch["page"], "main_ids": p["main_ids"],
            "members": p["member_ids"], "member_polygons": [(i, raw_map[(type(i).__name__, i)]["polygon"]) for i in p["member_ids"]],
            "annotation_parents": p["annotation_parents"], "direction": p["direction"]}
        if batch.get("ownership_policy", "production8") != "production8":
            identity["ownership_policy"] = batch["ownership_policy"]
        if batch.get("companion_policy", "disabled") != "disabled":
            identity["companion_policy"] = batch["companion_policy"]
        require(p["proposal_id"] == "proposal-"+digest(json.dumps(identity, sort_keys=True).encode())[:24], "source proposal identity drift")
    seen = set()
    for s in selection["units"]:
        uid = s["unit_id"]
        require(uid not in seen and uid in by_id, "duplicate/foreign selected unit")
        seen.add(uid)
        record, unit = by_id[uid]
        expected = strict_result_binding(record, unit)
        require(all(canonical(s[key]) == canonical(value) for key, value in expected.items()),
                'selection text/member/order/result drift')
        if selection_kind == 'candidate_only':
            require(type(s["geometry_intent"]) is str and bool(s["geometry_intent"].strip()), "explicit geometry review intent required")
    return BoundSelection(canonical(selection), digest(preview_bytes), digest(image_bytes), digest(fonts_bytes))


def run(mode, *, preview, image, selection, fonts, output, parent=None, compare=()):
    require(mode in ("plan", "compose"), "mode must be plan or compose")
    output = Path(output).absolute()
    require(not any(p.is_symlink() for p in (output, *output.parents)), "symlink output path")
    require(not output.exists() and output.parent.is_dir(), "output must be fresh with existing parent")
    paths = [Path(p).absolute() for p in (preview, image, selection, fonts)]
    contents = [read_explicit(p, limit) for p, limit in zip(paths, (64*1024*1024, 64*1024*1024, 4*1024*1024, 256*1024))]
    preview_doc, selection_doc, fonts_doc = (document(contents[i]) for i in (0, 2, 3))
    with Image.open(BytesIO(contents[1])) as opened:
        require(opened.mode in ("RGB", "L") and getattr(opened, "n_frames", 1) == 1
                and opened.width*opened.height <= MAX_PAGE_PIXELS, "unsupported source mode/frame/pixel budget")
        require(not opened.info.get("icc_profile"), "embedded color profile unsupported; no silent color conversion")
        require(opened.getexif().get(274, 1) == 1, "EXIF orientation unsupported; source coordinates must be explicit")
        rgb = opened.convert("RGB")
        source_mode = opened.mode
    bound = bind(preview_doc, paths[1], rgb, contents[1], contents[0], selection_doc, contents[3])
    from .review import POLICY, implementation_hash, job_revision, parent_link, verify_candidate, specimen_html
    parent_candidate = parent_link(parent, bound.image_sha256) if parent is not None else None
    require(type(compare) in (tuple, list) and len(compare) <= 4, "comparison candidate budget")
    comparisons = []
    for alternative in compare:
        previous, sha = verify_candidate(candidate=alternative, preview=preview, image=image,
            selection=Path(alternative)/'evidence-selection.json', fonts=fonts)
        comparisons.append({'manifest_sha256': sha, 'job_revision': previous['job_revision'],
                            'scope': 'source/preview/font-bound candidate comparison; no review inherited'})
    budget_check(len(selection_doc["units"])*rgb.width*rgb.height <= MAX_UNIT_PIXELS, "aggregate mask/layer pixel budget")
    font_map = load_fonts(fonts_doc)
    # Validate all geometry before reserving output. No paths are taken from it.
    prepared = []
    occupied = np.zeros((rgb.height, rgb.width), dtype=bool)
    raw = preview_doc["ocr"]["raw_detections"][0]
    for s in selection_doc["units"]:
        interior = union_mask(s["interior"], rgb.size)
        allowed = union_mask(s["layout"], rgb.size)
        permission = union_mask(s["cleanup_permission"], rgb.size)
        protected = union_mask(s["protected"], rgb.size)
        require(allowed.any() and not (allowed & (~interior | protected)).any(), "layout outside interior/protected")
        require(permission.any() and not (permission & (~interior | protected)).any(), "cleanup permission outside interior/protected")
        require(not ((interior | protected) & occupied).any(), "selected regions collide")
        occupied |= interior | protected
        member_keys = {(type(i).__name__, i) for i in s["member_ids"]}
        for d in raw:
            source_key = (type(d["source_id"]).__name__, d["source_id"])
            if source_key not in member_keys:
                neighbor = np.asarray(polygon_mask(d["polygon"], rgb.size, evidence=True)) != 0
                require(not (neighbor & (interior | allowed | permission)).any(), "neighbor/annotation contamination")
        require(s["font_id"] in font_map, "unknown explicit font")
        prepared.append((s, interior, allowed, permission, protected))
    output.mkdir(mode=0o700)
    artifacts = {}
    artifact_bytes = 0
    def save_bytes(name, data):
        nonlocal artifact_bytes
        artifact_bytes += len(data)
        budget_check(len(data) <= 64*1024*1024 and artifact_bytes <= 128*1024*1024, "artifact/report byte budget")
        with (output/name).open("xb") as stream:
            stream.write(data)
        artifacts[name] = digest(data)
    def save_image(name, value):
        stream = BytesIO()
        value.save(stream, format="PNG")
        save_bytes(name, stream.getvalue())
    for name, data in zip(("evidence-preview.json", "original-file.bin", "evidence-selection.json", "evidence-fonts.json"), contents):
        save_bytes(name, data)
    save_image("source.png", rgb)
    clean, composite = rgb.copy(), rgb.copy()
    report = {"schema_version": 1, "purpose": "candidate_only", "mode": mode,
        "policy": POLICY, "implementation_sha256": implementation_hash(),
        "selection_sha256": digest(contents[2]),
        "parent_candidate": parent_candidate, "comparison_candidates": comparisons,
        "approved_export": False, "binding": asdict(bound), "source_mode": source_mode,
        "canonical_mode": "RGB", "color_policy": "no color management; embedded ICC rejected",
        "coordinates": "source pixels; polygon vertices inclusive", "units": [],
        "resource_limits": {"page_pixels": MAX_PAGE_PIXELS, "aggregate_unit_pixels": MAX_UNIT_PIXELS,
            "local_pixels": MAX_LOCAL_PIXELS, "local_side": MAX_LOCAL_SIDE,
            "font_bytes": MAX_FONT_BYTES, "coverage_points_per_font": MAX_COVERAGE_POINTS},
        "engine": {"pillow": pillow_version, "opencv": cv2.__version__, **{k: features.version_feature(k) for k in ("raqm", "harfbuzz", "fribidi")},
                   "freetype": features.version_module("freetype2")}}
    report['job_revision'] = job_revision(report)
    for index, alternative in enumerate(compare):
        save_bytes(f'comparison-{index}.png', read_explicit(Path(alternative)/'composite.png'))
    layers = []
    text_tiles = {}
    cleanup_union = np.zeros((rgb.height, rgb.width), dtype=bool)
    alpha_union = np.zeros_like(cleanup_union)
    protected_union = np.zeros_like(cleanup_union)
    for index, (s, interior, allowed, permission, protected) in enumerate(prepared):
        prefix = f"unit-{index:03d}"
        row = {"unit_id": s["unit_id"], "status": "blocked", "text": s["text"], "font_id": s["font_id"],
               "aesthetic_state": "not_assessed", "linguistic_approval": "not_assessed",
               "cleanup_diagnostics": None,
               "resolved_style": None, "style_approximation": ["Style unresolved; see blocked reason."]}
        report["units"].append(row)
        protected_union |= protected
        for name, mask in (("interior", interior), ("layout", allowed), ("permission", permission), ("protected", protected)):
            save_image(f"{prefix}-{name}.png", Image.fromarray(mask.astype(np.uint8)*255))
        try:
            font = font_map[s["font_id"]]
            if isinstance(font, ResourceLimit):
                raise font
            row['resolved_style'], row['style_approximation'] = resolve_style(s['style'], face_metadata(font[0]))
            layer, provenance = layout(s["text"], *font,
                                      Image.fromarray(allowed.astype(np.uint8)*255), s["style"])
            save_image(f"{prefix}-text.png", layer)
            row["layout"] = provenance
            ink = layer.getchannel('A').getbbox()
            require(ink is not None, 'empty layout alpha')
            text_tiles[index] = (layer.crop(ink), ink[:2])
            alpha = np.asarray(layer.getchannel("A")) != 0
            row["outside_layout_alpha_pixels"] = int(np.count_nonzero(alpha & ~allowed))
            require(row["outside_layout_alpha_pixels"] == 0, "runtime alpha outside layout region")
            plate = rgb.copy()
            glyph = np.zeros_like(cleanup_union)
            if mode == "compose":
                plate, glyph, color = cleanup(rgb, interior, permission, protected)
                save_image(f"{prefix}-glyph-cleanup.png", Image.fromarray(glyph.astype(np.uint8)*255))
                row.update(background=color, cleanup_pixels=int(glyph.sum()))
            row["outside_cleanup_permission_pixels"] = int(np.count_nonzero(glyph & (~permission | ~interior | protected)))
            require(row["outside_cleanup_permission_pixels"] == 0, "runtime glyph mask outside cleanup permission/protection")
            unit_composite = Image.alpha_composite(plate.convert("RGBA"), layer).convert("RGB")
            row["pixel_audit"] = pixel_audit(rgb, plate, unit_composite, glyph, alpha, protected)
            verify_pixel_audit(row["pixel_audit"])
            ink = layer.getchannel("A").getbbox()
            require(ink is not None, "empty accepted alpha")
            if mode == "compose":
                next_clean = np.asarray(clean).copy()
                next_clean[glyph] = np.asarray(plate)[glyph]
                clean = Image.fromarray(next_clean, "RGB")
            # Retain compact tiles, not one full-page RGBA buffer per unit.
            layers.append(text_tiles[index])
            cleanup_union |= glyph
            alpha_union |= alpha
            row["status"] = "candidate_composed" if mode == "compose" else "candidate_layout_no_erase"
        except (ValueError, OSError) as error:
            row["reason"] = str(error)
            row["failure_kind"] = "resource_limit" if isinstance(error, ResourceLimit) else "validation"
            if isinstance(error, CleanupBlocked):
                row['cleanup_diagnostics'] = error.diagnostics
    composite = clean.convert("RGBA")
    for tile, xy in layers:
        composite.alpha_composite(tile, dest=xy)
    composite = composite.convert("RGB")
    report["pixel_audit"] = pixel_audit(rgb, clean, composite, cleanup_union, alpha_union, protected_union)
    try:
        verify_pixel_audit(report["pixel_audit"])
    except ValueError as error:
        # Never publish an unsafe composed raster as a candidate. Keep actual
        # failed counts and roll output images back to canonical source pixels.
        report["rejected_pixel_audit"] = report["pixel_audit"]
        for row in report["units"]:
            row.update(status="blocked", reason=str(error), failure_kind="runtime_locality")
        clean, composite = rgb.copy(), rgb.copy()
        report["pixel_audit"] = pixel_audit(rgb, clean, composite, np.zeros_like(cleanup_union),
                                          np.zeros_like(alpha_union), protected_union)
    save_image("clean.png", clean)
    save_image("composite.png", composite)
    for name, value in (("cleanup-diff", clean), ("composite-diff", composite)):
        diff = np.any(np.asarray(value) != np.asarray(rgb), axis=2)
        save_image(name+".png", Image.fromarray(diff.astype(np.uint8)*255))
    # Bounded static specimens, independently labelled as thumbnails. These do
    # not replace or rescale full-resolution layout/cleanup/composite evidence.
    for index, (_, interior, allowed, permission, protected) in enumerate(prepared):
        prefix = f'unit-{index:03d}-specimen-'
        metadata, rasters = specimen_rasters(rgb, clean, composite,
            {'interior': interior, 'layout': allowed, 'permission': permission, 'protected': protected,
             'glyph-cleanup': cleanup_union & interior if report['units'][index]['status'] == 'candidate_composed'
             else np.zeros_like(cleanup_union)}, text_tiles.get(index))
        report['units'][index]['specimen'] = metadata
        for name, raster in rasters.items():
            save_image(prefix+name+'.png', raster)
    # Recheck every explicit input before publishing the manifest.
    require(all(read_explicit(p) == data for p, data in zip(paths, contents)), "inputs changed during render")
    page = specimen_html(report).encode()
    budget_check(len(page) <= 4*1024*1024, 'static report byte budget')
    save_bytes('review.html', page)
    report["artifacts"] = dict(artifacts)
    manifest = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
    budget_check(len(manifest) <= 8*1024*1024, 'manifest byte budget')
    save_bytes("manifest.json", manifest)
    return report
