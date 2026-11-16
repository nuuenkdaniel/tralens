"""Original-image light interior hypotheses shared by OCR and rendering.

Geometry does not decide utterance membership, annotation roles, speakers or ink
ownership. No dependency on a successful glyph mask, renderer or model service.
"""
from dataclasses import dataclass
from collections import OrderedDict
from copy import deepcopy
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .translation.contracts import revision


POLICY = 'original_multilevel_closed_light_interior_v1'
MAX_PIXELS = 4_000_000
MAX_REGION = 262144
MAX_WORK = 512_000_000


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def key(value):
    require(type(value) in (str, int) and value != '', 'typed nonempty source ID required')
    return type(value).__name__, value


def polygon_mask(polygon, size):
    require(type(polygon) in (list, tuple) and 3 <= len(polygon) <= 64, 'polygon vertex budget')
    require(all(len(p) == 2 and all(type(v) in (int, float) and np.isfinite(v) for v in p)
        and 0 <= p[0] <= size[0] and 0 <= p[1] <= size[1] for p in polygon), 'invalid source polygon')
    result = Image.new('L', size)
    ImageDraw.Draw(result).polygon([tuple(p) for p in polygon], fill=255)
    return np.asarray(result) != 0


def erode(mask, radius):
    return cv2.erode(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0


def dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)) != 0


def mask_revision(mask):
    return revision({'shape':list(mask.shape), 'packed_sha256':hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()})


@dataclass
class Interior:
    envelope: np.ndarray
    core: np.ndarray
    background: list
    evidence: dict

    @property
    def revision(self):
        return revision(self.evidence)


class LightInteriors:
    def __init__(self, rgb, detections):
        require(rgb.mode == 'RGB' and rgb.width*rgb.height <= MAX_PIXELS, 'balloon original raster budget')
        require(type(detections) is list and 1 <= len(detections) <= 512
            and len(detections)*rgb.width*rgb.height <= MAX_WORK, 'balloon geometry work budget')
        self.pixels = np.asarray(rgb).copy()
        self.size = rgb.size
        self.source_revision = hashlib.sha256(rgb.tobytes()).hexdigest()
        self.geometry_revision = revision(detections)
        self.raw = {key(d['source_id']):deepcopy(d) for d in detections}
        require(len(self.raw) == len(detections), 'duplicate typed source IDs')
        self.masks = PackedMasks(self.raw,rgb.size)
        self.fields = []
        # Two levels and both connectivities must yield the same filled boundary.
        # This measures topology before assigning any original interior ink.
        for threshold in (220, 245):
            for connectivity in (4, 8):
                count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    (self.pixels.min(axis=2) >= threshold).astype(np.uint8), connectivity=connectivity)
                require(count <= 8192, 'balloon component budget')
                self.fields.append((threshold, connectivity, labels, stats))
        self.cache = OrderedDict()

    def locate(self, member_ids):
        require(type(member_ids) in (list, tuple) and 1 <= len(member_ids) <= 32, 'balloon member budget')
        keys = tuple(key(i) for i in member_ids)
        require(len(set(keys)) == len(keys) and set(keys) <= self.raw.keys(), 'foreign/duplicate balloon members')
        signatures, votes, envelopes = [], [], []
        for threshold, connectivity, labels, stats in self.fields:
            chosen = []
            for k in keys:
                mask = self.masks[k]
                counts = np.bincount(labels[mask], minlength=len(stats))
                require(len(counts) > 1 and int(counts[1:].sum()) > 0, 'balloon lacks multilevel light support')
                label = int(counts[1:].argmax()+1)
                light = int(counts[1:].sum())
                require(light >= .5*int(mask.sum()) and counts[label] >= .85*light,
                    'balloon ambiguous light support')
                x,y,w,h,area = map(int,stats[label])
                require(x > 0 and y > 0 and x+w < self.size[0] and y+h < self.size[1],
                    'balloon open/page-connected')
                require(64 <= area <= MAX_REGION, 'balloon region extent budget')
                chosen.append(label)
                votes.append({'source_id':k[1], 'threshold':threshold, 'connectivity':connectivity,
                    'light_fraction':light/int(mask.sum()), 'dominant_light_fraction':int(counts[label])/light})
            require(len(set(chosen)) == 1, 'balloon members have distinct interiors')
            signature = threshold, connectivity, chosen[0]
            signatures.append(signature)
            if signature not in self.cache:
                component = (labels == chosen[0]).astype(np.uint8)
                contours,_ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                envelope = np.zeros_like(component)
                cv2.drawContours(envelope, contours, -1, 1, -1)
                self.cache[signature] = np.packbits(envelope != 0)
                if len(self.cache) > 64:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(signature)
            envelopes.append(np.unpackbits(self.cache[signature],count=self.size[0]*self.size[1]).reshape(
                self.size[1],self.size[0]).view(bool))
        # Antialiased outlines/very light boundary intrusions may form thin
        # disagreements. Keep ALL of these protected, including narrow spurs;
        # never erase or bridge them. A broad disputed lobe rejects the region.
        envelope = np.logical_and.reduce(envelopes)
        disagreement = np.logical_or.reduce(envelopes) & ~envelope
        require(not erode(disagreement,2).any(),
            'balloon multilevel boundary disagreement')
        core = erode(envelope, 4)
        require(64 <= int(envelope.sum()) <= MAX_REGION and core.any(), 'balloon filled extent budget')
        outer = dilate(envelope,1) & ~envelope
        support = float((self.pixels[outer].min(axis=1) < 245).mean())
        require(support >= .95, 'balloon unsupported boundary')
        # Background measured over a spatial grid, not a glyph collar. Every
        # occupied tile needs the same dominant light surface; gradients cannot
        # hide in a whole-region median. Dark cells are NOT silently filled.
        sample = self.pixels[core & (self.pixels.min(axis=2) >= 220)]
        require(len(sample) >= 32, 'balloon background samples insufficient')
        color = np.median(sample, axis=0).astype(np.uint8)
        require(int(color.min()) >= 220 and int(np.ptp(color)) <= 8, 'balloon background not light neutral')
        residual = np.abs(self.pixels.astype(np.int16)-color.astype(np.int16)).max(axis=2)
        x,y,w,h = cv2.boundingRect(core.astype(np.uint8))
        tiles = []
        for top in range(y,y+h,32):
            for left in range(x,x+w,32):
                selection = core[top:top+32,left:left+32] & (self.pixels[top:top+32,left:left+32].min(axis=2) >= 220)
                if int(selection.sum()) < 16:
                    continue
                delta = residual[top:top+32,left:left+32][selection]
                flat = float((delta <= 2).mean())
                require(flat >= .9, 'balloon nonuniform spatial background')
                tiles.append({'bounds':[left,top,left+32,top+32], 'samples':len(delta), 'flat_fraction':flat})
        require(len(tiles) >= 2, 'balloon background sampling insufficient')
        evidence = {'policy':POLICY, 'source_pixels_sha256':self.source_revision,
            'raw_detection_revision':self.geometry_revision, 'boundary_mask_revision':mask_revision(envelope),
            'core_mask_revision':mask_revision(core), 'bounds':list(cv2.boundingRect(envelope.astype(np.uint8))),
            'boundary_inset':4, 'boundary_support_fraction':support, 'background_rgb':color.tolist(),
            'boundary_band_tolerance':2, 'boundary_disagreement_pixels':int((np.logical_or.reduce(envelopes) & ~envelope).sum()),
            'background_tiles':tiles, 'votes':votes, 'envelope_pixels':int(envelope.sum()),
            'semantic_balloon_verified':False, 'utterance_membership_inferred':False, 'speaker':'unknown'}
        return Interior(envelope.copy(), core, color.tolist(), evidence)


class PackedMasks:
    """Retain source geometry at one bit/pixel, not one byte per raw detection."""
    def __init__(self, raw, size):
        self.shape = size[1],size[0]
        self.packed = {k:np.packbits(polygon_mask(d['polygon'],size)) for k,d in raw.items()}

    def __getitem__(self,k):
        return np.unpackbits(self.packed[k],count=self.shape[0]*self.shape[1]).reshape(self.shape).view(bool)

    def items(self):
        for k in self.packed:
            yield k,self[k]
