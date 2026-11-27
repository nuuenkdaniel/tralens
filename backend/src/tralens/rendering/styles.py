"""Explicit deterministic approximations, never source-font matching or fake weight."""
from io import BytesIO
import math
import struct

from PIL import ImageFont


PRESETS = {
    'dialogue': dict(fill=[0, 0, 0], outline_fill=[255, 255, 255], outline=0,
                     leading=.2, align='center', rotation=0, requested_weight='regular'),
    'narration': dict(fill=[25, 25, 25], outline_fill=[255, 255, 255], outline=0,
                      leading=.3, align='left', rotation=0, requested_weight='regular'),
    'shout': dict(fill=[0, 0, 0], outline_fill=[255, 255, 255], outline=2,
                  leading=.15, align='center', rotation=0, requested_weight='bold'),
    'thought': dict(fill=[55, 55, 70], outline_fill=[255, 255, 255], outline=1,
                    leading=.3, align='center', rotation=0, requested_weight='regular'),
}
PRESETS['handwriting'] = dict(PRESETS['dialogue'])


def check(condition, message):
    if not condition:
        raise ValueError(message)


def face_metadata(data):
    """FreeType-reported names and actual OS/2 weight, or explicitly unknown.

    Called after the bounded sfnt/coverage validation. Names/weight are metadata
    from the chosen bytes, not independent font authenticity certification.
    """
    check(len(data) <= 32*1024*1024 and len(data) >= 12 and struct.unpack_from('>H', data, 4)[0] <= 128,
          'bounded sfnt metadata required')
    font = ImageFont.truetype(BytesIO(data), 16, layout_engine=ImageFont.Layout.RAQM)
    family, subfamily = font.getname()
    check(all(v is None or len(v) <= 512 for v in (family, subfamily)), 'font face name budget')
    weight = None
    for i in range(struct.unpack_from('>H', data, 4)[0]):
        tag, _, offset, length = struct.unpack_from('>4sIII', data, 12+16*i)
        if tag == b'OS/2' and length >= 6:
            value = struct.unpack_from('>H', data, offset+4)[0]
            weight = value if 1 <= value <= 1000 else None
    return {'family': family or None, 'subfamily': subfamily or None, 'weight_class': weight,
            'metadata_source': 'chosen sfnt bytes / FreeType; unknown fields are null'}


def resolve_style(style, face):
    check(type(style) is dict, 'style object required')
    allowed = {'preset', 'min_size', 'max_size', 'fill', 'outline_fill', 'outline', 'leading', 'align', 'rotation'}
    check(set(style) <= allowed, 'unknown style control')
    preset = style.get('preset', 'legacy')
    check(type(preset) is str and (preset == 'legacy' or preset in PRESETS), 'unknown explicit role preset')
    base = dict(PRESETS['dialogue'] if preset == 'legacy' else PRESETS[preset])
    base.update({k: v for k, v in style.items() if k != 'preset'})
    check('min_size' in base and 'max_size' in base, 'explicit size bounds required')
    for key, lo, hi in (('min_size', 8, 144), ('max_size', 8, 144), ('outline', 0, 8)):
        check(type(base[key]) is int and lo <= base[key] <= hi, 'bounded integer '+key)
    check(base['min_size'] <= base['max_size'], 'ordered size bounds required')
    for key, lo, hi in (('leading', 0, 1), ('rotation', -20, 20)):
        check(type(base[key]) in (int, float) and math.isfinite(base[key]) and lo <= base[key] <= hi,
              'bounded finite '+key)
    for key in ('fill', 'outline_fill'):
        check(type(base[key]) is list and len(base[key]) == 3 and
              all(type(v) is int and 0 <= v <= 255 for v in base[key]), 'RGB '+key+' required')
    check(base['align'] in ('left', 'center', 'right'), 'bounded alignment required')
    approximation = ['Explicit role approximation; no exact source-font match asserted.']
    weight = face['weight_class']
    if weight is None:
        approximation.append('Actual weight is unknown; no regular/bold certification.')
    elif (base['requested_weight'] == 'bold') != (weight >= 600):
        approximation.append('Requested preset weight differs from selected face; no synthetic weight applied.')
    base.update(preset=preset, actual_face=face, synthetic_weight=False)
    return base, approximation
