"""Discover missing page lettering before frontier translation and rendering."""
from . import frontier
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw


def propose(image, polygons, model, timeout):
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        with Image.open(image) as source:
            marked = source.convert('RGB')
        draw = ImageDraw.Draw(marked)
        for box in polygons:
            polygon = [tuple(p) for p in box]
            draw.line(polygon + polygon[:1], fill='#00ccff', width=2)
        outline = root/'coverage.png'
        marked.save(outline)
        fields = {'source': {'type':'string'}, 'english': {'type':'string'},
                  'box': {'type':'array', 'items':{'type':'integer'}, 'minItems':4, 'maxItems':4},
                  'landmark': {'type':'string'}}
        item = dict(type='object', properties=fields, required=list(fields), additionalProperties=False)
        schema = dict(type='object', properties={'regions':dict(type='array', items=item)}, required=['regions'],
                      additionalProperties=False)
        prompt = ('Scan the entire Japanese comic in image 1 for missed dialogue, handwritten asides, '
                  'signs and sound effects. Image 2 is the identical page with cyan outlines around EVERY '
                  'existing detection. Return ONLY visible Japanese lettering outside the existing polygons. '
                  'For partially detected phrases report only separate uncovered characters if independently meaningful. '
                  'Exclude furigana repeating a detected word, pure punctuation, Latin lettering, motion lines, '
                  'and drawings. Empty regions is a valid result. Give exact Japanese transcription, concise '
                  'English lettering, landmark and tight [left,top,right,bottom] box in normalized 0..1000 '
                  'page coordinates. Scan every panel in reading order and double-check the bounds. No tools.')
        result = frontier.infer(root, model, prompt, [image, outline], schema, timeout)
    return result['regions']


def extend(engine, image, size, model, timeout):
    """Rebuild one coherent batch so grouping sees the old and new lettering together."""
    import re
    from ..ocr.automatic_units import propose_page_from_boxes
    from ..ocr.types import OCREngine, Text_Box

    original = engine.raw_detections[0]
    polygons = [box.paddle_poly.tolist() for box in original]
    proposals = propose(image, polygons, model, timeout)
    if not isinstance(proposals, list):
        raise ValueError('invalid frontier discovery response')

    covered = Image.new('1', size)
    draw = ImageDraw.Draw(covered)
    for polygon in polygons:
        draw.polygon(list(map(tuple, polygon)), fill=1)

    added = []
    report = dict(model=model, proposed=len(proposals), added_ids=[], skipped=[])
    next_id = max((box.box_id for box in original if isinstance(box.box_id, int)), default=-1) + 1
    for i,row in enumerate(proposals):
        coords = row.get('box') if isinstance(row, dict) else None
        if (not isinstance(coords, list) or len(coords) != 4
                or any(type(v) is not int or not 0 <= v <= 1000 for v in coords)):
            report['skipped'].append([i,'invalid_box'])
            continue
        source = row.get('source')
        if not isinstance(source,str) or len(source)>512 or not re.search(r'[ぁ-ゖァ-ヺ一-鿿]',source):
            report['skipped'].append([i,'no_japanese_reading'])
            continue
        x,y,r,b = [round(n*size[j%2]/1000) for j,n in enumerate(coords)]
        if r-x<2 or b-y<2 or (r-x)*(b-y)>size[0]*size[1]//4:
            report['skipped'].append([i,'invalid_extent'])
            continue
        occupied = covered.crop((x,y,r,b))
        if occupied.histogram()[0] < .75 * occupied.width * occupied.height:
            report['skipped'].append([i,'already_covered'])
            continue
        if len(original)+len(added) >= 512:
            report['skipped'].append([i,'page_capacity'])
            continue
        identity = next_id+len(added)
        box = Text_Box(identity,0.0,[(x,y),(r-1,y),(r-1,b-1),(x,b-1)],source,review=1)
        box.reason = 'frontier_discovery'
        box.selected_engine = OCREngine.FRONTIER
        box.ocr_metadata = {'candidates':{'frontier':source},'frontier_discovery':{'model':model,'proposal':row}}
        added.append(box)
        draw.rectangle((x,y,r-1,b-1), fill=1)
        report['added_ids'].append(identity)

    if added:
        combined = [*original,*added]
        batch = propose_page_from_boxes(combined,str(image),complementary_ownership=True,regroup_companions=True)
        engine.raw_detections = [combined]
        engine.automatic_unit_proposals = [batch]
    return report
