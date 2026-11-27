"""Discover missing page lettering before frontier translation and rendering."""
from . import frontier
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw


def propose(image, polygons, model, timeout, *, repeated=()):
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
                  'existing reading. Return ONLY visible Japanese lettering outside the existing polygons. '
                  'For partially detected phrases report only separate uncovered characters if independently meaningful. '
                  'Exclude furigana repeating a detected word, pure punctuation, Latin lettering, motion lines, '
                  'and drawings. Empty regions is a valid result. Give exact Japanese transcription, concise '
                  'English lettering, landmark and tight [left,top,right,bottom] box in normalized 0..1000 '
                  'page coordinates. When the same glyph repeats as separate beats across artwork, give '
                  'each visible occurrence its own tight box. Keep the gaps between beats outside the boxes. '
                  'Scan every panel in reading order and double-check the bounds. No tools.')
        if repeated:
            prompt += (' Follow the visible trail of these repeated characters: '+', '.join(sorted(repeated))+
                       '. Inspect gaps between consecutive outlines, especially where a character crosses '
                       'clothing or changes direction. Return only an additional visible occurrence.')
        result = frontier.infer(root, model, prompt, [image, outline], schema, timeout)
    return result['regions']


def extend(engine, image, size, model, timeout, *, repeated=()):
    """Rebuild one coherent batch so grouping sees the old and new lettering together."""
    import re
    from ..ocr.automatic_units import propose_page_from_boxes
    from ..ocr.types import OCREngine, Text_Box

    original = engine.raw_detections[0]
    unread = {box.box_id:box for box in original if not box.selected_text.strip()}
    polygons = [box.paddle_poly.tolist() for box in original if box.box_id not in unread]
    proposals = propose(image, polygons, model, timeout, repeated=repeated) if repeated else propose(image, polygons, model, timeout)
    if not isinstance(proposals, list):
        raise ValueError('invalid frontier discovery response')
    counts = Counter(row['source'] for row in proposals if isinstance(row,dict)
                     and isinstance(row.get('source'),str) and re.fullmatch(r'[ぁ-ゖァ-ヺ]',row['source']))
    trail = set(repeated) | {word for word,count in counts.items() if 4<=count<=16}

    covered = Image.new('1', size)
    draw = ImageDraw.Draw(covered)
    for polygon in polygons:
        draw.polygon(list(map(tuple, polygon)), fill=1)

    added = []
    recovered = set()
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
        if repeated and source not in repeated:
            report['skipped'].append([i,'outside_repeated_sequence'])
            continue
        x,y,r,b = [round(n*size[j%2]/1000) for j,n in enumerate(coords)]
        if r-x<2 or b-y<2 or (r-x)*(b-y)>size[0]*size[1]//4:
            report['skipped'].append([i,'invalid_extent'])
            continue
        occupied = covered.crop((x,y,r,b))
        free = occupied.histogram()[0] / (occupied.width * occupied.height)
        if free < .75:
            # Two angled beats can have intersecting box corners while their
            # centers and most of both glyphs remain separate.
            adjacent = source in trail and free >= .60 and not covered.getpixel(((x+r)//2,(y+b)//2))
            if adjacent:
                for other in (*original,*added):
                    left,top,right,bottom = other.bbox
                    if (other.box_id in unread or right<x or left>=r or bottom<y or top>=b):
                        continue
                    if (other.reason!='frontier_discovery' or other.selected_text!=source
                            or (x<=(left+right)/2<r and y<=(top+bottom)/2<b)):
                        adjacent = False
                        break
        else:
            adjacent = True
        if not adjacent:
            report['skipped'].append([i,'already_covered'])
            continue
        # An unread detector footprint and a new reading of the same ink must
        # not become two overlapping text units.
        matches = []
        for key,empty in unread.items():
            if key in recovered:
                continue
            left,top,right,bottom = empty.bbox
            overlap = max(0,min(r,right+1)-max(x,left))*max(0,min(b,bottom+1)-max(y,top))
            if overlap >= .65*(r-x)*(b-y) and overlap >= .45*(right+1-left)*(bottom+1-top):
                matches.append(key)
        if len(matches)>1:
            report['skipped'].append([i,'ambiguous_empty_detections'])
            continue
        if not matches and len(original)-len(recovered)+len(added) >= 512:
            report['skipped'].append([i,'page_capacity'])
            continue
        identity = next_id+len(added)
        box = Text_Box(identity,0.0,[(x,y),(r-1,y),(r-1,b-1),(x,b-1)],source,review=1)
        box.reason = 'frontier_discovery'
        box.selected_engine = OCREngine.FRONTIER
        box.ocr_metadata = {'candidates':{'frontier':source},'frontier_discovery':{'model':model,'proposal':row}}
        if matches:
            prior = unread[matches[0]]
            box.ocr_metadata['empty_detection'] = {'source_id':prior.box_id,'text':prior.paddle_text,
                'polygon':prior.paddle_poly.tolist(),'reason':prior.reason,'confidence':float(prior.paddle_confidence)}
            recovered.add(prior.box_id)
            report.setdefault('recovered_ids',[]).append(prior.box_id)
        added.append(box)
        report['added_ids'].append(identity)
        draw.rectangle((x,y,r-1,b-1), fill=1)

    if added:
        counts = Counter(box.selected_text for box in added if re.fullmatch(r'[ぁ-ゖァ-ヺ]', box.selected_text))
        report['repeated_kana'] = [word for word,count in counts.items() if 4<=count<=16]
        combined = [box for box in original if box.box_id not in recovered] + added
        batch = propose_page_from_boxes(combined,str(image),complementary_ownership=True,regroup_companions=True)
        engine.raw_detections = [combined]
        engine.automatic_unit_proposals = [batch]
    return report
