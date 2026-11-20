"""Replay visually selected frontier discoveries through the existing renderer.

The input is an existing rendered page and its original OCR preview. Rendering
only the new regions isolates the effect from the original translation run.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tralens.ocr.automatic_units import propose_page_from_boxes
from tralens.ocr.recognition_units import image_fingerprint
from tralens.ocr.types import Text_Box
from tralens.rendering.automatic import run_automatic
from tralens.translation import coverage, luna
from tralens.translation.contracts import TranslationRequest, automatic_units
from tralens.translation.workflow import _raw_box, _snapshot


def render(args):
    proposals = json.loads(args.proposals.read_bytes())
    checkpoint = Path(proposals['preview'])
    original = json.loads(checkpoint.read_bytes())['input']
    baseline = (checkpoint.parents[1]/'rendering/composite.png').resolve(strict=True)
    manifest = json.loads(baseline.with_name('manifest.json').read_bytes())
    if proposals['image_sha256'] != original['sha256'] or manifest['source'] != original:
        raise ValueError('proposals and baseline must belong to the same page')
    if len(set(args.region)) != len(args.region):
        raise ValueError('duplicate proposal index')
    with Image.open(baseline) as source:
        image = source.convert('RGB')
    boxes = []
    for i in args.region:
        if not 0 <= i < len(proposals['regions']):
            raise ValueError('unknown proposal index')
        row = proposals['regions'][i]
        x,y,r,b = [round(z*image.size[j%2]/1000) for j,z in enumerate(row['box'])]
        if not (0 <= x < r <= image.width and 0 <= y < b <= image.height and row['english'].strip()):
            raise ValueError('invalid proposal bounds or empty lettering')
        polygon = np.array([(x,y),(r,y),(r,b),(x,b)],np.float32)
        box = Text_Box(i,0.0,polygon,row['source'],review=1)
        box.reason = 'visually_reviewed_frontier_proposal'
        boxes.append(box)
    batch = propose_page_from_boxes(boxes,str(baseline),complementary_ownership=True,regroup_companions=True)
    records = []
    for unit in automatic_units(batch):
        if len(unit.member_ids) != 1:
            raise ValueError('selected regions regrouped; evaluate them separately')
        i = unit.member_ids[0]
        target = unit
        kind = 'strict_utterance'
        record = dict(unit=asdict(unit),unit_revision=unit.revision,binding_current=True,fragment_previews=[])
        if unit.disposition == 'blocked':
            target = coverage.derive(unit)
            kind = coverage.KIND
            record.update(coverage=coverage.provenance(unit),translation_input=asdict(target))
        request = TranslationRequest((target,),(target.unit_id,))
        decision = dict(**proposals['regions'][i],reason='Selected for supplemental discovery trial')
        result = luna.result_for(request,decision,'page-discovery',kind,model=proposals['model'])
        record.update(request=asdict(request),request_revision=request.revision,input_kind=kind,
                      status=result.status,result=asdict(result))
        records.append(record)
    source = dict(path=str(baseline),size=list(image.size),sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                  pixels_sha256=image_fingerprint(image))
    args.output.mkdir(exist_ok=False,parents=False)
    preview = args.output/'supplement.json'
    doc = dict(schema_version=4,preview_only=True,input=source,status='complete',units=records,
               discovery=dict(proposals=str(args.proposals.resolve()),selected_indices=args.region),
               settings=dict(model=proposals['model'],translation_policy='coverage-first'),
               ocr=dict(batches=[batch],raw_detections=[[_raw_box(b) for b in boxes]],automatic_recognition_results=[]))
    preview.write_text(json.dumps(_snapshot(doc),ensure_ascii=False))
    return run_automatic(preview=preview,image=baseline,fonts=args.fonts,output=args.output/'rendering',
                         font_id=args.font_id,text_mask_model=args.text_mask_model,replacement_policy='all',
                         visibility_policy='always-show',removal_policy='aggressive',reconstruction_policy='lama',
                         reconstruction_config=args.reconstruction_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('proposals',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--region',type=int,action='append',required=True,help='visually checked zero-based proposal index')
    parser.add_argument('--fonts',type=Path,required=True)
    parser.add_argument('--font-id',required=True)
    parser.add_argument('--text-mask-model',type=Path,required=True)
    parser.add_argument('--reconstruction-config',type=Path,required=True)
    render(parser.parse_args())
