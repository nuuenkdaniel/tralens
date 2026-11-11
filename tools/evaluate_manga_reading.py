"""Read isolated saved dialogue with the existing local Manga OCR checkpoint.

Raw OCR, crops, model identity, timings and alternative readings are retained.
This diagnostic does not promote candidates or use translation fluency as truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,default=ROOT/'test-results/production-best-effort-v1/frozen')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--maximum',type=int,default=48)
    args=p.parse_args()
    from PIL import Image
    from tralens.translation.contracts import automatic_units
    from tralens.ocr.automatic_recognition import script_evidence
    from tralens.scheduling import inference_lease
    options=[]
    for path in sorted(args.frozen.glob('*/preview.json')):
        doc=json.loads(path.read_bytes());batch=doc['ocr']['batches'][0]
        for proposal,unit in zip(batch['proposals'],automatic_units(batch)):
            if (len(unit.selected.text)<4 or script_evidence(unit.selected.text)[1]
                    or not proposal['crop_safe'] or proposal['direction'] not in ('vertical','horizontal')):continue
            options.append({'page':path.parent.name,'source':unit.selected.text,'unit_id':unit.unit_id,
                'bounds':proposal['crop_bounds'],'members':proposal['main_ids'],'source_revision':unit.revision,
                'image':str(path.parent/'original.png')})
    # Round-robin pages so controls are not all from the first chapter.
    pages={e['page']:[] for e in options}
    for e in options:pages[e['page']].append(e)
    jobs=[]
    while any(pages.values()) and len(jobs)<args.maximum:
        for items in pages.values():
            if items and len(jobs)<args.maximum:jobs.append(items.pop(0))
    args.output.mkdir(parents=True,exist_ok=False)
    identity={name:hashlib.sha256((args.model/name).read_bytes()).hexdigest() for name in ('config.json','pytorch_model.bin','vocab.txt')}
    (args.output/'plan.json').write_text(json.dumps({'jobs':jobs,'model':str(args.model),'digests':identity},ensure_ascii=False,indent=2))
    results=[]
    with inference_lease():
        import torch
        torch.set_num_threads(2)
        from tralens.ocr.manga_adapter import MangaOCRAdapter
        start=time.monotonic();reader=MangaOCRAdapter(args.model)
        load=time.monotonic()-start
        for index,item in enumerate(jobs):
            image=Image.open(item['image']).convert('RGB');crop=image.crop(item['bounds'])
            crop.save(args.output/f'{index:03d}.png')
            start=time.monotonic();value=reader(crop,source_box_id=item['unit_id'],crop_bounds=item['bounds'])
            row={**item,'manga':value,'seconds':time.monotonic()-start,'load_seconds':load}
            results.append(row)
            (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
            print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
