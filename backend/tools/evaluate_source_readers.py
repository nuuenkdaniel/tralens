"""Compare two local reader presentations on immutable saved source proposals.

Diagnostic only: includes unsafe crops to expose their errors, never promotes
agreement into production evidence or supplies expected readings to a model.
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
    p.add_argument('--renders',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    args=p.parse_args()
    from PIL import Image
    from tralens.ocr.manga_dialogue import presentations,normalized,valid_bounds
    from tralens.translation.contracts import automatic_units
    from tralens.scheduling import inference_lease
    jobs=[]
    for path in sorted(args.renders.glob('*/evidence-preview.json')):
        doc=json.loads(path.read_bytes());batch=doc['ocr']['batches'][0]
        raw={r['source_id']:r for r in batch['detections']}
        for i,(proposal,unit) in enumerate(zip(batch['proposals'],automatic_units(batch))):
            bounds=proposal['crop_bounds']
            if not valid_bounds(bounds,doc['input']['size']):continue
            if not any(c.isalnum() for c in unit.selected.text):continue
            jobs.append({'page':path.parent.name,'index':i,'source':unit.selected.text,
                'unit_id':unit.unit_id,'bounds':bounds,'member_ids':list(unit.member_ids),
                'scores':[raw[k]['confidence'] for k in unit.order],
                'safe':proposal['crop_safe'],'direction':proposal['direction'],
                'source_image':str(path.parent/'source.png'),'source_binding':doc['input'],
                'unit_revision':unit.revision})
    args.output.mkdir(exist_ok=False)
    identity={n:hashlib.sha256((args.model/n).read_bytes()).hexdigest()
        for n in ('config.json','pytorch_model.bin','vocab.txt')}
    (args.output/'plan.json').write_text(json.dumps({'jobs':jobs,'model':str(args.model),
        'model_digests':identity,'selection':'diagnostic only'},ensure_ascii=False,indent=2))
    results=[]
    with inference_lease():
        import torch
        torch.set_num_threads(2)
        from tralens.ocr.manga_adapter import MangaOCRAdapter
        reader=MangaOCRAdapter(args.model)
        for i,job in enumerate(jobs):
            image=Image.open(job['source_image']).convert('RGB')
            crops=presentations(image,job['bounds']);readings=[];hashes=[]
            clock=time.monotonic()
            for j,crop in enumerate(crops):
                name=f'{i:03d}-{j}.png';crop.save(args.output/name)
                hashes.append(hashlib.sha256((args.output/name).read_bytes()).hexdigest())
                readings.append(reader(crop,source_box_id=job['unit_id'],crop_bounds=job['bounds']))
            record={**job,'readings':readings,'crop_hashes':hashes,'seconds':time.monotonic()-clock,
                'agree':normalized(readings[0])==normalized(readings[1])}
            results.append(record)
            (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
            if any(normalized(v)!=normalized(job['source']) for v in readings):
                print(job['page'],job['index'],repr(job['source']),'->',readings,flush=True)


if __name__=='__main__':main()
