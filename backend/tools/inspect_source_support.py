"""Retain source geometry and explicit CTD evidence for saved render diagnostics."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--renders',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    args=p.parse_args()
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw
    from tralens.rendering.segmentation import TextMask,model_spec
    cv2.setNumThreads(4)
    args.output.mkdir(exist_ok=False)
    spec=model_spec(args.model)
    for old in sorted(args.renders.glob('*/manifest.json')):
        old=old.parent
        rgb=Image.open(old/'source.png').convert('RGB')
        preview=json.loads((old/'evidence-preview.json').read_bytes())
        manifest=json.loads((old/'manifest.json').read_bytes())
        output=args.output/old.name;output.mkdir()
        mask=TextMask(rgb,spec)
        Image.fromarray(mask.raw).save(output/'probability.png')
        overlay=np.asarray(rgb).copy();support=mask.raw>=77
        overlay[support]=(overlay[support].astype(float)*.45+np.array([255,0,80])*.55).astype('uint8')
        overlay=Image.fromarray(overlay);draw=ImageDraw.Draw(overlay);rows=[]
        for index,row in enumerate(manifest['units']):
            ids=row.get('binding',{}).get('member_ids',[])
            detections=[v for v in preview['ocr']['batches'][0]['detections'] if v['source_id'] in ids]
            for detection in detections:
                points=[tuple(pt) for pt in detection['polygon']]
                draw.line(points+[points[0]],fill='blue',width=2)
                draw.text(points[0],str(index),fill='red')
            rows.append({'index':index,'source':[v['text'] for v in detections],'members':ids,
                'target':row.get('binding',{}).get('text'),'method':row.get('cleanup_method'),
                'bounds':row.get('reconstruction',{}).get('bounds'),
                'final_bounds':row.get('removal_fallback',{}).get('bounds')})
        overlay.save(output/'overlay.png')
        (output/'evidence.json').write_text(json.dumps({'inference':mask.evidence,'rows':rows},ensure_ascii=False,indent=2))
        print(old.name,round(mask.evidence['seconds'],2),flush=True)


if __name__=='__main__':main()
