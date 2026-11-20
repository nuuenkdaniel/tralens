"""Reread retained disagreements using the existing local Paddle crop pipeline.

New predictions remain diagnostic evidence; full-page detections are immutable.
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
    p.add_argument('--readings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-root',type=Path,required=True)
    args=p.parse_args()
    import cv2
    import numpy as np
    from PIL import Image
    from paddle_model_benchmark import local_only
    from tralens.ocr.manga_dialogue import normalized
    from tralens.ocr.paddle_profiles import resolve_profile
    from tralens.ocr.detection import create_paddle_ocr
    from tralens.ocr.paddle_result import evidence_value
    from tralens.scheduling import inference_lease
    jobs=[r for r in json.loads(args.readings.read_bytes())
        if any(normalized(s)!=normalized(r['source']) for s in r['readings'])]
    args.output.mkdir(exist_ok=False)
    profile=resolve_profile('v6_medium',device='cpu',model_root=args.model_root,cpu_threads=2)
    (args.output/'plan.json').write_text(json.dumps({'jobs':jobs,'profile':profile.identity()},ensure_ascii=False,indent=2))
    results=[]
    with inference_lease(),local_only():
        engine=create_paddle_ocr(**profile.settings())
        try:
            for i,job in enumerate(jobs):
                native=Image.open(job['source_image']).convert('RGB').crop(job['bounds'])
                row={**job,'predictions':[]};results.append(row)
                for scale in (1,2):
                    view=native if scale==1 else native.resize((native.width*scale,native.height*scale),Image.Resampling.LANCZOS)
                    path=args.output/f'{i:03d}-{scale}.png';view.save(path)
                    clock=time.monotonic()
                    result=engine.predict(cv2.cvtColor(np.asarray(view),cv2.COLOR_RGB2BGR),return_word_box=True)
                    prediction=getattr(result[0],'res',result[0])
                    keep=evidence_value({k:prediction[k] for k in ('dt_polys','rec_texts','rec_scores','rec_polys','text_word_region')})
                    row['predictions'].append({'scale':scale,'seconds':time.monotonic()-clock,
                        'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'result':keep})
                (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
                print(job['page'],job['index'],repr(job['source']),'->',
                    [v['result']['rec_texts'] for v in row['predictions']],flush=True)
        finally:engine.close()


if __name__=='__main__':main()
