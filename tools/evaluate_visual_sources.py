"""Compare saved reader disagreements with the configured local visual reader.

All requests/results are retained; this diagnostic never selects source text.
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
    p.add_argument('--endpoint',required=True)
    args=p.parse_args()
    from PIL import Image
    from experimental import simple_transcription as st
    from tralens.ocr.manga_dialogue import normalized
    from tralens.translation.runtime import OllamaRuntime
    from tralens.scheduling import inference_lease
    jobs=[r for r in json.loads(args.readings.read_bytes())
        if any(normalized(s)!=normalized(r['source']) for s in r['readings'])]
    args.output.mkdir(exist_ok=False)
    plan=[]
    for i,job in enumerate(jobs):
        rgb=Image.open(job['source_image']).convert('RGB')
        request,crop=st.request(rgb,job['bounds'],source=job['source_binding'],
            member_ids=job['member_ids'],presentation='block')
        crop.save(args.output/f'{i:03d}.png')
        (args.output/f'{i:03d}-request.json').write_text(json.dumps(request,ensure_ascii=False))
        plan.append({**job,'request_revision':request['revision'],
            'request_file_sha256':hashlib.sha256((args.output/f'{i:03d}-request.json').read_bytes()).hexdigest()})
    (args.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    results=[]
    with inference_lease():
        runtime=OllamaRuntime(args.endpoint)
        try:
            for i,job in enumerate(plan):
                request=json.loads((args.output/f'{i:03d}-request.json').read_bytes())
                start=time.monotonic()
                try:
                    response=runtime.generate(model='qwen3.8:latest',payload=request['payload'],timeout=120)
                except Exception as error:
                    failure={'index':i,'request_revision':request['revision'],
                        'error':type(error).__name__+': '+str(error),'response_json':getattr(error,'response_json',None)}
                    (args.output/'failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2))
                    raise
                (args.output/f'{i:03d}-response.json').write_text(response.response_json)
                result={**job,'visual':st.parse(request,response.text),'seconds':time.monotonic()-start}
                results.append(result)
                (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
                print(job['page'],job['index'],repr(job['source']),'->',result['visual'],flush=True)
        finally:
            from tralens.scheduling import remote_state_path
            if not remote_state_path(runtime.endpoint).exists():
                runtime.unload(model='qwen3.8:latest',timeout=30)


if __name__=='__main__':main()
