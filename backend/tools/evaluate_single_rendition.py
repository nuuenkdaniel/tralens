"""Compare a single-rendition specialist instruction on saved source/English pairs.

Runs only with an explicit local endpoint and fresh output directory. Retains raw
requests/responses; does not select translations or rewrite historical previews.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

INSTRUCTION=('Give one natural translation suitable for dialogue, not a list of alternative phrasings. '
    'Preserve repetitions, hesitations, punctuation, and unfinished sentences. '
    'Do not add explanations, alternatives in parentheses, or words absent from the source. ')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,default=ROOT/'test-results/production-best-effort-v1/frozen')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--endpoint',required=True)
    parser.add_argument('--maximum',type=int,default=16)
    parser.add_argument('--controls',action='store_true')
    parser.add_argument('--minimal',action='store_true')
    args=parser.parse_args()
    instruction='Give exactly one translation, not alternative translations separated by slashes. ' if args.minimal else INSTRUCTION
    from tralens.translation.contracts import automatic_units
    from tralens.translation.runtime import OllamaRuntime,TRANSLATEGEMMA_PROMPT,GenerationOptions
    from tralens.scheduling import inference_lease
    examples={}
    for path in sorted(args.frozen.glob('*/preview.json')):
        doc=json.loads(path.read_bytes())
        for row,unit in zip(doc['units'],automatic_units(doc['ocr']['batches'][0],doc['ocr'].get('automatic_recognition_results',[[]])[0])):
            result=row.get('result') or {}
            if result.get('status')!='preview' or not result.get('translations'):continue
            target=result['translations'][0][1]
            examples.setdefault(unit.selected.text,{'source':unit.selected.text,'primary':target,'page':path.parent.name,
                'unit_id':unit.unit_id,'unit_revision':unit.revision})
    pool=[e for e in examples.values() if not args.controls or (' / ' not in e['primary'] and len(e['source'])>=8)]
    examples=sorted(pool,key=lambda e:(' / ' not in e['primary'],len(e['source']),e['source']))[:args.maximum]
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=OllamaRuntime(args.endpoint)
    (args.output/'plan.json').write_text(json.dumps({'instruction':instruction,'examples':examples},ensure_ascii=False,indent=2))
    results=[]
    with inference_lease():
        try:
            for i,item in enumerate(examples):
                prompt=TRANSLATEGEMMA_PROMPT.format(source_name='Japanese',source_code='ja',target_name='English',target_code='en',source_text=item['source'])
                prompt=prompt.replace('Please translate the following',instruction+'Please translate the following',1)
                payload={'messages':[{'role':'user','content':prompt}],**GenerationOptions(num_ctx=4096,num_predict=512,keep_alive=120).transport_fields()}
                (args.output/f'{i:03d}-request.json').write_text(json.dumps(payload,ensure_ascii=False))
                clock=time.monotonic()
                response=runtime.generate(model='translategemma:27b',payload=payload,timeout=180)
                (args.output/f'{i:03d}-response.json').write_text(response.response_json)
                results.append({**item,'candidate':response.text,'seconds':time.monotonic()-clock})
                (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
                print(json.dumps(results[-1],ensure_ascii=False),flush=True)
        finally:
            runtime.unload(model='translategemma:27b',timeout=30)


if __name__=='__main__':main()
