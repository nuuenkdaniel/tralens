"""Serial source-bound pilot over frozen cases, with GLOBAL 20/8/8 ceilings.

Uses only explicit existing local models. Default prepares immutable requests and
original/overlay crops without inference; --run-local-model executes a fresh run.
Cases are evaluation metadata, never production corrections.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from language_review import IdleGuard,paddle_reader_factory


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--model-root',required=True,type=Path)
    parser.add_argument('--endpoint',default='http://100.121.133.88:11434')
    parser.add_argument('--run-local-model',action='store_true')
    args=parser.parse_args(argv)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from main import run_language_review
    from tralens import language_review as lr
    from tralens.ocr import source_review as sr
    from tralens.translation.contracts import canonical
    from tralens.translation.runtime import local_endpoint
    from PIL import Image
    root=args.cases.resolve().parent
    cases=json.loads(args.cases.read_bytes())
    selected=[c for c in cases if c['inference']]
    if len(selected)>5:raise ValueError('pilot limited to five pages')
    model_root=args.model_root.expanduser().resolve(strict=True)
    if args.output.exists() or not args.output.parent.is_dir():raise ValueError('fresh output required')
    args.output.mkdir(mode=0o700)
    summary={'limits':{'source_calls':20,'reader_calls':8,'translation_calls':8},'pages':[],
        'calls':{'source_calls':0,'reader_calls':0,'translation_calls':0},'status':'running',
        'reference_status':'provisional agent inspection, not independently verified gold',
        'model_root':str(model_root),'cases_sha256':hashlib.sha256(args.cases.read_bytes()).hexdigest()}
    def save():(args.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    idle=IdleGuard(local_endpoint(args.endpoint))
    if args.run_local_model:idle()
    save()
    try:
        for case in selected:
            preview,image=root/case['preview'],root/case['image']
            output=args.output/case['name']
            if not args.run_local_model:
                doc=json.loads(preview.read_bytes())
                with Image.open(image) as original:rgb=original.convert('RGB')
                plan=lr.plan(doc,rgb,max_requests=4)
                output.mkdir()
                (output/'plan.json').write_text(canonical(plan))
                for i,uid in enumerate(plan['requests']):
                    request,native,view,overlay=sr.prepare(doc,plan,uid,rgb)
                    for name,data in [('request.json',canonical(request).encode()),('original.png',sr.png(native)),
                            ('presentation.png',sr.png(view)),('overlay.png',sr.png(overlay))]:
                        (output/f'{i:03d}-{name}').write_bytes(data)
                summary['pages'].append({'name':case['name'],'status':'frozen_requests_no_inference'})
            else:
                calls=summary['calls']
                state=run_language_review(preview=preview,image=image,output=output,run_local_model=True,
                    endpoint=args.endpoint,max_reviews=min(4,20-calls['source_calls']),
                    max_reader_calls=8-calls['reader_calls'],max_translation_calls=8-calls['translation_calls'],
                    reader_factory=paddle_reader_factory(image,model_root),idle_check=idle)
                counts=state['counts']
                calls['source_calls']+=counts['attempted'];calls['reader_calls']+=counts['corroborating_reader_calls']
                calls['translation_calls']+=counts['translation_calls']
                summary['pages'].append({'name':case['name'],'status':state['status'],'counts':counts})
            save()
            print(case['name'],summary['pages'][-1],flush=True)
        summary['status']='complete_candidates_only' if args.run_local_model else 'frozen_no_inference'
    except BaseException as error:
        summary.update(status='stopped_no_automatic_retry',error=type(error).__name__+': '+str(error))
        raise
    finally:
        # Recover cost from persisted page ledgers even if a page stopped early.
        if args.run_local_model:
            ledgers=[json.loads(p.read_bytes()) for p in args.output.glob('*/review.json')]
            summary['calls']={name:sum(p['counts'][field] for p in ledgers) for name,field in
                [('source_calls','attempted'),('reader_calls','corroborating_reader_calls'),('translation_calls','translation_calls')]}
        save()


if __name__=='__main__':main()
