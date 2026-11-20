"""Independent experimental language-review queue and bound source candidates.

Plan-only by default. No rendering/cleanup changes or automatic candidate promotion.
"""
import argparse
from pathlib import Path
import sys


class IdleGuard:
    """Fail closed on viewer work or unrelated model residency; never evict it.

    This is observation, not a server-wide inference lock. External clients using
    the same model cannot be distinguished by Ollama's residency API.
    """
    def __init__(self,endpoint):
        self.endpoint=endpoint.rstrip('/')
        self.started=False

    def __call__(self):
        import json
        from urllib.request import build_opener,ProxyHandler
        opener=build_opener(ProxyHandler({}))
        def get(url):
            with opener.open(url,timeout=10) as response:return json.load(response)
        jobs=get('http://127.0.0.1:7860/api/jobs')
        if not isinstance(jobs,list) or any(j['status']=='running' for j in jobs):
            raise RuntimeError('active viewer inference; optional language review stopped')
        loaded={m['name'] for m in get(self.endpoint+'/api/ps')['models']}
        permitted={'qwen3.8:latest','translategemma:27b'} if self.started else set()
        if loaded-permitted:
            raise RuntimeError('occupied model server; do not interrupt: '+str(sorted(loaded)))
        self.started=True


def paddle_reader_factory(image,model_root):
    """Construct the explicitly installed crop reader only when first needed."""
    def factory():
        import cv2
        import numpy as np
        from tralens.ocr import OCR
        engine = OCR(str(image),device='cpu',model_root=str(model_root),model_profile='v6_medium',enable_manga_ocr=False)
        def read(crop):
            result = engine.ocr.predict(cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR),return_word_box=True)
            if len(result) != 1:raise ValueError('one independent original-crop page required')
            data = getattr(result[0],'res',result[0])
            texts,polys,scores = data['rec_texts'],data['rec_polys'],data['rec_scores']
            if not len(texts) == len(polys) == len(scores) or len(texts) > 32:
                raise ValueError('independent crop reader association/budget')
            records = [{'text':text,'polygon':poly.tolist() if hasattr(poly,'tolist') else poly,
                'confidence':float(score)} for text,poly,score in zip(texts,polys,scores)]
            # Crops here are a single proposed original text line, not a group.
            # Retain every detection and deterministic geometric order, not a
            # substring chosen to match Qwen. Mixed/neighbour output disagrees.
            vertical = crop.height > 1.3*crop.width
            records.sort(key=lambda r: (-sum(p[0] for p in r['polygon'])/len(r['polygon']),
                min(p[1] for p in r['polygon'])) if vertical else
                (min(p[1] for p in r['polygon']),min(p[0] for p in r['polygon'])))
            return {'text':''.join(r['text'] for r in records),'records':records,
                'model_identity':engine.paddle_provenance,'order':'geometric, one-line hypothesis; not verified'}
        return read,engine.ocr.close
    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preview','image','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--run-local-model',action='store_true')
    parser.add_argument('--model-root',type=Path,help='existing local Paddle v6 medium directory; no acquisition')
    parser.add_argument('--endpoint',default='http://100.121.133.88:11434')
    parser.add_argument('--max-reviews',type=int,default=4)
    parser.add_argument('--max-reader-calls',type=int,default=8)
    parser.add_argument('--max-translation-calls',type=int,default=8)
    args=parser.parse_args(argv)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from main import run_language_review
    from tralens.translation.runtime import local_endpoint
    idle=IdleGuard(local_endpoint(args.endpoint))
    if args.run_local_model:idle()
    factory = paddle_reader_factory(args.image,args.model_root.expanduser().resolve(strict=True)) if args.model_root else None
    settings=vars(args);settings.pop('model_root')
    state=run_language_review(**settings,reader_factory=factory,idle_check=idle)
    print(state['status'],state['counts'])


if __name__=='__main__':main()
