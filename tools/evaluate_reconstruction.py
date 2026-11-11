"""Explicit local reconstruction evaluation; --freeze must precede --run.

Saved source-bound previews only; no OCR/translation calls or model acquisition.
Evaluation annotations and synthetic words never enter production routing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def freeze(root,cases,config):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from tralens.rendering import reconstruction as rec
    root.mkdir(parents=True,exist_ok=True)
    if (root/'evaluation-plan.json').exists():raise ValueError('already frozen')
    frozen=[]
    for case in json.loads(cases.read_bytes()):
        baseline=Path(case['baseline']);out=root/'frozen'/case['name'];out.mkdir(parents=True)
        files={'preview.json':baseline/'evidence-preview.json','fonts.json':baseline/'evidence-fonts.json',
            'baseline-manifest.json':baseline/'manifest.json','baseline-clean.png':baseline/'clean.png',
            'baseline-text.png':baseline/'text.png','baseline-erase.png':baseline/'erase-mask.png',
            'baseline-composite.png':baseline/'composite.png','original.png':Path(case['original_path'])}
        for name,path in files.items():(out/name).write_bytes(path.read_bytes())
        frozen.append({'name':case['name'],'original_path':case['original_path'],
            'files':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files}})
    synthetic=[]
    font_doc=json.loads((root/'frozen'/frozen[0]['name']/'fonts.json').read_bytes())
    font=ImageFont.truetype(font_doc['fonts'][0]['path'],32)
    for kind in ('flat','lines','tone','colored'):
        out=root/'synthetic'/kind;out.mkdir(parents=True)
        clean=Image.new('RGB',(320,256),'white');draw=ImageDraw.Draw(clean)
        if kind=='lines':
            for offset in range(-200,300,24):draw.line((0,offset,320,offset+210),fill=(35,35,35),width=3)
            draw.ellipse((45,15,280,240),outline=(60,60,60),width=4)
        if kind=='tone':
            for y in range(0,256,6):
                for x in range(0,320,6):draw.ellipse((x,y,x+2,y+2),fill=(90,90,90))
        if kind=='colored':
            a=np.asarray(clean).copy()
            for y in range(256):a[y,:]=(80+y//2,120+y//3,180-y//4)
            clean=Image.fromarray(a);draw=ImageDraw.Draw(clean)
            draw.line((10,230,290,20),fill=(30,40,50),width=6)
        overlay=Image.new('RGBA',clean.size);d=ImageDraw.Draw(overlay)
        d.text((92,82),'Outline!?',font=font,fill=(150,20,45,255) if kind=='colored' else (0,0,0,255),stroke_width=3,stroke_fill=(255,255,255,255))
        source=Image.alpha_composite(clean.convert('RGBA'),overlay).convert('RGB')
        alpha=np.asarray(overlay.getchannel('A'))>0
        box=overlay.getchannel('A').getbbox();x,y,r,b=box
        own=np.zeros(alpha.shape,bool);own[y+3:b-3,x+3:r-3]=True
        for name,image in (('clean',clean),('source',source),('alpha',Image.fromarray(alpha.astype(np.uint8)*255)),('geometry',Image.fromarray(own.astype(np.uint8)*255))):image.save(out/(name+'.png'))
        synthetic.append({'name':kind,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}})
    plan={'scope':'reused saved pages, provisional visual assessment; not independent bilingual or untouched-title confirmation',
        'criteria_document':'docs/production-best-effort-plan.md','model':rec.configuration(config),
        'maximum_learned_forwards':64,'synthetic_calls':8,'maximum_page_calls':8,
        'cases':frozen,'synthetic':synthetic,'primary_words_changed':False}
    save(root/'evaluation-plan.json',plan)


def run(root,config,stage):
    import numpy as np
    from PIL import Image
    from tralens.rendering import reconstruction as rec,aggressive,automatic
    plan=json.loads((root/'evaluation-plan.json').read_bytes())
    if rec.configuration(config)!=plan['model']:raise ValueError('frozen model/config drift')
    backend=rec.LocalLama(plan['model'])
    for group,folder in (('cases','frozen'),('synthetic','synthetic')):
        for case in plan[group]:
            for name,digest in case['files'].items():
                if hashlib.sha256((root/folder/case['name']/name).read_bytes()).hexdigest()!=digest:raise ValueError('frozen evidence drift')
    results=[]
    if stage=='synthetic':
        for case in plan['synthetic']:
            folder=root/'synthetic'/case['name'];crop=np.asarray(Image.open(folder/'source.png')).copy()
            truth=np.asarray(Image.open(folder/'clean.png'));alpha=np.asarray(Image.open(folder/'alpha.png'))>0
            own=np.asarray(Image.open(folder/'geometry.png'))>0;protected=np.zeros_like(own)
            evidence,ladder,_=aggressive.masks(crop,own,protected)
            for index,(name,mask) in enumerate(ladder):
                destination=folder/f'attempt-{index}';destination.mkdir()
                req=rec.request({'size':[320,256],'sha256':case['files']['source.png']},
                    {'member_ids':['synthetic'],'request_revision':case['files']['geometry.png'],'text':'Outline!?'},
                    [0,0,320,256],crop,mask,mask,protected,attempt=index)
                save(destination/'request.json',req)
                candidate,cheap=aggressive.cheap(crop,mask,protected)
                candidates={cheap:candidate}
                clock=time.monotonic()
                try:
                    value=backend.reconstruct(req,crop,mask,mask,protected)
                    candidates['lama']=rec.compose(req,crop,mask,mask,protected,value)
                    cost={k:v for k,v in value.items() if k!='pixels'}
                except Exception as error:cost={'error':type(error).__name__+': '+str(error)}
                metrics={}
                for method,pixels in candidates.items():
                    pixels=rec.compose(req,crop,mask,mask,protected,{'request_revision':req['revision'],'pixels':pixels})
                    Image.fromarray(pixels).save(destination/(method+'.png'))
                    metrics[method]={'mask_mae':float(np.abs(pixels.astype(float)-truth)[mask].mean()),
                        'known_lettering_mae':float(np.abs(pixels.astype(float)-truth)[alpha].mean()),
                        'collateral_art_mae':float(np.abs(pixels.astype(float)-truth)[mask&~alpha].mean()),
                        'outside_changed':int(np.any(pixels!=crop,axis=2)[~mask].sum()),
                        'assessment':aggressive.assess(crop,pixels,mask,evidence,protected)}
                Image.fromarray(mask.astype(np.uint8)*255).save(destination/'mask.png')
                row={'case':case['name'],'mask':name,'known_alpha_mask_coverage':float(mask[alpha].mean()),
                    'seconds':time.monotonic()-clock,'metrics':metrics,'cost':cost}
                save(destination/'result.json',row);results.append(row);print(case['name'],index,metrics,flush=True)
        save(root/'synthetic-results.json',results)
    else:
        for case in plan['cases']:
            folder=root/'frozen'/case['name'];out=root/'pages'/case['name'];out.mkdir(parents=True,exist_ok=True)
            baseline=json.loads((folder/'baseline-manifest.json').read_bytes())
            common=dict(preview=folder/'preview.json',image=case['original_path'],fonts=folder/'fonts.json',
                font_id=baseline['font_id'],min_size=baseline['style']['min_size'],max_size=baseline['style']['max_size'],
                replacement_policy='confidence-gated',visibility_policy='always-show',removal_policy='aggressive',
                layout_policy=baseline.get('layout_policy','source-local'),cleanup_policy=baseline.get('cleanup_policy','native'))
            for policy in ('classical','lama'):
                clock=time.monotonic()
                report=automatic.run_automatic(**common,output=out/policy,reconstruction_policy=policy,
                    reconstruction_config=config if policy=='lama' else None)
                row={'case':case['name'],'policy':policy,'coverage':report['coverage'],'reconstruction':report['reconstruction_summary'],
                    'display':report['display_summary'],'seconds':time.monotonic()-clock,'pixel_audit':report['pixel_audit']}
                results.append(row);save(root/'page-results.json',results);print(case['name'],policy,row['reconstruction'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--cases',type=Path)
    p.add_argument('--freeze',action='store_true');p.add_argument('--run',choices=('synthetic','pages'))
    args=p.parse_args()
    if args.freeze:freeze(args.root,args.cases,args.config)
    elif args.run:run(args.root,args.config,args.run)
    else:p.error('choose --freeze or --run')


if __name__=='__main__':main()
