"""Freeze a native-scale, fixed-commit context matrix; explicit approval precedes run.

No acquisition, OCR or translation. Case IDs/categories are evaluation inputs only.
Identical mask controls reuse a forward, never consume duplicate inference budget.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root,cases,config):
    import numpy as np
    from PIL import Image,ImageDraw
    from tralens.rendering import reconstruction as rec,preparation as p
    root.mkdir();spec=rec.configuration(config);requests=[];records=[]
    selection=read(cases)
    if len(selection)!=8:raise ValueError('exactly eight preselected categories required')
    for index,case in enumerate(selection):
        folder=Path(case['folder']);out=root/f'case-{index:02d}';out.mkdir()
        if case.get('synthetic'):
            source=Image.open(folder/'source.png').convert('RGB');mask=Image.open(folder/'attempt-1/mask.png').convert('L')
            binding=read(folder/'attempt-1/request.json')['binding'];raw=[]
            protected=np.zeros((source.height,source.width),bool)
            source_doc={'size':list(source.size),'sha256':digest(folder/'source.png')}
            references={'source.png':folder/'source.png','reference.png':folder/'clean.png'}
        else:
            manifest=read(folder/'manifest.json');row=manifest['units'][case['row']]
            for name,value in manifest['artifacts'].items():
                if digest(folder/name)!=value:raise ValueError('baseline artifact drift: '+name)
            source=Image.open(folder/'source.png').convert('RGB');binding=row['binding'];source_doc=manifest['source']
            raw=read(folder/'evidence-preview.json')['ocr']['raw_detections'][0]
            protected=(np.asarray(Image.open(folder/'erase-mask.png'))>0)|(np.asarray(Image.open(folder/'text.png'))[:,:,3]>0)
            recovery=row.get('reconstruction',{})
            if recovery.get('status')=='composed_unverified':
                mask=Image.new('L',source.size);bounds=recovery['bounds']
                local=Image.open(folder/f'recovery-{case["row"]:03d}-{recovery["selected_attempt"]}-commit-mask.png')
                mask.paste(local,bounds[:2])
                lettering=Image.new('RGBA',source.size)
                lettering.paste(Image.open(folder/f'recovery-{case["row"]:03d}-text.png'),bounds[:2])
            else:
                mask=Image.open(folder/f'unit-{case["row"]:03d}-applied-erase.png').convert('L')
                lettering=Image.open(folder/f'unit-{case["row"]:03d}-text.png')
            protected &= ~((np.asarray(mask)>0)|(np.asarray(lettering)[:,:,3]>0))
            references={name:folder/name for name in ('source.png','clean.png','text.png','patch.png','composite.png','manifest.json')}
        for name,path in references.items():(out/name).write_bytes(path.read_bytes())
        fixed=np.asarray(mask)>0
        if not fixed.any() or (fixed&protected).any():raise ValueError('invalid fixed support')
        mask.save(out/'fixed-commit.png')
        Image.fromarray(protected.astype(np.uint8)*255).save(out/'protected.png')
        neighbor=Image.new('L',source.size)
        members={(type(v).__name__,v) for v in binding['member_ids']}
        for detection in raw:
            if (type(detection['source_id']).__name__,detection['source_id']) not in members:
                ImageDraw.Draw(neighbor).polygon([tuple(point) for point in detection['polygon']],fill=255)
        neighbor=p.dilate(np.asarray(neighbor)>0,3)
        left,top,right,bottom=mask.getbbox()
        record={**case,'index':index,'source':source_doc,'binding':binding,'fixed_commit_pixels':int(fixed.sum()),
            'prior_edit_pixels':int(protected.sum()),'references':{name:digest(out/name) for name in references}}
        records.append(record)
        for extent in (16,48):
            bounds=[max(0,left-extent),max(0,top-extent),min(source.width,right+extent),min(source.height,bottom+extent)]
            x,y,r,b=bounds
            if max(r-x,b-y)>512:raise ValueError('preflight crop exclusion: adjust matrix before freezing, never resize')
            crop=np.asarray(source.crop(bounds)).copy();commit=fixed[y:b,x:r];protection=protected[y:b,x:r]
            for policy in ('target_only','hide_neighbors'):
                inference=commit.copy() if policy=='target_only' else commit|neighbor[y:b,x:r]
                req=rec.request(source_doc,binding,bounds,crop,inference,commit,protection,attempt=0,model=rec.model_identity(spec))
                name=f'case-{index:02d}-{extent}-{policy}';destination=root/name;destination.mkdir()
                for label,values in (('input',crop),('inference',inference),('commit',commit),('protected',protection)):
                    Image.fromarray(values if label=='input' else values.astype(np.uint8)*255).save(destination/(label+'.png'))
                write(destination/'request.json',req)
                # Deduplicate only exactly identical same-case/context inputs.
                reuse=next((v['name'] for v in requests if v['revision']==req['revision']),None)
                requests.append({'name':name,'case':index,'extent':extent,'policy':policy,'revision':req['revision'],
                    'reuse':reuse,'bounds':bounds,'scale':[1,1],'padding_bottom_right':[512-(b-y),512-(r-x)],
                    'added_inference_pixels':int((inference&~commit).sum())})
    code={str(path.relative_to(ROOT)):digest(path) for path in
        [Path(__file__),ROOT/'src/tralens/rendering/reconstruction.py',ROOT/'src/tralens/rendering/reconstruction_worker.py']}
    files={str(path.relative_to(root)):digest(path) for path in root.rglob('*') if path.is_file()}
    write(root/'plan.json',{'authorization':'requires separate explicit user allocation; this file grants none',
        'maximum_forwards':32,'actual_planned_forwards':sum(v['reuse'] is None for v in requests),
        'maximum_seconds_per_forward':30,'automatic_retries':0,'model':spec,'cases':records,'requests':requests,
        'files':files,'code':code,'criteria_document':'docs/production-best-effort-v2-plan.md',
        'scope':'reused local pages and one constructed tone control; no untouched-title claim'})


def run(root,config,limit):
    import numpy as np
    from PIL import Image
    from tralens.rendering import reconstruction as rec
    from tralens.scheduling import inference_lease
    plan=read(root/'plan.json')
    if limit is None or not plan['actual_planned_forwards']<=limit<=32:raise ValueError('explicit bounded allocation required')
    if rec.configuration(config)!=plan['model']:raise ValueError('model/config drift')
    for name,value in plan['files'].items():
        if digest(root/name)!=value:raise ValueError('frozen matrix drift: '+name)
    for name,value in plan['code'].items():
        if digest(ROOT/name)!=value:raise ValueError('frozen code drift: '+name)
    output=root/'outputs';output.mkdir();backend=rec.LocalLama(plan['model'])
    with inference_lease():
        for row in plan['requests']:
            src=root/row['name'];dest=output/row['name'];dest.mkdir()
            if row['reuse']:
                write(dest/'reuse.json',{'reused_forward':row['reuse'],'request_revision':row['revision']});continue
            req=read(src/'request.json')
            with Image.open(src/'input.png') as im:crop=np.asarray(im).copy()
            masks=[]
            for name in ('inference','commit','protected'):
                with Image.open(src/(name+'.png')) as im:masks.append(np.asarray(im)>0)
            # Durable dispatch precedes inference; restarting refuses this output root.
            from tralens.storage import atomic_bytes
            atomic_bytes(dest/'dispatch.json',(json.dumps(row)+'\n').encode())
            try:
                result=backend.reconstruct(req,crop,*masks,timeout=30)
                plate=rec.compose(req,crop,*masks,result);Image.fromarray(plate).save(dest/'cleanplate.png')
                metadata={key:value for key,value in result.items() if key!='pixels'}
                metadata['outside_commit_changed']=int(np.any(plate!=crop,axis=2)[~masks[1]].sum())
                reference=root/f'case-{row["case"]:02d}'
                if (reference/'clean.png').exists():
                    page=Image.open(reference/'clean.png').convert('RGB');pixels=np.asarray(page).copy()
                    x,y,r,b=req['bounds'];pixels[y:b,x:r][masks[1]]=plate[masks[1]]
                    final=Image.alpha_composite(Image.fromarray(pixels).convert('RGBA'),Image.open(reference/'patch.png'))
                    final=Image.alpha_composite(final,Image.open(reference/'text.png')).convert('RGB');final.save(dest/'final-page.png')
                    before=np.asarray(Image.open(reference/'composite.png'));support=np.asarray(Image.open(reference/'fixed-commit.png'))>0
                    metadata['outside_commit_final_changed']=int(np.any(np.asarray(final)!=before,axis=2)[~support].sum())
                    if metadata['outside_commit_final_changed']:raise rec.BindingError('final page locality failure')
                write(dest/'result.json',metadata)
            except Exception as error:
                write(dest/'failure.json',{'type':type(error).__name__,'message':str(error)})
                raise  # Failure is retained; no implicit retries/extra allocation.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--cases',type=Path);parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--run',action='store_true');parser.add_argument('--approved-forward-limit',type=int)
    args=parser.parse_args()
    if args.freeze and not args.run:freeze(args.root,args.cases,args.config)
    elif args.run and not args.freeze:run(args.root,args.config,args.approved_forward_limit)
    else:parser.error('choose freeze or run')


if __name__=='__main__':main()
