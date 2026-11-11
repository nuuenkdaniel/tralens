"""Offline verification and inspection sheets for a frozen context experiment."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image,ImageDraw,ImageFont


def read(path):return json.loads(path.read_bytes())
def pixels(path):
    with Image.open(path) as image:return np.asarray(image).copy()


def inspect(root,output_name='inspection'):
    if Path(output_name).name!=output_name or output_name in ('.','..'):raise ValueError('one fresh output directory name required')
    plan=read(root/'plan.json');destination=root/output_name;destination.mkdir()
    verified=0
    for name,digest in plan['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        verified+=1
    assert len(plan['requests'])==32 and plan['actual_planned_forwards']<=32
    by_name={row['name']:row for row in plan['requests']}
    outputs=root/'outputs';results=[];case_rows=[];slots=[];links=[]
    font=ImageFont.load_default(size=15)
    for case in plan['cases']:
        reference=root/f'case-{case["index"]:02d}'
        source=pixels(reference/'source.png');fixed=pixels(reference/'fixed-commit.png')>0
        protected=pixels(reference/'protected.png')>0
        native=not case.get('synthetic')
        clean=pixels(reference/'clean.png') if native else source.copy()
        baseline=pixels(reference/'composite.png') if native else source.copy()
        if native:
            manifest=read(reference/'manifest.json')
            assert manifest['units'][case['row']]['binding']==case['binding']
        for extent in (16,48):
            pair=[]
            for policy in ('target_only','hide_neighbors'):
                row=next(row for row in plan['requests'] if row['case']==case['index']
                    and row['extent']==extent and row['policy']==policy)
                folder=root/row['name'];request=read(folder/'request.json')
                out=outputs/(row['reuse'] or row['name'])
                if row['reuse']:
                    reuse=read(outputs/row['name']/'reuse.json')
                    assert reuse['request_revision']==row['revision']==by_name[row['reuse']]['revision']
                    assert reuse['reused_forward']==row['reuse']
                else:
                    assert read(out/'dispatch.json')==row
                metadata=read(out/'result.json');assert not (out/'failure.json').exists()
                assert metadata['request_revision']==request['revision']==row['revision']
                assert metadata['model_sha256']==plan['model']['sha256']
                assert metadata['providers']==['CPUExecutionProvider'] and metadata['wall_seconds']<=30
                crop=pixels(folder/'input.png');commit=pixels(folder/'commit.png')>0
                inference=pixels(folder/'inference.png')>0;local_protected=pixels(folder/'protected.png')>0
                x,y,r,b=row['bounds'];assert row['scale']==[1,1]
                assert row['padding_bottom_right']==[512-crop.shape[0],512-crop.shape[1]]
                assert np.array_equal(source[y:b,x:r],crop) and np.array_equal(fixed[y:b,x:r],commit)
                assert int(commit.sum())==int(fixed.sum()) and not (commit&local_protected).any()
                assert np.array_equal(local_protected,protected[y:b,x:r]) and not (commit&~inference).any()
                plate=pixels(out/'cleanplate.png');assert np.array_equal(plate[~commit],crop[~commit])
                page=clean.copy();page[y:b,x:r][commit]=plate[commit]
                assert np.array_equal(page[protected],clean[protected])
                final=Image.fromarray(page).convert('RGBA')
                if native:
                    for name in ('patch.png','text.png'):
                        with Image.open(reference/name) as image:final=Image.alpha_composite(final,image)
                final=np.asarray(final.convert('RGB'))
                assert np.array_equal(final[~fixed],baseline[~fixed])
                assert np.array_equal(final[protected],baseline[protected])
                if native:assert np.array_equal(final,pixels(out/'final-page.png'))
                Image.fromarray(page).save(destination/(row['name']+'-clean-page.png'))
                Image.fromarray(final).save(destination/(row['name']+'-final-page.png'))
                mask_view=crop.copy()
                mask_view[inference&~commit]=(mask_view[inference&~commit].astype(np.uint16)+[0,220,255])//2
                mask_view[commit]=(mask_view[commit].astype(np.uint16)+[255,0,0])//2
                measured={**row,'outside_commit_changed':0,'prior_edits_changed':0,
                    'composed_page_exact':True,'result':metadata}
                if case.get('synthetic'):
                    truth=pixels(reference/'reference.png')[y:b,x:r]
                    measured['known_reference_commit_mae']=float(np.abs(plate.astype(float)-truth)[commit].mean())
                slots.append(measured)
                if not row['reuse']:results.append(metadata)
                pair.append((row,crop,mask_view,plate,final[y:b,x:r],commit))
            a,crop,mask_a,plate_a,final_a,commit=pair[0]
            b,_,mask_b,plate_b,final_b,_=pair[1]
            x,y,r,bottom=a['bounds'];w,h=crop.shape[1],crop.shape[0]
            panels=[('Original',crop),('A target mask (red)',mask_a),('B extra context (cyan)',mask_b),
                ('Retained baseline clean' if native else 'Known constructed clean',
                    clean[y:bottom,x:r] if native else pixels(reference/'reference.png')[y:bottom,x:r]),
                ('A cleanplate',plate_a),('B cleanplate'+(' (reuse)' if b['reuse'] else ''),plate_b),
                ('Retained final' if native else 'Original',baseline[y:bottom,x:r]),
                ('A final crop',final_a),('B final crop',final_b)]
            sheet=Image.new('RGB',(w*3,(h+25)*3+30),'#e5e5e5');draw=ImageDraw.Draw(sheet)
            draw.text((4,4),f'{case["category"]} / margin {extent}px / native scale',font=font,fill='black')
            for index,(title,values) in enumerate(panels):
                ox=(index%3)*w;oy=(index//3)*(h+25)+30
                draw.text((ox+3,oy+3),title,font=font,fill='black');sheet.paste(Image.fromarray(values),(ox,oy+25))
            name=f'case-{case["index"]:02d}-{extent}-sheet.png';sheet.save(destination/name)
            links.append(f'<h3>{html.escape(case["category"])} / {extent}px</h3><a href="{name}">native sheet</a>')
            for row in (a,b):
                final_name=row['name']+'-final-page.png';clean_name=row['name']+'-clean-page.png'
                links.append(f'<p>{row["policy"]}: <a href="{clean_name}">clean page</a> · '
                    f'<a href="{final_name}">full final page</a></p><img src="{final_name}" style="width:800px;max-width:100%">')
            changed=np.any(plate_a!=plate_b,axis=2)
            case_rows.append({'case':case['index'],'category':case['category'],'extent':extent,
                'identical_inference':bool(b['reuse']),'different_commit_pixels':int(changed[commit].sum()),
                'ab_commit_mae':float(np.abs(plate_a.astype(float)-plate_b)[commit].mean())})
        links.append('<hr>')
    assert len(results)==plan['actual_planned_forwards']==len(list(outputs.glob('*/dispatch.json')))
    assert len(results)==len(list(outputs.glob('*/result.json'))) and not list(outputs.glob('*/failure.json'))
    costs={name:{'sum':sum(r[name] for r in results),'minimum':min(r[name] for r in results),
        'median':statistics.median(r[name] for r in results),'maximum':max(r[name] for r in results)}
        for name in ('load_seconds','inference_seconds','wall_seconds','peak_rss_kib')}
    # Sum of per-process RSS peaks is not simultaneous or cumulative live memory.
    del costs['peak_rss_kib']['sum']
    assert costs['wall_seconds']['sum']<=840
    report={'frozen_artifact_hashes_checked':verified,'actual_forwards':len(results),'slots':slots,
        'reuse_slots':sum(bool(row['reuse']) for row in plan['requests']),'comparisons':case_rows,'costs':costs,
        'outside_commit_changed':0,'prior_edits_changed':0,'real_ocr_translation_review_calls':0,
        'scope':'reused diagnostic images; visual and language judgments are not independent release gold'}
    (destination/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (destination/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Context inspection</title>'
        '<h1>Fixed-commit context experiment</h1><p>Reused diagnostics; A/B variants are not selected production outputs.</p>'
        +''.join(links))
    artifacts={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (outputs,destination) for path in folder.rglob('*') if path.is_file()}
    (destination/'artifact-hashes.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    print(json.dumps({'forwards':len(results),'hashes_checked':verified,'costs':costs},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-name',default='inspection',help='fresh directory inside root; refuses overwrite')
    args=parser.parse_args();inspect(args.root,args.output_name)
