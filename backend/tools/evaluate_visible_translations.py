"""Renderer-only saved-case comparison. No OCR, translation or downloads.

Cases are explicit JSON evaluation inputs, not production source substitutions.
Each names preview/image/fonts/baseline plus existing renderer options. Outputs
are fresh; successful clean layers, exact strings and declared locality are checked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tralens.rendering.automatic import run_automatic


def run(cases, output):
    output=Path(output).absolute()
    if output.exists() or any(p.is_symlink()for p in(output,*output.parents)):
        raise ValueError('evaluation output must be fresh without symlinks')
    output.mkdir(mode=0o700)
    root=Path(__file__).resolve().parents[1]
    hashes={}
    for path in [*(root/'src').rglob('*.py'),*(root/'src/tralens/devui/static').iterdir(),Path(__file__).resolve()]:
        target=output/'frozen-code'/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes());hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'source-hashes.json').write_text(json.dumps(hashes,indent=2))
    results=[]
    for case in cases:
        name=case['name']
        if Path(name).name!=name or name in('.','..'):raise ValueError('bounded evaluation name required')
        folder=output/name;folder.mkdir(mode=0o700)
        baseline=Path(case['baseline']);before=json.loads((baseline/'manifest.json').read_bytes())
        settings=dict(case.get('options',{}))
        if set(settings)-{'replacement_policy','min_ocr_confidence','layout_policy','cleanup_policy','render_policy','min_size','max_size'}:
            raise ValueError('unknown evaluation renderer options')
        inputs={k:case[k]for k in('preview','image','fonts')}
        started=time.monotonic()
        off=run_automatic(**inputs,output=folder/'off',**settings)
        pngs=list(baseline.glob('*.png'))
        assert all(p.read_bytes()==(folder/'off'/p.name).read_bytes()for p in pngs),'mode-off pixels changed'
        candidate=run_automatic(**inputs,output=folder/'candidate',visibility_policy='always-show',**settings)
        assert before['source']==candidate['source'] and before['fonts_sha256']==candidate['fonts_sha256']
        assert (baseline/'clean.png').read_bytes()==(folder/'candidate/clean.png').read_bytes()
        preserved=0
        for i,(a,b)in enumerate(zip(before['units'],candidate['units'])):
            assert a['unit_id']==b['unit_id']
            if a['status']=='automatic_composed':
                assert b['status']=='automatic_composed' and a['binding']==b['binding'];preserved+=1
                for suffix in('foreground','applied-erase','text'):
                    assert(baseline/f'unit-{i:03d}-{suffix}.png').read_bytes()==(folder/'candidate'/f'unit-{i:03d}-{suffix}.png').read_bytes()
        for target in case.get('required_targets',[]):
            matching=[r for r in candidate['units']if r.get('binding',{}).get('text')==target]
            assert len(matching)==1 and matching[0]['display_status']in('clean','patch','caption'),target
            row=matching[0]
            assert row['display_fallback']['layout']['size']>=candidate['style']['min_size']
            assert row['display_fallback']['layout']['original_text']==target
        clean=Image.open(folder/'candidate/clean.png').convert('RGBA')
        patch=Image.open(folder/'candidate/patch.png').convert('RGBA')
        text=Image.open(folder/'candidate/text.png').convert('RGBA')
        final=Image.open(folder/'candidate/composite.png').convert('RGB')
        assert Image.alpha_composite(Image.alpha_composite(clean,patch),text).convert('RGB').tobytes()==final.tobytes()
        native=np.asarray(Image.open(folder/'candidate/native-composite.png').convert('RGB'))
        mask=np.asarray(Image.open(folder/'candidate/fallback-mask.png'))!=0
        assert np.array_equal(native[~mask],np.asarray(final)[~mask])
        pages=[Image.open(case['image']).convert('RGB'),Image.open(baseline/'composite.png').convert('RGB'),final]
        font_doc=json.loads(Path(case['fonts']).read_bytes());label_font=ImageFont.truetype(font_doc['fonts'][0]['path'],18)
        width=620;height=round(pages[0].height*width/pages[0].width)
        sheet=Image.new('RGB',(3*width,height+35),'#ddd');draw=ImageDraw.Draw(sheet)
        for i,(title,page)in enumerate(zip(('Original','Existing clean replacements','Always show available translations'),pages)):
            sheet.paste(page.resize((width,height)),(i*width,35));draw.text((i*width+6,7),title,font=label_font,fill='black')
        sheet.save(folder/'comparison-full.png')
        result={'name':name,'scope':'saved reused evidence, not independent accuracy','mode_off_identical_pngs':len(pngs),
            'clean_replacements_preserved':preserved,'display_summary':candidate['display_summary'],
            'seconds_including_off_and_candidate':time.monotonic()-started,
            'degraded':[{'index':i,'target':r['binding']['text'],'display':r['display_status'],**r['display_fallback']}
                for i,r in enumerate(candidate['units'])if'display_fallback'in r]}
        results.append(result)
        (output/'report.json').write_text(json.dumps({'inference_calls':0,'results':results},ensure_ascii=False,indent=2))
        print(name,candidate['display_summary'],flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(json.loads(args.cases.read_bytes()),args.output)


if __name__=='__main__':main()
