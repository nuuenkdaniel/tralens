"""Bounded serial saved-page balloon evaluation. No page OCR or downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-local-model',action='store_true')
    parser.add_argument('--replay-root',type=Path)
    args = parser.parse_args(argv)
    cases = json.loads(args.cases.read_bytes())
    if not 1 <= len(cases) <= 5:
        raise ValueError('evaluation budget: one to five explicit reused pages')
    if args.output.exists() or any(p.is_symlink() for p in (args.output,*args.output.absolute().parents)):
        raise ValueError('fresh evaluation output required')
    args.output.mkdir(mode=0o700)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    from main import run_balloon_experiment
    hashes = {}
    for path in [*(root/'src').rglob('*.py'),Path(__file__).resolve(),root/'tools/balloon_experiment.py']:
        relative = path.relative_to(root)
        target = args.output/'frozen-code'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'implementation.json').write_text(json.dumps(hashes,indent=2))
    results = []
    for case in cases:
        name = case['name']
        if Path(name).name != name or name in ('.','..'):
            raise ValueError('bounded case name required')
        output = args.output/name
        replay = (args.replay_root/name/'experiment.json') if args.replay_root else case.get('replay')
        state = run_balloon_experiment(**{k:case[k] for k in ('preview','image','fonts','baseline')},
            output=output,run_local_model=args.run_local_model and replay is None,replay=replay)
        manifest = json.loads((output/'rendering/manifest.json').read_bytes())
        results.append({'name':name,'proposal_calls':state['proposal_calls'],'translation_calls':state['translation_calls'],
            'seconds':state['seconds'],'native_preserved':state['native_preserved'],
            'native_atomically_superseded':state['native_atomically_superseded'],
            'accepted':[{'parents':r['parent_indices'],'target':r['binding']['text'],'size':r['layout']['size'],
                'flow':r['layout']['balloon_flow']['kind']} for r in manifest['accepted']],
            'failures':manifest['failures'],'display_summary':state['display_summary']})
        pages = [Image.open(case['image']).convert('RGB'),Image.open(Path(case['baseline'])/'composite.png').convert('RGB'),
            Image.open(output/'rendering/clean-composite.png').convert('RGB'),Image.open(output/'rendering/composite.png').convert('RGB')]
        font = ImageFont.truetype(json.loads(Path(case['fonts']).read_bytes())['fonts'][0]['path'],18)
        width = 620;height = round(pages[0].height*width/pages[0].width)
        sheet = Image.new('RGB',(width*4,height+35),'#ddd');draw = ImageDraw.Draw(sheet)
        for i,(title,page) in enumerate(zip(('Original','Native baseline','Balloon clean experiment','Clean + existing safety net'),pages)):
            sheet.paste(page.resize((width,height)),(i*width,35));draw.text((i*width+6,7),title,font=font,fill='black')
        sheet.save(output/'comparison-full.png')
        (args.output/'report.json').write_text(json.dumps({'reused_not_held_out':True,'page_ocr_calls':0,'results':results},ensure_ascii=False,indent=2))
        print(name,'accepted',results[-1]['accepted'],'calls',state['proposal_calls'],state['translation_calls'],flush=True)


if __name__ == '__main__':
    main()
