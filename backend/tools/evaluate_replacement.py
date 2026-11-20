"""Replay frozen, bound page inputs through the integrated renderer, never overwrite.

For visual iteration only: reused pages are diagnostic, not independent quality
labels. Inference is enabled only by supplying an existing reconstruction config.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,default=ROOT/'test-results/production-best-effort-v1/frozen')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='*')
    parser.add_argument('--reconstruction-config',type=Path)
    args=parser.parse_args()
    from tralens.rendering.automatic import run_automatic
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for folder in sorted(args.frozen.iterdir()):
        if not folder.is_dir() or args.cases and folder.name not in args.cases:continue
        baseline=json.loads((folder/'baseline-manifest.json').read_bytes())
        clock=time.monotonic()
        report=run_automatic(preview=folder/'preview.json',image=baseline['source']['path'],fonts=folder/'fonts.json',
            output=args.output/folder.name,font_id=baseline['font_id'],**{k:baseline['style'][k] for k in ('min_size','max_size')},
            replacement_policy='confidence-gated',visibility_policy='always-show',removal_policy='aggressive',
            layout_policy=baseline.get('layout_policy','source-local'),cleanup_policy=baseline.get('cleanup_policy','native'),
            reconstruction_policy='lama' if args.reconstruction_config else 'classical',
            reconstruction_config=args.reconstruction_config)
        row={'case':folder.name,'seconds':time.monotonic()-clock,'coverage':report['coverage'],
            'removal':report['removal_summary'],'display':report['display_summary'],'pixel_audit':report['pixel_audit']}
        results.append(row)
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
