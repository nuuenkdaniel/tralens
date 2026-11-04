"""Replay existing crop evidence on reused pages; never starts any model.

This is a source/mask/cost regression comparison, not an accuracy benchmark.
Supply an existing corpus report and approved fonts; output is always fresh.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from tralens.ocr.source_localization import execute_requests
from tralens.rendering.automatic import run_automatic


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(corpus, fonts, output):
    corpus, fonts, output = (Path(p).absolute() for p in (corpus, fonts, output))
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('evaluation output must be fresh without symlinks')
    cases = json.loads(corpus.read_bytes())['results']
    output.mkdir(mode=0o700)
    snapshot = {}
    root = Path(__file__).resolve().parents[1]
    for path in [*(root/'src').rglob('*.py'), Path(__file__)]:
        relative = path.relative_to(root)
        target = output/'frozen-code'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
        snapshot[str(relative)] = digest(path)
    (output/'source-hashes.json').write_text(json.dumps(snapshot,indent=2))
    results = []
    total_started = time.monotonic()
    for case in cases:
        started = time.monotonic()
        old_path = Path(case['output'])
        old = json.loads((old_path/'manifest.json').read_bytes())
        saved_preview = json.loads((old_path/'evidence-preview.json').read_bytes())
        evidence = saved_preview['ocr']['source_localization']
        preview = copy.deepcopy(saved_preview)
        del preview['ocr']['source_localization']
        folder = output/case['file']
        folder.mkdir(mode=0o700)
        path = folder/'preview.json'
        path.write_text(json.dumps(preview,ensure_ascii=False))
        options = dict(image=case['image'], fonts=fonts, replacement_policy='confidence-gated',
                       layout_policy='enclosed-retry')
        t = time.monotonic()
        native = run_automatic(preview=path,output=folder/'native',plan_localization=True,**options)
        native_seconds = time.monotonic()-t
        plan = native['localization_request_plan']
        placed = {r['unit_id'] for r in native['units'] if r['status']=='automatic_composed'}
        assert not placed.intersection(plan['requests']), 'already-placed crop request'
        derived = execute_requests(preview,plan,image=case['image'],output=folder/'replay',
            model_identity=evidence['model_identity'],replay=evidence)
        stage = derived['ocr']['source_localization']
        t = time.monotonic()
        candidate = run_automatic(preview=folder/'replay/preview.json',output=folder/'candidate',
                                  cleanup_policy='localized',**options)
        retry_seconds = time.monotonic()-t
        assert candidate['source'] == old['source']
        assert candidate['fonts_sha256'] == old['fonts_sha256']
        assert derived['units'] == saved_preview['units']
        changes = []
        for index,(before,after) in enumerate(zip(old['units'],candidate['units'])):
            assert before['unit_id'] == after['unit_id']
            if before['status'] == 'automatic_composed':
                assert after['status'] == 'automatic_composed', ('lost source replacement',case['file'],index)
                assert before['binding'] == after['binding']
                for suffix in ('foreground','applied-erase','text'):
                    name = f'unit-{index:03d}-{suffix}.png'
                    assert (old_path/name).read_bytes() == (folder/'candidate'/name).read_bytes(), (case['file'],name)
            elif after['status']=='automatic_composed':
                changes.append(index)
        same = (old_path/'composite.png').read_bytes() == (folder/'candidate/composite.png').read_bytes()
        result = {'file':case['file'],'group':case['group'],'image':case['image'],
            'old_path':str(old_path),'native_path':str(folder/'native'),'candidate_path':str(folder/'candidate'),
            'preview_path':str(path),'old_preview_sha256':digest(old_path/'evidence-preview.json'),
            'old_calls':evidence['calls'],'requests':len(plan['requests']),'calls':stage['calls'],
            'reused_predictions':stage['reused_predictions'],
            'predictions_consumed':candidate['localized_cleanup_summary']['retries'],
            'budget_missed':[r for r in plan['records'] if 'budget' in r['reason']],
            'unavailable_saved_predictions':[r['unit_id'] for r in stage['records'] if r['reason']=='saved_prediction_unavailable'],
            'before':old['coverage']['rendered'],'after':candidate['coverage']['rendered'],
            'new_replacements':changes,'composite_byte_identical':same,
            'old_successful_masks_text_and_targets_identical':True,
            'native_seconds':native_seconds,'planning_seconds':native['localization_planning_seconds'],
            'replay_seconds':stage['execution_seconds'],'retry_render_seconds':retry_seconds,
            'total_seconds':time.monotonic()-started}
        results.append(result)
        report = {'scope':'21 reused pages; saved-evidence replay, not fresh inference or linguistic accuracy',
            'corpus_sha256':digest(corpus),'fonts_sha256':digest(fonts),
            'total_seconds':time.monotonic()-total_started,'results':results}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(case['file'],result['requests'],'requests;',result['predictions_consumed'],'consumed;',
              result['after'],'retained replacements;', 'identical' if same else 'changed',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True,type=Path)
    parser.add_argument('--fonts',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    run(args.corpus,args.fonts,args.output)


if __name__ == '__main__':
    main()
