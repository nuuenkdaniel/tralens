"""Build a local, native-scale review of already saved model experiments."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path


def summarize(masks, restoration, output):
    mask_report = json.loads((masks/'report.json').read_text())
    restoration_report = json.loads((restoration/'report.json').read_text())
    assert mask_report['frozen_manifest_sha256'] == restoration_report['frozen_manifest_sha256']
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(frozen_manifest_sha256=mask_report['frozen_manifest_sha256'],
                   no_production_adoption=True, mask_splits={}, restoration_splits={})
    for split in ('development', 'held_out'):
        real = [row for row in mask_report['cases'] if row['split'] == split and row['kind'] == 'real_no_ground_truth']
        synthetic = [row for row in mask_report['cases'] if row['split'] == split and row['kind'] == 'synthetic_known_overlay']
        summary['mask_splits'][split] = dict(real_cases=len(real), synthetic_cases=len(synthetic),
            current_real_plans=sum('current_erase' in row['candidates'] for row in real),
            current_synthetic_plans=sum('current_erase' in row['candidates'] for row in synthetic), learned={})
        for method in ('ctd_raw', 'ctd_refined_source_bound'):
            counts = Counter()
            for row in synthetic:
                values = row['candidates'][method]['ink_metrics']
                counts.update({k: values[k] for k in ('tp', 'fp', 'fn')})
            tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
            summary['mask_splits'][split]['learned'][method] = dict(
                synthetic_micro_ink_precision=tp/(tp+fp) if tp+fp else None,
                synthetic_micro_ink_recall=tp/(tp+fn) if tp+fn else None,
                real_nonmember_polygon_collision_cases=sum(row['candidates'][method]['nonmember_polygon_pixels'] > 0 for row in real),
                warning='zero collision with OCR polygons does NOT certify art or unannotated neighboring text safety')
        known = [row for row in restoration_report['cases'] if row['split'] == split and row['kind'] == 'synthetic_known_overlay']
        wins = losses = ties = 0
        by_category = {}
        for row in known:
            candidates = row['candidates']
            if 'masked_mae' not in candidates['lama']:
                continue
            classical = [candidates[name]['masked_mae'] for name in ('telea', 'navier_stokes') if 'masked_mae' in candidates[name]]
            if not classical:
                continue
            baseline, learned = min(classical), candidates['lama']['masked_mae']
            wins += learned < baseline
            losses += learned > baseline
            ties += learned == baseline
            by_category.setdefault(row['category'], []).append(dict(margin=row['margin'], best_classical_mae=baseline,
                                                                  lama_mae=learned))
        summary['restoration_splits'][split] = dict(synthetic_context_comparisons=len(known),
            unique_synthetic_cases=len({row['id'] for row in known}),
            lama_lower_mae_than_best_classical=wins, lama_higher_mae_than_best_classical=losses, exact_ties=ties,
            categories=by_category,
            warning='best-classical comparison is an oracle diagnostic using known reference, not an automatic selector')
    summary['outside_mask_changed_pixels'] = sum(value['composited_outside_changed_pixels']
        for row in restoration_report['cases'] for value in row['candidates'].values() if value['status'] == 'completed')
    summary['restoration_contexts'] = len(restoration_report['cases'])
    summary['restoration_skips'] = restoration_report['skipped']
    summary['timing_warning'] = 'Feasibility wall times were collected under mixed system load, including concurrent tests; not controlled performance claims.'
    summary['checkpoint_rights'] = 'CTD artifact-specific terms unresolved; LaMa card Apache-2.0 but original-checkpoint chain not digest-verified; no redistribution/commercial clearance.'
    summary['fixture_limitation'] = ('The frozen synthetic line_crossing cases actually have zero dark-reference '
        'pixels under the fixed mask: they test rotated text NEAR lines, not interrupted-line recovery. '
        'The separate outline_touch cases do intersect a known art line. No fixtures were changed after inference.')
    summary['corrected_rerun_caveat'] = ('Restoration v3 corrects a padding defect discovered after inspecting '
        'both splits. The held_out label remains the original frozen split, not a fresh confirmatory '
        'validation. New confirmatory pages are required before production adoption.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    def relative(path):
        import os
        return escape(os.path.relpath(path, output), quote=True)

    sections = ['<h1>Learned-mask / fixed-mask restoration feasibility</h1>',
        '<p>Noncommercial local evaluation. No production adoption, no translated-output coverage claim. '
        'Real samples: Ubunchu! by Hiroshi Seo (Seotch), ASCII MEDIA WORKS Inc., CC BY-NC 3.0. '
        'Reference pixels exist only for synthetic overlays. Images display at native scale; scroll horizontally.</p>',
        '<p>CTD raw threshold is 0.3. Refinement uses the pinned upstream INPAINT code with the frozen source-unit '
        'union box, not CTD detected grouping. Neither is certified erase permission.</p>',
        '<p>Restoration columns: source | ring-median fill | Telea | Navier–Stokes | LaMa | '
        'known pre-text reference (synthetic only). All methods share the exact same mask and context per row. '
        'No lettering conceals cleanplate defects.</p>', '<pre>'+escape(json.dumps(summary, indent=2))+'</pre>']
    for row in mask_report['cases']:
        sections += ['<h2>'+escape(row['id'])+' — '+escape(row['split'])+'</h2>',
            '<p>Mask columns: source | current erase (gray if skipped) | CTD raw | CTD source-bound refined.</p>',
            '<div class="strip"><img alt="source and mask candidates" src="'+relative(masks/row['id']/'source-current-raw-refined.png')+'"></div>',
            '<details><summary>Mask measurements / current skips</summary><pre>'+escape(json.dumps(row, indent=2))+'</pre></details>']
        for restored in [case for case in restoration_report['cases'] if case['id'] == row['id']]:
            path = restoration/(row['id']+f"-context-{restored['margin']}")/'comparison.png'
            sections += ['<p>Fixed-mask restoration, context margin '+str(restored['margin'])+'px.</p>',
                '<div class="strip"><img alt="fixed-mask restoration candidates" src="'+relative(path)+'"></div>',
                '<details><summary>Restoration metrics / provenance</summary><pre>'+escape(json.dumps(restored, indent=2))+'</pre></details>']
    html = ('<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" '
            'content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'">'
            '<title>Local learned restoration experiment</title><style>body{font-family:sans-serif;margin:24px;}'
            '.strip{overflow-x:auto;}img{max-width:none;}pre{white-space:pre-wrap;}h2{margin-top:48px;}</style>'
            + '\n'.join(sections))
    (output/'index.html').write_text(html)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--restoration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.masks, args.restoration, args.output), indent=2))
