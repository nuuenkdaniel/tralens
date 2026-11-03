"""Image pipeline coordinator. Stage algorithms live in tralens/, not this file.

Import and --help are engine-free. Without --render, the existing preview-only
CLI and its output format remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path


def run_pipeline(image, output, *, fonts, font_id=None, min_size=16, max_size=28,
                 max_units=None, background_method='auto', **preview_options):
    """Connect the existing OCR/translation preview to automatic rendering.

    No stage policy is relaxed. A fresh job owns preview/ and rendering/ plus a
    pipeline.json checkpoint; failures retain evidence, never resume/overwrite.
    OCR factories/providers may be injected through the existing preview API.
    """
    if preview_options.get('run_local_ocr') is not True or preview_options.get('run_local_model') is not True:
        raise ValueError('the rendered pipeline requires explicit OCR and translation opt-ins')
    output = Path(output).expanduser().absolute()
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or any(path.is_symlink() for path in output.parents)):
        raise ValueError('pipeline output must be fresh with an existing parent and no symlinks')
    from tralens.rendering.automatic import preflight_automatic, run_automatic
    from tralens.translation.workflow import run_preview, _atomic_json

    rendering_options = dict(image=Path(image).expanduser().absolute(), fonts=Path(fonts).expanduser().absolute(),
        font_id=font_id, min_size=min_size, max_size=max_size, max_units=max_units, background_method=background_method)
    checked = preflight_automatic(**rendering_options)
    report = {'schema_version': 1, 'purpose': 'automatic_pipeline_candidate', 'status': 'running',
        'stage': 'preview', 'source': checked['source'], 'fonts_sha256': checked['fonts_sha256'],
        'font_id': checked['font_id'], 'background_method': background_method,
        'approved_export': False, 'linguistic_approval': 'not_assessed',
        'full_page_translation_verified': False, 'preview': None,
        'rendering': {'status': 'not_started'}, 'coverage': None, 'error': None}
    output.mkdir(mode=0o700)
    checkpoint = output/'pipeline.json'
    _atomic_json(checkpoint, report)

    def persist():
        _atomic_json(checkpoint, report, replace_owned=True)

    def fingerprint(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        preview = run_preview(rendering_options['image'], output/'preview', **preview_options)
        records = preview['units']
        statuses = Counter(row['status'] for row in records)
        strict_count = sum(row['status'] == 'preview' and row.get('input_kind', 'strict_utterance') == 'strict_utterance'
                           for row in records)
        report['preview'] = {'status': preview['status'], 'path': 'preview/preview.json',
            'sha256': fingerprint(output/'preview/preview.json'), 'attempts': preview['attempts'],
            'unit_status_counts': dict(statuses), 'error': preview['error'],
            'ocr_cleanup': preview['cleanup'], 'fragment_previews': preview.get('fragment_preview_count', 0),
            'visual_alternates': preview.get('alternate_preview_count', 0)}
        report['coverage'] = {'saved_units': len(records), 'strict_translations': strict_count,
            'ocr_blocked_units': statuses['blocked'],
            'ocr_blocking_reasons': dict(Counter(reason for row in records for reason in row['blocking_reasons'])),
            'translation_budget_exhausted': statuses['budget_exhausted'], 'rendered': 0,
            'render_skip_reasons': {}, 'partial_output': True}
        if preview['status'] in ('failed', 'interrupted'):
            report.update(status=preview['status'], error=preview['error'])
            report['rendering'] = {'status': 'skipped', 'reason': 'upstream_failed_or_interrupted'}
        else:
            # Check the original preflight binding again, not just the current
            # preview binding. A source/font change cannot become a new job input.
            if preflight_automatic(**rendering_options) != checked or preview['input'] != checked['source']:
                raise ValueError('pipeline image/font inputs changed after preflight')
            if not strict_count:
                report['status'] = 'partial'
                report['rendering'] = {'status': 'skipped', 'reason': 'no_strict_translations',
                    'note': 'No composite was produced; original image and preview evidence remain unchanged.'}
            else:
                report['stage'] = 'rendering'
                report['rendering'] = {'status': 'running'}
                persist()
                rendered = run_automatic(preview=output/'preview/preview.json', output=output/'rendering', **rendering_options)
                if (rendered['source'] != checked['source'] or rendered['fonts_sha256'] != checked['fonts_sha256']
                        or rendered['preview_sha256'] != report['preview']['sha256']):
                    raise ValueError('rendered output does not bind the original pipeline inputs')
                report['rendering'] = {'status': 'complete', 'manifest': 'rendering/manifest.json',
                    'manifest_sha256': fingerprint(output/'rendering/manifest.json'),
                    'composite': 'rendering/composite.png', 'review': 'rendering/review.html',
                    'coverage': rendered['coverage'], 'background_summary': rendered['background_summary']}
                report['coverage'].update(rendered=rendered['coverage']['rendered'],
                    render_skip_reasons=dict(Counter(row['reason_code'] for row in rendered['units']
                        if row['machine_selected'] and row['status'] == 'skipped')),
                    partial_output=rendered['coverage']['partial_output'])
                report['status'] = ('complete' if preview['status'] == 'complete'
                    and not rendered['coverage']['partial_output'] else 'partial')
        report['stage'] = 'finished'
    except (Exception, KeyboardInterrupt) as error:
        failed_stage = report['stage']
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            error={'stage': failed_stage, 'type': type(error).__name__, 'message': str(error)[:2000]})
        if failed_stage == 'rendering':
            report['rendering'] = {'status': report['status']}
        else:
            report['rendering'] = {'status': 'skipped', 'reason': 'upstream_failed_or_invalid'}
    persist()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create an OCR/translation preview; add --render for an automatic candidate page.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="fresh output directory; parent must exist")
    parser.add_argument("--device", required=True, help="cpu or one gpu:N")
    parser.add_argument("--model-root", required=True, type=Path, help="existing local medium Paddle exports")
    parser.add_argument("--run-local-ocr", action="store_true")
    parser.add_argument("--run-local-model", action="store_true")
    parser.add_argument("--allow-fragment-previews", action="store_true",
                        help="opt in to review-required raw singleton text fragments; original structural blocks remain")
    parser.add_argument("--review-ocr", action="store_true",help="optional local visual reread proposals before translation; requires model opt-in")
    parser.add_argument("--review-model",default="qwen3.8:latest")
    parser.add_argument("--max-review-attempts",type=int,default=4,help="independent review-generation budget, 0..20")
    parser.add_argument("--review-timeout",type=float,default=90,help="review timeout, at most 120 seconds")
    parser.add_argument("--model", default="translategemma:27b", help="installed model (default: translategemma:27b); no download/fallback")
    parser.add_argument("--profile", choices=("generalist", "translategemma"), default="translategemma",
                        help="capability profile (default: translategemma); Qwen overrides require generalist")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--max-attempts", type=int, default=2, help="global model call budget, 0..20; no retries")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--keep-alive", type=int, default=60)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--unit-id", action="append", default=[], help="optional selected proposal ID; repeat to set call order")
    parser.add_argument('--render', action='store_true', help='connect OCR/translation to automatic cleanup and rendering')
    parser.add_argument('--fonts', type=Path, help='explicit local licensed font manifest; required with --render')
    parser.add_argument('--font-id', help='global font ID; required if the manifest has multiple faces')
    parser.add_argument('--min-size', type=int, help='rendering minimum size, default 16; never shrink below it')
    parser.add_argument('--max-size', type=int, help='rendering maximum size, default 28')
    parser.add_argument('--max-units', type=int, help='optional rendering evaluation cap, 1..16; not a translation budget')
    parser.add_argument('--background-method', choices=('auto', 'flat'),
                        help='render: flat fill then classical restoration (default auto); flat disables restoration')
    args = parser.parse_args(argv)
    if not args.run_local_ocr:
        parser.error("image processing requires explicit --run-local-ocr; no engine was initialized")
    if args.review_ocr and not args.run_local_model:
        parser.error("--review-ocr requires --run-local-model; OCR-only performs no image review")
    if args.render and (not args.run_local_model or args.fonts is None):
        parser.error('--render requires --run-local-model and --fonts; no engine was initialized')
    if not args.render and any(getattr(args, name) is not None for name in ('fonts', 'font_id', 'min_size', 'max_size', 'max_units', 'background_method')):
        parser.error('font/rendering options require --render')
    # Keep help/import and missing opt-in paths free of OCR/provider imports.
    from tralens.translation.workflow import run_preview
    try:
        preview_options = dict(device=args.device, model_root=args.model_root,
            run_local_ocr=args.run_local_ocr, run_local_model=args.run_local_model, model=args.model,
            profile=args.profile, endpoint=args.endpoint, max_attempts=args.max_attempts, timeout=args.timeout,
            keep_alive=args.keep_alive, num_predict=args.num_predict, selected_ids=tuple(args.unit_id),
            allow_fragment_previews=args.allow_fragment_previews,review_ocr=args.review_ocr,review_model=args.review_model,
            max_review_attempts=args.max_review_attempts,review_timeout=args.review_timeout)
        if args.render:
            report = run_pipeline(args.image, args.output, fonts=args.fonts, font_id=args.font_id,
                min_size=16 if args.min_size is None else args.min_size,
                max_size=28 if args.max_size is None else args.max_size, max_units=args.max_units,
                background_method=args.background_method or 'auto', **preview_options)
            coverage = report['coverage'] or {}
            print(f"{report['status']}: {args.output / 'pipeline.json'}; "
                  f"strict translations={coverage.get('strict_translations', 0)}; rendered={coverage.get('rendered', 0)}; "
                  'candidate only, not verified full-page translation')
            if report['rendering'].get('composite'):
                print(f"Composite: {args.output / report['rendering']['composite']}; review: {args.output / report['rendering']['review']}")
            if report['error']:
                print(f"Pipeline error: {report['error']['message']}")
            return 130 if report['status'] == 'interrupted' else 0 if report['status'] == 'complete' else 1
        preview = run_preview(args.image, args.output, **preview_options)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, ("pipeline" if args.render else "preview") + " rejected: " + str(error) + "\n")
    print(f"{preview['status']}: {args.output / 'preview.json'}; model calls={preview['attempts']}; "
          f"strict previews={preview.get('strict_preview_count',0)}; unverified fragments={preview.get('fragment_preview_count',0)}")
    if args.review_ocr:
        stage = preview.get("ocr_review",{})
        print(f"visual review calls={stage.get('attempts',0)}; release/control calls={stage.get('control_calls',0)}; "
              f"unverified alternates={preview.get('alternate_preview_count',0)}")
    return 0 if preview["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
