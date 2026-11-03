"""Offline candidate rendering with flat fill/restoration; never initializes OCR."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "compose", "record-review", "verify-review", "prepare", "confirm-preparation", "auto-compose"))
    for name in ("preview", "image", "fonts"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument('--selection', type=Path)
    for name in ('output', 'parent', 'candidate', 'review'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--compare', action='append', type=Path, default=[])
    parser.add_argument('--unit-id', action='append')
    parser.add_argument('--state', choices=('not_assessed', 'accepted', 'rejected', 'rework'))
    parser.add_argument('--operator')
    parser.add_argument('--note')
    parser.add_argument('--proposal-bundle', type=Path)
    parser.add_argument('--proposal-id', action='append')
    parser.add_argument('--intent')
    parser.add_argument('--font-id')
    parser.add_argument('--max-units', type=int, help='auto-compose evaluation cap only; default processes all strict eligible candidates within resource budgets')
    parser.add_argument('--background-method', choices=('auto', 'flat'),
                        help='auto-compose: flat fill then classical restoration (default auto); flat disables restoration')
    for name in ('min-size', 'max-size', 'clearance', 'member-padding', 'glyph-padding', 'layout-inset'):
        parser.add_argument('--'+name, type=int)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
    from tralens.rendering.natural import run
    inputs = {name: getattr(args, name) for name in ('preview', 'image', 'selection', 'fonts')}
    preparation_options = ('font_id', 'min_size', 'max_size', 'clearance', 'member_padding', 'glyph_padding', 'layout_inset')
    if args.mode == 'auto-compose':
        forbidden = ('selection', 'unit_id', 'operator', 'note', 'state', 'proposal_bundle', 'proposal_id', 'intent',
                     'parent', 'candidate', 'review', 'clearance', 'member_padding', 'glyph_padding', 'layout_inset')
        if args.output is None or args.compare or any(getattr(args, key) is not None for key in forbidden):
            parser.error('auto-compose requires --output; no selection, per-unit IDs, confirmation/review or mask overrides')
        from tralens.rendering.automatic import run_automatic
        try:
            report = run_automatic(preview=args.preview, image=args.image, fonts=args.fonts, output=args.output,
                font_id=args.font_id, min_size=16 if args.min_size is None else args.min_size,
                max_size=28 if args.max_size is None else args.max_size, max_units=args.max_units,
                background_method=args.background_method or 'auto')
        except (ValueError, OSError, KeyError, TypeError) as error:
            parser.exit(2, f'automatic render rejected: {error}\n')
        print(f"Automatic candidate output: {report['coverage']['rendered']}/{report['coverage']['strict_eligible']} strict candidates rendered; "
              f"{report['coverage']['skipped']} saved units skipped; {args.output}/review.html; no human approval")
        return 0
    if args.max_units is not None:
        parser.error('--max-units is an automatic evaluation cap, not a manual/confirmation option')
    if args.background_method is not None:
        parser.error('--background-method requires auto-compose')
    if args.mode in ('prepare', 'confirm-preparation'):
        if args.output is None or args.selection is not None or args.parent is not None or args.compare or any(
                getattr(args, k) is not None for k in ('candidate', 'review', 'state', 'note')):
            parser.error('preparation modes require fresh --output and do not consume rendering/review inputs')
        source_inputs = {name: inputs[name] for name in ('preview', 'image', 'fonts')}
        try:
            if args.mode == 'prepare':
                if any(getattr(args, k) is None for k in ('unit_id', 'font_id', 'min_size', 'max_size')) or any(
                        getattr(args, k) is not None for k in ('proposal_bundle', 'proposal_id', 'operator', 'intent')):
                    parser.error('prepare requires explicit --unit-id(s) --font-id --min-size --max-size; no confirmation is inferred')
                from tralens.rendering.preparation import prepare
                options = {key: getattr(args, key) for key in ('clearance', 'member_padding', 'glyph_padding', 'layout_inset') if getattr(args, key) is not None}
                report = prepare(unit_ids=args.unit_id, font_id=args.font_id, min_size=args.min_size, max_size=args.max_size,
                                 options=options, output=args.output, **source_inputs)
                print(f"Unconfirmed geometry: {sum(r['status'] == 'proposed' for r in report['units'])}/{len(report['units'])} proposed; {args.output}/review.html")
            else:
                if any(getattr(args, k) is None for k in ('proposal_bundle', 'proposal_id', 'operator', 'intent')) or args.unit_id is not None or any(
                        getattr(args, k) is not None for k in preparation_options):
                    parser.error('confirm-preparation requires --proposal-bundle --proposal-id(s) --operator --intent; no automatic all-selection')
                from tralens.rendering.preparation import confirm_preparation
                selected = confirm_preparation(proposal_bundle=args.proposal_bundle, proposal_ids=args.proposal_id,
                    operator=args.operator, intent=args.intent, output=args.output, **source_inputs)
                print(f"Explicit geometry confirmation: {len(selected['units'])} candidate-only units; {args.output}; no identity/linguistic approval")
        except (ValueError, OSError, KeyError, TypeError) as error:
            parser.exit(2, f"preparation rejected: {error}\n")
        return 0
    if args.selection is None or any(getattr(args, k) is not None for k in (*preparation_options, 'proposal_bundle', 'proposal_id', 'intent')):
        parser.error('render/review modes require --selection and do not consume preparation options')
    if args.mode in ('plan', 'compose'):
        if args.output is None or any(getattr(args, k) is not None for k in ('candidate', 'review', 'unit_id', 'state', 'operator', 'note')):
            parser.error('render modes require --output and do not consume review decisions')
    elif args.parent is not None or args.compare or args.candidate is None:
        parser.error('review modes require --candidate and do not consume parent/comparison inputs')
    elif args.mode == 'record-review' and any(getattr(args, k) is None for k in ('output', 'unit_id', 'state', 'operator', 'note')):
        parser.error('record-review requires --output --unit-id --state --operator --note')
    elif args.mode == 'record-review' and args.review is not None:
        parser.error('record-review creates a fresh record; it does not consume --review')
    elif args.mode == 'verify-review' and (args.review is None or any(getattr(args, k) is not None for k in ('output', 'unit_id', 'state', 'operator', 'note'))):
        parser.error('verify-review requires --review and does not consume new decisions')
    if args.unit_id is not None and len(args.unit_id) != 1:
        parser.error('record-review requires exactly one --unit-id')
    try:
        if args.mode == 'record-review':
            from tralens.rendering.review import record_review
            record = record_review(candidate=args.candidate, unit_id=args.unit_id[0], state=args.state,
                                   operator=args.operator, note=args.note, output=args.output, **inputs)
            print(f"Aesthetic record {record['state']}: {args.output}; identity/translation not verified")
            return 0
        if args.mode == 'verify-review':
            from tralens.rendering.review import verify_review
            record = verify_review(candidate=args.candidate, review=args.review, **inputs)
            print(f"Current aesthetic record: {record['state']}; consistency only, no identity/linguistic approval")
            return 0
        report = run(args.mode, output=args.output, parent=args.parent, compare=args.compare, **inputs)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"render rejected: {error}\n")
    print(f"Candidate only: {len(report['units'])} units; "
          f"{sum(r['status'] == 'blocked' for r in report['units'])} blocked; {args.output}/review.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
