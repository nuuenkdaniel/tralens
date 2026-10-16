"""Offline candidate typography and flat-interior composition; never initializes OCR."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "compose", "record-review", "verify-review"))
    for name in ("preview", "image", "selection", "fonts"):
        parser.add_argument("--"+name, required=True, type=Path)
    for name in ('output', 'parent', 'candidate', 'review'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--compare', action='append', type=Path, default=[])
    parser.add_argument('--unit-id')
    parser.add_argument('--state', choices=('not_assessed', 'accepted', 'rejected', 'rework'))
    parser.add_argument('--operator')
    parser.add_argument('--note')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
    from tralens.rendering.natural import run
    inputs = {name: getattr(args, name) for name in ('preview', 'image', 'selection', 'fonts')}
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
    try:
        if args.mode == 'record-review':
            from tralens.rendering.review import record_review
            record = record_review(candidate=args.candidate, unit_id=args.unit_id, state=args.state,
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
