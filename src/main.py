"""Explicit image-to-utterance preview CLI. Import and --help are engine-free."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a JSON OCR/translation preview; never render or erase.")
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
    args = parser.parse_args(argv)
    if not args.run_local_ocr:
        parser.error("image processing requires explicit --run-local-ocr; no engine was initialized")
    if args.review_ocr and not args.run_local_model:
        parser.error("--review-ocr requires --run-local-model; OCR-only performs no image review")
    # Keep help/import and missing opt-in paths free of OCR/provider imports.
    from tralens.translation.workflow import run_preview
    try:
        preview = run_preview(args.image, args.output, device=args.device, model_root=args.model_root,
            run_local_ocr=args.run_local_ocr, run_local_model=args.run_local_model, model=args.model,
            profile=args.profile, endpoint=args.endpoint, max_attempts=args.max_attempts, timeout=args.timeout,
            keep_alive=args.keep_alive, num_predict=args.num_predict, selected_ids=tuple(args.unit_id),
            allow_fragment_previews=args.allow_fragment_previews,review_ocr=args.review_ocr,review_model=args.review_model,
            max_review_attempts=args.max_review_attempts,review_timeout=args.review_timeout)
    except (ValueError, TypeError, OSError) as error:
        parser.exit(2, "preview rejected: " + str(error) + "\n")
    print(f"{preview['status']}: {args.output / 'preview.json'}; model calls={preview['attempts']}; "
          f"strict previews={preview.get('strict_preview_count',0)}; unverified fragments={preview.get('fragment_preview_count',0)}")
    if args.review_ocr:
        stage = preview.get("ocr_review",{})
        print(f"visual review calls={stage.get('attempts',0)}; release/control calls={stage.get('control_calls',0)}; "
              f"unverified alternates={preview.get('alternate_preview_count',0)}")
    return 0 if preview["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
