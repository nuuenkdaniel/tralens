"""Bounded clean-text/reference comparison helper; model use requires explicit opt-in."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from dataclasses import asdict
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tralens.translation import (Reading, Utterance, TranslationRequest, TranslationResult, UnitOutcome,
                                 PreviewTranslator, OllamaRuntime)
from tralens.translation.contracts import canonical, revision, strict_json
from tralens.translation.runtime import GENERALIST, TRANSLATEGEMMA, GenerationOptions, local_endpoint, make_payload, validate_model_profile
from tools.translation_diagnostics import read_bounded, retain_response


def load_cases(path, max_cases):
    if type(max_cases) is not int or not 1 <= max_cases <= 150:
        raise ValueError("max_cases must be 1..150")
    raw = read_bounded(path, 16 * 1024 * 1024)
    cases = strict_json(raw.decode("utf-8"))
    if type(cases) is not list or not cases:
        raise ValueError("cases must be a nonempty list")
    ids = set()
    for case in cases:
        if type(case) is not dict or set(case) != {"id", "source", "reference", "split", "origin"}:
            raise ValueError("exact case fields required: id/source/reference/split/origin")
        for key, limit in (("id", 128), ("source", 8192), ("reference", 8192), ("split", 64), ("origin", 1024)):
            if type(case[key]) is not str or len(case[key]) > limit or (key != "source" and not case[key].strip()):
                raise ValueError("invalid or oversized case field: " + key)
        if case["id"] in ids:
            raise ValueError("duplicate case ID")
        ids.add(case["id"])
    return tuple(cases[:max_cases]), hashlib.sha256(raw).hexdigest(), len(cases)


def text_request(case):
    """Only source and an opaque case key enter this explicitly non-OCR contract."""
    key = "benchmark-" + hashlib.sha256(case["id"].encode("utf-8")).hexdigest()
    evidence = {"benchmark_id": "clean_text_only_quality_v1", "case_key": key,
                "provenance": "synthetic_text_only_benchmark_not_OCR",
                "source_sha256": hashlib.sha256(case["source"].encode("utf-8")).hexdigest()}
    reading = Reading(case["source"], "benchmark_supplied_text_not_OCR", canonical(evidence))
    unit = Utterance(key, canonical({"benchmark_id": "clean_text_only_quality_v1", "case_key": key,
        "provenance": "synthetic_text_only_benchmark_not_OCR"}), (key,), (key,), (reading,),
        reading.revision, revision(evidence), completeness="benchmark_supplied_text_no_OCR_claim")
    return TranslationRequest((unit,), (key,))


def frozen_request(case, profile, options):
    request = text_request(case)
    payload = make_payload(request, profile)
    payload.update(options.transport_fields())
    return request, {"case_id": case["id"], "request_revision": request.revision,
        "source_sha256": hashlib.sha256(case["source"].encode("utf-8")).hexdigest(),
        "prompt_sha256": revision(payload), "request": asdict(request), "payload": payload}


def metric_availability():
    available = importlib.util.find_spec("sacrebleu") is not None
    return {"name": "sacrebleu_corpus_bleu", "available": available,
            "availability_note": "installed package only; no installation attempted" if available else "metric unavailable: sacrebleu not installed"}


def score_records(cases, records):
    """Optional independent reference diagnostic. Every selected case is in the denominator."""
    if [r["case_id"] for r in records] != [c["id"] for c in cases]:
        raise ValueError("score requires exact ordered case IDs/counts")
    info = {**metric_availability(), "denominator": len(cases), "semantic_quality_claimed": False}
    if any(r["status"] == "preflight_only" for r in records):
        return {**info, "available": False, "reason": "preflight has no model outputs"}
    if not info["available"]:
        return {**info, "reason": "sacrebleu is not installed; no installation attempted"}
    predictions = [r["candidate"] if r["status"] == "preview" else "" for r in records]
    try:
        module = importlib.import_module("sacrebleu")
        metric = module.corpus_bleu(predictions, [[c["reference"] for c in cases]])
        value = float(metric.score)
        if not math.isfinite(value):
            raise ValueError("nonfinite metric")
        return {**info, "score": value, "version": getattr(module, "__version__", "unknown"),
                "settings": "installed sacrebleu.corpus_bleu defaults; unmodified Unicode candidates/references"}
    except Exception as error:
        return {**info, "available": False, "reason": "installed metric unusable: " + type(error).__name__}


def new_output(path):
    output = Path(path)
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    output.mkdir(exist_ok=False)
    return output


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_review(output, cases, runs):
    """Distribute review.json alone; identity/status evidence stays out of reviewer data."""
    labels = ["candidate-" + secrets.token_hex(8) for _ in runs]
    mapping = {label: {"model": plan["model"], "model_digest_parent_asserted": plan["model_record"]["digest"],
                       "profile": plan["profile"], "run": str(path)}
               for label, (path, plan, _) in zip(labels, runs)}
    review = []
    for index, case in enumerate(cases):
        candidates = [{"candidate_id": label, "text": records[index]["candidate"],
                       "fidelity_notes": None, "omissions_notes": None, "pragmatics_notes": None}
                      for label, (_, _, records) in zip(labels, runs)]
        secrets.SystemRandom().shuffle(candidates)
        review.append({**case, "candidates": candidates})
    write_json(output / "review.json", {"case_count": len(cases), "candidate_count_per_case": len(runs), "cases": review})
    write_json(output / "model-mapping.json", mapping)


def run(cases_path, output, *, model, model_record, profile="generalist", endpoint="http://127.0.0.1:11434",
        run_local_model=False, max_cases=150, num_predict=1024, keep_alive=60, runtime=None, score=False):
    cases, corpus_hash, input_count = load_cases(cases_path, max_cases)
    profiles = {"generalist": GENERALIST, "translategemma": TRANSLATEGEMMA}
    if profile not in profiles:
        raise ValueError("unknown profile")
    capability = profiles[profile]
    validate_model_profile(model, capability)
    endpoint = local_endpoint(endpoint)
    identity = strict_json(read_bounded(model_record, 65536).decode("utf-8"))
    if (type(identity) is not dict or identity.get("model") != model or type(identity.get("digest")) is not str
            or len(identity["digest"]) != 64 or any(c not in "0123456789abcdef" for c in identity["digest"])):
        raise ValueError("parent model record must match model and contain full SHA256 digest")
    options = GenerationOptions(num_ctx=4096, num_predict=num_predict, keep_alive=keep_alive)
    prepared = [frozen_request(case, capability, options) for case in cases]
    plan = {"schema_version": 1, "preview_only": True, "benchmark": "clean_text_only_not_OCR", "cases_sha256": corpus_hash,
        "cases_path": str(Path(cases_path).resolve()), "input_case_count": input_count,
        "selected_ids": [c["id"] for c in cases], "selection": "first max_cases in file order; no sorting",
        "model": model, "model_record": identity, "profile": profile, "capability_profile": asdict(capability),
        "endpoint": endpoint, "identity_scope": "parent assertions, not daemon/weight authentication",
        "run_local_model": run_local_model, "generation_options": asdict(options), "timeout_seconds": 120,
        "max_attempts_per_case": 1, "preload_calls": 0, "references_in_prompts": False,
        "requests": [frozen for _, frozen in prepared]}
    output = new_output(output)
    write_json(output / "run.json", plan)
    translator = None
    if run_local_model:
        runtime = runtime if runtime is not None else OllamaRuntime(endpoint)
        translator = PreviewTranslator(runtime, model=model, profile=capability, generation_options=options,
                                       max_attempts=1, timeout=120)
    records = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        for index, (case, (request, frozen)) in enumerate(zip(cases, prepared)):
            evidence = []
            def observe(event):
                item = {k: v for k, v in event.items() if k != "response_json"}
                if event.get("response_json") is not None:
                    item.update(retain_response(output, f"case-{index:03d}-attempt-{event['attempt']:02d}",
                                                event["response_json"], 4 * 1024 * 1024))
                evidence.append(item)
            started = time.monotonic()
            result = translator.translate(request, observe=observe) if translator is not None else None
            if result is not None:
                result.assert_current(request)
            candidate = dict(result.translations).get(request.target_ids[0], "") if result is not None else ""
            record = {"case_id": case["id"], "request_revision": request.revision,
                "source_sha256": frozen["source_sha256"], "prompt_sha256": frozen["prompt_sha256"],
                "model": model, "model_digest_parent_asserted": identity["digest"], "profile": profile,
                "generation_options": asdict(options),
                "status": result.status if result is not None else "preflight_only", "candidate": candidate,
                "attempts": result.attempts if result is not None else 0, "wall_seconds": time.monotonic() - started,
                "result": asdict(result) if result is not None else None, "generation_evidence": evidence}
            stream.write(canonical(record) + "\n")
            stream.flush()
            records.append(record)
    summary = {"preview_only": True, "selected_cases": len(cases), "record_count": len(records), "denominator": len(cases),
        "attempts": sum(r["attempts"] for r in records), "nonblank_candidates": sum(bool(r["candidate"]) for r in records),
        "statuses": {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})},
        "metric": score_records(cases, records) if score else {**metric_availability(), "computed": False},
        "semantic_quality_claimed": False, "timing_scope": "serial per-case evidence; load state/prefix cache not controlled"}
    write_review(output, cases, [(output, plan, records)])
    write_json(output / "summary.json", summary)
    return summary


def compare(cases_path, output, run_dirs, *, max_cases=150, score=False):
    cases, corpus_hash, _ = load_cases(cases_path, max_cases)
    if not 1 <= len(run_dirs) <= 8 or len({str(Path(p).resolve()) for p in run_dirs}) != len(run_dirs):
        raise ValueError("compare requires 1..8 distinct run directories")
    runs, prompts, shared_options = [], {}, None
    for path in map(Path, run_dirs):
        plan = strict_json(read_bounded(path / "run.json", 16 * 1024 * 1024).decode("utf-8"))
        records = [strict_json(line) for line in read_bounded(path / "results.jsonl", 64 * 1024 * 1024).decode("utf-8").splitlines()]
        ids = [c["id"] for c in cases]
        if plan["cases_sha256"] != corpus_hash or plan["selected_ids"] != ids or [r["case_id"] for r in records] != ids:
            raise ValueError("comparison requires exact corpus/order/IDs/counts, including failures")
        options = GenerationOptions(**plan["generation_options"])
        if shared_options is not None and asdict(options) != shared_options:
            raise ValueError("comparison generation options must match")
        shared_options = asdict(options)
        capability = {"generalist": GENERALIST, "translategemma": TRANSLATEGEMMA}[plan["profile"]]
        if len(plan["requests"]) != len(cases):
            raise ValueError("frozen request count mismatch")
        for case, record, saved in zip(cases, records, plan["requests"]):
            request, expected = frozen_request(case, capability, options)
            if (record["model"] != plan["model"] or record["model_digest_parent_asserted"] != plan["model_record"]["digest"]
                    or record["profile"] != plan["profile"] or record["generation_options"] != asdict(options)):
                raise ValueError("per-case model identity/options mismatch")
            for key in ("case_id", "request_revision", "source_sha256", "prompt_sha256"):
                if record[key] != expected[key] or saved[key] != expected[key]:
                    raise ValueError("stale or mismatched source/prompt binding")
            if canonical(saved["payload"]) != canonical(expected["payload"]) or canonical(saved["request"]) != canonical(expected["request"]):
                raise ValueError("frozen request/payload mismatch")
            if type(record["candidate"]) is not str or (record["status"] != "preview" and record["candidate"] != ""):
                raise ValueError("failed/abstained/preflight candidates must remain blank")
            if record["result"] is None:
                if record["status"] != "preflight_only" or record["attempts"] != 0:
                    raise ValueError("missing result for executed case")
            else:
                data = dict(record["result"])
                data["translations"] = tuple(tuple(p) for p in data["translations"])
                data["outcomes"] = tuple(UnitOutcome(**{**o, "reasons": tuple(o["reasons"])}) for o in data["outcomes"])
                result = TranslationResult(**data)
                result.assert_current(request)
                if (result.model != plan["model"] or result.profile != capability.name or result.status != record["status"]
                        or result.attempts != record["attempts"] or result.attempts not in (0, 1)
                        or record["candidate"] != dict(result.translations).get(request.target_ids[0], "")):
                    raise ValueError("candidate/status/model does not match bound result")
        profile_prompts = [r["prompt_sha256"] for r in records]
        if plan["profile"] in prompts and prompts[plan["profile"]] != profile_prompts:
            raise ValueError("same-profile prompts must be identical")
        prompts[plan["profile"]] = profile_prompts
        runs.append((path, plan, records))
    output = new_output(output)
    write_review(output, cases, runs)
    summary = {"paired_cases": len(cases), "runs": len(runs), "records_per_run": [len(r) for _, _, r in runs],
        "semantic_quality_claimed": False, "generation_options": shared_options,
        "metrics": [score_records(cases, records) for _, _, records in runs] if score else {**metric_availability(), "computed": False}}
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model-record", type=Path)
    parser.add_argument("--profile", choices=("generalist", "translategemma"), default="generalist")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--run-local-model", action="store_true")
    parser.add_argument("--max-cases", type=int, default=150)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--keep-alive", type=int, default=60)
    parser.add_argument("--compare", nargs="+", type=Path, help="offline exact-pair review of saved run directories")
    parser.add_argument("--score", action="store_true", help="optional installed-only SacreBLEU diagnostic, all-case denominator")
    args = parser.parse_args(argv)
    try:
        if args.compare:
            if args.run_local_model or args.model or args.model_record:
                raise ValueError("comparison is saved-record-only; no model flags")
            result = compare(args.cases, args.output, args.compare, max_cases=args.max_cases, score=args.score)
        else:
            if not args.model or args.model_record is None:
                raise ValueError("run/preflight requires --model and --model-record")
            result = run(args.cases, args.output, model=args.model, model_record=args.model_record,
                profile=args.profile, endpoint=args.endpoint, run_local_model=args.run_local_model,
                max_cases=args.max_cases, num_predict=args.num_predict, keep_alive=args.keep_alive, score=args.score)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, "translation quality helper failed: " + str(error) + "\n")
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
