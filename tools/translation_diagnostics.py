"""Saved-evidence translation preflight; explicit bounded parent-only live opt-in."""
import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tralens.translation.contracts import (Style, TranslationRequest, automatic_units,
                                           canonical, strict_json)
from tralens.translation.runtime import (GENERALIST, TRANSLATEGEMMA, CapabilityProfile,
    OllamaRuntime, PreviewTranslator, compact_unit, compile_source, GenerationOptions,
    validate_model_profile, make_payload, execution_metadata, local_endpoint, TRANSLATEGEMMA_PROVENANCE)


def read_bounded(path, limit):
    path = Path(path)
    if not path.is_file():
        raise ValueError("explicit evidence input must be a regular file")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("evidence input exceeds byte budget")
    return content


def preflight(manifest, *, selected_ids=(), max_units=3, mode="A", max_context_chars=4000,
              style=Style()):
    if type(max_units) is not int or not 1 <= max_units <= 20:
        raise ValueError("max_units must be 1..20")
    if type(max_context_chars) is not int or not 0 <= max_context_chars <= 8000:
        raise ValueError("context character budget must be 0..8000")
    if mode not in ("A", "B", "C"):
        raise ValueError("unknown experiment mode")
    pages = []
    all_ids = []
    for page in manifest["pages"]:
        candidates = [c for entry in page.get("units", ())
                      for c in entry.get("recognition", {}).get("candidates", ())]
        units = automatic_units(page["batch"], candidates)
        explicit_order = page.get("translation_page_order")
        status = "saved_proposal_spatial_order_unverified"
        if explicit_order is not None:
            order = explicit_order["unit_ids"]
            if len(order) != len(units) or set(order) != {u.unit_id for u in units}:
                raise ValueError("explicit page order must be an exact proposal permutation")
            by_id = {u.unit_id: u for u in units}
            units = tuple(by_id[i] for i in order)
            status = explicit_order["status"]
        bounds = {p["proposal_id"]: p["crop_bounds"] for p in page["batch"]["proposals"]}
        pages.append((units, bounds, status))
        all_ids.extend(u.unit_id for u in units)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate page/unit IDs")
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= set(all_ids):
        raise ValueError("selected IDs must be unique known units")
    chosen = tuple(selected_ids) if selected_ids else tuple(all_ids[:max_units])
    if len(chosen) > max_units:
        raise ValueError("selection exceeds unit budget")
    requests, dispositions = [], []
    lookup = {u.unit_id: (u, units, bounds, status) for units, bounds, status in pages for u in units}
    for chosen_id in chosen:
        unit, units, bounds, status = lookup[chosen_id]
        page_order = tuple(u.unit_id for u in units)
        if len(canonical(compact_unit(unit, page_order))) > 32768:
            raise ValueError("selected unit exceeds 32768-character serialized source budget")
        dispositions.append({"unit_id": unit.unit_id, "disposition": unit.disposition,
                             "blocked_reasons": unit.blocking_reasons,
                             "unit_revision": unit.revision})
        context = []
        if mode in ("B", "C"):
            target = bounds[unit.unit_id]

            def distance(other):
                box = bounds[other.unit_id]
                return ((box[0] + box[2] - target[0] - target[2]) ** 2
                        + (box[1] + box[3] - target[1] - target[3]) ** 2,
                        page_order.index(other.unit_id))

            for other in sorted(units, key=distance):
                if other.unit_id == unit.unit_id or other.disposition == "blocked":
                    continue
                # Budget exactly the compact list, including alternatives/labels.
                size = len(canonical([compact_unit(u, page_order) for u in (*context, other)]))
                if size <= max_context_chars:
                    context.append(other)
            context.sort(key=lambda u: page_order.index(u.unit_id))
        requests.append(TranslationRequest((unit, *context), (unit.unit_id,),
                         tuple(u.unit_id for u in context), style, mode, page_order, status))
    return tuple(requests), dispositions


def image_presentation(request, path, *, max_side=1536):
    """Explicit original only; bind original bytes/pixels and derived PNG, never write source."""
    import base64
    from io import BytesIO
    from PIL import Image
    if request.mode != "C" or any(u.disposition == "blocked" for u in request.units):
        raise ValueError("image condition requires structurally safe C targets/context")
    if type(max_side) is not int or not 0 <= max_side <= 2048:
        raise ValueError("image max side must be 0..2048; zero disables resizing")
    path = Path(path)
    original = read_bounded(path, 32 * 1024 * 1024)
    page = strict_json(request.units[0].page_json)
    digest = hashlib.sha256(original).hexdigest()
    if digest != page["image_sha256"]:
        raise ValueError("original image file hash mismatch")
    with Image.open(BytesIO(original)) as opened:
        if opened.width * opened.height > 40_000_000 or getattr(opened, "n_frames", 1) != 1:
            raise ValueError("image dimensions/frames exceed presentation bounds")
        image = opened.convert("RGB")
    pixels = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()).hexdigest()
    if pixels != page["pixels_sha256"]:
        raise ValueError("original image pixel hash mismatch")
    import math
    for unit in request.units:
        raw = next((r for r in unit.readings if r.origin == "raw_main_reading"), None)
        if raw is None:
            raise ValueError("C requires original raw-main geometry evidence")
        for detection in strict_json(raw.evidence_json):
            polygon = detection["polygon"]
            if (len(polygon) < 3 or any(len(p) != 2 or any(type(v) not in (int, float)
                    or not math.isfinite(v) for v in p) or not 0 <= p[0] <= image.width
                    or not 0 <= p[1] <= image.height for p in polygon)):
                raise ValueError("unsafe target/context source geometry")
            area = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(polygon, polygon[1:] + polygon[:1]))
            if abs(area) < 1e-7:
                raise ValueError("degenerate target/context source geometry")
    original_size = image.size
    if max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    elif max(image.size) > 4096:
        raise ValueError("unresized presentation exceeds 4096 pixels per side")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    presentation = buffer.getvalue()
    if len(presentation) > 8 * 1024 * 1024:
        raise ValueError("presentation exceeds 8 MiB")
    evidence = {"original_path": str(path.resolve()), "original_sha256": digest,
                "original_pixels_sha256": pixels, "original_size": original_size,
                "presentation_sha256": hashlib.sha256(presentation).hexdigest(),
                "presentation_size": image.size, "format": "PNG", "max_side": max_side,
                "transform": "RGB conversion; bounded LANCZOS thumbnail" if image.size != original_size else "RGB conversion"}
    return replace(request, image_evidence_json=canonical(evidence)), base64.b64encode(presentation).decode("ascii")


def load_specialist_template(path, *, model, model_digest):
    """Parent-supplied literal template only. Never execute code or invent a template."""
    from string import Template
    path = Path(path)
    spec = strict_json(read_bounded(path, 65536).decode("utf-8"))
    if (spec["encoding"] != "literal_substitution_v1" or spec["model"] != model
            or spec["model_digest"] != model_digest or not spec["verified_by"]
            or not spec["verification_source"] or spec["schema_version"] != 1
            or hashlib.sha256(spec["template"].encode()).hexdigest() != spec["template_sha256"]):
        raise ValueError("specialist template verification/identity mismatch")
    template = Template(spec["template"])
    placeholders = [match.group("named") or match.group("braced") for match in template.pattern.finditer(spec["template"])
                    if match.group("named") or match.group("braced")]
    if (set(placeholders) != {"source_lang_code", "target_lang_code", "text"}
            or placeholders.count("text") != 1):
        raise ValueError("template requires explicit language fields and exactly one text substitution")
    template.substitute(source_lang_code="ja", target_lang_code="en", text="test")
    return lambda **fields: template.substitute(**fields), spec


def retain_response(output, name, raw_response, limit):
    content = raw_response.encode("utf-8")
    truncated = len(content) > limit
    name += ".txt" if truncated else ".json"
    with (output / name).open("xb") as stream:
        stream.write(content[:limit])
    parsed = strict_json(raw_response)
    metrics = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
               "prompt_eval_count", "eval_count", "done", "done_reason", "model", "created_at")
    return {"raw_response_path": name, "raw_response_sha256": hashlib.sha256(content).hexdigest(),
            "raw_response_bytes": len(content), "retained_bytes": min(len(content), limit),
            "truncated": truncated, "generation_metadata":
                {k: parsed[k] for k in metrics if k in parsed} if type(parsed) is dict else {}}


def run(manifest_path, output, *, runtime=None, run_local_model=False,
        endpoint="http://127.0.0.1:11434", model="qwen3.8:latest", model_digest="",
        model_details="", profile=GENERALIST, max_attempts=3, timeout=60,
        preload=False, images=None, image_max_side=1536, specialist_template=None,
        max_response_bytes=1024 * 1024, num_ctx=8192, num_predict=2048, keep_alive=300,
        max_output_chars=32768, **options):
    generation_options = GenerationOptions(num_ctx, num_predict, keep_alive, max_output_chars)
    endpoint = local_endpoint(endpoint)
    if (not isinstance(model_digest, str) or len(model_digest) > 64
            or not isinstance(model_details, str) or len(model_details.encode("utf-8")) > 65536):
        raise ValueError("parent model identity assertions exceed bounded string fields")
    if type(max_attempts) is not int or not 0 <= max_attempts <= 20:
        raise ValueError("global attempt budget must be 0..20")
    if type(timeout) not in (int, float) or not 0 < timeout <= 300:
        raise ValueError("timeout must be 0..300 seconds")
    if run_local_model and (not model_digest.strip() or not model_details.strip()):
        raise ValueError("live opt-in requires parent-verified model digest and details")
    if preload and (not run_local_model or max_attempts == 0):
        raise ValueError("preload requires live opt-in and consumes one attempt")
    if type(max_response_bytes) is not int or not 1024 <= max_response_bytes <= 4 * 1024 * 1024:
        raise ValueError("raw response retention bound must be 1024..4194304 bytes per attempt")
    template_encoder, template_spec = None, None
    if specialist_template is not None:
        if not profile.specialist_template:
            raise ValueError("specialist template requires specialist profile")
        template_encoder, template_spec = load_specialist_template(specialist_template, model=model, model_digest=model_digest)
        profile = replace(profile, name="parent_verified_raw_specialist", specialist_chat=False)
    validate_model_profile(model, profile)
    path = Path(manifest_path)
    raw = read_bounded(path, 16 * 1024 * 1024)
    manifest = strict_json(raw.decode("utf-8"))
    requests, dispositions = preflight(manifest, **options)
    presentations = {}
    if options.get("mode", "A") == "C":
        if not profile.images:
            raise ValueError("C requires an explicitly verified image-capable profile")
        bound = []
        for request in requests:
            if any(u.disposition == "blocked" for u in request.units):
                bound.append(request)
                continue
            page_id = strict_json(request.units[0].page_json)["page_id"]
            if images is None or page_id not in images:
                raise ValueError("C requires an explicit original image mapping for page " + page_id)
            request, presentation = image_presentation(request, images[page_id], max_side=image_max_side)
            presentations[request.revision] = presentation
            bound.append(request)
        requests = tuple(bound)
    elif images:
        raise ValueError("explicit images are supported only in C")
    for request in requests:
        if all(u.disposition != "blocked" for u in request.units):
            make_payload(request, profile,
                images=(presentations[request.revision],) if request.revision in presentations else ())
    output = Path(output)
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    # Exclusive new directory: no existing evidence or experiment can be overwritten.
    output.mkdir(exist_ok=False)
    presentation_paths = {}
    if presentations:
        import base64
        for request_revision, encoded in presentations.items():
            content = base64.b64decode(encoded, validate=True)
            name = "image-" + hashlib.sha256(content).hexdigest() + ".png"
            if name not in presentation_paths.values():
                with (output / name).open("xb") as stream:
                    stream.write(content)
            presentation_paths[request_revision] = name
    plan = {"schema_version": 3, "preview_only": True,
            "saved_evidence_sha256": hashlib.sha256(raw).hexdigest(),
            "preflight_scope": "saved evidence consistency only; no pixel/semantic/model verification",
            "input_page_errors": manifest.get("page_errors", []), "dispositions": dispositions,
            "requests": [{"revision": r.revision, "request": asdict(r),
                "execution_metadata": strict_json(execution_metadata(r, profile, generation_options))} for r in requests],
            "model_sources": [{"request_revision": r.revision, "source": compile_source(r)
                if not profile.specialist_template else {"primary_text": r.units[0].selected.text,
                    "alternatives_supplied_to_model": False}} for r in requests],
            "selected_ids": [i for r in requests for i in r.target_ids],
            "specialist_template": template_spec, "max_response_bytes_per_attempt": max_response_bytes,
            "native_template_provenance": TRANSLATEGEMMA_PROVENANCE if profile.specialist_chat else None,
            "generation_options": asdict(generation_options),
            "image_presentations": presentation_paths,
            "run_local_model": run_local_model, "model": model, "model_digest_parent_asserted": model_digest,
            "endpoint": endpoint, "endpoint_policy": "literal loopback/private/overlay scope; localhost pinned to 127.0.0.1",
            "server_authentication": "not verified; HTTP origin and model digest remain parent assertions",
            "model_details_parent_asserted": model_details, "profile": asdict(profile),
            "max_attempts": max_attempts, "timeout_per_batch_seconds": timeout,
            "cancellation": "between calls only; socket timeout cannot guarantee remote cancellation"}
    (output / "requests.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    attempts = 0
    counts = {}
    unit_counts = {}
    review_batches = 0
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        def append(record):
            stream.write(canonical(record) + "\n")
            stream.flush()
        if run_local_model:
            runtime = runtime if runtime is not None else OllamaRuntime(endpoint, specialist_template=template_encoder)
        if preload:
            attempts += 1
            try:
                timing = runtime.preload(model, timeout, generation_options=generation_options)
                raw_response = timing.pop("response_json", None)
                evidence = retain_response(output, "preload-response", raw_response, max_response_bytes) if raw_response else {}
                append({"kind": "preload", "status": "loaded", "timing": timing, "generation_evidence": evidence})
            except Exception as error:
                raw_response = getattr(error, "response_json", None)
                evidence = retain_response(output, "preload-error-response", raw_response, max_response_bytes) if raw_response else {}
                append({"kind": "preload", "status": "failed", "error": type(error).__name__, "generation_evidence": evidence})
        for batch_index, request in enumerate(requests):
            observations = []
            def observe(event):
                raw_response = event.pop("response_json", None)
                evidence = {**event, "request_revision": request.revision}
                if raw_response is not None:
                    name = f"batch-{batch_index:03d}-attempt-{event['attempt']:02d}-response"
                    evidence.update(retain_response(output, name, raw_response, max_response_bytes))
                observations.append(evidence)
            if any(u.disposition == "blocked" for u in request.units):
                record = {"request_revision": request.revision, "status": "blocked", "attempts": 0}
            elif not run_local_model:
                record = {"request_revision": request.revision, "status": "preflight_only", "attempts": 0}
            elif attempts >= max_attempts:
                record = {"request_revision": request.revision, "status": "budget_exhausted", "attempts": 0}
            else:
                # Exactly one call per batch. Errors consume the same global budget.
                result = PreviewTranslator(runtime, model=model, profile=profile, timeout=timeout,
                    generation_options=generation_options).translate(
                    request, images=(presentations[request.revision],) if request.revision in presentations else (),
                    observe=observe)
                record = asdict(result)
                attempts += result.attempts
            record["generation_evidence"] = observations
            review_batches += int(record.get("review_needed", False))
            for outcome in record.get("outcomes", ()):
                unit_counts[outcome["disposition"]] = unit_counts.get(outcome["disposition"], 0) + 1
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            append({"kind": "translation_batch", **record})
    summary = {"preview_only": True, "selected_units": len(requests), "attempts": attempts,
               "counts": counts, "unit_dispositions": unit_counts, "review_needed_batches": review_batches,
               "preload_separate_from_generation": True, "semantic_quality_assessed": False,
               "timing_scope": "per-call evidence; repeated-prefix warm timings are cached lower bounds, not page throughput"}
    (output / "summary.json").write_text(canonical(summary) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unit-id", action="append", default=[])
    parser.add_argument("--selection", type=Path, help="saved JSON list of target IDs, preserving caller order")
    parser.add_argument("--max-units", type=int, default=3)
    parser.add_argument("--mode", choices=("A", "B", "C"), default="A")
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--source-language", default="Japanese")
    parser.add_argument("--target-language", default="English")
    parser.add_argument("--run-local-model", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.8:latest")
    parser.add_argument("--model-digest", default="")
    parser.add_argument("--model-details", default="")
    parser.add_argument("--model-record", type=Path, help="parent-saved model.json; no daemon lookup")
    parser.add_argument("--profile", choices=("generalist", "translategemma"), default="generalist")
    parser.add_argument("--vision-capable", action="store_true", help="parent asserts selected generalist supports images")
    parser.add_argument("--image", action="append", default=[], metavar="PAGE_ID=ORIGINAL_PATH")
    parser.add_argument("--image-max-side", type=int, default=1536)
    parser.add_argument("--specialist-template", type=Path)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--keep-alive", type=int, default=300, help="bounded residency in seconds, 0 unloads after request")
    parser.add_argument("--max-output-chars", type=int, default=32768)
    parser.add_argument("--max-response-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.model_record:
            metadata = strict_json(read_bounded(args.model_record, 65536).decode("utf-8"))
            if (metadata["model"] != args.model or not isinstance(metadata["digest"], str)
                    or len(metadata["digest"]) != 64
                    or any(c not in "0123456789abcdef" for c in metadata["digest"])
                    or (args.model_digest and args.model_digest != metadata["digest"])):
                raise ValueError("parent model record identity mismatch")
            if args.vision_capable and "vision" not in metadata.get("capabilities", []):
                raise ValueError("parent model record does not support asserted vision capability")
            args.model_digest, args.model_details = metadata["digest"], canonical(metadata)
        selected = tuple(args.unit_id)
        if args.selection:
            if selected:
                raise ValueError("selection cannot be combined with --unit-id")
            value = strict_json(read_bounded(args.selection, 65536).decode("utf-8"))
            if type(value) is not list or any(type(i) is not str for i in value):
                raise ValueError("selection must be a JSON string-ID list")
            selected = tuple(value)
        image_paths = {}
        for item in args.image:
            page_id, separator, path = item.partition("=")
            if not separator or not page_id or not path or page_id in image_paths:
                raise ValueError("images require unique PAGE_ID=ORIGINAL_PATH mappings")
            image_paths[page_id] = path
        if args.vision_capable and args.profile != "generalist":
            raise ValueError("specialist profile has no verified image support")
        profile = (CapabilityProfile("parent_verified_generalist_vision", True, True, True)
                   if args.vision_capable else GENERALIST) if args.profile == "generalist" else TRANSLATEGEMMA
        result = run(args.manifest, args.output, selected_ids=selected, max_units=args.max_units,
            mode=args.mode, max_context_chars=args.max_context_chars,
            style=Style(source_language=args.source_language, target_language=args.target_language),
            run_local_model=args.run_local_model, endpoint=args.endpoint, model=args.model,
            model_digest=args.model_digest, model_details=args.model_details,
            profile=profile, images=image_paths, image_max_side=args.image_max_side,
            specialist_template=args.specialist_template, max_response_bytes=args.max_response_bytes,
            num_ctx=args.num_ctx, num_predict=args.num_predict, keep_alive=args.keep_alive,
            max_output_chars=args.max_output_chars,
            max_attempts=args.max_attempts, timeout=args.timeout, preload=args.preload)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, "translation diagnostics failed: " + str(error) + "\n")
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
