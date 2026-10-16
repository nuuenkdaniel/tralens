"""Offline, read-only HTML review of a hash-bound workflow preview. No engines or scripts."""
import argparse
import base64
from collections import Counter
from dataclasses import asdict
import hashlib
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from tralens.translation.contracts import (Reading, Utterance, TranslationRequest,
    TranslationResult, UnitOutcome, FragmentRequest, FragmentResult, fragment_plan,
    automatic_units, canonical, strict_json, revision, FRAGMENT_POLICY_REVISION)
from tralens.translation import ocr_review as visual
from tralens.translation.runtime import (GENERALIST, TRANSLATEGEMMA, GenerationOptions, execution_metadata)
from tralens.translation.workflow import fragment_scheduling_info


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    # JSON equality otherwise conflates True/1 and integer/string source IDs.
    require(canonical(actual) == canonical(expected), message)


def bounded_tree(value, depth=0, budget=None):
    """Bound even unused retained metadata; never interpret its strings as resources."""
    if budget is None:
        budget = [0]
    budget[0] += 1
    require(depth <= 48 and budget[0] <= 500_000, "JSON nesting/item budget exceeded")
    if type(value) is dict:
        require(len(value) <= 4096, "JSON object budget exceeded")
        bools = {"preview_only", "binding_current", "review_needed", "crop_safe", "reviewed",
                 "human_reviewed", "run_local_model", "allow_fragment_previews", "review_ocr",
                 "thinking", "truncated", "response_available"}
        for key, item in value.items():
            require(type(key) is str and len(key) <= 4096, "invalid JSON key")
            if key in bools:
                require(type(item) is bool, "boolean required: " + key)
            if key == "raw_response_path":
                safe_sidecar(item)
            bounded_tree(item, depth + 1, budget)
    elif type(value) is list:
        require(len(value) <= 8192, "JSON array budget exceeded")
        for item in value:
            bounded_tree(item, depth + 1, budget)
    elif type(value) is str:
        require(len(value) <= 1024 * 1024, "JSON string budget exceeded")
    elif type(value) in (int, float):
        require(math.isfinite(value), "nonfinite number")
    else:
        require(value is None or type(value) is bool, "unsupported JSON type")


def safe_sidecar(value):
    require(type(value) is str and value not in ("", ".", "..") and len(value) <= 255
            and not any(c in value for c in ("/", "\\", ":", "\x00")),
            "invalid sidecar name; sidecars are never opened")


def count(value, limit=20):
    require(type(value) is int and 0 <= value <= limit, "invalid attempt/count budget")
    return value


def check_count(mapping, key, expected):
    if key in mapping:
        same(mapping[key], expected, "inconsistent counter: " + key)


def read_bounded(path, limit):
    path = Path(path)
    if not path.is_file():
        raise ValueError("explicit input must be a regular file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("input exceeds byte budget")
    return raw


def _unit(data):
    data = dict(data)
    data["readings"] = tuple(Reading(**r) for r in data["readings"])
    for key in ("member_ids", "order", "structural_reasons", "lexical_reasons", "blocked_reasons"):
        data[key] = tuple(data[key])
    return Utterance(**data)


def _result(data, cls=TranslationResult):
    data = dict(data)
    pairs = "translations" if cls is TranslationResult else "fragments" if cls is FragmentResult else "alternates"
    data[pairs] = tuple(tuple(p) for p in data[pairs])
    data["outcomes"] = tuple(UnitOutcome(**{**o, "reasons": tuple(o["reasons"])}) for o in data["outcomes"])
    count(data["attempts"])
    require(type(data["generation_seconds"]) in (int, float) and data["generation_seconds"] >= 0,
            "invalid generation duration")
    for key in ("failure", "detail", "provider", "model", "profile", "request_revision"):
        require(type(data[key]) is str, "result strings required")
    return cls(**data)


def check_execution(record, request, original, prefix="", cls=TranslationResult, settings=None):
    require(record["binding_current"] is True, "stale source binding")
    same(record["request"], asdict(request), "request/source/style binding drift")
    same(record["request_revision"], request.revision, "request revision drift")
    started = count(record["attempts_started"], 1)
    evidence = record["generation_evidence"]
    require(type(evidence) is list and len(evidence) <= 2, "generation evidence budget exceeded")
    allowed = {prefix + s for s in ("pending", "not_selected", "budget_exhausted", "ocr_only",
               "not_executed", "interrupted", "preview", "abstained", "blocked", "failed", "cancelled")}
    require(type(record["status"]) is str and record["status"] in allowed, "unknown input status")
    result = _result(record["result"], cls) if record["result"] is not None else None
    if result is not None:
        if cls is TranslationResult:
            result.assert_current(request)
        else:
            result.assert_current(request, original)
        require(result.status == record["status"] and result.attempts <= started, "result/status/attempt mismatch")
        if settings is not None:
            profile = {"generalist": GENERALIST, "translategemma": TRANSLATEGEMMA}[settings["profile"]]
            same(result.model, settings["model"], "result model drift")
            same(result.profile, profile.name, "result profile drift")
            same(result.execution_metadata_json, execution_metadata(request, profile,
                 GenerationOptions(**settings["generation_options"])), "result execution metadata drift")
    else:
        require(record["status"] not in {prefix + s for s in ("preview", "abstained", "failed", "cancelled")},
                "output status has no retained result")
    if record["status"] in {prefix + s for s in ("pending", "not_selected", "budget_exhausted", "ocr_only", "blocked")}:
        require(started == 0 and result is None, "unexecuted/blocked input has execution evidence")
    return result


def check_review(stage, units, records, preview, rgb, image_bytes):
    settings = preview["settings"]
    limit = count(settings["max_review_attempts"])
    require(settings["review_ocr"] is True and settings["run_local_model"] is True, "review requires model opt-in")
    require(stage["status"] in {"not_started", "running", "complete", "complete_with_failures", "release_failed",
                               "interrupted_or_binding_failed"}, "unknown review phase status")
    rows = stage["units"]
    if not rows and stage["status"] == "not_started":
        require(count(stage["attempts"]) == count(stage["control_calls"], 1) == 0, "unstarted review has calls")
        return {}
    same([r["unit_id"] for r in rows], [u.unit_id for u in units], "review coverage/order mismatch")
    same(stage["model"], settings["review_model"], "review model mismatch")
    same(stage["policy_revision"], visual.REVIEW_POLICY, "unknown review policy")
    same(stage["options"], asdict(visual.REVIEW_OPTIONS), "review options drift")
    same(stage["thinking"], False, "review thinking flag drift")
    same(stage["timeout"], settings["review_timeout"], "review timeout drift")
    visual.validate_options(True, stage["model"], settings["model"], limit, stage["timeout"], True)
    validated = {}
    selected = settings["selected_ids"]
    for unit, row in zip(units, rows):
        same(row["original_unit_revision"], unit.revision, "review original revision drift")
        if unit.unit_id in records:
            same(records[unit.unit_id].get("ocr_review"), row, "duplicated review row drift")
        started = count(row["attempts_started"], 1)
        require(row["status"] in {"review_pending", "review_skipped", "review_failed", "review_cancelled",
                "review_interrupted", "review_binding_or_phase_failed", "review_unchanged",
                "review_proposed_correction", "review_unreadable", "review_uncertain"}, "unknown review status")
        reject = "not_selected" if selected and unit.unit_id not in selected else visual.target_rejection(unit, settings["allow_fragment_previews"])
        if reject:
            require(row["status"] == "review_skipped" and row["skip_reason"] == reject, "review bypassed selection/structural gate")
        require(type(row["generation_evidence"]) is list and len(row["generation_evidence"]) <= 1,
                "review evidence budget exceeded")
        request = verdict = None
        if row["request"] is not None:
            require(not reject, "ineligible visual review request")
            # Regenerate from explicit source pixels, never open row['views'][*]['path'].
            request, views = visual.prepare_input(unit, rgb, image_bytes,
                preview["ocr"]["raw_detections"][0], stage["model"])
            same(row["request"], asdict(request), "review source/crop/view binding drift")
            same(row["request_revision"], request.revision, "review request revision drift")
            expected_body = revision({"model": request.model, "stream": False, "think": False,
                                      **visual.review_payload(request, views)})
            same(row["requested_http_body_sha256"], expected_body, "review requested HTTP body drift")
            binding = strict_json(request.binding_json)
            require(len(row["views"]) == 2, "review needs two retained view records")
            for i, view in enumerate(row["views"]):
                safe_sidecar(view["path"])
                same(view["sha256"], binding["view_sha256"][i], "review view hash drift")
                same(view["pixels_sha256"], binding["view_pixels_sha256"][i], "review view pixel drift")
        else:
            require(started == 0 and not row["views"] and row["request_revision"] is None, "review call without request")
        if row["verdict"] is not None:
            require(request is not None and started == 1, "verdict without dispatched request")
            data = row["verdict"]
            verdict = visual.ReviewVerdict(**{**data, "source_ids": tuple(data["source_ids"]), "reasons": tuple(data["reasons"])})
            same(verdict.request_revision, request.revision, "verdict request drift")
            same(verdict.target_id, unit.unit_id, "verdict target drift")
            same(verdict.source_ids, unit.member_ids, "verdict source drift")
            same(verdict.original_text, unit.selected.text, "verdict original echo drift")
            same(row["verdict_revision"], verdict.revision, "verdict revision drift")
            require(row["status"] == "review_" + verdict.decision, "verdict/status mismatch")
            evidence = row["generation_evidence"]
            require(len(evidence) == 1 and evidence[0].get("raw_response_sha256") == verdict.raw_response_sha256,
                    "verdict/retained response digest mismatch")
        else:
            require(row["verdict_revision"] is None and row["status"] not in {"review_" + s for s in
                    ("unchanged", "proposed_correction", "unreadable", "uncertain")}, "review verdict missing")
        if row["status"] in ("review_pending", "review_skipped"):
            require(started == 0, "unexecuted review has attempts")
        if row["skip_reason"] == "review_budget_exhausted":
            require(stage["attempts"] == limit, "review budget skip without exhausted budget")
        validated[unit.unit_id] = (request, verdict)
    same(stage["attempts"], sum(r["attempts_started"] for r in rows), "review attempt counter drift")
    require(count(stage["attempts"]) <= limit, "review budget exceeded")
    check_count(stage, "status_counts", dict(Counter(r["status"] for r in rows)))
    controls = count(stage["control_calls"], 1)
    residency = stage["residency"]
    release = residency.get("release")
    ownership = residency["ownership"]
    require(ownership in ("caller_injected", "workflow"), "unknown review ownership")
    if ownership == "caller_injected":
        require(controls == 0 and release in (None, "caller_responsibility; no unload dispatched"), "caller-owned reviewer release contradiction")
    elif release is not None:
        require((stage["attempts"] > 0 and controls == 1 and release in ("acknowledged", "failed")) or
                (stage["attempts"] == 0 and controls == 0 and release == "not_loaded"), "review release counter contradiction")
    if release == "failed":
        require(stage["status"] == "release_failed" and preview["attempts"] == 0, "translation after failed reviewer release")
    return validated


def check_schedule(preview, entries):
    """Check retained scheduling references without executing the workflow."""
    if "input_schedule" not in preview:
        return
    selected = preview["settings"]["selected_ids"]
    by_id = {e["unit"].unit_id: e for e in entries}
    originals, alternates, fragments = [], [], []
    for identifier in selected or tuple(by_id):
        entry = by_id[identifier]
        unit, record = entry["unit"], entry["record"]
        if unit.disposition != "blocked":
            originals.append((unit, record, "strict_inputs_first"))
        for child in record.get("visual_alternates", []):
            if unit.disposition == "preview":
                alternates.append((unit, child, "visual_alternate_after_strict_originals"))
        for child in record.get("fragment_previews", []):
            expected = fragment_scheduling_info(unit.selected.text, explicit_selection=bool(selected))
            actual = {k: v for k, v in child["scheduling"].items() if k != "planned_position"}
            same(actual, expected, "fragment scheduling evidence drift")
            fragments.append((unit, child, expected["reason"]))
    if not selected:
        fragments.sort(key=lambda job: job[1]["scheduling"]["priority_tier"])
    jobs = originals + alternates
    for job in fragments:
        for child in by_id[job[0].unit_id]["record"].get("visual_alternates", []):
            jobs.append((job[0], child, "visual_alternate_before_original_fragment"))
        jobs.append(job)
    expected = []
    for position, (unit, record, reason) in enumerate(jobs, 1):
        if record["input_kind"] != "strict_utterance":
            same(record["scheduling"]["planned_position"], position, "planned position drift")
            same(record["scheduling"]["reason"], reason, "alternate scheduling reason drift")
        expected.append({"planned_position": position, "input_kind": record["input_kind"],
            "original_unit_id": unit.unit_id, "request_revision": record["request_revision"],
            "source_ids": list(unit.member_ids), "reason": reason})
    same(preview["input_schedule"], expected, "input schedule/source/request binding drift")


def _polygon(polygon, width, height):
    if (type(polygon) is not list or not 3 <= len(polygon) <= 64 or any(
            type(p) is not list or len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)
            or not 0 <= p[0] <= width or not 0 <= p[1] <= height for p in polygon)):
        return False
    return abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(polygon, polygon[1:] + polygon[:1]))) > 1e-7


def _bounds(bounds, width, height):
    return (type(bounds) is list and len(bounds) == 4 and all(type(v) is int for v in bounds)
            and 0 <= bounds[0] < bounds[2] <= width and 0 <= bounds[1] < bounds[3] <= height)


def crop_reason(proposal, detections, unit, size):
    """Conservative presentation filter, never bubble ownership/completeness proof."""
    width, height = size
    bounds = proposal["crop_bounds"]
    if (unit.disposition == "blocked" or proposal["crop_safe"] is not True or proposal["crop_conflicts"]
            or not _bounds(bounds, width, height)):
        return "blocked, unsafe or conflicting proposal; use full page"
    x0, y0, x1, y1 = bounds
    for raw in detections:
        if not _polygon(raw["polygon"], width, height):
            return "invalid raw geometry prevents an isolation check; use full page"
        xs, ys = zip(*raw["polygon"])
        if raw["source_id"] in proposal["member_ids"]:
            if min(xs) < x0 or max(xs) > x1 or min(ys) < y0 or max(ys) > y1:
                return "crop does not contain member geometry; use full page"
        elif max(min(xs), x0) < min(max(xs), x1) and max(min(ys), y0) < min(max(ys), y1):
            return "nonmember geometry may intersect the crop; use full page"
    return ""


def validate(preview, image_path, rgb, image_hash, image_bytes):
    """Validate saved consistency, not OCR accuracy, ownership truth or approval."""
    width, height = rgb.size
    pixels = hashlib.sha256(f"RGB:{width}:{height}:".encode() + rgb.tobytes()).hexdigest()
    source = preview["input"]
    bounded_tree(preview)
    schema = preview["schema_version"]
    require(type(schema) is int and schema in (1, 2, 3), "unsupported preview schema")
    require(preview["status"] in ("running", "complete", "complete_with_failures", "failed", "interrupted"),
            "unknown workflow status")
    require(preview["stage"] in ("ocr", "ocr_review", "translation", "finished"), "unknown workflow stage")
    if (preview["preview_only"] is not True
            or source["sha256"] != image_hash or source["pixels_sha256"] != pixels
            or canonical(source["size"]) != canonical([width, height])):
        raise ValueError("original image/hash/pixel provenance mismatch")
    # Only the explicitly supplied image is opened. Never resolve a path from JSON.
    if source["path"] != str(image_path):
        raise ValueError("saved original path must match the explicit image; no path substitution")
    settings = preview["settings"]
    require(type(settings["run_local_model"]) is bool, "model opt-in must be boolean")
    fragments_enabled = settings.get("allow_fragment_previews", False)
    require(type(fragments_enabled) is bool and (schema != 1 or not fragments_enabled), "schema/fragment opt-in contradiction")
    require((schema == 3) == settings.get("review_ocr", False), "schema/review opt-in contradiction")
    require(schema == 3 or "ocr_review" not in preview, "review evidence outside schema 3")
    selected = settings["selected_ids"]
    require(type(selected) is list and all(type(i) is str and i for i in selected)
            and len(set(selected)) == len(selected), "invalid selected IDs")
    limit = count(settings["max_attempts"])
    count(preview["attempts"])
    ocr = preview["ocr"]
    if not isinstance(ocr, dict) or len(ocr["batches"]) != 1 or len(ocr["raw_detections"]) != 1:
        raise ValueError("review needs a retained single OCR page")
    raw = ocr["raw_detections"][0]
    if len(raw) > 2048 or len(preview["units"]) > 512:
        raise ValueError("review detection/unit budget exceeded")
    ids = [r["source_id"] for r in raw]
    if any(type(i) not in (int, str) or i == "" for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("duplicate or invalid raw source IDs")
    batch = ocr["batches"][0]
    if batch is None:
        if raw or preview["units"] or ocr.get("automatic_recognition_results"):
            raise ValueError("missing proposal batch for retained source data")
        require(preview["attempts"] == 0, "empty OCR cannot have translation calls")
        require(not selected, "empty OCR has selected source IDs")
        for key in ("strict_eligible_units", "fragment_candidate_inputs", "strict_attempts", "fragment_attempts",
                    "alternate_attempts", "strict_preview_count", "fragment_preview_count", "alternate_preview_count"):
            check_count(preview, key, 0)
        for key in ("unit_status_counts", "fragment_status_counts", "alternate_status_counts"):
            check_count(preview, key, {})
        if schema == 3:
            check_review(preview["ocr_review"], (), {}, preview, rgb, image_bytes)
        return [], {"scope": "saved source-ID coverage, not detection/translation accuracy", "raw_detections": 0,
                    "main_detections": 0, "annotation_detections": 0, "unassigned_detections": 0,
                    "missing_source_ids": [], "unit_count": 0, "unit_status_counts": {}, "blocked_reason_counts": {}}
    page = batch["page"]
    require(batch["policy"] == "automatic_manga_structure_v2" and batch["reviewed"] is False,
            "unsupported automatic batch policy/review claim")
    if page["image_sha256"] != image_hash or page["pixels_sha256"] != pixels or page["image_path"] != source["path"]:
        raise ValueError("stale or foreign proposal page")
    by_id = {r["source_id"]: r for r in raw}
    detection_ids = [d["source_id"] for d in batch["detections"]]
    if (any(type(i) not in (int, str) or i == "" for i in detection_ids)
            or len(set(detection_ids)) != len(detection_ids) or set(by_id) != set(detection_ids)):
        raise ValueError("raw/batch source IDs differ")
    for r in raw:
        require(_polygon(r["polygon"], width, height), "invalid/nonfinite/out-of-page raw polygon")
        require(type(r["text"]) is str and type(r["selected_text"]) is str and type(r["reason"]) is str
                and type(r["selected_engine"]) is str and type(r["ocr_metadata"]) is dict
                and r["status"] in ("accepted", "review", "rejected")
                and type(r["confidence"]) in (int, float) and math.isfinite(r["confidence"]), "malformed raw source evidence")
    comparison = [{"id": r["source_id"], "text": r["text"], "confidence": float(r["confidence"]),
                   "polygon": r["polygon"]} for r in raw]
    comparison.sort(key=lambda r: (type(r["id"]).__name__, str(r["id"])))
    same(page["comparison_sha256"], hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest(),
         "original detection snapshot hash drift")
    for d in batch["detections"]:
        original = by_id[d["source_id"]]
        for key in ("text", "confidence", "polygon", "status", "reason", "selected_text"):
            same(original[key], d[key], "raw/batch source evidence drift")
        same(strict_json(d["metadata_json"]), original["ocr_metadata"], "raw/batch metadata drift")
        bounded_tree(strict_json(d["metadata_json"]))
    require(type(page["page_id"]) is str and bool(page["page_id"]) and
            type(page["comparison_sha256"]) is str and len(page["comparison_sha256"]) == 64 and
            all(c in "0123456789abcdef" for c in page["comparison_sha256"]), "invalid page identity")
    for p in batch["proposals"]:
        require(_bounds(p["crop_bounds"], width, height), "malformed proposal crop bounds")
        for key in ("member_ids", "main_ids", "annotation_ids"):
            values = p[key]
            require(type(values) is list and all(type(i) in (int, str) and i != "" for i in values)
                    and len(set(values)) == len(values) and set(values) <= set(ids), "invalid proposal membership")
        require(type(p["crop_safe"]) is bool and p["reviewed"] is False
                and p["structural_status"] == "proposal_unverified", "invalid automatic proposal flags/status")
        identity = {"policy": batch["policy"], "page": page, "main_ids": p["main_ids"],
            "members": p["member_ids"], "member_polygons": [(i, by_id[i]["polygon"]) for i in p["member_ids"]],
            "annotation_parents": p["annotation_parents"], "direction": p["direction"]}
        if batch.get("ownership_policy", "production8") != "production8":
            identity["ownership_policy"] = batch["ownership_policy"]
        if batch.get("companion_policy", "disabled") != "disabled":
            identity["companion_policy"] = batch["companion_policy"]
        same(p["proposal_id"], "proposal-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24],
             "proposal membership/identity hash drift")
    require(all(type(row) is list and len(row) == 3 and type(row[0]) in (int, str)
            and row[0] in by_id for row in batch["regions"]), "invalid region binding")
    candidates = ocr.get("automatic_recognition_results", [])
    if candidates and len(candidates) != 1:
        raise ValueError("foreign candidate page")
    if candidates:
        same([r["proposal_id"] for r in candidates[0]], [p["proposal_id"] for p in batch["proposals"]],
             "retained OCR candidate coverage/order drift")
    alternatives = [c for r in candidates[0] for c in r["candidates"]] if candidates else []
    units = automatic_units(batch, alternatives)
    require(len(units) <= 512, "derived unit count budget exceeded")
    retained_records = preview["units"]
    # OCR failure can retain a complete batch before workflow requests were built.
    source_only = not retained_records and bool(units) and preview["status"] in ("failed", "interrupted") and preview["stage"] == "ocr"
    if source_only:
        require(preview["attempts"] == 0, "source-only failure has translation calls")
        retained_records = []
        for unit in units:
            request = TranslationRequest((unit,), (unit.unit_id,))
            retained_records.append({"unit": asdict(unit), "unit_revision": unit.revision,
                "request": asdict(request), "request_revision": request.revision,
                "status": "blocked" if unit.disposition == "blocked" else "not_executed",
                "blocking_reasons": list(unit.blocking_reasons), "binding_current": True,
                "input_kind": "strict_utterance", "result": None, "attempts_started": 0,
                "generation_evidence": [], "fragment_previews": [], "visual_alternates": []})
    if [r["unit"]["unit_id"] for r in retained_records] != [u.unit_id for u in units]:
        raise ValueError("unit/proposal IDs or saved order differ; no units may disappear")
    require(set(selected) <= {u.unit_id for u in units}, "foreign selected unit IDs")
    records = {r["unit"]["unit_id"]: r for r in preview["units"]}
    reviews = check_review(preview["ocr_review"], units, records, preview, rgb, image_bytes) if schema == 3 else {}
    entries = []
    all_fragments, all_alternates = [], []
    for index, (record, expected, proposal) in enumerate(zip(retained_records, units, batch["proposals"]), 1):
        unit = _unit(record["unit"])
        same(record["unit"], asdict(expected), "exact original unit/selected reading drift")
        if unit != expected or record["unit_revision"] != unit.revision:
            raise ValueError("unit/source revision drift")
        if record["blocking_reasons"] != list(unit.blocking_reasons):
            raise ValueError("blocking reasons do not match source evidence")
        if unit.disposition == "blocked" and record["status"] != "blocked":
            raise ValueError("structurally blocked source cannot become output")
        same(record.get("input_kind", "strict_utterance"), "strict_utterance", "strict input kind mismatch")
        request = TranslationRequest((unit,), (unit.unit_id,))
        result = check_execution(record, request, unit, settings=settings)
        if unit.disposition != "blocked":
            if selected and unit.unit_id not in selected:
                require(record["status"] == "not_selected", "nonselected strict input executed")
            else:
                require(record["status"] not in ("blocked", "not_selected"), "eligible selected strict input incorrectly blocked")
        fragment_records = record.get("fragment_previews", [])
        derived, rejected = fragment_plan(unit, allow_fragment_previews=True) if fragments_enabled and unit.disposition == "blocked" else ((), ())
        if not source_only:
            same(record.get("fragment_rejection_reasons", []), rejected, "fragment rejection reasons drift")
            require(len(fragment_records) == len(derived), "fragment candidate coverage drift")
        bases = [request] if unit.disposition == "preview" else []
        for child, fragment in zip(fragment_records, derived):
            same(child["input_kind"], "unverified_text_fragment", "fragment kind drift")
            same(child["input"], asdict(fragment), "fragment original/source binding drift")
            same(child["input_revision"], fragment.revision, "fragment input revision drift")
            base = FragmentRequest(fragment)
            check_execution(child, base, unit, "fragment_", FragmentResult, settings)
            if selected and unit.unit_id not in selected:
                require(child["status"] == "fragment_not_selected", "nonselected fragment executed")
            bases.append(base)
            all_fragments.append(child)
        alternate_records = record.get("visual_alternates", [])
        require(schema == 3 or not alternate_records, "visual alternates outside schema 3")
        review_request, verdict = reviews.get(unit.unit_id, (None, None))
        expected_alternates = []
        if verdict is not None and verdict.decision == "proposed_correction":
            if not selected or unit.unit_id in selected:
                expected_alternates = [visual.AlternateRequest(visual.AlternateInput(base, review_request, verdict)) for base in bases]
        # An aborted review phase has not built translation alternates yet.
        if preview["stage"] in ("translation", "finished"):
            require(len(alternate_records) == len(expected_alternates), "visual alternate coverage drift")
        else:
            require(not alternate_records, "alternate before completed review phase")
        for child, alternate in zip(alternate_records, expected_alternates):
            same(child["input_kind"], "unverified_visual_alternate", "alternate kind drift")
            same(child["input"], asdict(alternate.alternate), "alternate original/review binding drift")
            same(child["input_revision"], alternate.alternate.revision, "alternate input revision drift")
            check_execution(child, alternate, unit, "alternate_", visual.AlternateResult, settings)
            all_alternates.append(child)
        if not _bounds(proposal["crop_bounds"], width, height) or any(
                not _polygon(by_id[i]["polygon"], width, height) for i in unit.member_ids):
            raise ValueError("unit geometry is malformed or outside the original image")
        entries.append({"number": index, "unit": unit, "record": record, "proposal": proposal, "result": result,
                        "crop_omission": crop_reason(proposal, raw, unit, rgb.size)})
    main = {i for p in batch["proposals"] for i in p["main_ids"]}
    annotations = {i for p in batch["proposals"] for i in p["annotation_ids"]}
    unassigned = [r[0] for r in batch["unassigned"]]
    if (len(set(unassigned)) != len(unassigned) or not set(unassigned) <= set(ids)
            or set(unassigned) & (main | annotations)):
        raise ValueError("invalid unassigned source coverage")
    require(main | annotations | set(unassigned) == set(ids), "missing retained source-ID coverage")
    require(all(type(row) is list and len(row) == 2 and type(row[0]) in (int, str)
                and type(row[1]) is str for row in batch["unassigned"]), "invalid unassigned IDs/reasons")
    if not source_only:
        check_count(preview, "strict_eligible_units", sum(u.disposition == "preview" for u in units))
        check_count(preview, "fragment_candidate_inputs", len(all_fragments))
        for key, rows, suffix in (("unit", retained_records, ""), ("fragment", all_fragments, "fragment_"),
                                   ("alternate", all_alternates, "alternate_")):
            check_count(preview, key + "_status_counts", dict(Counter(r["status"] for r in rows)))
            check_count(preview, ("strict" if key == "unit" else key) + "_preview_count", sum(r["status"] == suffix + "preview" for r in rows))
    for key, rows in (("strict_attempts", retained_records), ("fragment_attempts", all_fragments), ("alternate_attempts", all_alternates)):
        check_count(preview, key, sum(r["attempts_started"] for r in rows))
    same(preview["attempts"], sum(r["attempts_started"] for r in retained_records + all_fragments + all_alternates), "global attempt counter drift")
    require(preview["attempts"] <= limit and (settings["run_local_model"] or preview["attempts"] == 0), "translation budget/opt-in contradiction")
    for record in retained_records + all_fragments + all_alternates:
        if record["status"].endswith("budget_exhausted"):
            require(settings["run_local_model"] and preview["attempts"] == limit, "budget skip before exhausted budget")
        if record["status"].endswith("ocr_only"):
            require(not settings["run_local_model"], "OCR-only status with model enabled")
    if preview["status"] in ("complete", "complete_with_failures"):
        require(preview["stage"] == "finished" and preview["error"] is None, "completed workflow has error/unfinished phase")
        failed = any(r["status"] in ("failed", "fragment_failed", "alternate_failed")
                     for r in retained_records + all_fragments + all_alternates)
        require((preview["status"] == "complete_with_failures") == failed, "workflow failure status contradiction")
        require(not any(r["status"] in ("pending", "not_executed", "interrupted", "fragment_pending", "alternate_pending")
                        for r in retained_records + all_fragments + all_alternates), "completed workflow has pending inputs")
    if fragments_enabled:
        same(settings["fragment_policy_revision"], FRAGMENT_POLICY_REVISION, "unsupported fragment policy")
    check_schedule(preview, entries)
    summary = {"scope": "saved source-ID coverage, not detection/translation accuracy",
        "raw_detections": len(ids), "main_detections": len(main), "annotation_detections": len(annotations),
        "unassigned_detections": len(unassigned), "missing_source_ids": [i for i in ids if i not in main | annotations | set(unassigned)],
        "unit_count": len(units), "unit_status_counts": dict(Counter(r["status"] for r in retained_records)),
        "blocked_reason_counts": dict(Counter(reason for u in units for reason in u.blocking_reasons)),
        "unassigned_reason_counts": dict(Counter(r[1] for r in batch["unassigned"])),
        "invalid_unassigned_geometry": sum(not _polygon(by_id[i]["polygon"], width, height) for i in unassigned),
        "main_source_ids": [i for i in ids if i in main], "annotation_source_ids": [i for i in ids if i in annotations],
        "unassigned_source_ids": unassigned, "rendering_approved": False,
        "source_only_failure": source_only, "fragment_status_counts": dict(Counter(r["status"] for r in all_fragments)),
        "alternate_status_counts": dict(Counter(r["status"] for r in all_alternates)),
        "validation": "saved data consistency and explicit original binding only; not semantic/model truth",
        "raw_http_authenticity": "not verified: raw response sidecars are not opened",
        "review_views": "regenerated from explicit original; sidecar files are not opened or certified"}
    return entries, summary


def _png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("embedded presentation exceeds 16 MiB")
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def build_html(preview, entries, summary, rgb, *, crops=False):
    def pretty(value):
        return escape(json.dumps(value, ensure_ascii=False, indent=2), quote=True)
    def details(label, value, *, identifier=None):
        attribute = ' id="' + escape(identifier, quote=True) + '"' if identifier else ''
        return '<details' + attribute + '><summary>' + escape(label) + '</summary><pre>' + pretty(value) + '</pre></details>'
    def output(record, kind):
        result = record.get("result")
        pairs = result.get(kind, []) if result else []
        text = "\n".join(pair[1] for pair in pairs)
        outcomes = result.get("outcomes", []) if result else []
        return ('<p><strong>' + escape(record["status"]) + '</strong> · Review needed</p><pre>'
                + escape(text if pairs else "No translation output.") + '</pre>'
                + details("Typed outcomes, reasons and execution evidence", {"outcomes": outcomes, "record": record}))
    colors = {"blocked": "#b91c1c", "preview": "#047857", "failed": "#b45309", "budget_exhausted": "#6d28d9"}
    raw = preview["ocr"]["raw_detections"][0]
    by_id = {r["source_id"]: r for r in raw}
    overlay, cards, card_bytes = [], [], 0
    for entry in entries:
        n, unit, record, proposal = entry["number"], entry["unit"], entry["record"], entry["proposal"]
        color = colors.get(record["status"], "#374151")
        polygons = "".join('<polygon points="' + " ".join(f"{x:g},{y:g}" for x, y in by_id[i]["polygon"]) + '"/>' for i in unit.member_ids)
        x0, y0, _, _ = proposal["crop_bounds"]
        overlay.append(f'<a href="#unit-{n}"><g style="stroke:{color};fill:none"><title>U{n}: {escape(unit.unit_id)}</title>'
            + polygons + f'<text x="{x0}" y="{max(16, y0)}" style="fill:{color};stroke:white;stroke-width:0.7;paint-order:stroke">U{n}</text></g></a>')
        readings = [{"selected": r.revision == unit.selected_revision, "text": r.text, "origin": r.origin, "revision": r.revision} for r in unit.readings]
        crop = ""
        if crops and not entry["crop_omission"]:
            with rgb.crop(tuple(proposal["crop_bounds"])) as clipped:
                crop = '<img class="crop" alt="Derived source crop, not ownership proof" src="' + _png(clipped) + '">'
        crop_note = entry["crop_omission"] or ("Derived detection crop is context only, never ownership/completeness proof." if crops else "Crops disabled; inspect original full page.")
        review = record.get("ocr_review", {})
        verdict = review.get("verdict") or {}
        correction = ("<p>" + escape(review.get("status", "Review disabled / not started")) + '</p><pre>'
            + escape(verdict.get("proposed_text") or "No proposed correction.") + '</pre>'
            + details("Visual verdict, uncertainty and saved bindings", review))
        cards.append(f'<section id="unit-{n}"><h2>U{n} — {escape(record["status"])} — {escape(unit.unit_id)}</h2>'
            + '<p><strong>Review needed — no automatic acceptance.</strong></p>'
            + f'<p>Members/order: {pretty(list(unit.member_ids))} / {pretty(list(unit.order))}</p>'
            + '<h3>Original OCR → proposed correction → translation previews</h3>'
            + '<p>Original selected OCR (preserved):</p><pre>' + escape(unit.selected.text) + '</pre>'
            + '<p>Visual review / proposed correction (unverified; never replaces original OCR):</p>' + correction
            + '<h3>Original strict utterance translation</h3>' + output(record, "translations")
            + '<h3>Original blocked-source fragment translations</h3><p>Unverified text fragments; original blockers retained.</p>'
            + (''.join(output(child, "fragments") for child in record.get("fragment_previews", [])) or '<p>No fragment input.</p>')
            + '<h3>Alternate translations of proposed readings</h3><p>Unverified visual alternates.</p>'
            + (''.join(output(child, "alternates") for child in record.get("visual_alternates", [])) or '<p>No visual alternate input.</p>')
            + '<h3>Blocking reasons</h3><pre>' + pretty(record["blocking_reasons"]) + '</pre>'
            + '<h3>Lexical / structural uncertainty</h3><pre>' + pretty({"lexical": unit.lexical_reasons, "structural": unit.structural_reasons, "completeness": unit.completeness}) + '</pre>'
            + details("Source readings, revisions and provenance (blank strings retained)", {
                "readings": readings, "unit_revision": unit.revision, "request_revision": record["request_revision"]})
            + '<p>' + escape(crop_note) + '</p>' + crop + '</section>')
        card_bytes += len(cards[-1].encode("utf-8"))
        require(card_bytes <= 32 * 1024 * 1024, "unit-card HTML exceeds 32 MiB")
    width, height = rgb.size
    roles = {i: "main" for i in summary.get("main_source_ids", [])}
    roles.update({i: "tentative annotation" for i in summary.get("annotation_source_ids", [])})
    roles.update({i: "unassigned" for i in summary.get("unassigned_source_ids", [])})
    for index, r in enumerate(raw, 1):
        if roles.get(r["source_id"]) == "unassigned":
            points = " ".join(f"{x:g},{y:g}" for x, y in r["polygon"])
            x, y = r["polygon"][0]
            overlay.append(f'<g style="stroke:#6b7280;fill:none"><title>Raw {index} — unassigned source '
                + escape(str(r["source_id"]), quote=True) + f'</title><polygon points="{points}"/>'
                + f'<text x="{x:g}" y="{max(16,y):g}" style="fill:#374151">R{index}</text></g>')
    rows = "".join('<tr><td>' + escape(str(r["source_id"])) + '</td><td>' + escape(roles.get(r["source_id"], "missing assignment"))
        + '</td><td><pre>' + escape(r["text"]) + '</pre></td><td>' + escape(r["status"]) + '</td></tr>' for r in raw)
    review_stage = preview.get("ocr_review", {})
    settings = preview["settings"]
    strict_count = sum(e["record"]["status"] == "preview" for e in entries)
    fragment_count = sum(child["status"] == "fragment_preview" for e in entries
                         for child in e["record"].get("fragment_previews", []))
    alternate_count = sum(child["status"] == "alternate_preview" for e in entries
                          for child in e["record"].get("visual_alternates", []))
    quick_summary = ('<div id="quick-summary" aria-label="Quick summary">'
        + '<p><strong>Workflow: ' + escape(preview["status"]) + '</strong> · '
        + f'Raw IDs: {len(raw)} · Units: {len(entries)}<br>'
        + f'Outputs — strict: {strict_count} · fragment: {fragment_count} · alternate: {alternate_count}<br>'
        + f'Translation calls: {preview["attempts"]}/{settings["max_attempts"]} · '
        + f'Review calls: {review_stage.get("attempts", 0)}/{settings.get("max_review_attempts", 0)} · '
        + f'Release calls: {review_stage.get("control_calls", 0)} — '
        + escape(review_stage.get("residency", {}).get("release", review_stage.get("status", "disabled")))
        + '</p></div>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Translation preview review</title>'
        '<style>body{font-family:system-ui,sans-serif;margin:1.5rem;max-width:1300px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        '.page{position:relative}.page img{display:block;width:100%;height:auto}.page svg{position:absolute;inset:0;width:100%;height:100%;font-size:18px;stroke-width:2}'
        '#geometry:not(:checked)~.page svg{display:none}section{border-top:2px solid #bbb;margin-top:2rem;padding-top:1rem}'
        'td{vertical-align:top;border-bottom:1px solid #ddd;padding:.3rem}.crop{max-width:100%;height:auto}</style></head><body>'
        '<h1>Read-only translation preview review</h1>' + quick_summary
        + '<p>All outputs require review. No accuracy, human approval or erasure permission is claimed; geometry is diagnostic only.</p>'
        '<input id="geometry" type="checkbox" checked><label for="geometry">Show labelled source-unit geometry (click a label for its unit)</label>'
        f'<div class="page"><img alt="Original full page, RGB presentation" src="{_png(rgb)}">'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" aria-label="Retained unit geometry">' + ''.join(overlay) + '</svg></div>'
        + details("Full execution schedule, review evidence and counters", {
            "workflow_status": preview["status"], "error": preview.get("error"),
            "translation_limit": settings["max_attempts"],
            **{k: preview.get(k) for k in ("attempts", "strict_attempts", "fragment_attempts", "alternate_attempts")},
            "review_limit": settings.get("max_review_attempts"), "review": review_stage,
            "input_schedule": preview.get("input_schedule", [])}, identifier="execution-details")
        + details("Full coverage JSON — retained source IDs, not accuracy", summary, identifier="coverage-details")
        + details("Original image provenance and validation limits", {
            "original_file_sha256": preview["input"]["sha256"],
            "presentation": "Full original page shown as lossless RGB PNG; no source edits.",
            "validation": "Validated data consistency does not establish OCR/model truth or confirmed bubble ownership.",
            "raw_responses": "Raw HTTP response sidecars are not opened: their authenticity is not verified.",
            "review_views": "Regenerated from the explicit original to check saved bindings; existing sidecar files are not certified.",
            "outputs": "Strict previews require review. Fragments and visual alternates are separately typed, unverified outputs."})
        + '<details><summary>Every raw source ID and retained reading</summary><table><tr><th>ID</th><th>Role</th><th>Raw text</th><th>OCR status</th></tr>' + rows + '</table></details>'
        + ''.join(cards) + '</body></html>')


def run(preview_path, image_path, output, *, crops=False):
    require(type(crops) is bool, "crops must be a boolean")
    output = Path(output).expanduser()
    require(not output.exists() and not output.is_symlink(), "output directory must be new and not a symlink")
    require(output.parent.is_dir() and not any(p.is_symlink() for p in (output.parent, *output.parent.parents)),
            "output parent must exist without symlink components")
    raw = read_bounded(preview_path, 64 * 1024 * 1024)
    preview = strict_json(raw.decode("utf-8"))
    image_path = Path(image_path).expanduser().resolve(strict=True)
    content = read_bounded(image_path, 32 * 1024 * 1024)
    with Image.open(BytesIO(content)) as opened:
        if getattr(opened, "n_frames", 1) != 1 or opened.width * opened.height > 40_000_000:
            raise ValueError("one bounded original image frame required")
        rgb = opened.convert("RGB")
    try:
        entries, summary = validate(preview, image_path, rgb, hashlib.sha256(content).hexdigest(), content)
        summary["preview_sha256"] = hashlib.sha256(raw).hexdigest()
        summary["image_sha256"] = hashlib.sha256(content).hexdigest()
        html = build_html(preview, entries, summary, rgb, crops=crops).encode("utf-8")
    finally:
        rgb.close()
    if len(html) > 64 * 1024 * 1024:
        raise ValueError("HTML exceeds 64 MiB")
    require(hashlib.sha256(read_bounded(image_path, 32 * 1024 * 1024)).hexdigest() == summary["image_sha256"],
            "explicit original changed before report publication")
    require(hashlib.sha256(read_bounded(preview_path, 64 * 1024 * 1024)).hexdigest() == summary["preview_sha256"],
            "explicit preview changed before report publication")
    output.mkdir(mode=0o700, exist_ok=False)
    with (output / "review.html").open("xb") as stream:
        stream.write(html)
    with (output / "coverage.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path, help="explicit original file; never inferred from JSON")
    parser.add_argument("--output", required=True, type=Path, help="new exclusive directory, parent must exist")
    parser.add_argument("--crops", action="store_true", help="optional conservative source crops; not ownership proof")
    args = parser.parse_args(argv)
    try:
        summary = run(args.preview, args.image, args.output, crops=args.crops)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, RecursionError, OSError) as error:
        parser.exit(2, "review rejected: " + escape(str(error), quote=True)[:2000] + "\n")
    print(canonical(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
