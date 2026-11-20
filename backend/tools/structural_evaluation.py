"""Deterministic saved-artifact structural evaluation; no production/model imports."""
import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CATEGORIES = ("tentative_furigana", "relationship_conflict", "unsafe_crop", "unknown_region",
              "horizontal_layout", "latin_or_mixed_script", "unknown_orientation_sfx_candidate",
              "multi_main_columns", "crop_completeness_unverified")
VISUAL_CATEGORIES = ("horizontal_narration", "latin_label", "mixed_script", "stylized_sfx",
                     "ambiguous_ownership", "tentative_furigana", "companion_columns",
                     "crop_completeness_unverified", "overlapping_footprints", "rectangular_gap_contamination")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def tags(record, page):
    proposal = record["proposal"]
    reasons = set(proposal["structural_reasons"])
    result = []
    if proposal["annotation_parents"]:
        result.append("tentative_furigana")
    conflict_reasons = {"ambiguous_annotation_parents", "ambiguous_annotation_chain_or_role",
                        "unresolved_annotation_relationship"}
    relevant = [r for r in page["batch"]["relationships"] if r["source_id"] in proposal["member_ids"]]
    if reasons & conflict_reasons or any(r["relationship"] == "unresolved" for r in relevant):
        result.append("relationship_conflict")
    if proposal["crop_safe"] is False:
        result.append("unsafe_crop")
    if "unknown_region_ownership" in reasons or not proposal["region_ids"]:
        result.append("unknown_region")
    if proposal["direction"] == "horizontal":
        result.append("horizontal_layout")
    if record["diagnosis"]["script_evidence"] in ("clearly_non_japanese", "mixed_script"):
        result.append("latin_or_mixed_script")
    if proposal["direction"] == "unknown":
        result.append("unknown_orientation_sfx_candidate")
    if len(proposal["main_ids"]) > 1:
        result.append("multi_main_columns")
    if proposal["completeness"] == "unverified_full_utterance":
        result.append("crop_completeness_unverified")
    return result


def load_saved_artifact(path):
    content = Path(path).read_bytes()
    document = json.loads(content, object_pairs_hook=unique_keys)
    pages = document["pages"]
    if not isinstance(pages, list) or not 1 <= len(pages) <= 32:
        raise ValueError("one bounded saved artifact with 1..32 pages is required")
    page_ids, proposal_ids, items, images = set(), set(), [], {}
    for page in sorted(pages, key=lambda p: p["page"]["page_id"]):
        provenance = page["page"]
        page_id = provenance["page_id"]
        if page_id in page_ids or page["batch"]["page"] != provenance:
            raise ValueError("duplicate/foreign page provenance")
        page_ids.add(page_id)
        raw_ids = [raw["source_id"] for raw in page["batch"]["detections"]]
        if (any(type(i) not in (int, str) or i == "" for i in raw_ids)
                or len(set(raw_ids)) != len(raw_ids)
                or any(raw["page"] != provenance for raw in page["batch"]["detections"])):
            raise ValueError("invalid raw snapshot IDs/provenance")
        image_bytes = Path(provenance["image_path"]).read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != provenance["image_sha256"]:
            raise ValueError("stale original image SHA256")
        with Image.open(provenance["image_path"]) as source:
            image = source.convert("RGB")
        prefix = f"RGB:{image.width}:{image.height}:".encode()
        if hashlib.sha256(prefix + image.tobytes()).hexdigest() != provenance["pixels_sha256"]:
            raise ValueError("stale original RGB fingerprint")
        images[page_id] = image
        saved_proposals = {p["proposal_id"]: p for p in page["batch"]["proposals"]}
        if len(saved_proposals) != len(page["batch"]["proposals"]) or len(page["units"]) != len(saved_proposals):
            raise ValueError("duplicate/missing saved proposals")
        assigned = []
        for record in page["units"]:
            proposal = record["proposal"]
            proposal_id = proposal["proposal_id"]
            if (proposal_id in proposal_ids or record["case_id"] != proposal_id
                    or saved_proposals.get(proposal_id) != proposal or proposal["page"] != provenance
                    or record["source_ids"] != proposal["member_ids"] or proposal["reviewed"] is not False
                    or proposal["structural_status"] != "proposal_unverified"):
                raise ValueError("ambiguous/drifted saved proposal")
            proposal_ids.add(proposal_id)
            members = proposal["member_ids"]
            if (not members or any(type(i) not in (int, str) or i not in raw_ids for i in members)
                    or len(set(members)) != len(members)
                    or proposal["main_ids"] + proposal["annotation_ids"] != members):
                raise ValueError("invalid saved membership partition")
            assigned.extend(members)
            bounds = proposal["crop_bounds"]
            if (len(bounds) != 4 or any(type(v) is not int for v in bounds)
                    or not 0 <= bounds[0] < bounds[2] <= image.width
                    or not 0 <= bounds[1] < bounds[3] <= image.height):
                raise ValueError("invalid saved crop bounds")
            # Required rich saved evidence; do not regenerate proposals or fill gaps.
            record["diagnosis"]["nonmember_conflicts"]
            record["eligibility_preflight"]["reason"]
            for conflict in proposal["crop_conflicts"]:
                if conflict["source_id"] not in raw_ids or conflict["source_id"] in members:
                    raise ValueError("foreign crop conflict ID")
            items.append({"page_id": page_id, "proposal_id": proposal_id, "tags": tags(record, page),
                          "page": page, "record": record})
        unassigned = [entry[0] for entry in page["batch"].get("unassigned", [])]
        if (len(set(assigned + unassigned)) != len(assigned + unassigned)
                or set(assigned + unassigned) != set(raw_ids)):
            raise ValueError("duplicate/missing membership across proposals/unassigned")
    return document, hashlib.sha256(content).hexdigest(), sorted(items, key=lambda i: (i["page_id"], i["proposal_id"])), images


def select_cases(items, max_cases, per_page):
    if type(max_cases) is not int or not 1 <= max_cases <= 64 or type(per_page) is not int or not 1 <= per_page <= 8:
        raise ValueError("max_cases 1..64 and per_page 1..8 required")
    selected, counts = [], Counter()
    priority = {"unsafe_crop", "unknown_region", "relationship_conflict"}
    def add(item):
        if item not in selected and len(selected) < max_cases and counts[item["page_id"]] < per_page:
            selected.append(item)
            counts[item["page_id"]] += 1
            return True
        return False
    # Cover observable strata first, then balanced per-page high-risk cases.
    for category in CATEGORIES[:-1]:
        if any(category in item["tags"] for item in selected):
            continue
        candidates = sorted((i for i in items if category in i["tags"]),
                            key=lambda i: (not bool(priority & set(i["tags"])), i["page_id"], i["proposal_id"]))
        for item in candidates:
            if add(item):
                break
    page_ids = sorted({i["page_id"] for i in items})
    queues = {page_id: sorted((i for i in items if i["page_id"] == page_id),
        key=lambda i: (not bool(priority & set(i["tags"])), "unsafe_crop" not in i["tags"],
                       "relationship_conflict" not in i["tags"], i["proposal_id"])) for page_id in page_ids}
    for _ in range(per_page):
        for page_id in page_ids:
            for item in queues[page_id]:
                if add(item):
                    break
    return sorted(selected, key=lambda i: (i["page_id"], i["proposal_id"]))


def binding(item, artifact_hash):
    proposal = item["record"]["proposal"]
    return {"artifact_sha256": artifact_hash, "page": copy.deepcopy(item["page"]["page"]),
            "proposal_id": item["proposal_id"], "proposal_sha256": digest(proposal),
            "source_record_sha256": digest(item["record"]), "member_ids": proposal["member_ids"],
            "crop_bounds": proposal["crop_bounds"]}


def validate_references(document, cases):
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or document.get("scope") != "structural_observations_only" or not isinstance(document.get("records"), list)):
        raise ValueError("structural-only reference schema required")
    expected = {case["case_id"]: case["binding"] for case in cases}
    seen = set()
    for record in document["records"]:
        if not isinstance(record, dict):
            raise ValueError("structural records must be objects")
        case_id = record["case_id"]
        if case_id not in expected or case_id in seen or record["binding"] != expected[case_id]:
            raise ValueError("foreign/duplicate/stale structural reference binding")
        seen.add(case_id)
        if (record["candidate_reviewed"] is not False or record["independent_transcription_reference"] is not False
                or record.get("reviewed", False) is not False
                or record["state"] not in ("unannotated", "provisional_visual_observation")
                or record["completeness"] != "unverified"):
            raise ValueError("references cannot grant OCR review/truth/completeness")
        if (not isinstance(record["observed_categories"], list)
                or any(c not in VISUAL_CATEGORIES for c in record["observed_categories"])
                or not isinstance(record["observer"], str) or not isinstance(record["notes"], str)):
            raise ValueError("invalid structural observation category")
        if record["state"] == "provisional_visual_observation" and (not record["observer"].strip() or not record["notes"].strip()):
            raise ValueError("visual observations require observer/source notes")
        if record["state"] == "unannotated" and (record["observed_categories"] or record["observer"] or record["notes"]):
            raise ValueError("unannotated records cannot claim visual observations")
    return document


def _save(image, path):
    image.save(path, format="PNG")
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _plottable(raw):
    points = raw["polygon"]
    return (len(points) >= 3 and all(len(p) == 2 and all(type(v) in (int, float)
            and math.isfinite(v) for v in p) for p in points))


def evaluate(artifact, output_dir, *, max_cases=21, per_page=3, reference=None):
    document, artifact_hash, items, images = load_saved_artifact(artifact)
    selected = select_cases(items, max_cases, per_page)
    cases = []
    for item in selected:
        bound = binding(item, artifact_hash)
        case_id = "structure-" + digest(bound)[:24]
        cases.append({"case_id": case_id, "binding": bound, "observed_saved_tags": item["tags"],
                      "source_record": copy.deepcopy(item["record"]), "evidence": {}})
    references = {"schema_version": 1, "scope": "structural_observations_only", "records": [
        {"case_id": case["case_id"], "binding": case["binding"], "state": "unannotated", "observer": "",
         "observed_categories": [], "notes": "", "completeness": "unverified",
         "candidate_reviewed": False, "independent_transcription_reference": False} for case in cases]}
    if reference:
        references = validate_references(json.loads(Path(reference).read_bytes(), object_pairs_hook=unique_keys), cases)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for index, (item, case) in enumerate(zip(selected, cases)):
        image = images[item["page_id"]]
        proposal = item["record"]["proposal"]
        bounds = proposal["crop_bounds"]
        case["evidence"]["actual_crop"] = _save(image.crop(bounds), output / f"case-{index:02d}-original.png")
        # Diagnostic context only; the actual saved candidate/proposal bounds stay untouched.
        context = [max(0, bounds[0]-24), max(0, bounds[1]-24), min(image.width, bounds[2]+24), min(image.height, bounds[3]+24)]
        related_ids = set(proposal["member_ids"]) | {c["source_id"] for c in proposal["crop_conflicts"]}
        case["evidence"]["unplottable_source_ids"] = [raw["source_id"] for raw in item["page"]["batch"]["detections"]
            if raw["source_id"] in related_ids and not _plottable(raw)]
        for raw in item["page"]["batch"]["detections"]:
            if raw["source_id"] in related_ids and _plottable(raw):
                xs, ys = zip(*raw["polygon"])
                context = [max(0, min(context[0], int(min(xs))-24)), max(0, min(context[1], int(min(ys))-24)),
                           min(image.width, max(context[2], int(max(xs))+24)), min(image.height, max(context[3], int(max(ys))+24))]
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        for raw in item["page"]["batch"]["detections"]:
            if raw["source_id"] not in related_ids or not _plottable(raw):
                continue
            color = "cyan" if raw["source_id"] in proposal["annotation_ids"] else "red" if raw["source_id"] in proposal["main_ids"] else "magenta"
            points = [tuple(p) for p in raw["polygon"]]
            draw.polygon(points, outline=color, width=2)
            draw.text((points[0][0], points[0][1]), str(raw["source_id"]), fill=color, font=ImageFont.load_default())
        draw.rectangle(bounds, outline="orange", width=2)
        case["evidence"]["context_bounds"] = context
        case["evidence"]["context_original"] = _save(image.crop(context), output / f"case-{index:02d}-context-original.png")
        case["evidence"]["context_overlay"] = _save(marked.crop(context), output / f"case-{index:02d}-context-overlay.png")
    counts = Counter(tag for item in items for tag in item["tags"])
    selected_counts = Counter(tag for item in selected for tag in item["tags"])
    blockers = Counter(item["record"]["eligibility_preflight"]["reason"] for item in items
                       if item["record"]["eligibility_preflight"]["reason"] is not None)
    dominant = sorted(blockers, key=lambda reason: (-blockers[reason], reason))[0] if blockers else None
    representatives = {reason: [{"page_id": i["page_id"], "proposal_id": i["proposal_id"],
                                "member_ids": i["record"]["proposal"]["member_ids"]}
                               for i in items if i["record"]["eligibility_preflight"]["reason"] == reason][:3]
                       for reason in sorted(blockers)}
    report = {"schema_version": 1, "scope": "saved_structural_evidence_not_accuracy", "artifact_sha256": artifact_hash,
              "artifact_path": str(Path(artifact).resolve()), "selection_policy": "strata_then_balanced_risk_v1",
              "settings": {"max_cases": max_cases, "per_page": per_page},
              "counts": {"pages": len(document["pages"]), "raw_detections": sum(len(p["batch"]["detections"]) for p in document["pages"]),
                         "proposals": len(items), "selected_cases": len(cases),
                         "priority_proposals": sum(bool(set(i["tags"]) & {"unsafe_crop", "unknown_region", "relationship_conflict"}) for i in items)},
              "category_counts_overlapping": dict(sorted(counts.items())), "selected_category_counts_overlapping": dict(sorted(selected_counts.items())),
              "saved_eligibility_blockers": dict(sorted(blockers.items())), "dominant_saved_blocker": dominant,
              "representative_proposals": representatives, "source_page_errors": document.get("page_errors", []),
              "full_utterance_completeness_is_baseline_uncertainty_not_an_exclusive_failure": True,
              "candidate_review_or_acceptance_performed": False, "provider_calls": 0,
              "provisional_visual_reference_cases": sum(r["state"] == "provisional_visual_observation" for r in references["records"])}
    evidence = {"schema_version": 1, "artifact_sha256": artifact_hash,
                "pages": [{"page": p["page"], "source_batch": p["batch"], "source_overlay_path": p["overlay_path"]}
                          for p in sorted(document["pages"], key=lambda p: p["page"]["page_id"])], "cases": cases}
    for name, value in (("evaluation.json", report), ("evidence.json", evidence), ("structural-reference.json", references)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "category_counts": report["category_counts_overlapping"],
                      "blockers": report["saved_eligibility_blockers"], "dominant": dominant, "provider_calls": 0}))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="existing rich automatic-proposals.json; never regenerate OCR")
    parser.add_argument("--output-dir", required=True, help="new-only evaluation directory")
    parser.add_argument("--max-cases", type=int, default=21)
    parser.add_argument("--per-page", type=int, default=3)
    parser.add_argument("--reference", help="optional hash-bound structural observations, never OCR review records")
    args = parser.parse_args(argv)
    evaluate(args.artifact, args.output_dir, max_cases=args.max_cases, per_page=args.per_page, reference=args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
