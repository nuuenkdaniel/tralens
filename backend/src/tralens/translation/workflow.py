"""One explicit image -> immutable utterance previews. No diagnostics or rendering imports."""
from dataclasses import asdict, is_dataclass
from enum import Enum
import subprocess
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import time
import unicodedata
from ..scheduling import serialized


FRAGMENT_SCHEDULING_REVISION = "strict_then_kana_han_stable_v1"


def fragment_scheduling_info(text, *, explicit_selection=False):
    """Weak presence hint only. Never normalize source/model input or filter candidates."""
    if type(text) is not str or type(explicit_selection) is not bool:
        raise ValueError("scheduling requires raw text and a boolean selection mode")
    view = unicodedata.normalize("NFKC",text)
    prefixes = ("HIRAGANA LETTER ","KATAKANA LETTER ","HENTAIGANA LETTER ",
                "CJK UNIFIED IDEOGRAPH-","CJK COMPATIBILITY IDEOGRAPH-")
    detected = any(unicodedata.name(char,"").startswith(prefixes) for char in view)
    return {"policy_revision":FRAGMENT_SCHEDULING_REVISION,"contains_kana_or_han":detected,
        "priority_tier":None if explicit_selection else 0 if detected else 1,
        "reason":"explicit_selected_ids_order" if explicit_selection else
                 "automatic_kana_or_han_hint_first" if detected else "automatic_other_script_hint_later",
        "inspection":"NFKC view only; original text unchanged","unicode_version":unicodedata.unidata_version,
        "certifies_language_quality_readability_or_dialogue":False}


class SourceChanged(ValueError):
    pass


class _CountedRuntime:
    """Count dispatch before provider entry, including interrupted calls."""
    def __init__(self, provider, state, limit):
        self.provider, self.state, self.limit = provider, state, limit
        self.record = None

    def generate(self, **kwargs):
        if self.state["attempts"] >= self.limit:
            raise RuntimeError("global model call budget exhausted")
        self.state["attempts"] += 1
        kind = ("alternate_attempts" if self.record.get("input_kind") == "unverified_visual_alternate" else
                "fragment_attempts" if self.record.get("input_kind") == "unverified_text_fragment" else
                "coverage_attempts" if self.record.get("input_kind") == "coverage_text_region" else "strict_attempts")
        self.state[kind] = self.state.get(kind,0) + 1
        self.record["attempts_started"] += 1
        return self.provider.generate(**kwargs)


def _panel_bounds(page, target, fallback):
    """Follow uninterrupted paper gutters around the target in a bounded thumbnail."""
    import numpy as np
    preview = page.convert('RGB')
    preview.thumbnail((620, 800))
    ink = np.asarray(preview).min(axis=2) < 244
    height, width = ink.shape
    scale = np.array([width / page.width, height / page.height] * 2)
    x, y, right, bottom = np.asarray(target) * scale
    left, top, end, base = 0, 0, width, height
    divided = False
    for _ in range(4):
        region = ink[top:base, left:end]
        previous = left, top, end, base
        for axis, origin in ((1, top), (0, left)):
            paper = region.mean(axis=axis) < .01
            edges = np.flatnonzero(np.diff(np.r_[False, paper, False]))
            for start, stop in zip(edges[::2], edges[1::2]):
                if stop - start < 3:
                    continue
                middle = origin + (start + stop) // 2
                if start > 0 and stop < len(paper):
                    divided |= middle < (y if axis else x) or middle > (bottom if axis else right)
                if axis == 1:
                    if middle < y:
                        top = max(top, middle)
                    elif middle > bottom:
                        base = min(base, middle)
                elif middle < x:
                    left = max(left, middle)
                elif middle > right:
                    end = min(end, middle)
        if previous == (left, top, end, base):
            break
    if not divided or (end-left)*(base-top) > .8*width*height or min(end-left, base-top) < 40:
        return fallback
    return tuple(round(v) for v in (np.array([left, top, end, base]) / scale))


def _translation_crop(image, image_hash, detections, unit, *, panel=False, toolbar=False, adjacent=False):
    """A bounded view around the original target. Neighboring lettering is context only."""
    from PIL import Image
    content = _read_image(image)
    if hashlib.sha256(content).hexdigest() != image_hash:
        raise SourceChanged('translation context source changed')
    polygons = {row['source_id']:row['polygon'] for row in detections}
    points = [point for member in unit.member_ids for point in polygons[member]]
    xs, ys = zip(*points)
    punctuation = ''
    for row in detections:
        # A separate punctuation column immediately left of a vertical line
        # carries tone even when the OCR unit omits it. Do not import dialogue.
        mark = row['text'].strip()
        if (not mark or len(mark) > 8 or any(ch not in '…!?！？.' for ch in mark)
                or not any(ch in '!?！？' for ch in mark)):
            continue
        px, py = zip(*row['polygon'])
        gap = min(xs) - max(px)
        overlap = min(max(ys),max(py)) - max(min(ys),min(py))
        if 0 <= gap <= max(3, (max(px)-min(px))*.5) and overlap > .6*(max(py)-min(py)):
            punctuation = mark
            break
    with Image.open(BytesIO(content)) as page:
        pad = min(320, max(72, round(min(page.size)*.15)))
        if adjacent:
            pad = max(55, int(max(max(xs)-min(xs), max(ys)-min(ys)))+1)
        bounds = (max(0, int(min(xs))-pad), max(0, int(min(ys))-pad),
                  min(page.width, int(max(xs))+pad+1), min(page.height, int(max(ys))+pad+1))
        if panel:
            bounds = _panel_bounds(page, (min(xs), min(ys), max(xs), max(ys)), bounds)
        if toolbar:
            w,h = max(xs)-min(xs),max(ys)-min(ys)
            bounds = (max(0,round(min(xs)-.45*w)),max(0,round(min(ys)-1.5*h)),
                      min(page.width,round(max(xs)+.45*w)),min(page.height,round(max(ys)+.3*h)))
        view = page.crop(bounds).convert('RGB')
        if toolbar:
            view = view.resize((view.width*3,view.height*3),Image.Resampling.LANCZOS)
        view.thumbnail((768,768), Image.Resampling.LANCZOS)
        stream = BytesIO()
        view.save(stream, format='PNG')
        return stream.getvalue(), bounds, punctuation


def _read_image(path):
    if not path.is_file():
        raise ValueError("image must be an existing regular file")
    with path.open("rb") as stream:
        content = stream.read(32 * 1024 * 1024 + 1)
    if len(content) > 32 * 1024 * 1024:
        raise ValueError("image exceeds 32 MiB")
    return content


def _snapshot(value):
    def encode(item):
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, Enum):
            return item.value
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError("unsupported evidence type: " + type(item).__name__)
    return json.loads(json.dumps(value, default=encode, ensure_ascii=False, allow_nan=False))


def _raw_box(box):
    return {"source_id": box.paddle_box_id, "text": box.paddle_text, "confidence": float(box.paddle_confidence),
        "polygon": box.paddle_poly.tolist(), "selected_text": box.selected_text,
        "selected_engine": box.selected_engine, "status": box.status, "reason": box.reason,
        "ocr_metadata": box.ocr_metadata}


def _atomic_json(path, value, *, replace_owned=False):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if len(content) > 64 * 1024 * 1024:
        raise ValueError("preview JSON exceeds 64 MiB")
    from ..storage import atomic_bytes
    atomic_bytes(path,content,replace_owned=replace_owned)


def _close_ocr(engine):
    close = getattr(engine, "close", None)
    if not callable(close):
        close = getattr(getattr(engine, "ocr", None), "close", None)
    if not callable(close):
        raise RuntimeError("OCR engine exposes no resource close method")
    close()


def _retain_event(output, prefix, event):
    from .contracts import strict_json
    evidence = {k:v for k,v in event.items() if k != "response_json"}
    raw = event.get("response_json")
    if raw is not None:
        content = raw.encode("utf-8")
        truncated = len(content)>1024*1024
        name = prefix+f"-attempt-{event['attempt']:02d}-response."+("txt" if truncated else "json")
        with (output/name).open("xb") as stream:
            stream.write(content[:1024*1024])
            stream.flush()
            os.fsync(stream.fileno())
        parsed = strict_json(raw)
        keys = ("model","done","done_reason","total_duration","load_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration")
        evidence.update(raw_response_path=name,raw_response_sha256=hashlib.sha256(content).hexdigest(),
            response_bytes=len(content),retained_bytes=min(len(content),1024*1024),truncated=truncated,
            metadata={k:parsed[k] for k in keys if k in parsed} if type(parsed) is dict else {})
    return evidence


@serialized
def run_preview(image, output, *, device, model_root, run_local_ocr=False, run_local_model=False,
                model="translategemma:27b", profile="translategemma", endpoint="http://127.0.0.1:11434", max_attempts=2,
                timeout=120, keep_alive=60, num_predict=1024, selected_ids=(), ocr_factory=None, runtime=None,
                allow_fragment_previews=False, review_ocr=False, review_model="qwen3.8:latest",
                max_review_attempts=4, review_timeout=90, review_runtime=None, cancelled=lambda:False,
                translation_policy='coverage-first', ocr_detect=None, ocr_refine=None, translation_context='isolated', source_localization=False,
                luna_mode=None, related_pages=(), source_language="ja"):
    """Injected OCR factory is workflow-owned; an injected translation runtime is caller-owned.

    Invalid arguments/paths are rejected before creating any engine. Runtime/engine
    failures after output reservation yield a failed, reviewable preview artifact.
    The sole mutable checkpoint belongs to this exclusively created output directory.
    The user-selected default pair is TG27B/native; generalists require both overrides.
    """
    if type(run_local_ocr) is not bool or not run_local_ocr:
        raise ValueError("explicit run_local_ocr=True is required; no OCR was initialized")
    if type(run_local_model) is not bool:
        raise ValueError("run_local_model must be a boolean")
    if type(allow_fragment_previews) is not bool:
        raise ValueError("allow_fragment_previews must be a boolean")
    if type(review_ocr) is not bool or not callable(cancelled):
        raise ValueError("review_ocr must be a boolean and cancelled must be callable")
    if ocr_detect is not None and not callable(ocr_detect):
        raise ValueError('OCR detection must be an explicit callback')
    if ocr_refine is not None and not callable(ocr_refine):
        raise ValueError('OCR refinement must be an explicit callback')
    if type(source_localization) is not bool:
        raise ValueError('source localization requires explicit boolean enablement')
    if review_ocr and not run_local_model:
        raise ValueError("--review-ocr requires --run-local-model; no image review in OCR-only mode")
    from . import coverage
    coverage.validate_policy(translation_policy)
    if type(max_attempts) is not int or not 0 <= max_attempts <= coverage.MAX_CALLS:
        raise ValueError(f"max_attempts must be 0..{coverage.MAX_CALLS} globally")
    if type(timeout) not in (int, float) or not 0 < timeout <= 300:
        raise ValueError("timeout must be finite and 0..300 seconds")
    if (type(selected_ids) is not tuple or any(type(i) is not str or not i for i in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)):
        raise ValueError("selected IDs must be an immutable unique tuple")
    if not isinstance(device, str) or model_root is None:
        raise ValueError("explicit device and local model_root are required")
    from .contracts import (TranslationRequest, FragmentRequest, FRAGMENT_POLICY_REVISION, fragment_plan,
                            automatic_units, canonical, strict_json)
    from .runtime import GENERALIST, TRANSLATEGEMMA, GenerationOptions, PreviewTranslator, OllamaRuntime, local_endpoint, validate_model_profile
    from ..ocr.paddle_profiles import DEFAULT_PROFILE, resolve_profile, validate_runtime
    from ..ocr.recognition_units import image_fingerprint, snapshot_detections
    from PIL import Image

    from . import luna
    frontier = model in luna.MODELS
    if frontier:
        luna_mode = luna_mode or "reviewed"
        if len(related_pages) > 6:
            raise ValueError('select at most six related manga pages')
        if luna_mode not in luna.MODES or (luna_mode == 'isolated' and related_pages):
            raise ValueError('invalid frontier mode or context')
        if allow_fragment_previews or review_ocr or translation_context != 'isolated':
            raise ValueError('Frontier models use their own page context and review')
        profile = 'generalist'
    elif related_pages or luna_mode is not None:
        raise ValueError('Frontier input options require a supported frontier model')
    profiles = {"generalist": GENERALIST, "translategemma": TRANSLATEGEMMA}
    if profile not in profiles:
        raise ValueError("unknown translation profile")
    capability = profiles[profile]
    validate_model_profile(model, capability)
    from . import context
    context.validate_policy(translation_context, capability)
    if translation_context != 'isolated' and review_ocr:
        raise ValueError('nearby translation context and visual alternate review must be separate experiments')
    if review_ocr:
        from . import ocr_review as visual
        visual.validate_options(review_ocr,review_model,model,max_review_attempts,review_timeout,run_local_model)
    endpoint = local_endpoint(endpoint)
    options = GenerationOptions(num_ctx=8192 if translation_context == 'nearby' else 4096,
                                num_predict=num_predict, keep_alive=keep_alive)
    from ..ocr.chapter_language import LANGUAGES
    from .contracts import Style
    style = Style(source_language=LANGUAGES[source_language])
    ocr_profile = "v6_korean" if source_language == "ko" else DEFAULT_PROFILE
    paddle = resolve_profile(ocr_profile, device=device, model_root=model_root)
    if ocr_factory is None:
        validate_runtime(paddle.settings())
    image = Path(image).expanduser().resolve(strict=True)
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError("output directory must be new: " + str(output))
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    content = _read_image(image)
    with Image.open(BytesIO(content)) as opened:
        if getattr(opened, "n_frames", 1) != 1 or opened.width * opened.height > 40_000_000:
            raise ValueError("one raster frame of at most 40 million pixels is required")
        rgb = opened.convert("RGB")
    image_hash = hashlib.sha256(content).hexdigest()
    pixels_hash = image_fingerprint(rgb)
    size = rgb.size
    rgb.close()
    del content
    related_hashes = {}
    for path in related_pages:
        path = Path(path).expanduser().resolve(strict=True)
        related_hashes[path] = hashlib.sha256(_read_image(path)).hexdigest()
    related_pages = tuple(related_hashes)
    output.mkdir(mode=0o700, exist_ok=False)
    state = {"schema_version": 2 if allow_fragment_previews else 1, "preview_only": True, "status": "running", "stage": "ocr",
        "input": {"path": str(image), "sha256": image_hash, "pixels_sha256": pixels_hash, "size": list(size)},
        "settings": {"source_language": source_language, "ocr_profile": paddle.identity(), "run_local_model": run_local_model, "model": model,
            "profile": profile, "endpoint": endpoint, "model_identity": "configured tag only; weights/digest not authenticated",
            "max_attempts": max_attempts, "timeout": timeout, "generation_options": asdict(options),
            "allow_fragment_previews": allow_fragment_previews, "fragment_policy_revision": FRAGMENT_POLICY_REVISION,
            "call_order": "selected strict inputs first, then selected singleton fragments; order preserved within each kind",
            "selected_ids": list(selected_ids), "context": "utterance text only; no image sent to translator"},
        "ocr": None, "units": [], "attempts": 0, "strict_attempts": 0, "fragment_attempts": 0,
        "strict_eligible_units": 0, "fragment_candidate_inputs": 0, "cleanup": None, "error": None}
    state['settings']['translation_policy'] = translation_policy
    if frontier:
        state['settings'].update(provider=luna.frontier.provider(), profile=luna_mode, endpoint=None,
            context='isolated OCR' if luna_mode == 'isolated' else 'target image and explicit related pages',
            related_pages=[{'path': str(p), 'sha256': h} for p,h in related_hashes.items()],
            model_identity='requested ' + luna.frontier.provider() + ' model')
    if source_localization:
        state['settings']['source_localization'] = True
    if translation_context != 'isolated':
        state['settings'].update(translation_context=translation_context, context_policy=context.SOUND_POLICY if translation_context=='sound-effects' else context.POLICY,
            context='separately identified page OCR text; order/speaker/relevance unverified; no image sent to translator')
    state.update(coverage_attempts=0, coverage_eligible_units=0)
    if allow_fragment_previews:
        state["settings"]["call_order"] = ("selected strict inputs first; explicit fragment selection order"
            if selected_ids else "strict inputs first; automatic singleton fragments Kana/Han hint first, stable within tiers")
        state["settings"]["fragment_scheduling_policy"] = FRAGMENT_SCHEDULING_REVISION
    if review_ocr:
        state["schema_version"] = 3
        state["settings"].update(review_ocr=True,review_model=review_model,max_review_attempts=max_review_attempts,
                                 review_timeout=review_timeout,visual_alternates="unverified proposed readings; original base paths retained")
        state["ocr_review"] = {"endpoint":endpoint,"status":"not_started","attempts":0,"control_calls":0,"units":[]}
        state["alternate_attempts"] = 0
    if translation_policy == 'coverage-first':
        state['schema_version'] = 4
        state['settings']['call_order'] = ('strict inputs first, then source-bound uncertain text regions, then optional fragments; '
            'explicit selection order within each kind' if selected_ids else
            'strict inputs first, then uncertain text regions with weak grouped-text/script/length hints, then optional fragments')
    if frontier:
        state['settings']['call_order'] = 'page batch then skipped-region reviews' if luna_mode == 'reviewed' else luna_mode
    checkpoint = output / "preview.json"
    _atomic_json(checkpoint, state)

    def persist():
        _atomic_json(checkpoint, state, replace_owned=True)

    def assert_source():
        try:
            current = hashlib.sha256(_read_image(image)).hexdigest()
            for path, expected in related_hashes.items():
                if hashlib.sha256(_read_image(path)).hexdigest() != expected:
                    raise SourceChanged('related manga page changed during workflow')
        except SourceChanged:
            raise
        except (OSError, ValueError) as error:
            state["input"]["source_check_error"] = type(error).__name__
            raise SourceChanged("original image is no longer readable for source binding") from error
        if current != image_hash:
            state["input"]["current_sha256"] = current
            raise SourceChanged("original image bytes changed during workflow")

    engine = None
    batches, recognition = (), ()
    raw_bound = None
    try:
        started = time.monotonic()
        ocr_error = None
        try:
            if ocr_factory is None:
                from ..ocr import OCR
                ocr_factory = OCR
            engine = ocr_factory(str(image), device=device, model_root=str(Path(model_root).expanduser()),
                                 model_profile=ocr_profile, enable_manga_ocr=False)
            if source_language != "ja":
                engine.source_language = source_language
            engine.process_images(propose_units=True, complementary_ownership=True, regroup_companions=True)
            if ocr_detect is not None:
                ocr_detect(engine)
                if hasattr(engine,'masked_detection_error'):
                    state['masked_detection_error'] = engine.masked_detection_error
            if frontier and run_local_model and luna_mode != 'isolated' and not selected_ids and max_attempts >= 2:
                from . import discovery
                if cancelled():
                    raise KeyboardInterrupt('cancelled before page discovery')
                assert_source()
                state['attempts'] += 1
                persist()
                try:
                    discovery_options = {} if source_language == "ja" else {"language": style.source_language}
                    state['frontier_discovery'] = discovery.extend(engine, image, size, model, timeout, **discovery_options)
                except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                    state['frontier_discovery'] = dict(model=model, added_ids=[], error=str(error)[:512])
                assert_source()
                discovery_result = state['frontier_discovery']
                repeated = discovery_result.get('repeated_kana')
                if repeated and state['attempts']+1 < max_attempts:
                    if cancelled():
                        raise KeyboardInterrupt('cancelled before repeated lettering check')
                    state['attempts'] += 1
                    persist()
                    try:
                        detail = discovery.extend(engine, image, size, model, timeout, repeated=repeated, **discovery_options)
                        discovery_result['repeated_check'] = detail
                        discovery_result['proposed'] += detail['proposed']
                        discovery_result['added_ids'].extend(detail['added_ids'])
                    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                        discovery_result['repeated_check'] = {'error':str(error)[:512]}
                    assert_source()
            if ocr_refine is not None:
                ocr_refine(engine, output=output)
            if source_localization:
                from ..ocr.source_localization import refine as localize
                localize(engine, output=output)
        except Exception as error:
            ocr_error = error
        finally:
            if engine is not None:
                try:
                    batches = tuple(getattr(engine, "automatic_unit_proposals", ()))
                    recognition = tuple(getattr(engine, "automatic_recognition_results", ()))
                    state["ocr"] = _snapshot({"raw_detections": [[_raw_box(b) for b in page]
                        for page in getattr(engine, "raw_detections", ())], "batches": batches,
                        "automatic_recognition_results": recognition,
                        "prediction_contract_reports": getattr(engine, "prediction_contract_reports", ()),
                         "model_provenance": getattr(engine, "paddle_provenance", {})})
                    if ocr_refine is not None:
                        state['ocr']['original_unit_proposals'] = _snapshot(getattr(engine, 'original_unit_proposals', ()))
                        state['ocr']['quality_rereads'] = _snapshot(getattr(engine, 'quality_rereads', {}))
                        state['ocr']['dialogue_readings'] = _snapshot(getattr(engine,'dialogue_readings',{}))
                        if hasattr(engine,'crop_readings'):
                            state['ocr']['crop_readings'] = _snapshot(engine.crop_readings)
                        if hasattr(engine,'reader_presentations'):
                            state['ocr']['reader_presentations'] = _snapshot(engine.reader_presentations)
                        if hasattr(engine,'outlined_sound_discovery'):
                            state['ocr']['outlined_sound_discovery'] = _snapshot(engine.outlined_sound_discovery)
                        if hasattr(engine,'ink_column_groups'):
                            state['ocr']['ink_column_groups'] = _snapshot(engine.ink_column_groups)
                    if source_localization:
                        state['ocr']['source_localization'] = _snapshot(getattr(engine, 'source_localization', {}))
                    if len(batches) == 1 and batches[0] is not None and len(engine.raw_detections) == 1:
                        raw_bound = snapshot_detections(engine.raw_detections[0], batches[0].page)
                except Exception as error:
                    state["evidence_error"] = type(error).__name__
                    ocr_error = ocr_error or error
                finally:
                    try:
                        _close_ocr(engine)
                        state["cleanup"] = "ocr_closed_before_translation"
                    except Exception as error:
                        state["cleanup"] = "failed: " + type(error).__name__
                        ocr_error = ocr_error or error
            state["ocr_seconds"] = time.monotonic() - started
        if ocr_error is not None:
            raise ocr_error
        assert_source()
        if len(batches) != 1 or len(state["ocr"]["raw_detections"]) != 1:
            raise ValueError("one current OCR page and proposal batch are required")
        batch = batches[0]
        if batch is None:
            if state["ocr"]["raw_detections"][0]:
                raise ValueError("nonempty OCR has no automatic proposal batch")
            units = ()
        else:
            if (batch.page.image_sha256 != image_hash or batch.page.pixels_sha256 != pixels_hash
                    or Path(batch.page.image_path).resolve() != image):
                raise SourceChanged("OCR proposal page does not bind the supplied original image")
            if raw_bound != batch.detections:
                raise SourceChanged("raw OCR evidence and proposal snapshot differ")
            if recognition and len(recognition) != 1:
                raise ValueError("candidate state does not bind the single OCR page")
            candidates = tuple(c for r in recognition[0] for c in r.candidates) if recognition else ()
            units = automatic_units(batch, candidates)
            from ..ocr import crop_readings, ink_groups, reader_presentations
            saved_crop_readings=[c for c in _snapshot(candidates) if c.get('engine')==crop_readings.ENGINE]
            grouped=any(p.grouping_method in ink_groups.POLICIES for p in batch.proposals)
            if saved_crop_readings or grouped or 'ink_column_groups' in state['ocr'] or 'reader_presentations' in state['ocr']:
                with Image.open(BytesIO(_read_image(image))) as source:
                    rgb=source.convert('RGB')
                    ink_groups.validate_saved(state['ocr'],rgb)
                    for candidate in saved_crop_readings:crop_readings.validate_pixels(candidate,rgb)
                    reader_presentations.validate_saved(state['ocr'],rgb,output=output)
        records,fragment_requests = {},{}
        for unit in units:
            request = TranslationRequest((unit,), (unit.unit_id,), style=style)
            status = "blocked" if unit.disposition == "blocked" else "not_selected" if selected_ids and unit.unit_id not in selected_ids else "pending"
            record = {"unit": _snapshot(unit), "unit_revision": unit.revision, "request": _snapshot(request),
                "request_revision": request.revision, "status": status, "blocking_reasons": list(unit.blocking_reasons),
                "input_kind": "strict_utterance", "binding_current": True, "result": None,
                "attempts_started": 0, "generation_evidence": [], "fragment_previews": [], "fragment_rejection_reasons": []}
            fragment_requests[unit.unit_id] = []
            if review_ocr: record["visual_alternates"] = []
            from ..ocr.manga_dialogue import preferred_reading as dialogue_reading
            preferred_dialogue=dialogue_reading(unit) if ocr_refine is not None else None
            if ocr_refine is not None and preferred_dialogue is None:
                from ..ocr.crop_readings import preferred_reading as crop_reading
                preferred_dialogue=crop_reading(unit)
            if unit.disposition == "preview" and preferred_dialogue is None:
                state["strict_eligible_units"] += 1
            elif translation_policy == 'coverage-first' and (not coverage.rejection_reasons(unit)
                    or (preferred_dialogue is not None and unit.disposition == 'preview')):
                reading_revision = preferred_dialogue.revision if preferred_dialogue else None
                if ocr_refine is not None:
                    from ..ocr.quality_reread import preferred_reading
                    preferred = preferred_reading(unit)
                    reading_revision = preferred.revision if preferred is not None else reading_revision
                derived = coverage.derive(unit, reading_revision)
                request = TranslationRequest((derived,), (derived.unit_id,), style=style)
                record.update(input_kind=coverage.KIND, translation_input=_snapshot(derived),
                    coverage=coverage.provenance(unit, reading_revision), request=_snapshot(request), request_revision=request.revision,
                    status='not_selected' if selected_ids and unit.unit_id not in selected_ids else 'pending',
                    scheduling={'reason': 'explicit_selected_ids_order' if selected_ids else 'coverage_weak_text_hints',
                        'priority': list(coverage.priority(unit)), 'certifies_readability_or_dialogue': False})
                state['coverage_eligible_units'] += 1
            if unit.disposition == 'blocked' and record['input_kind'] != coverage.KIND and allow_fragment_previews:
                derived,rejected = fragment_plan(unit,allow_fragment_previews=True)
                record["fragment_rejection_reasons"] = list(rejected)
                for fragment in derived:
                    fragment_request = FragmentRequest(fragment, style=style)
                    fragment_requests[unit.unit_id].append(fragment_request)
                    record["fragment_previews"].append({"input_kind":"unverified_text_fragment",
                        "input":_snapshot(fragment), "input_revision":fragment.revision,
                        "request":_snapshot(fragment_request), "request_revision":fragment_request.revision,
                        "status":"pending" if not selected_ids or unit.unit_id in selected_ids else "fragment_not_selected",
                        "scheduling":fragment_scheduling_info(fragment.selected.text,explicit_selection=bool(selected_ids)),
                        "binding_current":True,"result":None,"attempts_started":0,"generation_evidence":[]})
                    state["fragment_candidate_inputs"] += 1
            if translation_policy == 'coverage-first' and unit.disposition == 'blocked' and record['input_kind'] != coverage.KIND:
                record['coverage_rejection_reasons'] = list(coverage.rejection_reasons(unit))
            state["units"].append(record)
            records[unit.unit_id] = (record, request)
        if translation_context != 'isolated' and units:
            planned = context.page_plan(units, state['units'], state['ocr']['batches'][0],translation_context,style=style)
            for identity, (request, proof) in planned.items():
                record, _ = records[identity]
                record.update(request=_snapshot(request), request_revision=request.revision, context_selection=proof)
                records[identity] = (record, request)
        if not set(selected_ids) <= set(records):
            raise ValueError("selected unit IDs are not present on this page")
        review_proposals = {}
        if review_ocr:
            state["stage"] = "ocr_review"
            try:
                review_proposals = visual.run_pass(units,selected_ids,state["ocr"]["raw_detections"][0],image,output,state["ocr_review"],
                    model=review_model,max_attempts=max_review_attempts,timeout=review_timeout,allow_fragments=allow_fragment_previews,
                    injected_provider=review_runtime,assert_source=assert_source,persist=persist,
                    retain=lambda prefix,event:_retain_event(output,prefix,event),cancelled=cancelled)
            except visual.ReviewBindingError as error:
                raise SourceChanged(str(error)) from error
            except visual.ReviewCancelled as error:
                raise KeyboardInterrupt(str(error)) from error
            finally:
                for row in state["ocr_review"].get("units",[]):
                    records[row["unit_id"]][0]["ocr_review"] = row
        state["stage"] = "translation"
        translator = None
        strict_jobs,coverage_jobs,fragment_jobs = [],[],[]
        for index,unit_id in enumerate(selected_ids or tuple(records)):
            record,request = records[unit_id]
            if record["status"] != "blocked":
                original = next(u for u in units if u.unit_id == unit_id)
                if record['input_kind'] == coverage.KIND:
                    coverage_jobs.append((record,request,original,f"unit-{index:03d}",'coverage'))
                else:
                    strict_jobs.append((record,request,original,f"unit-{index:03d}",False))
            for n,(entry,fragment_request) in enumerate(zip(record["fragment_previews"],fragment_requests[unit_id])):
                fragment_jobs.append((entry,fragment_request,request.units[0],f"unit-{index:03d}-fragment-{n:02d}",True))
        if not selected_ids:
            fragment_jobs.sort(key=lambda job:job[0]["scheduling"]["priority_tier"])
            coverage_jobs.sort(key=lambda job:coverage.priority(job[2]))
        if review_ocr:
            strict_alternates,fragment_alternates = [],{}
            for base_record,base_request,original,prefix,is_fragment in strict_jobs+fragment_jobs:
                if original.unit_id not in review_proposals: continue
                review_input,verdict = review_proposals[original.unit_id]
                proposed = visual.AlternateRequest(visual.AlternateInput(base_request,review_input,verdict))
                entry = {"input_kind":"unverified_visual_alternate","input":_snapshot(proposed.alternate),
                    "input_revision":proposed.alternate.revision,"request":_snapshot(proposed),"request_revision":proposed.revision,
                    "status":"pending","binding_current":True,"attempts_started":0,"result":None,"generation_evidence":[],
                    "scheduling":{"reason":"visual_alternate_before_original_fragment" if is_fragment else "visual_alternate_after_strict_originals"}}
                records[original.unit_id][0]["visual_alternates"].append(entry)
                job = (entry,proposed,original,prefix+"-visual-alternate","alternate")
                if is_fragment: fragment_alternates[original.unit_id] = job
                else: strict_alternates.append(job)
            fragment_jobs = [candidate for job in fragment_jobs
                for candidate in ([fragment_alternates[job[2].unit_id],job] if job[2].unit_id in fragment_alternates else [job])]
            strict_jobs += strict_alternates
        jobs = strict_jobs+coverage_jobs+fragment_jobs
        if allow_fragment_previews or review_ocr or translation_policy == 'coverage-first':
            state["input_schedule"] = []
            for position,(entry,request,original,_,kind) in enumerate(jobs,1):
                if kind:
                    entry["scheduling"]["planned_position"] = position
                state["input_schedule"].append({"planned_position":position,"input_kind":entry["input_kind"],
                    "original_unit_id":original.unit_id,"request_revision":request.revision,
                    "source_ids":list(request.fragment.source_ids) if kind is True else list(original.member_ids),
                    "reason":entry["scheduling"]["reason"] if kind else "strict_inputs_first"})
        persist()  # Persist every candidate and planned order before provider entry.
        with (output / "results.jsonl").open("x", encoding="utf-8") as journal:
            if frontier:
                luna.translate_jobs(jobs, state, model=model, mode=luna_mode, image=image, related=related_pages,
                    budget=max_attempts, timeout=timeout, persist=persist, assert_source=assert_source,
                    cancelled=cancelled, journal=journal)
            for record,request,original,prefix,kind in (() if frontier else jobs):
                is_fragment,is_alternate = kind is True,kind == "alternate"
                status_prefix = "alternate_" if is_alternate else "fragment_" if is_fragment else ""
                unit_id = request.target_ids[0]
                if cancelled(): raise KeyboardInterrupt("workflow cancelled before translation")
                if not run_local_model:
                    record["status"] = status_prefix+"ocr_only"
                elif state["attempts"] >= max_attempts:
                    record["status"] = status_prefix+"budget_exhausted"
                else:
                    assert_source()
                    if translator is None:
                        provider = runtime if runtime is not None else OllamaRuntime(endpoint)
                        counted = _CountedRuntime(provider, state, max_attempts)
                        state["settings"]["provider"] = type(provider).__name__
                        translator = PreviewTranslator(counted, model=model, profile=capability,
                            max_attempts=1, timeout=timeout, generation_options=options,single_rendition=True,terminology_units=units,
                            page_context=translation_context == "sound-effects",
                            table_cells=(p.proposal_id for p in batch.proposals if p.grouping_method=="separate_printed_table_rows"),
                            context_crop=lambda unit, **crop_options: _translation_crop(image,image_hash,state["ocr"]["raw_detections"][0],unit,**crop_options))
                    counted.record = record
                    def observe(event):
                        record["generation_evidence"].append(_retain_event(output,prefix,event))
                    if is_alternate:
                        visual.assert_views(output,records[original.unit_id][0]["ocr_review"])
                        result = translator.translate_alternate(request,allow_visual_alternates=True,
                            allow_fragment_previews=allow_fragment_previews,observe=observe,cancelled=cancelled)
                        result.assert_current(request,original)
                    elif is_fragment:
                        request.fragment.assert_current(original)
                        result = translator.translate_fragment(request,allow_fragment_previews=True,observe=observe,cancelled=cancelled)
                        result.assert_current(request,original)
                    else:
                        result = translator.translate(request, observe=observe,cancelled=cancelled)
                        result.assert_current(request)
                    record["result"], record["status"] = _snapshot(result), result.status
                    assert_source()
                    if result.status.endswith("cancelled"): raise KeyboardInterrupt("translation cancelled")
                journal.write(canonical({"unit_id": unit_id, "request_revision": request.revision,
                    "original_unit_id":original.unit_id,"input_kind":record["input_kind"],
                    "scheduling":record.get("scheduling"),
                    "status": record["status"], "result": record["result"],
                    "generation_evidence": record["generation_evidence"]}) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                persist()
        assert_source()
        state["stage"] = "finished"
        state["status"] = "complete_with_failures" if any(r["status"] == "failed" or
            any(f["status"] == "fragment_failed" for f in r["fragment_previews"]) or
            any(a["status"] == "alternate_failed" for a in r.get("visual_alternates",[])) for r in state["units"]) else "complete"
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        if getattr(error, "provenance", None) is not None:
            state["initialization_provenance"] = _snapshot(error.provenance)
        all_records = [(r,False) for r in state["units"]] + [(f,True) for r in state["units"] for f in r["fragment_previews"]]
        all_records += [(a,"alternate") for r in state["units"] for a in r.get("visual_alternates",[])]
        for record,kind in all_records:
            prefix = "alternate_" if kind == "alternate" else "fragment_" if kind is True else ""
            if isinstance(error, SourceChanged):
                record["status_before_stale"] = record["status"]
                record["status"], record["binding_current"] = prefix+"stale_source", False
            elif record["status"] == "pending":
                record["status"] = prefix+("interrupted" if isinstance(error, KeyboardInterrupt) and record["attempts_started"] else "not_executed")
                if record["status"] == prefix+"interrupted":
                    record["generation_evidence"].append({"status": "interrupted", "response_available": False})
    state["unit_status_counts"] = {s: sum(r["status"] == s for r in state["units"])
        for s in sorted({r["status"] for r in state["units"]})}
    fragments = [f for r in state["units"] for f in r["fragment_previews"]]
    state["fragment_status_counts"] = {s:sum(f["status"] == s for f in fragments) for s in sorted({f["status"] for f in fragments})}
    state["strict_preview_count"] = sum(r["status"] == "preview" and r['input_kind'] == 'strict_utterance' for r in state["units"])
    state['coverage_preview_count'] = sum(r['status'] == 'preview' and r['input_kind'] == coverage.KIND for r in state['units'])
    state['translation_preview_count'] = state['strict_preview_count']+state['coverage_preview_count']
    state["fragment_preview_count"] = sum(f["status"] == "fragment_preview" for f in fragments)
    if review_ocr:
        alternates = [a for r in state["units"] for a in r["visual_alternates"]]
        state["alternate_preview_count"] = sum(a["status"] == "alternate_preview" for a in alternates)
        state["alternate_status_counts"] = {s:sum(a["status"] == s for a in alternates) for s in {a["status"] for a in alternates}}
    persist()
    return state
