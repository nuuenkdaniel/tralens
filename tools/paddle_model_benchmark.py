"""Parent-only serial, local-weight OCR benchmark. Import/help never loads Paddle."""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tralens.ocr.paddle_profiles import MODEL_FILES, PROFILES, resolve_profile
from tralens.ocr.paddle_result import associate_paddle_result, PaddleResultContractError, evidence_value

DEFAULT_MANIFEST = ROOT / "test-results/detector-scale-parent-v2/detector-scale-results.json"
GPU_QUERY = ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
             "--format=csv,noheader,nounits"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_records(manifest=None, images=None):
    from PIL import Image
    if manifest is not None and images is not None:
        raise ValueError("choose manifest or explicit images")
    saved = None
    if images is None:
        manifest = Path(manifest or DEFAULT_MANIFEST).resolve(strict=True)
        saved = json.loads(manifest.read_text())["plan"]["images"]
        if not isinstance(saved, list) or len(saved) != 7:
            raise ValueError("fixed manifest must contain exactly seven plan.images")
        images = [entry["path"] for entry in saved]
    if not images:
        raise ValueError("at least one existing raster image is required")
    records = []
    for index, image_path in enumerate(images):
        path = Path(image_path).expanduser().resolve(strict=True)
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("benchmark accepts single-frame raster images only")
            rgb = source.convert("RGB")
            record = {"path": str(path), "sha256": sha256_file(path), "size": list(rgb.size),
                      "rgb_pixels_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        if saved is not None and any(record[key] != saved[index][key]
                                     for key in ("sha256", "size", "rgb_pixels_sha256")):
            raise ValueError("manifest image provenance mismatch: " + str(path))
        records.append(record)
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("duplicate image paths")
    return records


def build_plan(args):
    if not args.run_local_ocr:
        raise ValueError("explicit --run-local-ocr is required; no models were loaded")
    if not 1 <= args.warm_passes <= 20:
        raise ValueError("warm passes must be 1..20")
    if not math.isfinite(args.sample_interval) or not .05 <= args.sample_interval <= 60:
        raise ValueError("sample interval must be .05..60 seconds")
    if not 1 <= args.cpu_threads <= 256:
        raise ValueError("cpu threads must be 1..256")
    profile = resolve_profile(args.profile, args.device, args.model_root, cpu_threads=args.cpu_threads)
    settings = dict(profile.settings(), cpu_threads=args.cpu_threads, enable_hpi=False,
                    enable_mkldnn=False, use_tensorrt=False, enable_cinn=False,
                    text_recognition_batch_size=1)
    weights = {}
    for name, directory in zip(PROFILES[profile.name], profile.model_dirs):
        weights[name] = {"directory": directory, "files": {
            file: {"sha256": sha256_file(Path(directory) / file),
                   "bytes": (Path(directory) / file).stat().st_size} for file in MODEL_FILES}}
    images = image_records(args.manifest, args.images)
    manifest = None if args.images else Path(args.manifest or DEFAULT_MANIFEST).resolve()
    sources = [Path(__file__), ROOT / "src/tralens/ocr/paddle_profiles.py",
               ROOT / "src/tralens/ocr/paddle_result.py", ROOT / "src/tralens/ocr/detection.py"]
    return {"schema_version": 1, "identity": profile.identity(), "parameters": settings,
            "weights": weights, "images": images, "warm_passes": args.warm_passes,
            "manifest": None if manifest is None else {"path": str(manifest), "sha256": sha256_file(manifest)},
            "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
            "python": sys.version, "platform": sys.platform, "pid": os.getpid(),
            "sample_interval_seconds": args.sample_interval,
            "timing_policy": "one first-page process-cold inference, then serial warm-process passes; OS caches not flushed",
            "quality_scope": "geometry/contract evidence only; counts are not accuracy; no output replacement"}


@contextmanager
def local_only():
    """Block network and official-model lookup throughout construction/inference."""
    if any(name in sys.modules for name in ("paddleocr", "paddlex", "paddle")):
        raise RuntimeError("benchmark requires a fresh process with no Paddle imports")
    flags = {"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True", "HF_HUB_OFFLINE": "1",
             "PADDLE_PDX_DISABLE_DEVICE_FALLBACK": "True",
             "PADDLE_PDX_INFER_BENCHMARK": "False", "PADDLE_PDX_PIPELINE_BENCHMARK": "False"}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, flags))
        for target, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                             (socket.socket, "sendto"), (socket, "create_connection"), (socket, "getaddrinfo")):
            stack.enter_context(patch.object(target, name, side_effect=RuntimeError("benchmark network forbidden")))
        # Source checks are disabled before import; all subsequent name lookups fail closed.
        manager = importlib.import_module("paddlex.inference.utils.official_models")._ModelManager
        stack.enter_context(patch.object(manager, "__getitem__", side_effect=RuntimeError(
            "model lookup/download fallback forbidden; supply complete local exports")))
        yield


def parse_gpu_rows(text, pid):
    rows = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] != str(pid):
            continue
        if len(fields) != 3:
            raise ValueError("malformed per-process GPU telemetry")
        try:
            memory = float(fields[2])
        except ValueError:
            memory = None
        if memory is not None and (not math.isfinite(memory) or memory < 0):
            memory = None
        rows.append({"gpu_uuid": fields[1], "used_memory_mib": memory})
    return rows


class GpuSampler:
    def __init__(self, interval, enabled):
        self.interval, self.enabled = interval, enabled
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.started = time.perf_counter()
        self.thread = None

    def sample(self):
        try:
            run = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=5, check=True)
            self.samples.append({"elapsed_seconds": time.perf_counter() - self.started,
                                 "process_gpus": parse_gpu_rows(run.stdout, os.getpid())})
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            message = type(error).__name__ + ": " + str(error)
            if message not in self.errors:
                self.errors.append(message)

    def loop(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.interval)

    def __enter__(self):
        if self.enabled:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
            self.sample()

    def report(self):
        measured = [row["used_memory_mib"] for sample in self.samples
                    for row in sample["process_gpus"] if row["used_memory_mib"] is not None]
        return {"enabled": self.enabled, "pid": os.getpid(), "query": GPU_QUERY,
                "interval_seconds": self.interval, "samples": self.samples, "errors": self.errors,
                "peak_single_gpu_process_mib": max(measured) if measured else None,
                "status": "sampled" if measured else "unavailable" if self.enabled else "not_requested_cpu",
                "limitations": "sampled, may miss peaks; empty PID rows are missing telemetry, never zero; no desktop totals"}


def usage():
    used = resource.getrusage(resource.RUSAGE_SELF)
    return {"cpu_process_seconds": time.process_time(), "user_seconds": used.ru_utime,
            "system_seconds": used.ru_stime, "peak_rss_raw": used.ru_maxrss,
            "peak_rss_unit": "bytes" if sys.platform == "darwin" else "KiB"}


def measured(call, synchronize, timing=None):
    timing = timing if timing is not None else {}
    synchronize()
    start, cpu = time.perf_counter(), time.process_time()
    try:
        result = call()
        synchronize()
        timing["status"] = "completed"
        return result, timing
    except Exception:
        timing["status"] = "failed"
        raise
    finally:
        timing.update(wall_seconds=time.perf_counter() - start,
                      cpu_process_seconds=time.process_time() - cpu)


def runtime_identity(engine, plan):
    pipeline = engine.paddlex_pipeline
    records = []
    for predictor, name, directory in zip((pipeline.text_det_model, pipeline.text_rec_model),
                                         PROFILES[plan["identity"]["profile"]],
                                         [v["directory"] for v in plan["weights"].values()]):
        record = {"model_name": predictor.model_name, "model_dir": str(Path(predictor.model_dir).resolve()),
                  "device": predictor.device}
        if record != {"model_name": name, "model_dir": directory, "device": plan["parameters"]["device"]}:
            raise RuntimeError("initialized predictor identity/device mismatch: " + repr(record))
        records.append(record)
    return {"predictors": records,
            "merged_pipeline_config": evidence_value(engine._merged_paddlex_config),
            "pipeline_parameters": {name: evidence_value(getattr(pipeline, name)) for name in (
                "text_det_limit_side_len", "text_det_limit_type", "text_det_max_side_limit", "text_det_thresh",
                "text_det_box_thresh", "text_det_unclip_ratio", "text_rec_score_thresh", "return_word_box")}}


def run_benchmark(plan, report):
    engine = None
    sampler = GpuSampler(plan["sample_interval_seconds"], plan["parameters"]["device"].startswith("gpu:"))
    try:
        with local_only(), sampler:
            # All live imports are inside the explicit parent-only run gate.
            from paddleocr import PaddleOCR
            import paddle
            synchronize = (lambda: paddle.device.cuda.synchronize(int(plan["parameters"]["device"].split(":")[1]))) if sampler.enabled else lambda: None
            report["gpu_timing_synchronization"] = "paddle.device.cuda.synchronize before/after" if sampler.enabled else "not_applicable_cpu"
            report["initialization"] = {}
            engine, _ = measured(lambda: PaddleOCR(**plan["parameters"]), synchronize, report["initialization"])
            report["runtime"] = runtime_identity(engine, plan)
            report["records"] = []
            schedule = [("cold", 0, 0)] + [("warm", pass_index + 1, page_index)
                for pass_index in range(plan["warm_passes"]) for page_index in range(len(plan["images"]))]
            for phase, pass_index, page_index in schedule:
                image = plan["images"][page_index]
                if sha256_file(image["path"]) != image["sha256"]:
                    raise ValueError("image changed since preflight")
                results, timing = measured(lambda: list(engine.predict(image["path"])), synchronize)
                if len(results) != 1:
                    raise ValueError("one raster must produce exactly one page")
                raw = getattr(results[0], "res", results[0])
                try:
                    _, contract = associate_paddle_result(raw, page_index)
                except PaddleResultContractError as error:
                    contract = error.report
                record = {"phase": phase, "pass": pass_index, "page_index": page_index,
                          "timing": timing, "resources": usage(), "contract": contract}
                # Contract raw contains every dt/rec polygon, text and score, including malformed evidence.
                report["records"].append(record)
                if phase == "cold":
                    report["first_inference"] = timing
            report["status"] = "completed"
    finally:
        try:
            if engine is not None:
                engine.close()
        finally:
            report["gpu_telemetry"] = sampler.report()
            report["resources_final"] = usage()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-local-ocr", action="store_true")
    result.add_argument("--model-root", required=True, type=Path)
    result.add_argument("--profile", choices=PROFILES, default="v5_server")
    result.add_argument("--device", required=True, help="cpu or one explicit gpu:N; no fallback")
    inputs = result.add_mutually_exclusive_group()
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--images", nargs="+")
    result.add_argument("--warm-passes", type=int, default=2)
    result.add_argument("--cpu-threads", type=int, default=1)
    result.add_argument("--sample-interval", type=float, default=.2)
    result.add_argument("--output", required=True, type=Path, help="new JSON file; existing files refused")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    # Reserve exclusively before any OCR initialization. Partial/failed evidence is retained.
    with args.output.open("x", encoding="utf-8") as output:
        report = {"plan": plan, "status": "started", "resources_initial": usage()}
        try:
            run_benchmark(plan, report)
        except Exception as error:
            report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
