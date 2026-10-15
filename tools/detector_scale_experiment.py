"""Bounded cached Paddle detector-only scale comparison; model-free unless opted in."""
import argparse
import ast
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
from io import StringIO
import json
import math
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "PP-OCRv5_server_det"
MODEL_DIR = Path("/home/Danuu/.paddlex/official_models/PP-OCRv5_server_det")
MODEL_FILES = ("inference.yml","inference.json","inference.pdiparams")
IMAGES = ("images/lmgJZ.jpg",)+tuple("images/ubunchu-ja-test/"+name+".png" for name in
    ("ubunchu01_02","ubunchu01_04","ubunchu02_02","ubunchu02_04","ubunchu03_0002","ubunchu03_0004"))
RUNTIME = {"device":"cpu","cpu_threads":1,"enable_hpi":False,"enable_mkldnn":False,
           "use_tensorrt":False,"enable_cinn":False}
INERT_TRANSFORMS = {"DetLabelEncode":None,
                    "KeepKeys":{"keep_keys":["image","shape","polys","ignore_tags"]}}
LIBC_PROBE = "ldd --version | awk '/ldd/{print $NF}'"
AVX_PROBE = "cat /proc/cpuinfo | grep -i avx"
LIBDL_SONAME = "libdl.so.2"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def parse_scale(value):
    try:
        number,kind = value.split(":")
        side = int(number)
    except (ValueError,AttributeError):
        raise argparse.ArgumentTypeError("scale must be LIMIT_SIDE_LEN:max or LIMIT_SIDE_LEN:min")
    if not 32 <= side <= 2048 or kind not in ("max","min"):
        raise argparse.ArgumentTypeError("scale requires 32..2048 and limit_type max/min")
    return {"limit_side_len":side,"limit_type":kind}


def validate_variants(variants):
    variants = tuple(variants)
    for variant in variants:
        if (not isinstance(variant,dict) or set(variant) != {"limit_side_len","limit_type"}
                or type(variant["limit_side_len"]) is not int or not 32 <= variant["limit_side_len"] <= 2048
                or variant["limit_type"] not in ("max","min")):
            raise ValueError("only validated scale overrides are permitted")
    if len(variants) > 2 or len({(v["limit_side_len"],v["limit_type"]) for v in variants}) != len(variants):
        raise ValueError("at most two distinct scale variants are permitted")
    return variants


def model_provenance(directory):
    import yaml  # Already installed Paddle dependency; no SDK/model import.
    directory = Path(directory).resolve(strict=True)
    if directory != MODEL_DIR.resolve(strict=True):
        raise ValueError("only the explicitly pinned cached detector directory is permitted")
    for name in ("model_state.pdparams","model.safetensors"):
        if (directory/name).exists():
            raise ValueError("dynamic model files are not permitted in this static-model experiment")
    files = {}
    for name in MODEL_FILES:
        path = directory/name
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise ValueError("missing/nonregular cached static model file: "+name)
        files[name] = {"sha256":sha256_file(path),"bytes":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns}
    if files["inference.yml"]["bytes"] > 1024*1024:
        raise ValueError("unexpectedly large detector export configuration")
    config = yaml.safe_load((directory/"inference.yml").read_text())
    if (not isinstance(config,dict) or not isinstance(config.get("Global"),dict)
            or config["Global"].get("model_name") != MODEL_NAME):
        raise ValueError("cached export must identify the pinned PP-OCRv5_server_det model")
    transforms = config.get("PreProcess",{}).get("transform_ops",[])
    allowed = {"DecodeImage","DetResizeForTest","NormalizeImage","ToCHWImage"}|set(INERT_TRANSFORMS)
    if (not isinstance(transforms,list) or any(not isinstance(op,dict) or len(op) != 1 or not set(op) <= allowed for op in transforms)
            or any(sum(name in op for op in transforms) != 1 for name in ("DetResizeForTest","NormalizeImage","ToCHWImage"))):
        raise ValueError("unexpected detector preprocessing configuration")
    ignored = []
    for name,expected in INERT_TRANSFORMS.items():
        entries = [op[name] for op in transforms if name in op]
        if len(entries) > 1 or entries and entries[0] != expected:
            raise ValueError("unsupported training-only transform fields: "+name)
        if entries:
            ignored.append({name:entries[0]})
    resize = next(op["DetResizeForTest"] for op in transforms if "DetResizeForTest" in op) or {}
    if any(resize.get(key) is not None for key in ("image_shape","input_shape")):
        raise ValueError("fixed-shape resizing is not a side-length experiment")
    normalize = next(op["NormalizeImage"] for op in transforms if "NormalizeImage" in op) or {}
    scale = normalize.get("scale")
    if scale is None:
        scale = 1/255
    if isinstance(scale,str):
        # Installed NormalizeImage evaluates string scales. Permit only a tiny
        # numeric literal/division grammar, never names/calls or arbitrary code.
        if len(scale) > 32:
            raise ValueError("unsupported normalization scale expression")
        node = ast.parse(scale,mode="eval").body
        literal = lambda n: float(n.value) if isinstance(n,ast.Constant) and type(n.value) in (int,float) else None
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
            a,b = literal(node.left),literal(node.right)
            scale = a/b if a is not None and b is not None and b != 0 else None
        else:
            scale = literal(node)
    if scale is None or isinstance(scale,bool) or not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("unsupported normalization scale")
    post = config.get("PostProcess",{})
    if (not isinstance(post,dict) or post.get("name") != "DBPostProcess" or post.get("box_type","quad") != "quad"
            or type(post.get("use_dilation",False)) is not bool):
        raise ValueError("only the pinned detector's quad DB postprocessor is supported")
    for name in ("thresh","box_thresh","unclip_ratio"):
        if name in post:
            value = post[name]
            if (type(value) not in (int,float) or not math.isfinite(value) or value <= 0
                    or (name != "unclip_ratio" and value > 1)):
                raise ValueError("unsupported cached postprocess parameter: "+name)
    return {"model_name":MODEL_NAME,"model_dir":str(directory),"files":files,
        "ignored_training_transforms":ignored,
        "export_resize_config":{key:resize[key] for key in ("limit_side_len","resize_long","limit_type","max_side_limit") if key in resize},
        "export_postprocess_config":{key:post[key] for key in ("name","thresh","box_thresh","unclip_ratio","use_dilation","max_candidates","score_mode","box_type") if key in post},
        "effective_defaults":"read from initialized predictor; YAML alone is not proof of wrapper defaults"}


def build_plan(variants=(),model_dir=None):
    variants = validate_variants(variants)
    model = model_provenance(model_dir or MODEL_DIR)
    images = []
    for relative in IMAGES:
        path = (ROOT/relative).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError("evaluation images must be existing local files in the checkout")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            images.append({"path":str(path),"relative_path":relative,"sha256":sha256_file(path),
                "size":list(image.size),"rgb_pixels_sha256":hashlib.sha256(rgb.tobytes()).hexdigest()})
    if len(images) != 7 or len({entry["path"] for entry in images}) != 7:
        raise ValueError("exactly seven distinct approved evaluation images are required")
    versions = {}
    for name in ("paddleocr","paddlex","paddlepaddle-gpu","paddlepaddle","numpy","Pillow","PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"schema_version":1,"model":model,"images":images,"runtime":dict(RUNTIME),"versions":versions,
        "python":sys.version,"platform":{"system":sys.platform,"machine":os.uname().machine,"release":os.uname().release},
        "source_script_sha256":sha256_file(__file__),
        "conditions":[{"name":"baseline","overrides":{}}]+[
            {"name":"scale-"+str(v["limit_side_len"])+"-"+v["limit_type"],"overrides":v} for v in variants],
        "maximum_page_conditions":7*(1+len(variants)),"historical_baseline_reproduced":False,
        "scope":"fresh detector-only baseline; historical saved OCR runtime settings are unknown; six Ubunchu pages are one CC BY-NC evaluation series"}


def read_glibc_version():
    """Equivalent to Paddle's fixed Linux libc query, without a child process."""
    value = os.confstr("CS_GNU_LIBC_VERSION")
    parts = value.split() if value else []
    if len(parts) != 2 or parts[0] != "glibc" or not all(p.isdigit() for p in parts[1].split(".")):
        raise RuntimeError("fixed read-only GNU libc capability query unavailable")
    return parts[1]


def read_avx_support():
    """Match installed Paddle's fixed grep query, using only the fixed kernel file."""
    return "avx" in Path("/proc/cpuinfo").read_text().lower()


def verified_libdl_soname():
    """Validate only cuda.pathfinder's documented stable Linux libdl fallback."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("fixed libdl resolution is Linux-only")
    try:
        library = ctypes.CDLL(LIBDL_SONAME)
        # These are exactly the symbols used by installed load_dl_linux.py.
        library.dlinfo
        library.dlerror
    except (OSError,AttributeError) as error:
        raise RuntimeError("fixed Linux libdl soname/symbols unavailable") from error
    return LIBDL_SONAME


class _CapabilityResult:
    """Only the communicate() interface used by installed core.run_shell_command."""
    def __init__(self,stdout):
        self.stdout = stdout
        self.returncode = 0

    def communicate(self):
        return self.stdout,b""


@contextmanager
def offline_local_only():
    flags = {"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK":"True","HF_HUB_OFFLINE":"1",
             "PADDLE_PDX_DISABLE_DEVICE_FALLBACK":"True","PADDLE_PDX_INFER_BENCHMARK":"False",
             "PADDLE_PDX_PIPELINE_BENCHMARK":"False"}
    previous = {name:os.environ.get(name) for name in flags}
    capabilities = {}
    original_find_library = ctypes.util.find_library

    def fixed_find_library(name):
        if name != "dl":
            return original_find_library(name)
        if "libdl_soname" not in capabilities:
            capabilities["libdl_soname"] = verified_libdl_soname()
        return capabilities["libdl_soname"]

    def fixed_popen(*args,**kwargs):
        if args != (LIBC_PROBE,) or kwargs != {"stdout":subprocess.PIPE,"stderr":subprocess.PIPE,"shell":True}:
            raise RuntimeError("child processes forbidden; only the fixed in-process libc query is supported")
        if "glibc_version" not in capabilities:
            capabilities["glibc_version"] = read_glibc_version()
        return _CapabilityResult((capabilities["glibc_version"]+"\n").encode("ascii"))

    def fixed_cpu_popen(command,mode="r",buffering=-1):
        if command != AVX_PROBE or mode != "r" or buffering != -1:
            raise RuntimeError("shell commands forbidden; only the fixed in-process AVX query is supported")
        if "avx_supported" not in capabilities:
            capabilities["avx_supported"] = read_avx_support()
        return StringIO("flags : avx\n" if capabilities["avx_supported"] else "")

    try:
        os.environ.update(flags)
        with ExitStack() as stack:
            for target,name in ((socket.socket,"connect"),(socket.socket,"connect_ex"),(socket.socket,"sendto"),
                                (socket,"create_connection"),(socket,"getaddrinfo"),(os,"system"),(os,"fork")):
                if hasattr(target,name):
                    stack.enter_context(patch.object(target,name,side_effect=RuntimeError("network/child processes forbidden in local detector experiment")))
            stack.enter_context(patch.object(BaseProcess,"start",side_effect=RuntimeError("child processes forbidden in local detector experiment")))
            # Installed paddle/base/core.py performs these exact read-only shell
            # queries during import. Emulate them from fixed OS data; never run
            # ldd, cat, grep, awk, or arbitrary caller-supplied commands.
            stack.enter_context(patch.object(subprocess,"Popen",side_effect=fixed_popen))
            stack.enter_context(patch.object(os,"popen",side_effect=fixed_cpu_popen))
            stack.enter_context(patch.object(ctypes.util,"find_library",side_effect=fixed_find_library))
            yield capabilities
    finally:
        for name,value in previous.items():
            if value is None:
                os.environ.pop(name,None)
            else:
                os.environ[name] = value


class LocalDetector:
    """One installed TextDetection instance, static cached weights, no lookup fallback."""
    def __init__(self,directory):
        if Path(directory).resolve() != MODEL_DIR.resolve():
            raise ValueError("cached detector path is pinned")
        if "paddleocr" in sys.modules or "paddlex" in sys.modules:
            raise RuntimeError("run the live detector experiment in a fresh process")
        from paddleocr import TextDetection
        from paddlex.inference.utils.official_models import _ModelManager
        self.lookup_guard = patch.object(_ModelManager,"__getitem__",side_effect=RuntimeError("model-name lookup/download fallback forbidden"))
        self.lookup_guard.start()
        self.model = None
        try:
            self.model = TextDetection(model_name=MODEL_NAME,model_dir=str(directory),**RUNTIME)
            predictor = self.model.paddlex_predictor
            required = {"limit_side_len","limit_type","max_side_limit","thresh","box_thresh","unclip_ratio"}
            if (type(predictor).__name__ != "TextDetPredictor" or not predictor._use_static_model
                    or predictor.model_name != MODEL_NAME or predictor.model_dir.resolve() != Path(directory).resolve()
                    or predictor.batch_sampler.batch_size != 1 or not required <= set(inspect.signature(predictor.process).parameters)):
                raise RuntimeError("installed detector API/model contract differs from inspected source")
            self.initial = self.settings()
        except Exception:
            self.close()
            raise

    def settings(self):
        predictor = self.model.paddlex_predictor
        resize,post = predictor.pre_tfs["Resize"],predictor.post_op
        if resize.resize_type != 0 or predictor.device != "cpu":
            raise RuntimeError("expected serial CPU side-length detector")
        result = {"limit_side_len":int(resize.limit_side_len),"limit_type":resize.limit_type,
            "max_side_limit":int(predictor.max_side_limit),"thresh":float(post.thresh),
            "box_thresh":float(post.box_thresh),"unclip_ratio":float(post.unclip_ratio),
            "use_dilation":bool(post.use_dilation),"max_candidates":int(post.max_candidates),
            "score_mode":post.score_mode,"box_type":post.box_type,"min_size":int(post.min_size)}
        if (not 32 <= result["limit_side_len"] <= 2048 or result["limit_type"] not in ("max","min")
                or not 32 <= result["max_side_limit"] <= 4000 or result["box_type"] != "quad"
                or not all(math.isfinite(result[key]) and result[key] > 0 for key in ("thresh","box_thresh","unclip_ratio"))):
            raise RuntimeError("unsupported effective detector parameters")
        return result

    def predict(self,path,params):
        if self.settings() != self.initial:
            raise RuntimeError("effective detector settings changed during the experiment")
        options = {key:params[key] for key in ("limit_side_len","limit_type","max_side_limit","thresh","box_thresh","unclip_ratio")}
        iterator = iter(self.model.predict_iter(str(path),**options))
        result = next(iterator)
        sentinel = object()
        if next(iterator,sentinel) is not sentinel:
            raise RuntimeError("one raster input must yield exactly one detector result")
        if self.settings() != self.initial:
            raise RuntimeError("effective detector settings drift")
        return getattr(result,"res",result)

    def close(self):
        model,self.model = self.model,None
        try:
            if model is not None:
                model.close()
        finally:
            self.lookup_guard.stop()


def native_result(result,image):
    if not isinstance(result,Mapping):
        raise ValueError("native detector result must be a mapping")
    if any(key in result for key in ("rec_texts","rec_scores","rec_polys")):
        raise ValueError("recognizer/pipeline output is forbidden in detector-only comparison")
    if result.get("input_path") is not None and Path(result["input_path"]).resolve() != Path(image["path"]):
        raise ValueError("detector returned a different input path")
    polys,scores = result["dt_polys"],result["dt_scores"]
    if len(polys) != len(scores):
        raise ValueError("native detector polygon/score length mismatch")
    arrays = [np.asarray(poly) for poly in polys]
    if any(array.ndim != 2 or array.shape != (4,2) or array.dtype.kind not in "iuf"
           or array.dtype.itemsize > 8 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("invalid native quad polygon; no repair or truncation is applied")
    if any(isinstance(score,(bool,np.bool_)) or not isinstance(score,(int,float,np.integer,np.floating))
           or isinstance(score,np.floating) and score.dtype.itemsize > 8
           or not math.isfinite(float(score)) or not 0 <= float(score) <= 1 for score in scores):
        raise ValueError("invalid native detector score")
    pixels = np.asarray(result["input_img"])
    if pixels.shape != (image["size"][1],image["size"][0],3) or pixels.dtype != np.uint8:
        raise ValueError("native input pixels are not the original raster dimensions/type")
    return {"dt_polys":[array.tolist() for array in arrays],"dt_scores":[float(score) for score in scores],
        "polygon_dtypes":[str(array.dtype) for array in arrays],"native_input_shape":list(pixels.shape),
        "native_input_dtype":str(pixels.dtype),"native_input_sha256":hashlib.sha256(pixels.tobytes()).hexdigest(),
        "detector_index_scope":"native order within this page/condition only; not cross-scale identity"}


def write_json(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def raw_native_evidence(result):
    if not isinstance(result,Mapping):
        return {"unsupported_result_type":type(result).__name__}
    def value(item):
        if isinstance(item,np.ndarray):
            return value(item.tolist())
        if isinstance(item,np.generic):
            plain = item.item()
            return {"type":str(item.dtype),"value":str(item)} if isinstance(plain,np.generic) else value(plain)
        if isinstance(item,float) and not math.isfinite(item):
            return {"type":"float","nonfinite":repr(item)}
        if item is None or isinstance(item,(str,bool,int,float)):
            return item
        if isinstance(item,(list,tuple)):
            return [value(v) for v in item]
        if isinstance(item,Mapping):
            return {str(k):value(v) for k,v in item.items()}
        return {"unsupported_type":type(item).__name__}
    return {key:value(result[key]) for key in ("input_path","page_index","dt_polys","dt_scores") if key in result}


def execute(plan,output):
    manifest = {"status":"running","detector_calls":0,"completed_conditions":[],"recognition_calls":0,
                "grouping_calls":0,"plan":plan,"started_at_utc":datetime.now(timezone.utc).isoformat()}
    detector = None
    try:
        if len(plan["images"]) != 7 or not 1 <= len(plan["conditions"]) <= 3:
            raise ValueError("execution requires seven images and baseline plus at most two variants")
        with offline_local_only() as capabilities:
            manifest["read_only_capability_probes"] = capabilities
            started = time.perf_counter()
            try:
                detector = LocalDetector(plan["model"]["model_dir"])
            finally:
                manifest["initialization_seconds"] = time.perf_counter()-started
            initial = detector.settings()
            manifest["effective_baseline"] = initial
            seen = {(initial["limit_side_len"],initial["limit_type"])}
            for condition in plan["conditions"][1:]:
                pair = (condition["overrides"]["limit_side_len"],condition["overrides"]["limit_type"])
                if pair in seen:
                    raise ValueError("variant duplicates the fresh effective baseline")
                seen.add(pair)
            for page_index,image in enumerate(plan["images"]):
                native_hash = None
                baseline_count = None
                for condition in plan["conditions"]:
                    params = dict(initial,**condition["overrides"])
                    manifest["active_condition"] = {"page_index":page_index,"condition":condition["name"],
                                                    "image":image,"effective_parameters":params}
                    if manifest["detector_calls"] >= 21:
                        raise RuntimeError("21-call resource bound exceeded")
                    if detector.settings() != initial:
                        raise RuntimeError("fixed effective detector settings drift")
                    if sha256_file(image["path"]) != image["sha256"]:
                        raise ValueError("original image drift before detector call")
                    for name,identity in plan["model"]["files"].items():
                        stat = (Path(plan["model"]["model_dir"])/name).stat()
                        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
                            raise ValueError("cached model file drift before detector call")
                    manifest["detector_calls"] += 1
                    started = time.perf_counter()
                    try:
                        result = detector.predict(image["path"],params)
                    finally:
                        elapsed = time.perf_counter()-started
                        manifest["active_condition"]["seconds"] = elapsed
                    try:
                        native = native_result(result,image)
                        if detector.settings() != initial:
                            raise RuntimeError("fixed effective detector settings drift after call")
                        if native_hash is not None and native["native_input_sha256"] != native_hash:
                            raise ValueError("native decoded source pixels changed across scale conditions")
                        observed_sha = sha256_file(image["path"])
                        manifest["active_condition"]["observed_image_sha256"] = observed_sha
                        if observed_sha != image["sha256"]:
                            raise ValueError("original image drift after detector call")
                    except Exception as error:
                        write_json(output/f"page-{page_index:04d}-{condition['name']}-failed-detector.json",
                            {"image":image,"effective_parameters":params,"seconds":elapsed,
                             "native_raw":raw_native_evidence(result),"error":{"type":type(error).__name__,"message":str(error)}})
                        raise
                    native_hash = native["native_input_sha256"]
                    if baseline_count is None:
                        baseline_count = len(native["dt_polys"])
                    record = {"page_index":page_index,"condition":condition["name"],"image":image,
                        "effective_parameters":params,"seconds":elapsed,"native":native,
                        "detection_count_delta_vs_fresh_baseline":len(native["dt_polys"])-baseline_count}
                    path = output/f"page-{page_index:04d}-{condition['name']}-detector.json"
                    write_json(path,record)
                    manifest["completed_conditions"].append({"path":str(path),"page_index":page_index,"condition":condition["name"]})
            closing,detector = detector,None
            closing.close()
        if model_provenance(plan["model"]["model_dir"]) != plan["model"]:
            raise ValueError("cached model/configuration drift during experiment")
        manifest["status"] = "completed"
        manifest.pop("active_condition",None)
    except Exception as error:
        manifest.update(status="failed",error={"type":type(error).__name__,"message":str(error)})
    finally:
        if detector is not None:
            try:
                with offline_local_only():
                    detector.close()
            except Exception as error:
                manifest["close_error"] = {"type":type(error).__name__,"message":str(error)}
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output/"detector-scale-results.json",manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,help="exclusive new output directory; omitted prints help")
    parser.add_argument("--model-dir",type=Path,default=None,help="must equal the explicitly pinned cached detector directory")
    parser.add_argument("--variant",type=parse_scale,action="append",default=[],metavar="LEN:TYPE",help="at most two scale-only overrides")
    parser.add_argument("--run-local-detector",action="store_true",help="explicitly initialize the pinned installed detector; no recognizer")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        if args.run_local_detector:
            parser.error("--run-local-detector requires --output-dir")
        parser.print_help()
        return 0
    output = args.output_dir.resolve()
    results_root = (ROOT/"test-results").resolve()
    if not output.is_relative_to(results_root) or output == results_root:
        parser.error("output must be a new directory under this checkout's test-results")
    plan = build_plan(args.variant,args.model_dir)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"detector-scale-plan.json",plan)
    if not args.run_local_detector:
        print(json.dumps({"status":"model_free_plan","maximum_page_conditions":plan["maximum_page_conditions"],"detector_calls":0}))
        return 0
    manifest = execute(plan,output)
    print(json.dumps({"status":manifest["status"],"detector_calls":manifest["detector_calls"],
                      "completed_conditions":len(manifest["completed_conditions"]),"recognition_calls":0}))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
