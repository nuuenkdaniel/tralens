"""Small, engine-free Paddle model selection and local export validation."""
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import re


PROFILES = {
    "v5_server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    "v6_tiny": ("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
    "v6_small": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    "v6_medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}
MODEL_FILES = ("inference.yml", "inference.json", "inference.pdiparams")
DEFAULT_PROFILE = "v6_medium"
V6_RUNTIME = {"paddleocr": "3.7.0", "paddlex": "3.7.2"}


def package_versions():
    versions = {}
    for name in ("paddleocr", "paddlex", "paddlepaddle-gpu", "paddlepaddle", "numpy", "Pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_runtime(settings):
    """Fail before SDK import for v6 outside the explicitly validated runtime."""
    names = (settings.get("text_detection_model_name"), settings.get("text_recognition_model_name"))
    if not any(isinstance(name, str) and name.startswith("PP-OCRv6_") for name in names):
        return
    installed = package_versions()
    if any(installed.get(name) != version for name, version in V6_RUNTIME.items()):
        required = ", ".join(name + "==" + version for name, version in V6_RUNTIME.items())
        found = ", ".join(name + "=" + str(installed.get(name)) for name in V6_RUNTIME)
        raise RuntimeError("Paddle v6 requires the validated runtime " + required + "; found " + found +
                           ". Use requirements-paddle6.txt in the isolated environment, or explicitly "
                           "select model_profile='v5_server' in the original runtime. No model fallback was attempted.")


def validate_export(directory, model_name):
    """Require a complete static export and matching public export identity."""
    import yaml  # Export parser only; does not import Paddle or initialize models.
    path = Path(directory).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("model directory is not a directory: " + str(path))
    for name in MODEL_FILES:
        file = path / name
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError("missing or empty local model file: " + str(file))
    config_path = path / "inference.yml"
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("model export configuration exceeds 1 MiB")
    try:
        config = yaml.safe_load(config_path.read_text())
    except (yaml.YAMLError, UnicodeError) as error:
        raise ValueError("invalid local model export configuration: " + str(config_path)) from error
    if (not isinstance(config, dict) or not isinstance(config.get("Global"), dict)
            or config["Global"].get("model_name") != model_name):
        raise ValueError("local export identity does not match " + model_name)
    return str(path)


@dataclass(frozen=True)
class PaddleProfile:
    name: str
    device: str | None
    model_dirs: tuple[str, str] | None
    cpu_threads: int = 1

    def settings(self):
        det, rec = PROFILES[self.name]
        result = dict(text_detection_model_name=det, text_recognition_model_name=rec,
                      use_doc_orientation_classify=False, use_doc_unwarping=False,
                      use_textline_orientation=False, enable_mkldnn=False,
                      cpu_threads=self.cpu_threads)
        if self.device is not None:
            result["device"] = self.device
        if self.model_dirs is not None:
            result.update(text_detection_model_dir=self.model_dirs[0],
                          text_recognition_model_dir=self.model_dirs[1])
        return result

    def identity(self):
        return {"profile": self.name, "requested_device": self.device,
                "device_policy": "upstream_auto" if self.device is None else "explicit_no_fallback",
                "settings": self.settings(), "versions": package_versions(),
                "local_weights": self.model_dirs is not None,
                "japanese_support": ("restricted_experimental_not_preferred_for_manga"
                                     if self.name == "v6_tiny" else "requires_local_quality_validation")}


def resolve_profile(model_profile=DEFAULT_PROFILE, device=None, model_root=None, model_dirs=None,
                    *, cpu_threads=1):
    if type(cpu_threads) is not int or cpu_threads <= 0:
        raise ValueError("cpu_threads must be a positive integer (not a boolean)")
    if not isinstance(model_profile, str) or model_profile not in PROFILES:
        raise ValueError("model_profile must be one of: " + ", ".join(PROFILES))
    if device is not None and (not isinstance(device, str) or
            re.fullmatch(r"cpu|gpu:(0|[1-9][0-9]*)", device) is None):
        raise ValueError("device must be None, 'cpu', or a single 'gpu:N'")
    if model_root is not None and model_dirs is not None:
        raise ValueError("choose model_root or model_dirs, not both")
    names = PROFILES[model_profile]
    if model_root is not None:
        if not isinstance(model_root, (str, Path)) or not str(model_root).strip():
            raise ValueError("model_root must be a nonempty path")
        model_dirs = tuple(Path(model_root).expanduser() / name for name in names)
    if model_dirs is not None:
        if not isinstance(model_dirs, tuple) or len(model_dirs) != 2:
            raise ValueError("model_dirs must be a (detector_dir, recognizer_dir) tuple")
        if any(not isinstance(path, (str, Path)) or not str(path).strip() for path in model_dirs):
            raise ValueError("model_dirs must contain two nonempty paths")
        model_dirs = tuple(validate_export(path, name) for path, name in zip(model_dirs, names))
    return PaddleProfile(model_profile, device, model_dirs, cpu_threads)


class PaddleInitializationError(RuntimeError):
    """Failed initialization keeps requested identity; no substitute engine is tried."""
    def __init__(self, identity, detail=None):
        message = "Paddle initialization failed for " + identity["profile"]
        super().__init__(message + (": " + detail if detail else ""))
        self.provenance = dict(identity, initialization_status="failed")
