"""Runtime-only availability checks for optional private saved-page integrations."""
from functools import wraps
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def require_local_evidence(*, files=(), comparisons=(), manifests=(), root=None):
    """Skip unavailable assets, not malformed JSON, failed assertions or hash drift.

    Only known image/comparison path fields are inspected. Never follow a saved
    absolute path into another checkout just because it happens to exist there.
    """
    root = Path(root if root is not None else ROOT).resolve()
    missing,outside = set(),set()

    def needed(value):
        if not isinstance(value,(str,Path)):
            raise ValueError("local evidence path must be a string or Path")
        path = Path(value)
        path = (path if path.is_absolute() else root/path).resolve()
        if not path.is_relative_to(root):
            outside.add(str(path))
            return None
        if not path.is_file():
            missing.add(str(path))
            return None
        return path

    def unavailable():
        reasons = []
        if missing:
            reasons.append("missing files: "+", ".join(sorted(missing)))
        if outside:
            reasons.append("references outside this checkout (regenerate local provenance): "+", ".join(sorted(outside)))
        if reasons:
            raise unittest.SkipTest("optional local evidence unavailable; "+"; ".join(reasons))

    for value in files:
        needed(value)
    comparison_paths = {needed(value) for value in comparisons}
    manifest_paths = [needed(value) for value in manifests]
    unavailable()
    for path in manifest_paths:
        document = json.loads(path.read_text())
        if not isinstance(document,dict) or not isinstance(document.get("pages"),list):
            raise ValueError("saved-page manifest must contain a pages list: "+str(path))
        for page in document["pages"]:
            comparison_paths.add(needed(page["comparison_path"]))
            if "image_path" in page.get("page",{}):
                needed(page["page"]["image_path"])
    unavailable()
    for path in sorted(comparison_paths):
        document = json.loads(path.read_text())
        image = document.get("input") or document.get("summary",{}).get("image")
        if not isinstance(image,str):
            raise ValueError("saved comparison must identify its original image: "+str(path))
        needed(image)
    unavailable()


def local_evidence(*, files=(), comparisons=(), manifests=()):
    """Decorate individual integration methods; no file I/O during collection."""
    def decorate(method):
        @wraps(method)
        def guarded(*args,**kwargs):
            require_local_evidence(files=files,comparisons=comparisons,manifests=manifests)
            return method(*args,**kwargs)
        return guarded
    return decorate
