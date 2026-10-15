"""Engine-free Paddle result association; detector indices are the source IDs."""
import copy
import math
from collections.abc import Mapping
from numbers import Real

import numpy as np

from .recognition_units import _valid_polygon
from .types import OCRStatus, Text_Box


class PaddleResultContractError(ValueError):
    """An unrepresentable detector page retains its selective raw report."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def evidence_value(value):
    """Detached JSON-safe evidence, including explicit malformed/nonfinite values."""
    if isinstance(value,np.ndarray):
        return evidence_value(value.tolist())
    if isinstance(value,np.generic):
        plain = value.item()
        if isinstance(plain,np.generic):
            return {"type":str(value.dtype),"value":str(value)}
        return evidence_value(plain)
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float):
        return value if math.isfinite(value) else {"type":"float","nonfinite":repr(value)}
    if isinstance(value,(list,tuple)):
        return [evidence_value(item) for item in value]
    if isinstance(value,Mapping):
        return {str(key):evidence_value(item) for key,item in value.items()}
    return {"unsupported_type":type(value).__module__+"."+type(value).__name__}


def polygon_key(poly):
    """Exact cyclic/winding-independent identity, never proximity/IoU matching."""
    try:
        array = np.asarray(poly)
        if array.dtype.kind not in "iuf" or array.ndim != 2 or array.shape[1] != 2:
            return None
        points = array.tolist()
        if len(points) > 1 and points[0] == points[-1]:
            points = points[:-1]
        if not _valid_polygon(points):
            return None
        points = tuple(tuple(point) for point in points)
        reverse = tuple(reversed(points))
        return min(sequence[i:]+sequence[:i] for sequence in (points,reverse) for i in range(len(points)))
    except (TypeError,ValueError,OverflowError):
        return None


def associate_paddle_result(res, page_index=0):
    """Return every representable detector polygon, with conservative OCR binding.

    Malformed detector arrays that Text_Box cannot represent fail the whole page,
    preserving all selective raw fields on the exception. Recognition defects
    leave detector boxes blank/reviewable, never truncate or shift detector IDs.
    """
    report = {"policy":"explicit_paddle_rec_polys_v1","page_index":page_index,
              "issues":[],"records":[],"raw":{}}
    if not isinstance(res,Mapping):
        report["issues"].append("invalid_page_result")
        report["status"] = "invalid_detector_page"
        raise PaddleResultContractError("Paddle page result must be a mapping",report)
    names = ("dt_polys","dt_scores","rec_polys","rec_texts","rec_scores")
    report["raw"] = {name:evidence_value(res[name]) for name in names if name in res}

    def sequence(name):
        value = res.get(name,[])
        if isinstance(value,(list,tuple)) or isinstance(value,np.ndarray) and value.ndim >= 1:
            return list(value)
        report["issues"].append("invalid_"+name+"_container")
        return None

    detectors = sequence("dt_polys")
    if detectors is None:
        report["status"] = "invalid_detector_page"
        raise PaddleResultContractError("Invalid detector polygon container",report)
    report["detector_count"] = len(detectors)
    keys,by_key = [],{}
    for index,poly in enumerate(detectors):
        try:
            array = np.asarray(poly,dtype=np.float64)
            representable = (array.ndim == 2 and array.shape[1] == 2 and len(array) > 0
                and np.isfinite(array).all() and np.max(np.abs(array)) <= np.finfo(np.float32).max)
        except (TypeError,ValueError,OverflowError):
            representable = False
        if not representable:
            report["issues"].append("unrepresentable_detector_polygon:"+str(index))
            report["status"] = "invalid_detector_page"
            raise PaddleResultContractError("Unrepresentable detector polygon at source index "+str(index),report)
        key = polygon_key(poly)
        keys.append(key)
        if key is not None:
            by_key.setdefault(key,[]).append(index)
    if any(key is None for key in keys):
        report["issues"].append("invalid_detector_geometry")
    if any(len(indices) > 1 for indices in by_key.values()):
        report["issues"].append("duplicate_detector_geometry")

    texts,scores = sequence("rec_texts"),sequence("rec_scores")
    explicit = "rec_polys" in res
    polys = sequence("rec_polys") if explicit else detectors
    report["mode"] = "explicit_rec_polys" if explicit else "legacy_full_length_alignment"
    report["field_lengths"] = {name:None if value is None else len(value) for name,value in
                              (("dt_polys",detectors),("rec_polys",polys),("rec_texts",texts),("rec_scores",scores))}
    aligned = texts is not None and scores is not None and polys is not None and len(texts) == len(scores) == len(polys)
    if not explicit:
        aligned = aligned and len(texts) == len(detectors) and all(key is not None for key in keys) and len(by_key) == len(detectors)
        if not aligned:
            report["mode"] = "legacy_unproven_alignment"
    if not aligned:
        report["issues"].append("recognition_field_length_or_legacy_alignment_mismatch")
    assignments,claims = {},{}
    if aligned:
        for index,(poly,text,score) in enumerate(zip(polys,texts,scores)):
            key = polygon_key(poly)
            candidates = by_key.get(key,[]) if key is not None else []
            row = {"recognition_index":index,"detector_indices":tuple(candidates),"issues":[],
                   "score_type":type(score).__name__,"text_type":type(text).__name__}
            report["records"].append(row)
            if key is None:
                row["issues"].append("invalid_recognition_polygon")
            elif not candidates:
                row["issues"].append("foreign_recognition_geometry")
            elif len(candidates) != 1:
                row["issues"].append("ambiguous_detector_geometry")
            if not isinstance(text,str):
                row["issues"].append("invalid_recognition_text")
            valid_score = isinstance(score,(Real,np.integer,np.floating)) and not isinstance(score,(bool,np.bool_))
            if isinstance(score,np.floating) and score.dtype.itemsize > 8:
                valid_score = False
            try:
                numeric_score = float(score) if valid_score else None
            except (TypeError,ValueError,OverflowError):
                numeric_score = None
            if numeric_score is None or not math.isfinite(numeric_score) or not 0 <= numeric_score <= 1:
                row["issues"].append("invalid_recognition_score")
            for detector_index in candidates:
                claims.setdefault(detector_index,[]).append(row)
            if not row["issues"]:
                assignments[candidates[0]] = (index,text,score,poly)
        for detector_index,rows in claims.items():
            if len(rows) > 1:
                assignments.pop(detector_index,None)
                for row in rows:
                    row["issues"].append("duplicate_recognition_geometry")
    report["unresolved_recognition_indices"] = tuple(row["recognition_index"] for row in report["records"] if row["issues"])
    report["matched_detector_indices"] = tuple(sorted(assignments))
    report["unrecognized_detector_indices"] = tuple(i for i in range(len(detectors)) if i not in assignments)
    report["status"] = "unresolved" if report["issues"] or report["unresolved_recognition_indices"] or report["unrecognized_detector_indices"] else "aligned"
    boxes = []
    for index,poly in enumerate(detectors):
        matched = assignments.get(index)
        box = Text_Box(index,matched[2] if matched else 0.,poly,matched[1] if matched else "",review=0 if matched else 1)
        association = {"status":"matched" if matched else "unresolved","detector_index":index,
            "mode":report["mode"],"recognition_indices":tuple(row["recognition_index"] for row in claims.get(index,[])),
            "issues":tuple(dict.fromkeys((report["issues"] if not aligned else [])+
                [issue for row in claims.get(index,[]) for issue in row["issues"]]))}
        if keys[index] is None:
            association["issues"] += ("invalid_detector_geometry",)
        elif len(by_key[keys[index]]) > 1:
            association["issues"] += ("ambiguous_detector_geometry",)
        if matched:
            association["recognition_index"] = matched[0]
        else:
            if not association["issues"]:
                association["issues"] = ("missing_recognition",)
            box.set_status(OCRStatus.REVIEW,"unresolved_paddle_association")
        box.ocr_metadata["paddle_association"] = association
        boxes.append(box)
    # One complete page report, rather than duplicating all OCR rows on every box.
    # This survives RawDetection snapshotting and retains orphan/malformed rows.
    if boxes:
        boxes[0].ocr_metadata["paddle_page_contract"] = copy.deepcopy(report)
    return boxes,report
