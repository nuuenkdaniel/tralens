import importlib
import warnings
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image

from .grouping import GroupingMixin
from .automatic_units import propose_page_from_boxes
from .automatic_recognition import recognize_automatic_batch
from .review import ReviewMixin
from .paddle_result import associate_paddle_result, PaddleResultContractError
from .paddle_profiles import DEFAULT_PROFILE, resolve_profile, validate_runtime, PaddleInitializationError
from .types import MediaType, OCREngine, OCRReviewPolicy, OCRStatus, Text_Box, Text_Group


def create_paddle_ocr(**settings):
    # PaddleX performs a host check on import: defer it to explicit construction.
    validate_runtime(settings)
    engine = importlib.import_module("paddleocr").PaddleOCR(**settings)
    if any(settings.get(key) is not None for key in (
            "device", "text_detection_model_name", "text_recognition_model_name",
            "text_detection_model_dir", "text_recognition_model_dir")):
        try:
            pipeline = engine.paddlex_pipeline
            predictors = (pipeline.text_det_model, pipeline.text_rec_model)
            devices = [predictor.device for predictor in predictors]
            if settings.get("device") is not None and devices != [settings["device"]] * 2:
                raise RuntimeError("Paddle runtime device differs from explicit request: " + repr(devices))
            for predictor, kind in zip(predictors, ("detection", "recognition")):
                name = settings.get("text_" + kind + "_model_name")
                directory = settings.get("text_" + kind + "_model_dir")
                if name is not None and predictor.model_name != name:
                    raise RuntimeError("Paddle runtime model differs from explicit request: " + str(name))
                if directory is not None and Path(predictor.model_dir).resolve() != Path(directory).resolve():
                    raise RuntimeError("Paddle runtime model directory differs from explicit request: " + str(directory))
        except Exception:
            engine.close()
            raise
    return engine


class OCR(GroupingMixin, ReviewMixin):
    def __init__(self, image_path: str, *, media_type=MediaType.MANGA,
                 manga_recognizer=None, enable_manga_ocr=False,
                 manga_model_path=None, review_policy=None,
                 model_profile=DEFAULT_PROFILE, device=None, model_root=None, model_dirs=None,
                 allow_restricted_paddle=False, cpu_threads=1):
        # Reject invalid model selections/paths before initializing either engine.
        if type(allow_restricted_paddle) is not bool:
            raise ValueError("allow_restricted_paddle must be a boolean")
        media = MediaType(media_type)
        profile = resolve_profile(model_profile, device, model_root, model_dirs,
                                  cpu_threads=cpu_threads)
        restricted = profile.name == "v6_tiny"
        if restricted and media == MediaType.MANGA and not allow_restricted_paddle:
            raise ValueError("v6_tiny lacks Japanese kana; manga use requires explicit experimental "
                             "allow_restricted_paddle=True")
        self.paddle_provenance = profile.identity()
        self.paddle_provenance.update(
            media_type=media.value, allow_restricted_paddle=allow_restricted_paddle,
            restricted_paddle_override=restricted and media == MediaType.MANGA and allow_restricted_paddle)
        if restricted:
            warning = ("Experimental v6_tiny: the inspected official recognizer dictionary has no "
                       "hiragana or katakana; general Japanese recognition is unsupported. "
                       "Not preferred for Japanese manga.")
            self.paddle_provenance.update(
                japanese_support="restricted_experimental_no_hiragana_or_katakana_not_general_japanese_support",
                restricted_paddle_warning=warning)
            warnings.warn(warning, RuntimeWarning, stacklevel=2)
        self.image_path = image_path
        self.media_type = media
        self.review_policy = review_policy or OCRReviewPolicy()
        if enable_manga_ocr and manga_recognizer is not None:
            raise ValueError("Choose an injected recognizer or the runtime adapter")
        if enable_manga_ocr:
            # Optional dependency: importing it at baseline module scope would
            # prevent Paddle-only use. Enablement errors deliberately propagate.
            module_name = f"{__package__}.manga_adapter"
            adapter = importlib.import_module(module_name)
            manga_recognizer = adapter.MangaOCRAdapter(manga_model_path)
        self.manga_recognizer = manga_recognizer
        try:
            self.ocr = create_paddle_ocr(**profile.settings())
        except Exception as error:
            raise PaddleInitializationError(self.paddle_provenance, detail=str(error)) from error
        self.paddle_provenance["initialization_status"] = "initialized"

    def predict(self) -> list[list[Text_Box]]:
        self.prediction_contract_reports = []
        results = self.ocr.predict(self.image_path)
        images_text_box = []
        for page_index,page in enumerate(results):
            res = getattr(page, "res", page)
            try:
                text_boxes,report = associate_paddle_result(res,page_index)
            except PaddleResultContractError as error:
                if hasattr(self, "paddle_provenance"):
                    error.report["model_identity"] = deepcopy(self.paddle_provenance)
                self.prediction_contract_reports.append(error.report)
                raise
            if hasattr(self, "paddle_provenance"):
                report["model_identity"] = deepcopy(self.paddle_provenance)
                for box in text_boxes:
                    box.ocr_metadata["paddle_model"] = deepcopy(self.paddle_provenance)
            self.prediction_contract_reports.append(report)
            images_text_box.append(text_boxes)
        return images_text_box

    def process_images(self, *, media_type=None, propose_units=False,
                       complementary_ownership=False, regroup_companions=False):
        if type(propose_units) is not bool:
            raise ValueError("propose_units must be a boolean")
        if (type(complementary_ownership) is not bool or type(regroup_companions) is not bool
                or ((complementary_ownership or regroup_companions) and not propose_units)
                or (regroup_companions and not complementary_ownership)):
            raise ValueError("ownership/regrouping flags require explicit propose_units and complementary ownership")
        media = MediaType(media_type if media_type is not None else
                          getattr(self, "media_type", MediaType.MANGA))
        identity = getattr(self, "paddle_provenance", {})
        if (media == MediaType.MANGA and identity.get("profile") == "v6_tiny"
                and not identity.get("allow_restricted_paddle", False)):
            raise ValueError("v6_tiny manga processing requires allow_restricted_paddle=True at construction")
        if propose_units and media != MediaType.MANGA:
            raise ValueError("automatic unit proposals require explicit manga mode")
        # A validated call starts a new pass, even when it fails or disables
        # proposals. Previously returned records remain untouched, but cannot
        # silently become this pass's candidates or consume its attempt budget.
        for attribute in ("automatic_unit_proposals","automatic_recognition_results","raw_detections","prediction_contract_reports"):
            self.__dict__.pop(attribute,None)
        images_text_boxes = self.predict()
        self.raw_detections = images_text_boxes
        images_text_groups = []
        proposals = []
        for text_boxes in images_text_boxes:
            image = cv2.imread(self.image_path) if text_boxes else None
            self.review_boxes(text_boxes, image, media_type=media)
            grouping = (self._group_boxes_manga if media == MediaType.MANGA
                        else self._group_boxes_proximity)
            images_text_groups.append(grouping(text_boxes))
            if propose_units:
                proposals.append(
                    propose_page_from_boxes(text_boxes, self.image_path,
                        complementary_ownership=complementary_ownership,
                        regroup_companions=regroup_companions) if text_boxes else None)
        if propose_units:
            self.automatic_unit_proposals = proposals
        return images_text_groups

    def recognize_automatic_proposals(self, recognizer, *, model_identity, model_settings=None, max_attempts=20):
        """Explicit candidate coordination only; never changes process_images' output."""
        if not hasattr(self,"automatic_unit_proposals"):
            raise ValueError("first request process_images(propose_units=True)")
        if type(max_attempts) is not int or not 0 <= max_attempts <= 20:
            raise ValueError("max_attempts must be 0..20 across all pages")
        previous = getattr(self,"automatic_recognition_results",None)
        if previous is not None and len(previous) != len(self.automatic_unit_proposals):
            raise ValueError("existing automatic results do not bind this page set")
        spent = sum(r.attempts for page in previous for r in page) if previous else 0
        collected = []
        for index,batch in enumerate(self.automatic_unit_proposals):
            if batch is None:
                collected.append(())
                continue
            content = Path(batch.page.image_path).read_bytes()
            with Image.open(BytesIO(content)) as source:
                image = source.convert("RGB")
            existing = previous[index] if previous else None
            prior_spent = sum(r.attempts for r in existing) if existing else 0
            allowance = min(20,prior_spent+max(0,max_attempts-spent))
            results = recognize_automatic_batch(batch,image,recognizer,original_image_bytes=content,
                model_identity=model_identity,model_settings=model_settings,max_attempts=allowance,results=existing)
            spent += sum(r.attempts for r in results)-prior_spent
            collected.append(results)
        self.automatic_recognition_results = tuple(collected)
        return self.automatic_recognition_results
