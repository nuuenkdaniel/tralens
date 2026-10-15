import importlib
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image

from .grouping import GroupingMixin
from .automatic_units import propose_page_from_boxes
from .automatic_recognition import recognize_automatic_batch
from .review import ReviewMixin
from .paddle_result import associate_paddle_result, PaddleResultContractError
from .types import MediaType, OCREngine, OCRReviewPolicy, OCRStatus, Text_Box, Text_Group


def create_paddle_ocr(**settings):
    # PaddleX performs a host check on import: defer it to explicit construction.
    return importlib.import_module("paddleocr").PaddleOCR(**settings)


class OCR(GroupingMixin, ReviewMixin):
    def __init__(self, image_path: str, *, media_type=MediaType.MANGA,
                 manga_recognizer=None, enable_manga_ocr=False,
                 manga_model_path=None, review_policy=None):
        self.image_path = image_path
        self.media_type = MediaType(media_type)
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
        self.ocr = create_paddle_ocr(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict(self) -> list[list[Text_Box]]:
        self.prediction_contract_reports = []
        results = self.ocr.predict(self.image_path)
        images_text_box = []
        for page_index,page in enumerate(results):
            res = getattr(page, "res", page)
            try:
                text_boxes,report = associate_paddle_result(res,page_index)
            except PaddleResultContractError as error:
                self.prediction_contract_reports.append(error.report)
                raise
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
