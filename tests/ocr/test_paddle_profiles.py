"""Offline constructor/selection contracts; fake exports and engines only."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tralens.ocr.detection import OCR, create_paddle_ocr
from tralens.ocr.types import MediaType
from tralens.ocr.paddle_profiles import (DEFAULT_PROFILE, PROFILES, MODEL_FILES, V6_RUNTIME,
                                        resolve_profile, validate_runtime, PaddleInitializationError)


def exports(root, profile="v6_medium"):
    directories = []
    for name in PROFILES[profile]:
        directory = Path(root) / name
        directory.mkdir()
        for file in MODEL_FILES:
            (directory / file).write_text("Global:\n  model_name: " + name + "\n" if file.endswith("yml") else "synthetic")
        directories.append(str(directory))
    return tuple(directories)


class PaddleProfileTests(unittest.TestCase):
    def test_default_pins_medium_pair_even_when_upstream_default_changes(self):
        with patch("tralens.ocr.detection.create_paddle_ocr") as factory:
            ocr = OCR("unused.png")
        self.assertEqual(factory.call_args.kwargs, dict(
            text_detection_model_name="PP-OCRv6_medium_det", text_recognition_model_name="PP-OCRv6_medium_rec",
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
            enable_mkldnn=False, cpu_threads=1))
        self.assertEqual(ocr.paddle_provenance["profile"], "v6_medium")
        self.assertEqual(resolve_profile().name, DEFAULT_PROFILE)
        self.assertEqual(resolve_profile().settings(), factory.call_args.kwargs)
        self.assertEqual(ocr.paddle_provenance["device_policy"], "upstream_auto")

    def test_every_profile_and_explicit_device_pass_exact_names_and_flags(self):
        for profile, names in PROFILES.items():
            for device in ("cpu", "gpu:0", "gpu:12"):
                with self.subTest(profile=profile, device=device), patch("tralens.ocr.detection.create_paddle_ocr") as factory:
                    if profile == "v6_tiny":
                        with self.assertWarnsRegex(RuntimeWarning, "no hiragana or katakana"):
                            ocr = OCR("unused", model_profile=profile, device=device, allow_restricted_paddle=True)
                    else:
                        ocr = OCR("unused", model_profile=profile, device=device)
                    settings = factory.call_args.kwargs
                    self.assertEqual((settings["text_detection_model_name"], settings["text_recognition_model_name"]), names)
                    self.assertEqual(settings["device"], device)
                    self.assertFalse(settings["use_textline_orientation"])
                    self.assertIs(settings["enable_mkldnn"], False)
                    self.assertEqual(settings["cpu_threads"], 1)
                    self.assertEqual(ocr.paddle_provenance["settings"], settings)
                    self.assertEqual(ocr.paddle_provenance["initialization_status"], "initialized")
                    if profile == "v6_tiny":
                        self.assertIn("restricted_experimental", ocr.paddle_provenance["japanese_support"])

    def test_invalid_selection_and_device_rejected_before_any_engine_or_adapter(self):
        settings = [{"model_profile": name} for name in (None, "v6", "", [], "v5_mobile")]
        settings += [{"device": device} for device in (True, 0, "gpu", "cuda:0", "gpu:-1", "gpu:00", "gpu:0,1", "cpu:0", "auto")]
        with patch("tralens.ocr.detection.create_paddle_ocr") as factory, patch("tralens.ocr.detection.importlib.import_module") as imports:
            for kwargs in settings:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    OCR("unused", enable_manga_ocr=True, **kwargs)
            factory.assert_not_called()
            imports.assert_not_called()

    def test_root_and_explicit_tuple_resolve_same_validated_exports(self):
        for profile in PROFILES:
            with tempfile.TemporaryDirectory() as directory:
                dirs = exports(directory, profile)
                root = resolve_profile(profile, "cpu", model_root=directory)
                explicit = resolve_profile(profile, "cpu", model_dirs=dirs)
                self.assertEqual(root, explicit)
                with patch("tralens.ocr.detection.create_paddle_ocr") as factory:
                    if profile == "v6_tiny":
                        with self.assertWarns(RuntimeWarning):
                            OCR("unused", model_profile=profile, model_dirs=dirs, allow_restricted_paddle=True)
                    else:
                        OCR("unused", model_profile=profile, model_dirs=dirs)
                self.assertEqual(factory.call_args.kwargs["text_detection_model_dir"], dirs[0])
                self.assertEqual(factory.call_args.kwargs["text_recognition_model_dir"], dirs[1])

    def test_expands_user_and_resolves_relative_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            dirs = exports(directory)
            relative = tuple(os.path.relpath(path) for path in dirs)
            self.assertEqual(resolve_profile(model_dirs=relative).model_dirs, dirs)
            with patch.dict(os.environ, {"HOME": directory}):
                self.assertEqual(resolve_profile(model_root="~").model_dirs, dirs)

    def test_invalid_local_paths_and_identity_fail_before_initialization(self):
        with tempfile.TemporaryDirectory() as directory, patch("tralens.ocr.detection.create_paddle_ocr") as factory:
            dirs = exports(directory)
            for kwargs in ({"model_root": directory, "model_dirs": dirs}, {"model_dirs": list(dirs)},
                           {"model_dirs": (dirs[0],)}, {"model_dirs": ("", dirs[1])},
                           {"model_root": str(Path(directory) / "absent")}):
                with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, FileNotFoundError)):
                    OCR("unused", **kwargs)
            with self.assertRaisesRegex(ValueError, "identity"):
                OCR("unused", model_dirs=tuple(reversed(dirs)))
            weights = Path(dirs[1]) / "inference.pdiparams"
            weights.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                OCR("unused", model_dirs=dirs)
            weights.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                OCR("unused", model_dirs=dirs)
            factory.assert_not_called()

    def test_failed_init_retains_identity_and_cause_without_retry(self):
        failure = RuntimeError("synthetic GPU failure")
        with patch("tralens.ocr.detection.create_paddle_ocr", side_effect=failure) as factory:
            with self.assertRaises(PaddleInitializationError) as caught:
                OCR("unused", model_profile="v6_medium", device="gpu:2")
        factory.assert_called_once()
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(caught.exception.provenance["initialization_status"], "failed")
        self.assertEqual(caught.exception.provenance["requested_device"], "gpu:2")

    def test_factory_rejects_silent_gpu_to_cpu_substitution_and_closes(self):
        engine = Mock(paddlex_pipeline=SimpleNamespace(text_det_model=SimpleNamespace(device="cpu"),
                                                      text_rec_model=SimpleNamespace(device="cpu")))
        with patch("tralens.ocr.detection.importlib.import_module", return_value=SimpleNamespace(PaddleOCR=Mock(return_value=engine))):
            with self.assertRaisesRegex(RuntimeError, "device differs"):
                create_paddle_ocr(device="gpu:0")
        engine.close.assert_called_once()

    def test_factory_accepts_exact_requested_runtime(self):
        engine = Mock(paddlex_pipeline=SimpleNamespace(text_det_model=SimpleNamespace(device="gpu:1"),
                                                      text_rec_model=SimpleNamespace(device="gpu:1")))
        with patch("tralens.ocr.detection.importlib.import_module", return_value=SimpleNamespace(PaddleOCR=Mock(return_value=engine))):
            self.assertIs(create_paddle_ocr(device="gpu:1"), engine)
        engine.close.assert_not_called()

    def test_provenance_on_raw_boxes_does_not_shift_filtered_geometry_or_text(self):
        polys = [[[0, 0], [5, 0], [5, 10], [0, 10]], [[10, 0], [15, 0], [15, 10], [10, 10]]]
        engine = Mock()
        engine.predict.return_value = [{"dt_polys": polys, "rec_polys": polys[1:], "rec_texts": ["fixture"], "rec_scores": [.9]}]
        with patch("tralens.ocr.detection.create_paddle_ocr", return_value=engine):
            ocr = OCR("unused", model_profile="v6_small", device="cpu")
        boxes = ocr.predict()[0]
        self.assertEqual([box.box_id for box in boxes], [0, 1])
        self.assertEqual([box.paddle_text for box in boxes], ["", "fixture"])
        self.assertEqual([box.paddle_poly.tolist() for box in boxes], polys)
        self.assertEqual(ocr.prediction_contract_reports[0]["model_identity"], ocr.paddle_provenance)
        for box in boxes:
            self.assertEqual(box.ocr_metadata["paddle_model"], ocr.paddle_provenance)
        boxes[0].ocr_metadata["paddle_model"]["profile"] = "mutated"
        self.assertEqual(boxes[1].ocr_metadata["paddle_model"]["profile"], "v6_small")
        self.assertEqual(ocr.paddle_provenance["profile"], "v6_small")

    def test_tiny_manga_rejected_by_default_before_either_engine(self):
        with patch("tralens.ocr.detection.create_paddle_ocr") as factory, patch("tralens.ocr.detection.importlib.import_module") as imports:
            for kwargs in ({}, {"media_type": "manga"}, {"allow_restricted_paddle": False}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "allow_restricted_paddle=True"):
                    OCR("unused", model_profile="v6_tiny", enable_manga_ocr=True, **kwargs)
            factory.assert_not_called()
            imports.assert_not_called()

    def test_explicit_tiny_manga_override_warns_and_survives_in_page_and_box_provenance(self):
        poly = [[0, 0], [5, 0], [5, 10], [0, 10]]
        engine = Mock()
        engine.predict.return_value = [{"dt_polys": [poly], "rec_polys": [poly],
                                        "rec_texts": ["fixture"], "rec_scores": [.9]}]
        with patch("tralens.ocr.detection.create_paddle_ocr", return_value=engine) as factory:
            with self.assertWarnsRegex(RuntimeWarning, "general Japanese recognition is unsupported"):
                ocr = OCR("unused", model_profile="v6_tiny", allow_restricted_paddle=True)
        factory.assert_called_once()
        self.assertNotIn("allow_restricted_paddle", factory.call_args.kwargs)
        boxes = ocr.predict()[0]
        for identity in (ocr.paddle_provenance, ocr.prediction_contract_reports[0]["model_identity"],
                         boxes[0].ocr_metadata["paddle_model"]):
            self.assertTrue(identity["allow_restricted_paddle"])
            self.assertTrue(identity["restricted_paddle_override"])
            self.assertEqual(identity["media_type"], "manga")
            self.assertIn("no_hiragana_or_katakana", identity["japanese_support"])
            self.assertIn("no hiragana or katakana", identity["restricted_paddle_warning"])

    def test_nonmanga_explicit_tiny_allowed_without_manga_override_but_warns(self):
        for media in (MediaType.OTHER, "other"):
            with self.subTest(media=media), patch("tralens.ocr.detection.create_paddle_ocr") as factory:
                with self.assertWarnsRegex(RuntimeWarning, "no hiragana or katakana"):
                    ocr = OCR("unused", media_type=media, model_profile="v6_tiny")
                factory.assert_called_once()
                self.assertEqual(ocr.media_type, MediaType.OTHER)
                self.assertFalse(ocr.paddle_provenance["allow_restricted_paddle"])
                self.assertFalse(ocr.paddle_provenance["restricted_paddle_override"])
                self.assertIn("not_general_japanese_support", ocr.paddle_provenance["japanese_support"])

    def test_invalid_override_or_media_rejected_before_either_engine(self):
        settings = [{"allow_restricted_paddle": value} for value in (None, 0, 1, "true", [], {})]
        settings += [{"media_type": value, "allow_restricted_paddle": True} for value in (None, "invalid", 1)]
        settings += [{"model_profile": "invalid", "allow_restricted_paddle": True}]
        with patch("tralens.ocr.detection.create_paddle_ocr") as factory, patch("tralens.ocr.detection.importlib.import_module") as imports:
            for kwargs in settings:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    OCR("unused", enable_manga_ocr=True, **kwargs)
            factory.assert_not_called()
            imports.assert_not_called()

    def test_tiny_override_retained_on_initialization_failure_without_retry(self):
        with patch("tralens.ocr.detection.create_paddle_ocr", side_effect=RuntimeError("synthetic")) as factory:
            with self.assertWarns(RuntimeWarning), self.assertRaises(PaddleInitializationError) as caught:
                OCR("unused", model_profile="v6_tiny", allow_restricted_paddle=True)
        factory.assert_called_once()
        self.assertTrue(caught.exception.provenance["restricted_paddle_override"])
        self.assertEqual(caught.exception.provenance["initialization_status"], "failed")

    def test_cpu_threads_and_disabled_onednn_reach_native_constructor_and_identity(self):
        for profile in ("v5_server", "v6_small", "v6_medium"):
            for threads in (1, 3):
                engine = Mock(paddlex_pipeline=SimpleNamespace(
                    text_det_model=SimpleNamespace(device="cpu", model_name=PROFILES[profile][0]),
                    text_rec_model=SimpleNamespace(device="cpu", model_name=PROFILES[profile][1])))
                constructor = Mock(return_value=engine)
                with self.subTest(profile=profile, threads=threads), patch(
                        "tralens.ocr.detection.importlib.import_module", return_value=SimpleNamespace(PaddleOCR=constructor)), patch(
                        "tralens.ocr.paddle_profiles.package_versions", return_value=dict(V6_RUNTIME)):
                    ocr = OCR("unused", model_profile=profile, device="cpu", cpu_threads=threads)
                constructor.assert_called_once()
                settings = constructor.call_args.kwargs
                self.assertIs(settings["enable_mkldnn"], False)
                self.assertEqual(settings["cpu_threads"], threads)
                self.assertEqual(ocr.paddle_provenance["settings"], settings)
                self.assertEqual(resolve_profile(profile, "cpu", cpu_threads=threads).identity()["settings"], settings)

    def test_invalid_cpu_threads_rejected_by_selector_and_facade_before_either_engine(self):
        with patch("tralens.ocr.detection.create_paddle_ocr") as factory, patch("tralens.ocr.detection.importlib.import_module") as imports:
            for value in (True, False, 0, -1, 1.0, "1", None, [], {}):
                with self.subTest(threads=value):
                    with self.assertRaisesRegex(ValueError, "cpu_threads"):
                        resolve_profile(cpu_threads=value)
                    with self.assertRaisesRegex(ValueError, "cpu_threads"):
                        OCR("unused", device="cpu", cpu_threads=value, enable_manga_ocr=True)
            factory.assert_not_called()
            imports.assert_not_called()

    def test_cpu_settings_survive_prediction_and_initialization_failure_without_retry(self):
        poly = [[0, 0], [5, 0], [5, 10], [0, 10]]
        engine = Mock()
        engine.predict.return_value = [{"dt_polys": [poly], "rec_polys": [poly],
                                        "rec_texts": ["fixture"], "rec_scores": [.9]}]
        with patch("tralens.ocr.detection.create_paddle_ocr", return_value=engine):
            ocr = OCR("unused", model_profile="v6_medium", device="cpu", cpu_threads=2)
        boxes = ocr.predict()[0]
        for identity in (ocr.paddle_provenance, ocr.prediction_contract_reports[0]["model_identity"],
                         boxes[0].ocr_metadata["paddle_model"]):
            self.assertIs(identity["settings"]["enable_mkldnn"], False)
            self.assertEqual(identity["settings"]["cpu_threads"], 2)
        with patch("tralens.ocr.detection.create_paddle_ocr", side_effect=RuntimeError("synthetic")) as factory:
            with self.assertRaises(PaddleInitializationError) as caught:
                OCR("unused", model_profile="v6_small", device="cpu", cpu_threads=2)
        factory.assert_called_once()
        self.assertIs(caught.exception.provenance["settings"]["enable_mkldnn"], False)
        self.assertEqual(caught.exception.provenance["settings"]["cpu_threads"], 2)

    def test_default_on_unsupported_runtime_fails_clearly_before_sdk_import(self):
        for versions in ({"paddleocr": "3.4.0", "paddlex": "3.4.2"},
                         {"paddleocr": "3.7.0", "paddlex": "3.7.1"},
                         {"paddleocr": None, "paddlex": None}):
            with self.subTest(versions=versions), patch("tralens.ocr.paddle_profiles.package_versions", return_value=versions), patch(
                    "tralens.ocr.detection.importlib.import_module") as imports:
                with self.assertRaisesRegex(PaddleInitializationError, "validated runtime.*requirements-paddle6.txt") as caught:
                    OCR("unused")
                imports.assert_not_called()
                self.assertEqual(caught.exception.provenance["profile"], "v6_medium")
                self.assertEqual(caught.exception.provenance["initialization_status"], "failed")
                self.assertIn("v5_server", str(caught.exception))

    def test_all_v6_profiles_require_validated_runtime_but_selector_remains_engine_free(self):
        for profile in ("v6_tiny", "v6_small", "v6_medium"):
            settings = resolve_profile(profile).settings()
            with patch("tralens.ocr.paddle_profiles.package_versions", return_value=dict(V6_RUNTIME)):
                validate_runtime(settings)
            with patch("tralens.ocr.paddle_profiles.package_versions", return_value={"paddleocr": "3.4.0"}), patch(
                    "tralens.ocr.detection.importlib.import_module") as imports:
                with self.assertRaisesRegex(RuntimeError, "No model fallback"):
                    create_paddle_ocr(**settings)
                imports.assert_not_called()

    def test_explicit_v5_works_on_original_runtime_without_changing_requested_names(self):
        versions = {"paddleocr": "3.4.0", "paddlex": "3.4.2"}
        engine = Mock(paddlex_pipeline=SimpleNamespace(
            text_det_model=SimpleNamespace(device="cpu", model_name="PP-OCRv5_server_det"),
            text_rec_model=SimpleNamespace(device="cpu", model_name="PP-OCRv5_server_rec")))
        constructor = Mock(return_value=engine)
        with patch("tralens.ocr.paddle_profiles.package_versions", return_value=versions), patch(
                "tralens.ocr.detection.importlib.import_module", return_value=SimpleNamespace(PaddleOCR=constructor)):
            ocr = OCR("unused", model_profile="v5_server", device="cpu")
        constructor.assert_called_once()
        self.assertEqual(constructor.call_args.kwargs["text_detection_model_name"], "PP-OCRv5_server_det")
        self.assertEqual(constructor.call_args.kwargs["text_recognition_model_name"], "PP-OCRv5_server_rec")
        self.assertEqual(ocr.paddle_provenance["profile"], "v5_server")
        self.assertEqual(ocr.paddle_provenance["versions"], versions)

    def test_native_model_name_or_local_directory_substitution_rejected_even_with_auto_device(self):
        with tempfile.TemporaryDirectory() as directory:
            dirs = exports(directory, "v5_server")
            settings = resolve_profile("v5_server", model_dirs=dirs).settings()
            for defect in ("name", "directory"):
                detector = SimpleNamespace(device="cpu", model_name="PP-OCRv5_server_det", model_dir=dirs[0])
                recognizer = SimpleNamespace(device="cpu", model_name="PP-OCRv5_server_rec", model_dir=dirs[1])
                if defect == "name":
                    detector.model_name = "PP-OCRv5_mobile_det"
                else:
                    recognizer.model_dir = directory
                engine = Mock(paddlex_pipeline=SimpleNamespace(text_det_model=detector, text_rec_model=recognizer))
                with self.subTest(defect=defect), patch("tralens.ocr.detection.importlib.import_module", return_value=SimpleNamespace(
                        PaddleOCR=Mock(return_value=engine))):
                    with self.assertRaisesRegex(RuntimeError, "differs from explicit request"):
                        create_paddle_ocr(**settings)
                engine.close.assert_called_once()

    def test_invalid_root_and_malformed_yaml_fail_before_initialization(self):
        with patch("tralens.ocr.detection.create_paddle_ocr") as factory:
            for root in ("", "  ", False, 7, []):
                with self.subTest(root=root), self.assertRaisesRegex(ValueError, "model_root"):
                    OCR("unused", model_root=root)
            with tempfile.TemporaryDirectory() as directory:
                dirs = exports(directory)
                (Path(dirs[0]) / "inference.yml").write_text("Global: [")
                with self.assertRaisesRegex(ValueError, "invalid local model export configuration"):
                    OCR("unused", model_root=directory)
            factory.assert_not_called()

    def test_fresh_import_and_profile_resolution_do_not_import_engines(self):
        source = Path(__file__).resolve().parents[2] / "src"
        code = ("import sys; sys.path.insert(0, " + repr(str(source)) + "); "
                "from tralens.ocr import OCR; from tralens.ocr.paddle_profiles import resolve_profile; "
                "assert resolve_profile().name == 'v6_medium'; "
                "assert not {'paddle', 'paddleocr', 'paddlex'}.intersection(sys.modules)")
        run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_process_media_override_cannot_bypass_tiny_manga_restriction(self):
        for allow in (False, True):
            with patch("tralens.ocr.detection.create_paddle_ocr"), self.assertWarns(RuntimeWarning):
                ocr = OCR("unused", model_profile="v6_tiny", media_type=MediaType.OTHER,
                          allow_restricted_paddle=allow)
            ocr.predict = Mock(return_value=[])
            if allow:
                self.assertEqual(ocr.process_images(media_type=MediaType.MANGA), [])
                ocr.predict.assert_called_once()
            else:
                with self.assertRaisesRegex(ValueError, "allow_restricted_paddle=True"):
                    ocr.process_images(media_type=MediaType.MANGA)
                ocr.predict.assert_not_called()
