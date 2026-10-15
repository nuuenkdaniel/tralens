"""Synthetic local-only benchmark contracts. No Paddle imports/inference."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from tools import paddle_model_benchmark as bench
from tralens.ocr.paddle_profiles import PROFILES


def make_fixture(root):
    for name in PROFILES["v5_server"]:
        directory = root / name
        directory.mkdir()
        (directory / "inference.yml").write_text("Global:\n  model_name: " + name + "\n")
        for file in ("inference.json", "inference.pdiparams"):
            (directory / file).write_text("synthetic")
    images = []
    for index in range(7):
        path = root / (str(index) + ".png")
        Image.new("RGB", (16, 24), (index, 255, 255)).save(path)
        images.append(str(path))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"plan": {"images": bench.image_records(images=images)}}))
    args = bench.parser().parse_args(["--run-local-ocr", "--model-root", str(root), "--device", "cpu",
                                      "--manifest", str(manifest), "--output", str(root / "result.json")])
    return args


class BenchmarkTests(unittest.TestCase):
    def test_preflight_hashes_exact_manifest_paths_weights_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_fixture(Path(directory))
            plan = bench.build_plan(args)
            self.assertEqual(len(plan["images"]), 7)
            self.assertEqual(plan["warm_passes"], 2)
            self.assertEqual(plan["identity"]["profile"], "v5_server")
            self.assertEqual(plan["parameters"]["text_recognition_batch_size"], 1)
            self.assertEqual(len(plan["weights"]), 2)
            self.assertEqual(plan["manifest"]["sha256"], bench.sha256_file(args.manifest))
            Path(plan["images"][0]["path"]).write_bytes(b"changed")
            with self.assertRaises(Exception):
                bench.build_plan(args)

    def test_refuses_without_gate_invalid_counts_or_device(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_fixture(Path(directory))
            for key, value in (("run_local_ocr", False), ("warm_passes", 0), ("sample_interval", float("nan")),
                               ("cpu_threads", 0), ("device", "gpu:0,1")):
                original = getattr(args, key)
                setattr(args, key, value)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    bench.build_plan(args)
                setattr(args, key, original)

    def test_explicit_image_list_and_manifest_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_fixture(Path(directory))
            records = bench.image_records(args.manifest)
            self.assertEqual(bench.image_records(images=[records[0]["path"]]), records[:1])
            saved = json.loads(args.manifest.read_text())
            saved["plan"]["images"][0]["sha256"] = "0" * 64
            args.manifest.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                bench.image_records(args.manifest)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                bench.image_records(images=[records[0]["path"]] * 2)

    def test_gpu_telemetry_filters_pid_not_desktop_and_reports_unavailable(self):
        rows = bench.parse_gpu_rows("2, GPU-A, 9000\n7, GPU-A, 123\n7, GPU-B, [N/A]\n", 7)
        self.assertEqual(rows, [{"gpu_uuid": "GPU-A", "used_memory_mib": 123.0},
                                {"gpu_uuid": "GPU-B", "used_memory_mib": None}])
        sampler = bench.GpuSampler(.2, True)
        with patch.object(bench.subprocess, "run", side_effect=FileNotFoundError("nvidia-smi unavailable")):
            sampler.sample()
        self.assertEqual(sampler.report()["status"], "unavailable")
        self.assertIsNone(sampler.report()["peak_single_gpu_process_mib"])
        self.assertTrue(sampler.report()["errors"])

    def test_local_guard_blocks_socket_and_model_name_fallback(self):
        class Manager:
            def __getitem__(self, key):
                raise AssertionError("unpatched fallback")
        with patch.object(bench.importlib, "import_module", return_value=SimpleNamespace(_ModelManager=Manager)):
            with bench.local_only():
                with self.assertRaisesRegex(RuntimeError, "fallback forbidden"):
                    Manager()["anything"]
                with self.assertRaisesRegex(RuntimeError, "network forbidden"):
                    bench.socket.create_connection(("invalid.test", 80))

    def test_serial_one_init_one_cold_two_all_page_passes_and_full_raw_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_fixture(Path(directory))
            plan = bench.build_plan(args)
            poly = [[0, 0], [5, 0], [5, 10], [0, 10]]
            raw = {"dt_polys": [poly], "rec_polys": [poly], "rec_texts": ["fixture"], "rec_scores": [.9]}
            engine = Mock()
            engine.predict.return_value = [raw]
            factory = Mock(return_value=engine)
            report = {}
            with patch.object(bench, "local_only", return_value=nullcontext()), patch.object(bench, "runtime_identity", return_value={}), patch.dict(
                    "sys.modules", {"paddleocr": SimpleNamespace(PaddleOCR=factory), "paddle": SimpleNamespace()}):
                bench.run_benchmark(plan, report)
            factory.assert_called_once_with(**plan["parameters"])
            engine.close.assert_called_once()
            self.assertEqual(engine.predict.call_count, 15)
            self.assertEqual([row["page_index"] for row in report["records"]], [0] + list(range(7)) * 2)
            self.assertEqual([row["phase"] for row in report["records"]], ["cold"] + ["warm"] * 14)
            self.assertEqual(report["records"][0]["contract"]["raw"], raw)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["gpu_telemetry"]["status"], "not_requested_cpu")

    def test_exclusive_output_and_failed_initialization_json(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_fixture(Path(directory))
            argv = ["--run-local-ocr", "--model-root", directory, "--device", "cpu",
                    "--manifest", str(args.manifest), "--output", str(args.output)]
            with patch.object(bench, "run_benchmark", side_effect=RuntimeError("synthetic init failure")) as run:
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    bench.main(argv)
                saved = json.loads(args.output.read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["plan"]["identity"]["profile"], "v5_server")
                with self.assertRaises(FileExistsError):
                    bench.main(argv)
                run.assert_called_once()

    def test_failed_constructor_keeps_timing_and_resources_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_fixture(Path(directory))
            plan = bench.build_plan(args)
            factory = Mock(side_effect=RuntimeError("synthetic constructor failure"))
            report = {}
            with patch.object(bench, "local_only", return_value=nullcontext()), patch.dict(
                    "sys.modules", {"paddleocr": SimpleNamespace(PaddleOCR=factory), "paddle": SimpleNamespace()}):
                with self.assertRaisesRegex(RuntimeError, "constructor failure"):
                    bench.run_benchmark(plan, report)
            factory.assert_called_once()
            self.assertEqual(report["initialization"]["status"], "failed")
            self.assertGreaterEqual(report["initialization"]["wall_seconds"], 0)
            self.assertIn("resources_final", report)

    def test_reference_default_is_v5_and_custom_thread_identity_matches_effective_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_fixture(Path(directory))
            self.assertEqual(args.profile, "v5_server")
            args.cpu_threads = 3
            plan = bench.build_plan(args)
            self.assertEqual(plan["parameters"]["cpu_threads"], 3)
            self.assertEqual(plan["identity"]["settings"]["cpu_threads"], 3)
            self.assertIs(plan["identity"]["settings"]["enable_mkldnn"], False)
