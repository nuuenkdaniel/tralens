"""Deterministic detector-scale orchestration with fake runtimes, never Paddle."""
import argparse
import copy
from io import StringIO
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from tools import detector_scale_experiment as s


class DetectorScaleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model = self.root/"cache"/s.MODEL_NAME
        self.model.mkdir(parents=True)
        self.config = {"Global":{"model_name":s.MODEL_NAME},"PreProcess":{"transform_ops":[
            {"DecodeImage":{"channel_first":False,"img_mode":"RGB"}},
            {"DetResizeForTest":{"limit_side_len":960,"limit_type":"max"}},
            {"NormalizeImage":{"scale":"1./255."}},{"ToCHWImage":None}]},
            "PostProcess":{"name":"DBPostProcess","thresh":.3,"box_thresh":.6,"unclip_ratio":2.,"use_dilation":False,"box_type":"quad"}}
        self.save_config()
        (self.model/"inference.json").write_text('{"test_only":true}')
        (self.model/"inference.pdiparams").write_bytes(b"test-only; not model weights")
        for index,relative in enumerate(s.IMAGES):
            path = self.root/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            image = Image.new("RGB",(80,96),"white")
            image.putpixel((index+1,2),(index,0,0))
            image.save(path)
        (self.root/"test-results").mkdir()
        for override in (patch.object(s,"ROOT",self.root),patch.object(s,"MODEL_DIR",self.model),
                         patch.object(s.importlib.metadata,"version",return_value="offline-test-installed")):
            override.start()
            self.addCleanup(override.stop)
        self.instances = []
        owner = self

        class FakeDetector:
            def __init__(self,directory):
                self.calls = []
                self.closed = 0
                self.fail_at = None
                self.malformed = False
                self.modify_source = False
                self.change_settings = False
                owner.instances.append(self)

            def settings(self):
                return {"limit_side_len":960,"limit_type":"max","max_side_limit":4000,"thresh":.3,
                    "box_thresh":.7 if self.change_settings and self.calls else .6,"unclip_ratio":2.,
                    "use_dilation":False,"max_candidates":1000,"score_mode":"fast","box_type":"quad","min_size":3}

            def predict(self,path,params):
                self.calls.append((path,copy.deepcopy(params)))
                if self.fail_at == len(self.calls):
                    raise RuntimeError("synthetic detector error")
                with Image.open(path) as source:
                    pixels = np.array(source.convert("RGB"))
                if self.modify_source:
                    Path(path).write_bytes(b"synthetic source drift")
                count = 1 if params["limit_side_len"] == 960 else 2
                polygons = np.array([[[10+20*i,10],[20+20*i,10],[20+20*i,50],[10+20*i,50]] for i in range(count)],dtype=np.float32)
                return {"input_path":str(path),"input_img":pixels,"dt_polys":polygons,
                        "dt_scores":np.array([] if self.malformed else [.9]*count,dtype=np.float32)}

            def close(self):
                self.closed += 1

        self.fake_class = FakeDetector

    def save_config(self):
        (self.model/"inference.yml").write_text(yaml.safe_dump(self.config))

    def actual_cached_config(self):
        # Parent-supplied public export metadata; no real cached file/model read.
        self.config = {"Global":{"model_name":s.MODEL_NAME},"PreProcess":{"transform_ops":[
            {"DecodeImage":{"img_mode":"BGR","channel_first":False}},
            {"DetLabelEncode":None},{"DetResizeForTest":{"resize_long":960}},
            {"NormalizeImage":{"mean":[.485,.456,.406],"std":[.229,.224,.225],"scale":"1./255."}},
            {"ToCHWImage":None},{"KeepKeys":{"keep_keys":["image","shape","polys","ignore_tags"]}}]},
            "PostProcess":{"name":"DBPostProcess","thresh":.3,"box_thresh":.6,
                           "max_candidates":1000,"unclip_ratio":1.5}}
        self.save_config()

    def output(self,name):
        return self.root/"test-results"/name

    def test_help_and_model_free_plan_never_initialize_sdk(self):
        with patch.object(s,"LocalDetector",side_effect=AssertionError("SDK initialization forbidden")),patch("sys.stdout",new=StringIO()):
            self.assertEqual(s.main([]),0)
            output = self.output("plan")
            self.assertEqual(s.main(["--output-dir",str(output),"--variant","1536:max","--variant","1920:max"]),0)
        self.assertEqual({p.name for p in output.iterdir()},{"detector-scale-plan.json"})
        plan = json.loads((output/"detector-scale-plan.json").read_text())
        self.assertEqual(plan["maximum_page_conditions"],21)
        self.assertEqual(len(plan["images"]),7)
        self.assertFalse(plan["historical_baseline_reproduced"])
        self.assertEqual(plan["model"]["files"]["inference.pdiparams"]["sha256"],s.sha256_file(self.model/"inference.pdiparams"))

    def test_scale_parsing_and_bounds_only_allow_two_scale_overrides(self):
        self.assertEqual(s.parse_scale("1536:max"),{"limit_side_len":1536,"limit_type":"max"})
        for value in ("0:max","4096:max","960:other","1536","960.5:max","true:max"):
            with self.assertRaises(argparse.ArgumentTypeError):
                s.parse_scale(value)
        one = s.parse_scale("1536:max")
        for variants in ((one,one),(one,s.parse_scale("1920:max"),s.parse_scale("1024:min")),
                         ({"limit_side_len":960,"limit_type":"max","thresh":.1},),
                         ({"limit_side_len":True,"limit_type":"max"},),({},)):
            with self.assertRaises(ValueError):
                s.validate_variants(variants)

    def test_exact_image_catalog_and_missing_source_fail_before_initialization(self):
        (self.root/s.IMAGES[-1]).unlink()
        with patch.object(s,"LocalDetector",side_effect=AssertionError("SDK forbidden")):
            with self.assertRaises(FileNotFoundError):
                s.main(["--output-dir",str(self.output("missing")),"--run-local-detector"])
        self.assertFalse(self.output("missing").exists())

    def test_cached_config_name_static_files_and_numeric_expression_are_validated(self):
        for change in ("name","dynamic","expression","dilation"):
            with self.subTest(change=change):
                original = copy.deepcopy(self.config)
                if change == "name":
                    self.config["Global"]["model_name"] = "other"
                elif change == "dynamic":
                    (self.model/"model.safetensors").write_bytes(b"not allowed")
                elif change == "expression":
                    self.config["PreProcess"]["transform_ops"][2]["NormalizeImage"]["scale"] = "__import__('os').system('false')"
                else:
                    self.config["PostProcess"]["use_dilation"] = "False"
                self.save_config()
                with self.assertRaises(ValueError):
                    s.model_provenance(self.model)
                self.config = original
                self.save_config()
                if (self.model/"model.safetensors").exists():
                    (self.model/"model.safetensors").unlink()
        (self.model/"inference.pdiparams").unlink()
        with self.assertRaises(ValueError):
            s.model_provenance(self.model)

    def test_only_pinned_cache_and_new_checkout_result_directory_are_permitted(self):
        other = self.root/"other-cache"
        other.mkdir()
        with self.assertRaises(ValueError):
            s.build_plan(model_dir=other)
        with self.assertRaises(SystemExit),patch("sys.stderr",new=StringIO()):
            s.main(["--output-dir",str(self.root/"elsewhere")])
        output = self.output("existing")
        output.mkdir()
        sentinel = output/"sentinel"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            s.main(["--output-dir",str(output)])
        self.assertEqual(sentinel.read_text(),"preserve")

    def test_serial_one_instance_21_conditions_preserve_hashes_native_values_and_fixed_parameters(self):
        output = self.output("live-fake")
        before = {relative:s.sha256_file(self.root/relative) for relative in s.IMAGES}
        with patch.object(s,"LocalDetector",self.fake_class),patch("sys.stdout",new=StringIO()):
            self.assertEqual(s.main(["--output-dir",str(output),"--run-local-detector",
                                    "--variant","1536:max","--variant","1920:max"]),0)
        self.assertEqual(len(self.instances),1)
        detector = self.instances[0]
        self.assertEqual(len(detector.calls),21)
        self.assertEqual(detector.closed,1)
        for page_index,relative in enumerate(s.IMAGES):
            group = detector.calls[page_index*3:page_index*3+3]
            self.assertEqual([path for path,_ in group],[str((self.root/relative).resolve())]*3)
            self.assertEqual([params["limit_side_len"] for _,params in group],[960,1536,1920])
            for _,params in group:
                self.assertEqual((params["thresh"],params["box_thresh"],params["unclip_ratio"],params["use_dilation"]),(.3,.6,2.,False))
        manifest = json.loads((output/"detector-scale-results.json").read_text())
        self.assertEqual(manifest["status"],"completed")
        self.assertEqual(manifest["recognition_calls"],0)
        self.assertEqual(manifest["grouping_calls"],0)
        native = json.loads((output/"page-0000-baseline-detector.json").read_text())["native"]
        self.assertEqual(native["dt_polys"],[[[10.,10.],[20.,10.],[20.,50.],[10.,50.]]])
        self.assertEqual(native["dt_scores"],[float(np.float32(.9))])
        self.assertEqual(native["polygon_dtypes"],["float32"])
        self.assertEqual(before,{relative:s.sha256_file(self.root/relative) for relative in s.IMAGES})

    def fake_with(self,**attributes):
        def factory(directory):
            instance = self.fake_class(directory)
            for name,value in attributes.items():
                setattr(instance,name,value)
            return instance
        return factory

    def test_error_stops_without_retry_and_retains_completed_sidecars(self):
        output = self.output("error")
        with patch.object(s,"LocalDetector",self.fake_with(fail_at=2)),patch("sys.stdout",new=StringIO()):
            self.assertEqual(s.main(["--output-dir",str(output),"--run-local-detector","--variant","1536:max"]),1)
        self.assertEqual(len(self.instances[0].calls),2)
        self.assertEqual(self.instances[0].closed,1)
        manifest = json.loads((output/"detector-scale-results.json").read_text())
        self.assertEqual(manifest["status"],"failed")
        self.assertEqual(len(manifest["completed_conditions"]),1)
        self.assertEqual(manifest["active_condition"]["condition"],"scale-1536-max")

    def test_malformed_native_result_is_saved_and_stops_after_first_call(self):
        output = self.output("malformed")
        with patch.object(s,"LocalDetector",self.fake_with(malformed=True)),patch("sys.stdout",new=StringIO()):
            self.assertEqual(s.main(["--output-dir",str(output),"--run-local-detector"]),1)
        self.assertEqual(len(self.instances[0].calls),1)
        failed = json.loads((output/"page-0000-baseline-failed-detector.json").read_text())
        self.assertEqual(len(failed["native_raw"]["dt_polys"]),1)
        self.assertEqual(failed["native_raw"]["dt_scores"],[])

    def test_source_and_runtime_parameter_drift_stop_without_retry(self):
        for name in ("modify_source","change_settings"):
            with self.subTest(name=name):
                output = self.output(name)
                with patch.object(s,"LocalDetector",self.fake_with(**{name:True})),patch("sys.stdout",new=StringIO()):
                    self.assertEqual(s.main(["--output-dir",str(output),"--run-local-detector"]),1)
                self.assertEqual(len(self.instances[-1].calls),1)
                if name == "modify_source":
                    # Restore this test-owned fixture, never a shared source.
                    Image.new("RGB",(80,96),"white").save(self.root/s.IMAGES[0])

    def test_duplicate_fresh_baseline_override_fails_before_any_detector_call(self):
        output = self.output("duplicate-baseline")
        with patch.object(s,"LocalDetector",self.fake_class),patch("sys.stdout",new=StringIO()):
            self.assertEqual(s.main(["--output-dir",str(output),"--run-local-detector","--variant","960:max"]),1)
        self.assertEqual(self.instances[0].calls,[])

    def test_network_and_child_process_guard_blocks_requests_and_restores_only_owned_flags(self):
        previous = os.environ.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK")
        with s.offline_local_only():
            self.assertEqual(os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"],"True")
            with self.assertRaises(RuntimeError):
                socket.create_connection(("127.0.0.1",1))
            with self.assertRaises(RuntimeError):
                s.subprocess.Popen(["not-executed"])
        self.assertEqual(os.environ.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"),previous)

    def test_native_recognizer_fields_are_rejected(self):
        plan = s.build_plan()
        image = plan["images"][0]
        result = {"rec_texts":[],"dt_polys":[],"dt_scores":[],"input_img":np.zeros((96,80,3),np.uint8)}
        with self.assertRaises(ValueError):
            s.native_result(result,image)

    def test_actual_cached_training_metadata_is_inert_valid_and_read_only(self):
        self.actual_cached_config()
        before = (self.model/"inference.yml").read_bytes()
        output = self.output("actual-plan")
        with patch.object(s,"LocalDetector",side_effect=AssertionError("no model initialization")),patch("sys.stdout",new=StringIO()):
            self.assertEqual(s.main(["--output-dir",str(output),"--variant","1280:max","--variant","1600:max"]),0)
        plan = json.loads((output/"detector-scale-plan.json").read_text())
        self.assertEqual(plan["model"]["ignored_training_transforms"],[
            {"DetLabelEncode":None},{"KeepKeys":{"keep_keys":["image","shape","polys","ignore_tags"]}}])
        self.assertEqual(plan["model"]["export_resize_config"],{"resize_long":960})
        self.assertEqual(plan["model"]["export_postprocess_config"]["unclip_ratio"],1.5)
        self.assertEqual(plan["maximum_page_conditions"],21)
        self.assertEqual((self.model/"inference.yml").read_bytes(),before)

    def test_unverified_training_fields_unknown_transforms_and_duplicates_remain_rejected(self):
        for replacement in ({"DetLabelEncode":{}},{"KeepKeys":None},
                            {"KeepKeys":{"keep_keys":["image","shape"]}},
                            {"KeepKeys":{"keep_keys":["image","shape","polys","ignore_tags"],"extra":True}},
                            {"UnknownTransform":None}):
            self.actual_cached_config()
            ops = self.config["PreProcess"]["transform_ops"]
            ops[1 if "DetLabelEncode" in replacement else -1] = replacement
            self.save_config()
            with self.assertRaises(ValueError):
                s.model_provenance(self.model)
        self.actual_cached_config()
        self.config["PreProcess"]["transform_ops"].append({"DetLabelEncode":None})
        self.save_config()
        with self.assertRaises(ValueError):
            s.model_provenance(self.model)

    def test_actual_unclip_value_is_fixed_across_parent_scale_choices(self):
        self.actual_cached_config()
        owner = self

        class ActualMetadataDetector(self.fake_class):
            def settings(self):
                result = super().settings()
                result["unclip_ratio"] = 1.5
                return result

        output = self.output("actual-live-fake")
        with patch.object(s,"LocalDetector",ActualMetadataDetector),patch("sys.stdout",new=StringIO()):
            self.assertEqual(s.main(["--output-dir",str(output),"--variant","1280:max","--variant","1600:max","--run-local-detector"]),0)
        self.assertEqual(len(owner.instances),1)
        self.assertEqual(len(owner.instances[0].calls),21)
        self.assertEqual({params["unclip_ratio"] for _,params in owner.instances[0].calls},{1.5})
        self.assertEqual({params["limit_side_len"] for _,params in owner.instances[0].calls},{960,1280,1600})

    def test_effective_resize_and_postprocess_readback_uses_initialized_objects(self):
        runtime = s.LocalDetector.__new__(s.LocalDetector)
        resize = SimpleNamespace(resize_type=0,limit_side_len=960,limit_type="max")
        post = SimpleNamespace(thresh=.3,box_thresh=.6,unclip_ratio=1.5,use_dilation=False,
                               max_candidates=1000,score_mode="fast",box_type="quad",min_size=3)
        runtime.model = SimpleNamespace(paddlex_predictor=SimpleNamespace(
            pre_tfs={"Resize":resize},post_op=post,max_side_limit=4000,device="cpu"))
        self.assertEqual(runtime.settings()["limit_side_len"],960)
        self.assertEqual(runtime.settings()["limit_type"],"max")
        self.assertEqual(runtime.settings()["unclip_ratio"],1.5)
        # A supported explicit initialized value wins over any YAML hint/default.
        resize.limit_side_len = 1280
        self.assertEqual(runtime.settings()["limit_side_len"],1280)

    def test_fixed_paddle_capability_queries_use_no_real_process_or_network(self):
        with patch.object(s,"read_glibc_version",return_value="2.39") as libc, \
             patch.object(s,"read_avx_support",return_value=True) as avx, \
             patch.object(s.subprocess,"Popen",side_effect=AssertionError("real process forbidden")) as real_process, \
             patch.object(s.os,"popen",side_effect=AssertionError("real shell forbidden")) as real_shell:
            with s.offline_local_only() as capabilities:
                for _ in range(2):
                    result = s.subprocess.Popen(s.LIBC_PROBE,stdout=s.subprocess.PIPE,stderr=s.subprocess.PIPE,shell=True)
                    self.assertEqual(result.communicate(),(b"2.39\n",b""))
                    self.assertEqual(result.returncode,0)
                    with s.os.popen(s.AVX_PROBE) as stream:
                        self.assertTrue(stream.read())
                self.assertEqual(capabilities,{"glibc_version":"2.39","avx_supported":True})
                with self.assertRaises(RuntimeError):
                    socket.create_connection(("127.0.0.1",1))
            libc.assert_called_once_with()
            avx.assert_called_once_with()
            real_process.assert_not_called()
            real_shell.assert_not_called()

    def test_capability_shim_rejects_command_variants_and_does_not_invent_probe_results(self):
        with patch.object(s,"read_glibc_version",side_effect=RuntimeError("unavailable")), \
             patch.object(s,"read_avx_support",return_value=False),s.offline_local_only() as capabilities:
            for command,kwargs in ((s.LIBC_PROBE+"; other",{}),(["ldd","--version"],{}),
                                   (s.LIBC_PROBE,{"stdout":s.subprocess.PIPE,"stderr":s.subprocess.PIPE,"shell":True,"cwd":"/"})):
                with self.assertRaises(RuntimeError):
                    s.subprocess.Popen(command,**kwargs)
            with self.assertRaises(RuntimeError):
                s.subprocess.Popen(s.LIBC_PROBE,stdout=s.subprocess.PIPE,stderr=s.subprocess.PIPE,shell=True)
            self.assertNotIn("glibc_version",capabilities)
            self.assertEqual(s.os.popen(s.AVX_PROBE).read(),"")
            self.assertIs(capabilities["avx_supported"],False)
            with self.assertRaises(RuntimeError):
                s.os.popen(s.AVX_PROBE+"; other")

    def test_read_only_capabilities_use_fixed_os_sources(self):
        with patch.object(s.os,"confstr",return_value="glibc 2.39") as confstr:
            self.assertEqual(s.read_glibc_version(),"2.39")
            confstr.assert_called_once_with("CS_GNU_LIBC_VERSION")
        for invalid in (None,"musl 1.2.3","glibc unknown"):
            with patch.object(s.os,"confstr",return_value=invalid),self.assertRaises(RuntimeError):
                s.read_glibc_version()
        with patch.object(Path,"read_text",autospec=True,return_value="flags : fpu AVX avx2\n") as read:
            self.assertTrue(s.read_avx_support())
            self.assertEqual(read.call_args.args[0],Path("/proc/cpuinfo"))
        with patch.object(Path,"read_text",autospec=True,return_value="flags : fpu sse\n"):
            self.assertFalse(s.read_avx_support())

    @unittest.skipUnless(sys.platform.startswith("linux"),"fixed Linux system-library compatibility")
    def test_installed_libdl_loads_without_linker_cache_process_or_model_import(self):
        # This loads a public system library only, never Paddle/Torch/a model.
        with patch.object(s.ctypes.util,"find_library",side_effect=AssertionError("no linker cache lookup")) as original:
            with s.offline_local_only() as capabilities:
                self.assertEqual(s.ctypes.util.find_library("dl"),"libdl.so.2")
                self.assertEqual(capabilities["libdl_soname"],"libdl.so.2")
            original.assert_not_called()

    def test_only_dl_lookup_is_intercepted_and_all_other_process_network_guards_remain(self):
        def original_lookup(name):
            # Represent the exact stdlib fallback reported by the parent. It
            # remains forbidden for names outside the one supported dl lookup.
            return s.subprocess.Popen(["/sbin/ldconfig","-p"])

        with patch.object(s.ctypes.util,"find_library",side_effect=original_lookup) as original, \
             patch.object(s,"verified_libdl_soname",return_value="libdl.so.2") as verify:
            with s.offline_local_only() as capabilities:
                self.assertEqual(s.ctypes.util.find_library("dl"),"libdl.so.2")
                self.assertEqual(s.ctypes.util.find_library("dl"),"libdl.so.2")
                verify.assert_called_once_with()
                with self.assertRaises(RuntimeError):
                    s.ctypes.util.find_library("other")
                original.assert_called_once_with("other")
                with self.assertRaises(RuntimeError):
                    s.subprocess.Popen(["/sbin/ldconfig","-p"])
                with self.assertRaises(RuntimeError):
                    socket.create_connection(("127.0.0.1",1))
                self.assertEqual(capabilities["libdl_soname"],"libdl.so.2")

    def test_missing_libdl_or_required_symbol_fails_without_lookup_fallback(self):
        for outcome in (OSError("missing system library"),SimpleNamespace(dlerror=object())):
            with self.subTest(outcome_type=type(outcome).__name__):
                arguments = {"side_effect":outcome} if isinstance(outcome,Exception) else {"return_value":outcome}
                with patch.object(s.ctypes,"CDLL",**arguments) as load, \
                     patch.object(s.ctypes.util,"find_library",side_effect=AssertionError("fallback forbidden")) as original:
                    with s.offline_local_only() as capabilities:
                        with self.assertRaises(RuntimeError):
                            s.ctypes.util.find_library("dl")
                        self.assertNotIn("libdl_soname",capabilities)
                    load.assert_called_once_with("libdl.so.2")
                    original.assert_not_called()
