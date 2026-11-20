"""Availability guards must not hide corrupt fixtures or skip synthetic assertions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import local_evidence as e


class LocalEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_decoration_is_lazy_and_present_evidence_runs_original_assertions(self):
        calls = []
        with patch.object(e,"ROOT",self.root):
            def body(case):
                calls.append(True)
                case.assertEqual((self.root/"required.json").read_text(),"fixture")

            with patch.object(Path,"is_file",side_effect=AssertionError("collection-time asset check")), \
                 patch.object(Path,"read_text",side_effect=AssertionError("collection-time asset read")):
                integration = e.local_evidence(files=("required.json",))(body)

            self.assertEqual(calls,[])
            with self.assertRaises(unittest.SkipTest) as skipped:
                integration(self)
            self.assertIn(str(self.root/"required.json"),str(skipped.exception))
            self.assertEqual(calls,[])
            (self.root/"required.json").write_text("fixture")
            integration(self)
            self.assertEqual(calls,[True])
            (self.root/"required.json").write_text("corrupt")
            with self.assertRaises(AssertionError):
                integration(self)

    def test_comparison_without_original_image_skips_with_precise_path(self):
        comparison = self.root/"comparison.json"
        comparison.write_text(json.dumps({"input":"images/original.png","boxes":[]}))
        with self.assertRaises(unittest.SkipTest) as skipped:
            e.require_local_evidence(comparisons=("comparison.json",),root=self.root)
        self.assertIn(str(self.root/"images/original.png"),str(skipped.exception))

    def test_manifest_never_follows_absolute_evidence_from_another_checkout(self):
        checkout = self.root/"checkout"
        checkout.mkdir()
        outside = self.root/"other-private-comparison.json"
        outside.write_text("must not be parsed")
        (checkout/"manifest.json").write_text(json.dumps({"pages":[{"comparison_path":str(outside)}]}))
        with self.assertRaises(unittest.SkipTest) as skipped:
            e.require_local_evidence(manifests=("manifest.json",),root=checkout)
        self.assertIn("outside this checkout",str(skipped.exception))
        self.assertIn(str(outside),str(skipped.exception))

    def test_present_malformed_manifests_fail_and_complete_dependencies_do_not_skip(self):
        manifest = self.root/"manifest.json"
        manifest.write_text("not json")
        with self.assertRaises(json.JSONDecodeError):
            e.require_local_evidence(manifests=("manifest.json",),root=self.root)
        (self.root/"image.png").write_bytes(b"availability only; image validity belongs to integration assertions")
        (self.root/"comparison.json").write_text(json.dumps({"input":"image.png","boxes":[]}))
        manifest.write_text(json.dumps({"pages":[{"comparison_path":"comparison.json","page":{"image_path":"image.png"}}]}))
        e.require_local_evidence(manifests=("manifest.json",),root=self.root)
        expected_hash = hashlib.sha256((self.root/"image.png").read_bytes()).hexdigest()
        with patch.object(e,"ROOT",self.root):
            @e.local_evidence(manifests=("manifest.json",))
            def integration(case):
                case.assertEqual(hashlib.sha256((self.root/"image.png").read_bytes()).hexdigest(),expected_hash)

            integration(self)
            (self.root/"image.png").write_bytes(b"fixture hash drift")
            with self.assertRaises(AssertionError):
                integration(self)
