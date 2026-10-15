"""Explicit source-only copy verification, excluded from normal test*.py discovery.

Run with the existing venv: python -m unittest tests.source_only_check -v
The child runs full discovery; no images, saved evidence, configs or venv are copied.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class SourceOnlySuiteTest(unittest.TestCase):
    def test_source_only_copy_passes_full_suite_with_explicit_optional_evidence_skips(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)/"source-only"
            checkout.mkdir()
            for name in ("src","tools","tests"):
                for path in (source/name).rglob("*.py"):
                    if "__pycache__" in path.parts:
                        continue
                    self.assertFalse(path.is_symlink(),"refuse symlinked code in source-only verification: "+str(path))
                    target = checkout/path.relative_to(source)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(path,target)
            for name in ("README.md",".gitignore"):
                shutil.copyfile(source/name,checkout/name)
            self.assertFalse((checkout/"images").exists())
            self.assertFalse((checkout/"test-results").exists())
            child = r'''
import json, pathlib, sys, unittest
root = pathlib.Path.cwd().resolve()
sys.path.insert(0, str(root))
suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
for name in ("tests", "tralens.ocr.detection", "tools.diagnostics"):
    assert pathlib.Path(sys.modules[name].__file__).resolve().is_relative_to(root), name
result = unittest.TextTestRunner(verbosity=2).run(suite)
print("TRALENS_SOURCE_ONLY_RESULT=" + json.dumps({
    "run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
    "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped]}))
raise SystemExit(0 if result.wasSuccessful() else 1)
'''
            # -I ignores PYTHONPATH/user-site leakage from the real checkout;
            # the original venv executable still supplies installed dependencies.
            run = subprocess.run([sys.executable,"-I","-c",child],cwd=checkout,
                                 capture_output=True,text=True,timeout=180)
            self.assertEqual(run.returncode,0,run.stdout+"\n"+run.stderr)
            lines = [line for line in run.stdout.splitlines() if line.startswith("TRALENS_SOURCE_ONLY_RESULT=")]
            self.assertEqual(len(lines),1,run.stdout+"\n"+run.stderr)
            summary = json.loads(lines[0].split("=",1)[1])
            self.assertEqual((summary["failures"],summary["errors"]),(0,0))
            self.assertTrue(summary["skipped"],"saved-page integrations must be explicitly skipped without evidence")
            for skip in summary["skipped"]:
                self.assertTrue(skip["reason"].startswith("optional local evidence unavailable;"),skip)
            print("Source-only suite: "+str(summary["run"])+" tests, "+str(len(summary["skipped"]))+" optional-local-evidence skips")
            for skip in summary["skipped"]:
                print(skip["test"]+": "+skip["reason"])
