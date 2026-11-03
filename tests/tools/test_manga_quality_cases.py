"""Portable saved-output quality freezing, with optional private-page verification."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tests.local_evidence import local_evidence
from tests.rendering import test_natural as fixtures
from tests.rendering import test_coverage as coverage_fixtures
from tools import manga_quality_cases as q
from tralens.rendering.automatic import run_automatic


class MangaQualityCasesTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CommandTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.root = self.f.root
        self.rendering = self.root / 'rendering'
        run_automatic(preview=self.root / 'preview.json', image=self.f.image, fonts=self.root / 'fonts.json',
                      output=self.rendering, min_size=14, max_size=28, render_policy='conservative')
        self.spec = {'schema_version': 1, 'purpose': 'native_manga_quality_development',
            'source_sha256': q.n.digest(self.f.image.read_bytes()), 'split': 'development',
            'exposure': 'Synthetic fixture, already inspected; no held-out claim.',
            'missing_strata': ['checked transcription references'], 'cases': [
                {'id': 'text', 'kind': 'detected_text', 'anchor_ids': ['main'], 'bounds': [10, 10, 80, 115],
                 'tags': ['strict_control'], 'observation': '<script>not executable</script>'},
                {'id': 'negative', 'kind': 'negative_art', 'anchor_ids': [], 'bounds': [80, 10, 99, 115],
                 'tags': ['art_only_negative'], 'observation': 'Provisional background negative.'}]}
        self.cases = self.root / 'cases.json'
        self.write_spec()

    def write_spec(self):
        self.cases.write_text(json.dumps(self.spec, ensure_ascii=False))

    def freeze(self, output='frozen'):
        return q.freeze(preview=self.root / 'preview.json', image=self.f.image, rendering=self.rendering,
                        cases=self.cases, output=self.root / output)

    def rewrite_manifest(self, mutate):
        path = self.root / 'frozen/manifest.json'
        doc = json.loads(path.read_bytes())
        mutate(doc)
        path.write_text(json.dumps(doc, ensure_ascii=False))
        (path.parent / 'manifest.sha256').write_text(q.n.digest(path.read_bytes()) + '\n')

    def test_fresh_offline_snapshot_preserves_all_evidence_and_native_pixels(self):
        before = {p: p.read_bytes() for p in [self.f.image, self.cases, self.root / 'preview.json',
                  *self.rendering.iterdir()] if p.is_file()}
        with patch('socket.socket', side_effect=AssertionError('offline only')):
            summary = self.freeze()
            verified = q.verify(self.root / 'frozen', expected_sha256=summary['manifest_sha256'])
        self.assertTrue(verified['verified'])
        self.assertEqual((summary['native_cases'], summary['saved_units'], summary['translation_results'],
                          summary['replacements'], summary['held_out_cases']), (2, 2, 1, 1, 0))
        self.assertFalse(summary['quality_improvement_claimed'])
        self.assertEqual(summary['new_inference_calls'], 0)
        manifest = json.loads((self.root / 'frozen/manifest.json').read_bytes())
        for row in manifest['cases']:
            self.assertEqual(row['quality_assessment'], {key: None for key in q.UNASSESSED})
        unit = manifest['cases'][0]['baseline_units'][0]
        self.assertEqual(unit['selected_text'], 'こんにちは')
        self.assertEqual(q.n.canonical(unit['readings']), q.n.canonical(self.f.selected_record['unit']['readings']))
        self.assertEqual(unit['translation'], 'Hi!\nAV?')
        with Image.open(self.f.image) as source, Image.open(self.root / 'frozen/cases/text/source.png') as crop:
            self.assertTrue(np.array_equal(np.asarray(source.crop(self.spec['cases'][0]['bounds'])), np.asarray(crop)))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        for path in self.rendering.iterdir():
            if path.is_file():
                self.assertEqual((self.root / 'frozen/baseline' / path.name).read_bytes(), path.read_bytes())
        html = (self.root / 'frozen/review.html').read_text()
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>', html)
        self.assertIn('Content-Security-Policy', html)
        with self.assertRaises(ValueError):
            self.freeze()

    def test_verify_is_self_contained_without_saved_absolute_paths_or_fonts(self):
        self.freeze()
        frozen = self.root / 'frozen'
        real_open = Path.open
        opened = []
        def local_only(path, *args, **kwargs):
            self.assertTrue(Path(path).is_relative_to(frozen), 'opened a non-snapshot path: ' + str(path))
            opened.append(path)
            return real_open(path, *args, **kwargs)
        with patch.object(Path, 'open', local_only), patch('socket.socket', side_effect=AssertionError('no network')):
            self.assertTrue(q.verify(frozen)['verified'])
        self.assertTrue(opened)

    def test_uncertain_coverage_bindings_and_unplaced_translations_stay_separate(self):
        helper = coverage_fixtures.CoverageRenderingTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.make_coverage()
        helper.p.run_pipeline(render_policy='best-effort')
        preview = helper.p.root / 'job/preview/preview.json'
        saved = json.loads(preview.read_bytes())
        spec = copy.deepcopy(self.spec)
        spec['source_sha256'] = saved['input']['sha256']
        spec['cases'] = [{**spec['cases'][0], 'anchor_ids': saved['units'][0]['unit']['member_ids'],
                          'bounds': [0, 0, *saved['input']['size']]}]
        cases = helper.p.root / 'cases.json'
        cases.write_text(json.dumps(spec))
        helper.render('unplaced', render_policy='conservative')
        for name, rendering, count in [('placed', helper.p.root / 'job/rendering', 2),
                                       ('unplaced', helper.p.root / 'unplaced', 0)]:
            output = self.root / name
            summary = q.freeze(preview=preview, image=helper.f.image, rendering=rendering, cases=cases, output=output)
            self.assertEqual(summary['translation_results'], 2)
            self.assertEqual(summary['replacements'], count)
            self.assertEqual(summary['best_effort_replacements'], count)
            self.assertTrue(q.verify(output)['verified'])
            manifest = json.loads((output / 'manifest.json').read_bytes())
            self.assertEqual(manifest['cases'][0]['baseline_units'][0]['input_kind'], 'coverage_text_region')
            self.assertIsNotNone(manifest['cases'][0]['baseline_units'][0]['translation'])
        self.assertTrue(all(record['unit']['blocked_reasons'] for record in saved['units']))

    def test_hash_drift_and_external_manifest_pin_fail(self):
        summary = self.freeze()
        for name in ('baseline/clean.png', 'cases/text/source.png', 'freeze-source.py'):
            path = self.root / 'frozen' / name
            content = path.read_bytes()
            path.write_bytes(content + b' ')
            with self.subTest(name=name), self.assertRaises(ValueError):
                q.verify(self.root / 'frozen')
            path.write_bytes(content)
        with self.assertRaises(ValueError):
            q.verify(self.root / 'frozen', expected_sha256='0' * 64)
        self.assertTrue(q.verify(self.root / 'frozen', expected_sha256=summary['manifest_sha256'])['verified'])

    def test_rehashed_semantic_drift_and_missing_artifacts_fail(self):
        self.freeze()
        path = self.root / 'frozen/manifest.json'
        original = path.read_bytes()
        changes = [lambda d: d['summary'].update(held_out_cases=1),
                   lambda d: d['summary'].update(quality_improvement_claimed=True),
                   lambda d: d.update(exposure='unseen'),
                   lambda d: d['cases'][0]['baseline_units'][0].update(selected_text='corrected'),
                   lambda d: d['cases'][0]['measurements'].update(page_cleanup_changed_pixels=0),
                   lambda d: d['cases'][0]['quality_assessment'].update(readability='passed'),
                   lambda d: d['artifacts'].pop('cases/text/source.png'),
                   lambda d: d['artifacts'].update({'../outside': '0' * 64})]
        for change in changes:
            path.write_bytes(original)
            self.rewrite_manifest(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                q.verify(self.root / 'frozen')

    def test_case_spec_rejects_truth_fields_wrong_source_heldout_and_bad_anchors(self):
        original = copy.deepcopy(self.spec)
        changes = [lambda d: d.update(source_sha256='0' * 64), lambda d: d.update(split='held-out'),
                   lambda d: d['cases'][0].update(reference='correct source'),
                   lambda d: d['cases'][0].update(id='../escape'),
                   lambda d: d['cases'][0].update(anchor_ids=[True]),
                   lambda d: d['cases'][0].update(anchor_ids=['foreign']),
                   lambda d: d['cases'][0].update(anchor_ids=['main', 'main']),
                   lambda d: d['cases'][0].update(bounds=[25, 20, 30, 30]),
                   lambda d: d['cases'][0].update(bounds=[0, 0, 181, 130]),
                   lambda d: d['cases'].append(copy.deepcopy(d['cases'][0]))]
        for change in changes:
            self.spec = copy.deepcopy(original)
            change(self.spec)
            self.write_spec()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.freeze()
            self.assertFalse((self.root / 'frozen').exists())

    def test_rehashed_render_target_source_and_audit_drift_fail_before_publication(self):
        path = self.rendering / 'manifest.json'
        original = json.loads(path.read_bytes())
        changes = [lambda d: next(r['binding'] for r in d['units'] if 'binding' in r).update(text='invented'),
                   lambda d: d['units'][0].update(source_record_sha256='0' * 64),
                   lambda d: d['pixel_audit'].update(clean_changed_pixels=0),
                   lambda d: d['coverage'].update(rendered=2),
                   lambda d: d['artifacts'].update({'../outside': '0' * 64})]
        for change in changes:
            doc = copy.deepcopy(original)
            change(doc)
            doc['job_revision'] = q.n.revision({k: v for k, v in doc.items() if k not in ('job_revision', 'artifacts')})
            path.write_text(json.dumps(doc))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.freeze()
            self.assertFalse((self.root / 'frozen').exists())

    def test_symlink_inputs_outputs_and_frozen_artifacts_rejected(self):
        alias = self.root / 'alias.json'
        alias.symlink_to(self.cases)
        self.cases = alias
        with self.assertRaises(ValueError):
            self.freeze()
        self.cases = self.root / 'cases.json'
        out = self.root / 'out'
        out.symlink_to(self.root / 'missing-directory')
        with self.assertRaises(ValueError):
            self.freeze('out')
        self.freeze()
        artifact = self.root / 'frozen/cases/text/source.png'
        content = artifact.read_bytes()
        replacement = self.root / 'other.png'
        replacement.write_bytes(content)
        artifact.unlink()
        artifact.symlink_to(replacement)
        with self.assertRaises(ValueError):
            q.verify(self.root / 'frozen')

    def test_cli_offline_verify_and_rejects_bad_pin(self):
        self.freeze()
        code = '''
import runpy, socket, sys, unittest.mock as m
def denied(*a, **kw): raise AssertionError('offline CLI used network')
with m.patch.object(socket, 'socket', denied), m.patch('subprocess.Popen', denied):
    sys.argv = ['tools/manga_quality_cases.py'] + sys.argv[1:]
    runpy.run_path('tools/manga_quality_cases.py', run_name='__main__')
'''
        args = [sys.executable, '-c', code, 'verify', '--frozen', str(self.root / 'frozen')]
        run = subprocess.run(args, cwd=q.ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(json.loads(run.stdout)['verified'])
        run = subprocess.run(args + ['--expected-sha256', '0' * 64], cwd=q.ROOT,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 2, run.stderr)


class SavedMangaQualityCasesTests(unittest.TestCase):
    @local_evidence(files=('test-results/manga-quality-frozen-v1/manifest.json',))
    def test_private_frozen_page_bindings_not_quality(self):
        result = q.verify(q.ROOT / 'test-results/manga-quality-frozen-v1')
        self.assertTrue(result['verified'])
        self.assertEqual(result['native_cases'], 14)
        self.assertFalse(result['quality_improvement_claimed'])


if __name__ == '__main__':
    unittest.main()
