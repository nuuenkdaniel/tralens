"""Offline Phase 3 specimens, transforms and explicit aesthetic-review workflow."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tralens.rendering import natural as n
from . import test_natural as fixtures
from .font_fixture import synthetic_font


class StyleTests(unittest.TestCase):
    def test_presets_are_explicit_and_do_not_invent_face_weight(self):
        from tralens.rendering.styles import resolve_style, face_metadata
        data = synthetic_font()
        face = face_metadata(data)
        for role in ('dialogue', 'narration', 'shout', 'thought'):
            resolved, approximation = resolve_style({'preset': role, 'min_size': 14, 'max_size': 28}, face)
            self.assertEqual(resolved['preset'], role)
            self.assertEqual(resolved['actual_face'], face)
            self.assertFalse(resolved['synthetic_weight'])
            self.assertTrue(approximation)
        self.assertIsNone(face['weight_class'])

    def test_actual_colored_stroked_rotated_glyphs_stay_contained(self):
        data = synthetic_font()
        mask = Image.new('L', (180, 150), 255)
        style = {'preset': 'shout', 'min_size': 28, 'max_size': 28, 'fill': [210, 30, 20],
                 'outline_fill': [20, 100, 220], 'outline': 3, 'rotation': 14, 'leading': .4, 'align': 'right'}
        layer, p = n.layout('AV!\nYo?', data, n.font_coverage(data), mask, style)
        pixels = np.asarray(layer)
        self.assertTrue(((pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 70) & (pixels[:, :, 3] > 200)).any())
        self.assertTrue(((pixels[:, :, 2] > 150) & (pixels[:, :, 0] < 70) & (pixels[:, :, 3] > 200)).any())
        self.assertEqual(p['resolved_style']['rotation'], 14)
        self.assertEqual(p['local_alpha_pixels'], np.count_nonzero(pixels[:, :, 3]))
        self.assertGreater(p['pretransform_alpha_pixels'], 0)

    def test_transformed_overflow_and_invalid_controls_block(self):
        data = synthetic_font()
        cov = n.font_coverage(data)
        base = {'preset': 'dialogue', 'min_size': 28, 'max_size': 28}
        for key, value in (('rotation', 21), ('rotation', float('nan')), ('leading', -1),
                           ('fill', [True, 0, 0]), ('outline', 9), ('align', 'justify')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                n.layout('AV!', data, cov, Image.new('L', (200, 150), 255), {**base, key: value})
        layer, p = n.layout('AV!', data, cov, Image.new('L', (200, 150), 255), base)
        box = layer.getchannel('A').getbbox()
        tight = Image.new('L', (box[2]-box[0], box[3]-box[1]), 255)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            n.layout('AV!', data, cov, tight, {**base, 'rotation': 20})

    def test_alignment_leading_presets_and_actual_face_remain_explicit(self):
        from tralens.rendering.styles import resolve_style
        face = {'family': 'Synthetic Regular', 'subfamily': 'Regular', 'weight_class': 400}
        shout, approximation = resolve_style({'preset': 'shout', 'min_size': 20, 'max_size': 20}, face)
        self.assertEqual(shout['actual_face']['weight_class'], 400)
        self.assertEqual(shout['requested_weight'], 'bold')
        self.assertIn('no synthetic weight', ' '.join(approximation))
        data = synthetic_font()
        outputs = []
        for role in ('dialogue', 'narration', 'shout', 'thought'):
            layer, provenance = n.layout('AV!\nYo?', data, n.font_coverage(data), Image.new('L', (180, 180), 255),
                {'preset': role, 'min_size': 28, 'max_size': 28})
            outputs.append(layer.tobytes())
            self.assertEqual(provenance['resolved_style']['preset'], role)
        self.assertEqual(len(set(outputs)), 4)

    def test_rotation_padding_and_allocation_guards_run_after_transform(self):
        data = synthetic_font()
        style = {'preset': 'shout', 'min_size': 28, 'max_size': 28, 'rotation': 20}
        with patch.object(n, 'MAX_LOCAL_PIXELS', 50000):
            with self.assertRaisesRegex(ValueError, 'rotated local raster budget'):
                n.layout('AV!', data, n.font_coverage(data), Image.new('L', (200, 150), 255), style)

    def test_independent_postrotation_crop_check_refuses_disagreement(self):
        data = synthetic_font()
        with patch.object(n.cv2, 'boundingRect', return_value=(0, 0, 1, 1)):
            with self.assertRaisesRegex(ValueError, 'post-transform crop bounds disagree'):
                n.layout('AV!', data, n.font_coverage(data), Image.new('L', (200, 150), 255),
                    {'preset': 'shout', 'min_size': 28, 'max_size': 28, 'rotation': 5})


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CommandTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def inputs(self):
        f = self.f
        return dict(preview=f.root/'preview.json', image=f.image,
                    selection=f.root/'selection.json', fonts=f.root/'fonts.json')

    def test_protected_checkout_manifests_remain_byte_identical(self):
        root = Path(__file__).resolve().parents[2]
        paths = [root/name for name in ('pyproject.toml', 'uv.lock') if (root/name).is_file()]
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.f.run_render()
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.assertEqual(before, after)
        if before:
            print('Protected manifest SHA256: '+json.dumps(before, sort_keys=True))

    def test_specimens_revision_and_explicit_review(self):
        from tralens.rendering.review import record_review, verify_review
        report = self.f.run_render()
        self.assertTrue((self.f.root/'out/unit-000-specimen-composite.png').is_file())
        self.assertEqual(report['units'][0]['aesthetic_state'], 'not_assessed')
        review = self.f.root/'aesthetic.json'
        record = record_review(candidate=self.f.root/'out', unit_id=report['units'][0]['unit_id'],
            state='accepted', operator='synthetic-test-operator', note='Synthetic aesthetic test only',
            output=review, **self.inputs())
        self.assertEqual(record['scope'], 'aesthetic_only')
        self.assertEqual(record['linguistic_approval'], 'not_assessed')
        self.assertFalse(record['identity_verified'])
        self.assertEqual(verify_review(candidate=self.f.root/'out', review=review, **self.inputs())['state'], 'accepted')

    def make_record(self, state='accepted', name='aesthetic.json'):
        from tralens.rendering.review import record_review
        report = self.f.run_render()
        path = self.f.root/name
        record = record_review(candidate=self.f.root/'out', unit_id=report['units'][0]['unit_id'], state=state,
            operator='synthetic-test-operator', note='Synthetic aesthetics only, not linguistic approval', output=path, **self.inputs())
        return record, path

    def test_all_review_states_require_explicit_action_and_never_modify_job(self):
        from tralens.rendering.review import record_review, verify_review
        report = self.f.run_render()
        manifest = (self.f.root/'out/manifest.json').read_bytes()
        for state in ('not_assessed', 'accepted', 'rejected', 'rework'):
            path = self.f.root/(state+'.json')
            record_review(candidate=self.f.root/'out', unit_id=report['units'][0]['unit_id'], state=state,
                operator='synthetic-test-operator', note='Synthetic state test', output=path, **self.inputs())
            self.assertEqual(verify_review(candidate=self.f.root/'out', review=path, **self.inputs())['state'], state)
        self.assertEqual(manifest, (self.f.root/'out/manifest.json').read_bytes())
        for state, operator in ((None, 'test'), ('automatic', 'test'), ('accepted', '')):
            with self.assertRaises(ValueError):
                record_review(candidate=self.f.root/'out', unit_id=report['units'][0]['unit_id'], state=state,
                    operator=operator, note='Synthetic test', output=self.f.root/'bad.json', **self.inputs())
        with self.assertRaises(ValueError):
            record_review(candidate=self.f.root/'out', unit_id=report['units'][0]['unit_id'], state='accepted',
                operator='synthetic-test-operator', note='Synthetic test', output=self.f.root/'accepted.json', **self.inputs())

    def test_selection_style_geometry_and_text_edits_invalidate_acceptance(self):
        from tralens.rendering.review import verify_review
        _, path = self.make_record()
        selection = self.f.root/'selection.json'
        before = selection.read_bytes()
        for kind in ('style', 'geometry', 'text', 'whitespace'):
            doc = json.loads(before)
            if kind == 'style':
                doc['units'][0]['style']['rotation'] = 4
            elif kind == 'geometry':
                doc['units'][0]['layout'][0][0][0] += 1
            elif kind == 'text':
                doc['units'][0]['text'] += '!'
            selection.write_text(n.canonical(doc)+(' ' if kind == 'whitespace' else ''))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                verify_review(candidate=self.f.root/'out', review=path, **self.inputs())
            selection.write_bytes(before)

    def test_current_image_preview_font_and_license_changes_invalidate_review(self):
        from tralens.rendering.review import verify_review
        _, path = self.make_record()
        for target in (self.f.image, self.f.root/'preview.json', self.f.font,
                       self.f.root/'synthetic-license.txt', self.f.root/'fonts.json'):
            before = target.read_bytes()
            target.write_bytes(before+b' ')
            with self.subTest(target=target.name), self.assertRaises(ValueError):
                verify_review(candidate=self.f.root/'out', review=path, **self.inputs())
            target.write_bytes(before)

    def test_tampered_output_masks_settings_specimens_and_html_are_rejected(self):
        from tralens.rendering.review import verify_review
        _, path = self.make_record()
        for name in ('composite.png', 'unit-000-layout.png', 'unit-000-specimen-text.png', 'review.html', 'manifest.json'):
            target = self.f.root/'out'/name
            before = target.read_bytes()
            target.write_bytes(before+b' ')
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_review(candidate=self.f.root/'out', review=path, **self.inputs())
            target.write_bytes(before)
        target = self.f.root/'out/manifest.json'
        doc = json.loads(target.read_bytes())
        doc['units'][0]['resolved_style']['rotation'] = 8
        target.write_text(n.canonical(doc))
        with self.assertRaisesRegex(ValueError, 'style/layout/text drift'):
            verify_review(candidate=self.f.root/'out', review=path, **self.inputs())

    def test_edited_review_state_or_linguistic_flags_rejected(self):
        from tralens.rendering.review import verify_review
        _, path = self.make_record('rejected')
        before = path.read_bytes()
        for key, value in (('state', 'accepted'), ('linguistic_approval', 'accepted'), ('identity_verified', True)):
            doc = json.loads(before)
            doc[key] = value
            path.write_text(n.canonical(doc))
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_review(candidate=self.f.root/'out', review=path, **self.inputs())
        path.write_bytes(before)

    def test_blocked_and_foreign_units_cannot_receive_review(self):
        from tralens.rendering.review import record_review
        report = self.f.run_render()
        with self.assertRaisesRegex(ValueError, 'foreign/blocked'):
            record_review(candidate=self.f.root/'out', unit_id='foreign', state='rejected', operator='synthetic-test-operator',
                          note='Synthetic test', output=self.f.root/'foreign.json', **self.inputs())
        self.f.selection['units'][0]['style'] = {'min_size': 144, 'max_size': 144, 'outline': 0}
        self.f.write_inputs()
        blocked = self.f.run_render(name='blocked')
        self.assertEqual(blocked['units'][0]['status'], 'blocked')
        with self.assertRaisesRegex(ValueError, 'foreign/blocked'):
            record_review(candidate=self.f.root/'blocked', unit_id=report['units'][0]['unit_id'], state='rework',
                operator='synthetic-test-operator', note='Synthetic test', output=self.f.root/'blocked.json', **self.inputs())

    def test_revision_determinism_fresh_outputs_and_parent_provenance(self):
        from tralens.rendering.review import verify_review
        record, review = self.make_record()
        second = self.f.run_render(name='same')
        self.assertEqual(record['binding']['job_revision'], second['job_revision'])
        self.assertEqual((self.f.root/'out/manifest.json').read_bytes(), (self.f.root/'same/manifest.json').read_bytes())
        with self.assertRaises(ValueError):
            self.f.run_render()
        self.f.selection['units'][0]['style'].update(preset='thought', fill=[80, 30, 100], rotation=5)
        self.f.write_inputs()
        revised = n.run('compose', output=self.f.root/'revised', parent=self.f.root/'out', **self.inputs())
        self.assertNotEqual(revised['job_revision'], second['job_revision'])
        self.assertEqual(revised['parent_candidate']['job_revision'], second['job_revision'])
        self.assertEqual(revised['units'][0]['aesthetic_state'], 'not_assessed')
        with self.assertRaises(ValueError):
            verify_review(candidate=self.f.root/'revised', review=review, **self.inputs())

    def test_explicit_alternative_comparison_is_source_bound(self):
        self.f.run_render()
        self.f.selection['units'][0]['style'].update(preset='thought')
        self.f.write_inputs()
        revised = n.run('compose', output=self.f.root/'compared', compare=[self.f.root/'out'], **self.inputs())
        self.assertEqual(len(revised['comparison_candidates']), 1)
        self.assertTrue((self.f.root/'compared/comparison-0.png').is_file())
        with self.assertRaises(ValueError):
            n.run('compose', output=self.f.root/'too-many', compare=[self.f.root/'out']*5, **self.inputs())
        self.f.selection['units'][0]['style']['rotation'] = 1
        self.f.write_inputs()
        before = (self.f.root/'preview.json').read_bytes()
        (self.f.root/'preview.json').write_bytes(before+b' ')
        with self.assertRaises(ValueError):
            n.run('compose', output=self.f.root/'foreign-comparison', compare=[self.f.root/'out'], **self.inputs())

    def test_closed_artifact_paths_symlinks_and_report_escaping(self):
        from tralens.rendering.review import read_bundle
        face = {'family': '<script src="https://invalid/">untrusted font name</script>',
                'subfamily': None, 'weight_class': None}
        with patch.object(n, 'face_metadata', return_value=face):
            self.f.run_render()
        html = (self.f.root/'out/review.html').read_text()
        self.assertIn('Content-Security-Policy', html)
        self.assertNotIn('<script', html)
        self.assertIn('&lt;script', html)
        self.assertNotIn('src="https://', html)
        path = self.f.root/'out/manifest.json'
        doc = json.loads(path.read_text())
        doc['artifacts']['../not-an-artifact.json'] = '0'*64
        path.write_text(n.canonical(doc))
        with self.assertRaisesRegex(ValueError, 'artifact path/name'):
            read_bundle(self.f.root/'out')

    def test_review_and_candidate_symlinks_and_implementation_changes_reject(self):
        from tralens.rendering import review as r
        record, path = self.make_record()
        link = self.f.root/'review-link.json'
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            r.verify_review(candidate=self.f.root/'out', review=link, **self.inputs())
        candidate_link = self.f.root/'candidate-link'
        candidate_link.symlink_to(self.f.root/'out', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            r.verify_review(candidate=candidate_link, review=path, **self.inputs())
        with patch.object(r, 'implementation_hash', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'implementation/policy'):
                r.verify_review(candidate=self.f.root/'out', review=path, **self.inputs())

    def test_rehashed_counter_tampering_is_rejected_before_record_creation(self):
        from tralens.rendering import review as r
        report = self.f.run_render()
        report['units'][0]['pixel_audit']['untouched_source_pixels'] += 1
        html = r.specimen_html(report).encode()
        (self.f.root/'out/review.html').write_bytes(html)
        report['artifacts']['review.html'] = n.digest(html)
        (self.f.root/'out/manifest.json').write_text(n.canonical(report))
        with self.assertRaisesRegex(ValueError, 'unit locality counters'):
            r.record_review(candidate=self.f.root/'out', unit_id=report['units'][0]['unit_id'], state='accepted',
                operator='synthetic-test-operator', note='Synthetic consistency test', output=self.f.root/'tampered.json', **self.inputs())
        self.assertFalse((self.f.root/'tampered.json').exists())

    def test_actual_cli_review_revision_workflow_and_required_operator_action(self):
        self.f.run_render()
        tool = Path(__file__).resolve().parents[2]/'tools/render_preview.py'
        common = [item for key, value in self.inputs().items() for item in ('--'+key, str(value))]
        def cli(*args):
            return subprocess.run([sys.executable, str(tool), *args, *common], capture_output=True, text=True)
        result = cli('record-review', '--candidate', str(self.f.root/'out'), '--unit-id', self.f.selection['units'][0]['unit_id'],
            '--state', 'accepted', '--operator', 'synthetic-test-operator', '--note', 'Synthetic aesthetic review only',
            '--output', str(self.f.root/'cli-review.json'))
        self.assertEqual(result.returncode, 0, result.stderr)
        result = cli('verify-review', '--candidate', str(self.f.root/'out'), '--review', str(self.f.root/'cli-review.json'))
        self.assertEqual(result.returncode, 0, result.stderr)
        result = cli('record-review', '--candidate', str(self.f.root/'out'))
        self.assertNotEqual(result.returncode, 0)
        self.f.selection['units'][0]['style'].update(preset='narration', fill=[80, 20, 40])
        self.f.write_inputs()
        result = cli('compose', '--parent', str(self.f.root/'out'), '--compare', str(self.f.root/'out'),
                     '--output', str(self.f.root/'cli-revision'))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_report_responsive_contract_wraps_long_ids_without_hiding_content(self):
        from tralens.rendering.review import specimen_html, CSP
        class ReportParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_style = False
                self.css = ''
                self.images = []
                self.scripts = 0
                self.csp = None
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                self.in_style = tag == 'style' or self.in_style
                if tag == 'img':
                    self.images.append(attrs['src'])
                if tag == 'script':
                    self.scripts += 1
                if tag == 'meta' and attrs.get('http-equiv') == 'Content-Security-Policy':
                    self.csp = attrs['content']
            def handle_endtag(self, tag):
                if tag == 'style':
                    self.in_style = False
            def handle_data(self, data):
                if self.in_style:
                    self.css += data
        long_id = 'proposal-'+('0123456789abcdef'*32)+'<untrusted>'
        page = specimen_html({'job_revision': 'a'*512, 'units': [{'unit_id': long_id, 'status': 'blocked',
            'reason': 'b'*512}], 'comparison_candidates': [{'job_revision': 'c'*512}]})
        parsed = ReportParser()
        parsed.feed(page)
        css = ''.join(parsed.css.split())
        self.assertIn('overflow-wrap:anywhere', css)
        self.assertIn('box-sizing:border-box', css)
        self.assertIn('figure{margin:0;min-width:0}', css)
        self.assertIn('.grid>*{min-width:0}', css)
        self.assertIn('@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}', css)
        self.assertIn('@media(max-width:420px){.grid{grid-template-columns:minmax(0,1fr)}}', css)
        self.assertNotIn('overflow:hidden', css)
        self.assertNotIn('overflow-x:hidden', css)
        self.assertNotIn('text-overflow:ellipsis', css)
        self.assertIn('a'*512, page)
        self.assertIn('&lt;untrusted&gt;', page)
        self.assertEqual(parsed.csp, CSP)
        self.assertEqual(parsed.scripts, 0)
        self.assertEqual(len(parsed.images), 10)
        self.assertIn('comparison-0.png', parsed.images)
        self.assertIn('unit-000-specimen-glyph-cleanup.png', parsed.images)

    def test_actual_cleanup_mask_specimen_matches_erasure_and_verifies(self):
        from tralens.rendering.review import verify_candidate
        report = self.f.run_render()
        root = self.f.root/'out'
        row = report['units'][0]
        name = 'unit-000-specimen-glyph-cleanup.png'
        self.assertIn(name, report['artifacts'])
        self.assertIn(name, (root/'review.html').read_text())
        with Image.open(root/'unit-000-glyph-cleanup.png') as glyph, Image.open(root/name) as specimen:
            expected = glyph.resize(tuple(row['specimen']['size']), resample=Image.Resampling.NEAREST,
                                    box=tuple(row['specimen']['source_bounds']))
            self.assertEqual(specimen.mode, 'L')
            self.assertTrue(np.array_equal(np.asarray(specimen), np.asarray(expected)))
            self.assertGreater(np.count_nonzero(np.asarray(specimen)), 0)
        verified, _ = verify_candidate(candidate=root, **self.inputs())
        self.assertEqual(verified['job_revision'], report['job_revision'])

    def test_plan_and_blocked_cleanup_specimens_show_no_applied_erasure(self):
        for mode, name in (('plan', 'plan'), ('compose', 'blocked')):
            if name == 'blocked':
                self.f.selection['units'][0]['style'] = {'min_size': 144, 'max_size': 144, 'outline': 0}
                self.f.write_inputs()
            report = self.f.run_render(mode, name=name)
            with Image.open(self.f.root/name/'unit-000-specimen-glyph-cleanup.png') as specimen:
                self.assertEqual(np.count_nonzero(np.asarray(specimen)), 0)
            self.assertEqual(report['pixel_audit']['cleanup_mask_pixels'], 0)


if __name__ == '__main__':
    unittest.main()
