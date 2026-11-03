"""Automatic machine policy: original tight masks, no confirmation or page assets."""
from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import natural as n
from . import test_natural as fixtures
from tests.local_evidence import local_evidence


def shared_source(color=(255, 255, 255)):
    image = Image.new('RGB', (280, 180), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 260, 165), outline='black', width=2)
    detections = []
    for name, x in (('left', 55), ('right', 190)):
        for y in (50, 80, 110):
            draw.rectangle((x, y, x+9, y+8), fill='black')
        detections.append({'source_id': name, 'polygon': fixtures.rect(x-5, 40, x+15, 135)})
    return image, detections


class AutomaticMaskTests(unittest.TestCase):
    def test_independent_tight_masks_in_shared_white_and_offwhite_region(self):
        from tralens.rendering.automatic import derive_plan
        for color in ((255, 255, 255), (241, 239, 233)):
            image, detections = shared_source(color)
            plans = [derive_plan(image, detections, [name]) for name in ('left', 'right')]
            self.assertFalse((plans[0].erase & plans[1].erase).any())
            self.assertFalse((plans[0].layout & plans[1].layout).any())
            for plan in plans:
                self.assertEqual(plan.background, list(color))
                self.assertEqual(plan.evidence['dark_components'], 3)
                self.assertEqual(plan.evidence['whole_component_spill_pixels'], 0)
                self.assertTrue((plan.foreground <= plan.erase).all())
                plate = np.asarray(image).copy()
                plate[plan.erase] = color
                self.assertTrue(np.array_equal(plate[~plan.erase], np.asarray(image)[~plan.erase]))

    def test_border_art_spill_gradient_and_annotation_refuse(self):
        from tralens.rendering.automatic import derive_plan, AutomaticSkip
        for kind in ('border', 'spill', 'gradient', 'annotation', 'colored'):
            image, detections = shared_source()
            draw = ImageDraw.Draw(image)
            if kind == 'border':
                draw.line((16, 53, 60, 53), fill='black', width=2)
            elif kind == 'spill':
                draw.line((60, 53, 95, 53), fill='black', width=2)
            elif kind == 'gradient':
                for x in range(45, 76):
                    draw.line((x, 43, x, 130), fill=(180+x,)*3)
            elif kind == 'annotation':
                detections.append({'source_id': 'annotation', 'polygon': fixtures.rect(63, 45, 70, 57)})
            else:
                draw.rectangle((56, 50, 60, 55), fill=(220, 20, 20))
            with self.subTest(kind=kind), self.assertRaises(AutomaticSkip):
                derive_plan(image, detections, ['left'])

    def test_antialias_is_measured_and_unassigned_gray_punctuation_is_not_ignored(self):
        from tralens.rendering.automatic import derive_plan, AutomaticSkip
        image, detections = shared_source()
        ImageDraw.Draw(image).line((54, 50, 54, 58), fill=(220, 220, 220))
        plan = derive_plan(image, detections, ['left'])
        self.assertTrue(plan.foreground[52, 54])
        self.assertTrue(plan.erase[52, 54])
        image.putpixel((62, 130), (220, 220, 220))
        with self.assertRaises(AutomaticSkip):
            derive_plan(image, detections, ['left'])

    def test_background_color_and_nearby_punctuation_clearance_negatives(self):
        from tralens.rendering.automatic import derive_plan, AutomaticSkip
        for color in ((90, 90, 90), (220, 230, 255)):
            image, detections = shared_source(color)
            with self.assertRaises(AutomaticSkip):
                derive_plan(image, detections, ['left'])
        image, detections = shared_source()
        # Different source annotation/ink in the local sampling band cannot be
        # silently captured as background or erased by a seed-box rectangle.
        ImageDraw.Draw(image).point((52, 50), fill='black')
        detections.append({'source_id': 'punctuation', 'polygon': fixtures.rect(50, 48, 54, 52)})
        with self.assertRaises(AutomaticSkip):
            derive_plan(image, detections, ['left'])

    def test_coincident_nonmember_polygon_is_not_lost_by_union_subtraction(self):
        from tralens.rendering.automatic import derive_plan, AutomaticSkip
        image, detections = shared_source()
        detections.append({'source_id': 'overlap', 'polygon': detections[0]['polygon']})
        with self.assertRaises(AutomaticSkip):
            derive_plan(image, detections, ['left'])

    def test_source_side_budget_is_checked_before_component_allocation(self):
        from tralens.rendering.automatic import OriginalComponents
        source = Image.new('RGB', (4097, 4), 'white')
        with patch.object(n.cv2, 'connectedComponentsWithStats', side_effect=AssertionError('must not allocate labels')):
            with self.assertRaises(n.ResourceLimit):
                OriginalComponents(source)


class AutomaticWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CommandTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def inputs(self):
        return dict(preview=self.f.root/'preview.json', image=self.f.image, fonts=self.f.root/'fonts.json')

    def test_machine_selection_produces_output_without_selection_or_confirmation(self):
        from tralens.rendering.automatic import run_automatic
        before = self.f.image.read_bytes()
        report = run_automatic(**self.inputs(), output=self.f.root/'auto', min_size=14, max_size=28)
        self.assertEqual(report['purpose'], 'automatic_candidate_output')
        self.assertEqual(report['coverage']['rendered'], 1)
        self.assertEqual(len(report['units']), 2)
        self.assertIsNone(report['operator_confirmation'])
        self.assertEqual(report['linguistic_approval'], 'not_assessed')
        self.assertFalse(report['approved_export'])
        self.assertEqual(report['pixel_audit']['outside_cleanup_changed_pixels'], 0)
        self.assertEqual(report['pixel_audit']['outside_alpha_changed_pixels'], 0)
        self.assertEqual(self.f.image.read_bytes(), before)
        self.assertTrue((self.f.root/'auto/clean.png').is_file())
        self.assertTrue((self.f.root/'auto/composite.png').is_file())
        self.assertFalse((self.f.root/'auto/selection.json').exists())

    def run_auto(self, name='auto', **options):
        from tralens.rendering.automatic import run_automatic
        options.setdefault('render_policy', 'conservative')
        return run_automatic(**self.inputs(), output=self.f.root/name, min_size=14, max_size=28, **options)

    def make_other_strict(self):
        from dataclasses import asdict
        from tralens.translation.contracts import TranslationResult, UnitOutcome
        row = next(r for r in self.f.preview['units'] if r is not self.f.selected_record)
        uid = row['unit']['unit_id']
        outcome = UnitOutcome(uid, 'preview', 'Hi!', (), 'unverified', row['unit']['selected_revision'])
        result = TranslationResult(row['request_revision'], 'preview', ((uid, 'Hi!'),), outcomes=(outcome,))
        row.update(status='preview', input_kind='strict_utterance', result=asdict(result))
        self.f.write_inputs()

    def test_all_strict_candidates_and_explicit_evaluation_cap(self):
        self.make_other_strict()
        complete = self.run_auto()
        self.assertEqual(complete['coverage']['strict_eligible'], 2)
        self.assertEqual(complete['coverage']['rendered'], 2)
        self.assertTrue(all(r['machine_selected'] for r in complete['units']))
        limited = self.run_auto(name='limited', max_units=1)
        self.assertEqual(limited['coverage']['rendered'], 1)
        self.assertEqual(limited['units'][1]['reason_code'], 'evaluation_cap')
        self.assertTrue(limited['coverage']['partial_output'])

    def test_stale_result_fragment_and_vlm_paths_are_never_promoted(self):
        for kind in ('request', 'fragment', 'alternate', 'blocked'):
            before = json.loads(n.canonical(self.f.preview))
            row = self.f.selected_record
            if kind == 'request':
                row['request_revision'] = '0'*64
            elif kind == 'fragment':
                row['input_kind'] = 'unverified_text_fragment'
            elif kind == 'alternate':
                row['input_kind'] = 'unverified_visual_alternate'
            else:
                row['status'] = 'blocked'
            row['fragment_previews'] = [{'status': 'fragment_preview', 'text': 'not selected'}]
            self.f.write_inputs()
            report = self.run_auto(name=kind)
            self.assertEqual(report['coverage']['rendered'], 0)
            self.assertEqual(report['pixel_audit']['untouched_source_pixels'], 180*130)
            self.f.preview = before
            self.f.selected_record = next(r for r in self.f.preview['units'] if r['unit']['unit_id'] == self.f.selection['units'][0]['unit_id'])
        self.f.write_inputs()

    def test_source_font_license_and_raw_pixel_drift_reject_before_output(self):
        for target in (self.f.image, self.f.font, self.f.root/'synthetic-license.txt'):
            before = target.read_bytes()
            target.write_bytes(before+b' ')
            with self.subTest(name=target.name), self.assertRaises(ValueError):
                self.run_auto(name=target.name+'-out')
            target.write_bytes(before)
        self.f.preview['input']['pixels_sha256'] = '0'*64
        self.f.write_inputs()
        with self.assertRaises(ValueError):
            self.run_auto()

    def test_overflow_missing_coverage_and_resource_failures_do_not_erase(self):
        from tralens.rendering import automatic as auto
        from tralens.rendering.automatic import run_automatic
        report = run_automatic(**self.inputs(), output=self.f.root/'overflow', min_size=144, max_size=144, render_policy='conservative')
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)
        actual_loader = n.load_fonts
        def missing(manifest):
            result = actual_loader(manifest)
            return {key: (data, coverage-{ord('H')}) for key, (data, coverage) in result.items()}
        with patch.object(n, 'load_fonts', missing):
            report = self.run_auto(name='coverage')
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertIn('coverage', next(r['reason'] for r in report['units'] if r['machine_selected']))
        with patch.object(auto, 'MAX_SLOTS', 0):
            report = self.run_auto(name='budget')
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['clean_changed_pixels'], 0)

    def test_collision_rejects_both_without_reading_order_priority(self):
        from tralens.rendering import automatic as auto
        self.make_other_strict()
        with Image.open(self.f.image) as image:
            rgb = image.convert('RGB')
        raw = self.f.preview['ocr']['raw_detections'][0]
        plan = auto.derive_plan(rgb, json.loads(n.canonical(raw)), ['main'])
        with patch.object(auto, 'derive_plan', return_value=plan):
            report = self.run_auto()
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertTrue(all(r['reason_code'] == 'region_collision' for r in report['units']))
        self.assertEqual(report['pixel_audit']['untouched_source_pixels'], 180*130)

    def test_runtime_composite_locality_violation_is_not_published(self):
        original = Image.alpha_composite
        def bad_composite(*args):
            image = original(*args)
            image.putpixel((179, 129), (255, 0, 0, 255))
            return image
        with patch.object(n.Image, 'alpha_composite', bad_composite):
            report = self.run_auto()
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertEqual(report['pixel_audit']['untouched_source_pixels'], 180*130)

    def test_final_combination_alpha_tampering_rolls_back_to_original(self):
        original = Image.Image.alpha_composite
        def bad_combination(image, *args, **kwargs):
            original(image, *args, **kwargs)
            image.putpixel((179, 129), (255, 0, 0, 255))
        with patch.object(Image.Image, 'alpha_composite', bad_combination):
            report = self.run_auto()
        self.assertEqual(report['coverage']['rendered'], 0)
        self.assertGreater(report['rejected_alpha_integrity']['disagreement_pixels'], 0)
        self.assertEqual(report['alpha_integrity']['actual_pixels'], 0)
        self.assertEqual(report['pixel_audit']['untouched_source_pixels'], 180*130)

    def test_fresh_output_symlinks_determinism_and_hash_inventory(self):
        one, two = self.run_auto(), self.run_auto(name='again')
        self.assertEqual(one, two)
        for name, expected in one['artifacts'].items():
            self.assertEqual(n.digest((self.f.root/'auto'/name).read_bytes()), expected)
        self.assertEqual((self.f.root/'auto/manifest.json').read_bytes(), (self.f.root/'again/manifest.json').read_bytes())
        with self.assertRaises(ValueError):
            self.run_auto()
        (self.f.root/'link').symlink_to(self.f.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_auto(name='link/output')
        from tralens.rendering.automatic import run_automatic
        link = self.f.root/'image-link.png'
        link.symlink_to(self.f.image)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            run_automatic(preview=self.inputs()['preview'], image=link, fonts=self.inputs()['fonts'], output=self.f.root/'bad-source')

    def test_actual_cli_needs_no_selection_and_refuses_operator_or_mask_inputs(self):
        tool = Path(__file__).resolve().parents[2]/'tools/render_preview.py'
        common = [value for key, path in self.inputs().items() for value in ('--'+key, str(path))]
        result = subprocess.run([sys.executable, str(tool), 'auto-compose', *common, '--output', str(self.f.root/'cli')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.f.root/'cli/manifest.json').read_text())
        self.assertEqual(report['coverage']['rendered'], 1)
        for option, value in (('--operator', 'not used'), ('--selection', str(self.f.root/'selection.json')), ('--unit-id', 'no per-unit selection')):
            result = subprocess.run([sys.executable, str(tool), 'auto-compose', *common, '--output', str(self.f.root/'forbidden'), option, value], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
        page = (self.f.root/'cli/review.html').read_text()
        self.assertIn('Content-Security-Policy', page)
        self.assertNotIn('<script', page)
        self.assertNotIn('src="http', page)

    def test_production_import_and_automatic_execution_do_not_load_engines_or_providers(self):
        root = Path(__file__).resolve().parents[2]
        script = '''
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'paddleocr','paddle','ollama','transformers','torch','requests','httpx'} or fullname == 'tralens.ocr.detection':
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Guard())
from tralens.rendering.automatic import run_automatic
run_automatic(preview=sys.argv[2],image=sys.argv[3],fonts=sys.argv[4],output=sys.argv[5])
'''
        result = subprocess.run([sys.executable, '-c', script, str(root/'src'), str(self.inputs()['preview']), str(self.f.image),
            str(self.inputs()['fonts']), str(self.f.root/'guarded')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class SavedTightMaskEvidence(unittest.TestCase):
    @local_evidence(files=('test-results/automatic-mask-spike-v1/measurements.json',
        'test-results/workflow-fragments-priority-parent-v1/preview.json',
        'images/ubunchu-ja-test/ubunchu01_04.png', 'test-results/render-parent-input-v1/fonts.json'))
    def test_spike_source_units_have_measured_automatic_plans_readonly(self):
        from io import BytesIO
        from tralens.rendering.automatic import OriginalComponents, derive_plan, AutomaticSkip
        root = Path(__file__).resolve().parents[2]
        pbytes = n.read_explicit(root/'test-results/workflow-fragments-priority-parent-v1/preview.json')
        ibytes = n.read_explicit(root/'images/ubunchu-ja-test/ubunchu01_04.png')
        fbytes = n.read_explicit(root/'test-results/render-parent-input-v1/fonts.json')
        preview = n.document(pbytes)
        spike = n.document(n.read_explicit(root/'test-results/automatic-mask-spike-v1/measurements.json'))
        wanted = {r['source_id']: r for r in spike}
        with Image.open(BytesIO(ibytes)) as image:
            rgb = image.convert('RGB')
        machine = {'schema_version': 1, 'purpose': 'automatic_candidate_only', 'source': preview['input'],
                   'preview_sha256': n.digest(pbytes), 'fonts_sha256': n.digest(fbytes), 'units': []}
        n.bind(preview, root/'images/ubunchu-ja-test/ubunchu01_04.png', rgb, ibytes, pbytes, machine, fbytes,
               selection_kind='automatic_candidate_only')
        context = OriginalComponents(rgb)
        units = n.automatic_units(preview['ocr']['batches'][0])
        results, found = [], set()
        for record, unit in zip(preview['units'], units):
            if record['status'] != 'preview' or record.get('input_kind', 'strict_utterance') != 'strict_utterance':
                continue
            binding = n.strict_result_binding(record, unit)
            try:
                plan = derive_plan(rgb, preview['ocr']['raw_detections'][0], binding['member_ids'], context=context)
                results.append({'members': binding['member_ids'], 'status': 'safe_plan', 'evidence': plan.evidence})
                for source_id in unit.member_ids:
                    if source_id in wanted:
                        self.assertEqual(plan.evidence['dark_components'], wanted[source_id]['components'])
                        found.add(source_id)
            except AutomaticSkip as error:
                results.append({'members': binding['member_ids'], 'status': 'skipped', 'code': error.code,
                                'reason': str(error), 'evidence': error.evidence})
        summaries = []
        for row in results:
            evidence = row['evidence']
            summaries.append({'members': row['members'], 'status': row['status'], 'code': row.get('code'),
                'reason': row.get('reason'), **{key: evidence.get(key) for key in
                ('dark_components', 'whole_component_spill_pixels', 'dark_pixels', 'foreground_pixels',
                 'antialias_pixels', 'erase_pixels', 'layout_bounds', 'background_rgb', 'unassigned')}})
        print('READ_ONLY_AUTOMATIC_TIGHT_MASK='+json.dumps(summaries, sort_keys=True))
        self.assertEqual(found, set(wanted), 'No useful automatic plans for the measured spike units; inspect actual blocker')


if __name__ == '__main__':
    unittest.main()
