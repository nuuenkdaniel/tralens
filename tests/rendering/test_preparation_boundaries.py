"""Local outline counterexamples and read-only optional saved-page measurement."""
from io import BytesIO
from pathlib import Path
import json
import unittest

import numpy as np
from PIL import Image, ImageDraw

from tralens.rendering import natural as n
from tralens.rendering import preparation as a
from tests.local_evidence import local_evidence
from . import test_natural as fixtures


def framed_pair(touch=False, seam=False, gap=45):
    image = Image.new('RGB', (300, 180), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, 296, 176), outline='black', width=2)
    left = (25, 35, 115, 145)
    right = (115 if touch else 115+gap, 35, 250 if not touch else 205, 145)
    draw.rectangle(left, outline='black', width=2)
    draw.rectangle(right, outline='black', width=2)
    # Both outlines join the same distant panel frame via narrow artwork lines.
    draw.line((40, 35, 40, 3), fill='black', width=2)
    draw.line((right[0]+20, 35, right[0]+20, 3), fill='black', width=2)
    if seam:
        draw.line((115, 80, right[0], 80), fill='black', width=2)
    draw.rectangle((55, 65, 65, 100), fill='black')
    draw.rectangle((right[0]+25, 65, right[0]+35, 100), fill='black')
    detections = [{'source_id': 'left', 'polygon': fixtures.rect(50, 60, 70, 110)},
                  {'source_id': 'right', 'polygon': fixtures.rect(right[0]+20, 60, right[0]+40, 110)}]
    return image, detections


class LocalBoundaryTests(unittest.TestCase):
    def test_independent_balloons_sharing_remote_frame_are_supported(self):
        image, detections = framed_pair()
        for member in ('left', 'right'):
            result = a.propose_geometry(image, detections, [member])
            self.assertEqual(result['status'], 'proposed', result)
            boundary = result['measured']['outline_adjacency']
            self.assertGreater(boundary['global_closed_contact_count'], 0)
            self.assertEqual(boundary['local_closed_contact_count'], 0)
            self.assertGreater(boundary['ignored_nonlocal_contact_count'], 0)
            geometry = result['geometry']
            masks = [n.union_mask(geometry[key], image.size) for key in ('interior', 'cleanup_permission', 'protected')]
            clean, glyph, _ = n.cleanup(image, *masks)
            self.assertTrue(np.array_equal(np.asarray(image)[~glyph], np.asarray(clean)[~glyph]))

    def test_actual_shared_seam_is_rejected_even_with_remote_frame(self):
        image, detections = framed_pair(touch=True)
        for member in ('left', 'right'):
            result = a.propose_geometry(image, detections, [member])
            self.assertEqual(result['status'], 'rejected')
            self.assertGreater(result['measured']['outline_adjacency']['local_closed_contact_count'], 0)

    def test_near_local_art_connection_stays_rejected(self):
        image, detections = framed_pair(seam=True, gap=5)
        result = a.propose_geometry(image, detections, ['left'])
        self.assertEqual(result['status'], 'rejected')
        self.assertGreater(result['measured']['outline_adjacency']['local_closed_contact_count'], 0)

    def test_near_but_separate_outlines_are_not_a_shared_seam(self):
        image, detections = framed_pair(gap=5)
        result = a.propose_geometry(image, detections, ['left'])
        self.assertEqual(result['status'], 'proposed', result)
        self.assertEqual(result['measured']['outline_adjacency']['local_closed_contact_count'], 0)

    def test_shared_interior_still_rejects_known_nonmember_source(self):
        image, detections = framed_pair()
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 30, 255, 150), fill='white', outline='black', width=2)
        draw.rectangle((55, 65, 65, 100), fill='black')
        draw.rectangle((185, 65, 195, 100), fill='black')
        for member in ('left', 'right'):
            result = a.propose_geometry(image, detections, [member])
            self.assertEqual(result['status'], 'rejected')
            self.assertIn('nonmember', result['reason'])

    def test_empty_edge_scanline_trim_keeps_required_pixels_and_rejects_internal_gaps(self):
        safe = np.zeros((90, 110), bool)
        safe[10:80, 10:100] = True
        safe[10, 45:55] = False
        required = np.zeros_like(safe)
        required[25:60, 25:80] = True
        actual, measured = a.trim_empty_boundary_spans(safe, required, 3)
        self.assertEqual(measured['trimmed_top_bottom_left_right'], [1, 0, 0, 0])
        self.assertEqual(measured['required_pixels_discarded'], 0)
        self.assertFalse((actual & ~safe).any())
        self.assertFalse((required & ~actual).any())
        required[10, 20] = True
        with self.assertRaises(a.Unsupported):
            a.trim_empty_boundary_spans(safe, required, 3)
        required[10, 20] = False
        safe[35, 45:55] = False
        with self.assertRaises(a.Unsupported):
            a.trim_empty_boundary_spans(safe, required, 3)

    def test_excessive_boundary_trim_and_unknown_holes_are_not_filled(self):
        safe = np.ones((70, 90), bool)
        safe[:4, 40:45] = False
        required = np.zeros_like(safe)
        required[20:40, 20:60] = True
        with self.assertRaises(a.Unsupported):
            a.trim_empty_boundary_spans(safe, required, 3)
        safe[:4, 40:45] = True
        safe[30:35, 40:45] = False
        with self.assertRaises(a.Unsupported):
            a.polygons_inside(safe, required)

    def test_pixel_vertex_contacts_split_without_expansion_and_stay_bounded(self):
        points = [[2, 2], [2, 8], [7, 8], [7, 2], [12, 8], [17, 8], [17, 2]]
        with self.assertRaises(ValueError):
            n.polygon_mask(points, (22, 14))
        polygons = a.split_pixel_contour(points, [0])
        self.assertEqual(len(polygons), 2)
        actual = n.union_mask(polygons, (22, 14))
        expected = Image.new('L', (22, 14))
        ImageDraw.Draw(expected).polygon([tuple(p) for p in points], fill=255)
        self.assertTrue(np.array_equal(actual, np.asarray(expected) != 0))
        with self.assertRaises(n.ResourceLimit):
            a.split_pixel_contour(points, [4_000_000])


class SavedPreparationBoundaryEvidence(unittest.TestCase):
    @local_evidence(files=('test-results/render-preparation-parent-v1/proposals.json',
        'test-results/workflow-fragments-priority-parent-v1/preview.json',
        'test-results/render-parent-input-v1/fonts.json', 'images/ubunchu-ja-test/ubunchu01_04.png'))
    def test_original_page_readonly_counterexamples_and_follow_on_guards(self):
        root = Path(__file__).resolve().parents[2]
        preview_path = root/'test-results/workflow-fragments-priority-parent-v1/preview.json'
        image_path = root/'images/ubunchu-ja-test/ubunchu01_04.png'
        fonts_path = root/'test-results/render-parent-input-v1/fonts.json'
        bundle = n.document(n.read_explicit(root/'test-results/render-preparation-parent-v1/proposals.json'))
        preview_bytes, image_bytes, fonts_bytes = [n.read_explicit(path) for path in (preview_path, image_path, fonts_path)]
        preview = n.document(preview_bytes)
        self.assertEqual(bundle['source']['sha256'], n.digest(image_bytes))
        self.assertEqual(bundle['preview_sha256'], n.digest(preview_bytes))
        self.assertEqual(bundle['fonts_sha256'], n.digest(fonts_bytes))
        with Image.open(BytesIO(image_bytes)) as original:
            rgb = original.convert('RGB')
        context = a.OriginalRegions(rgb)
        results = []
        for row in bundle['units']:
            binding = row['selection_binding']
            # Full renderer contract validation; no local-font lookup or layout,
            # output artifacts, source writes, providers or human-review flags.
            contract = {'schema_version': 1, 'purpose': 'candidate_only', 'source': bundle['source'],
                'preview_sha256': n.digest(preview_bytes), 'fonts_sha256': n.digest(fonts_bytes),
                'units': [{**binding, 'geometry_intent': 'Read-only original-pixel diagnostic; no confirmation.'}]}
            n.bind(preview, image_path, rgb, image_bytes, preview_bytes, contract, fonts_bytes)
            result = a.propose_geometry(rgb, preview['ocr']['raw_detections'][0], binding['member_ids'],
                                       bundle['request']['settings'], context=context)
            if result['status'] == 'proposed':
                geometry = result['geometry']
                masks = [n.union_mask(geometry[key], rgb.size) for key in ('interior', 'cleanup_permission', 'protected')]
                clean, glyph, _ = n.cleanup(rgb, *masks)
                self.assertTrue(np.array_equal(np.asarray(rgb)[~glyph], np.asarray(clean)[~glyph]))
                self.assertLessEqual(len(geometry['cleanup_permission']), 128)
                if 'boundary_span_trim' in result['measured']:
                    self.assertEqual(int(glyph.sum()), result['measured']['boundary_span_trim']['full_original_core_cleanup_pixels'])
                    self.assertEqual(result['measured']['boundary_span_trim']['required_pixels_discarded'], 0)
            boundary = result['measured'].get('outline_adjacency')
            if boundary is not None:
                boundary = {**boundary, 'contacts_sample': boundary['contacts_sample'][:4]}
            component = result['measured'].get('component_evidence')
            if component is not None:
                component = {key: value if not isinstance(value, dict) else
                    {k: v for k, v in value.items() if k in ('offending_count', 'offending_bounds')}
                    for key, value in component.items()}
            results.append({'unit_id': row['unit_id'], 'status': result['status'], 'reason': result.get('reason'),
                'outline_adjacency': boundary, 'component_evidence': component,
                'nonmember_contact': result['measured'].get('nonmember_contact'),
                'span_evidence': result['measured'].get('span_evidence'),
                'boundary_span_trim': result['measured'].get('boundary_span_trim'),
                'polygonization_stage': result['measured'].get('polygonization_stage'),
                'cleanup_diagnostics': result.get('cleanup_diagnostics')})
        summaries = []
        for row in results:
            boundary = row['outline_adjacency'] or {}
            components = row['component_evidence'] or {}
            diagnostic = row['cleanup_diagnostics'] or {}
            summaries.append({'unit_id': row['unit_id'], 'status': row['status'], 'reason': row['reason'],
                'boundary_counts': {key: boundary.get(key) for key in ('global_closed_contact_count', 'local_closed_contact_count',
                    'ignored_nonlocal_contact_count', 'unassigned_external_contact_count')},
                'contact_examples': boundary.get('contacts_sample', [])[:2],
                'owned_components': components.get('owned_component_count'),
                'unexplained_inside': components.get('unexplained_inside'),
                'nonmember': {k: v for k, v in (row['nonmember_contact'] or {}).items() if k in ('source_id', 'offending_count', 'offending_bounds')},
                'boundary_span_trim': row['boundary_span_trim'], 'polygonization_stage': row['polygonization_stage'],
                'cleanup_offending': diagnostic.get('offending_coordinates')})
        print('READ_ONLY_PREPARATION_EVIDENCE='+json.dumps(summaries, sort_keys=True))
        self.assertTrue(any(row['status'] == 'proposed' for row in results), 'No useful saved-page positive; investigate follow-on blockers before handoff')
