"""Routing with synthetic probability arrays only; no checkpoint inference/rights claim."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from tralens.ocr.automatic_units import propose_page_from_boxes
from tralens.ocr.types import Text_Box
from tralens.rendering import automatic as a, best_effort, segmentation


class CoverageSemanticRouteTests(unittest.TestCase):
    def setUp(self):
        from tests.test_pipeline import PipelineTests
        self.pipeline=PipelineTests();self.pipeline.setUp();self.addCleanup(self.pipeline.doCleanups)
        fixture=self.pipeline.fixture
        boxes=[Text_Box(b.paddle_box_id,.98,b.paddle_poly,b.paddle_text) for b in fixture.boxes]
        batch=propose_page_from_boxes(boxes,fixture.image,complementary_ownership=True,regroup_companions=True)
        self.coverage_id=batch.proposals[1].proposal_id
        proposals=list(batch.proposals)
        proposals[1]=replace(proposals[1],structural_reasons=proposals[1].structural_reasons+('unknown_region_ownership',))
        fixture.engine.raw_detections=[boxes]
        fixture.engine.automatic_unit_proposals=[replace(batch,proposals=tuple(proposals))]
        self.strict_members=list(proposals[0].member_ids)
        self.coverage_members=list(proposals[1].member_ids)
        self.forced_members=[self.coverage_members]
        self.original_derive=a.derive_plan
        self.model=self.pipeline.root/'synthetic-only.onnx'
        self.model.write_bytes(b'Not a model; test injection only.')
        self.spec={'path':str(self.model),'scope':'synthetic fixture, not deployment clearance'}
        with Image.open(fixture.image) as image:
            pixels=np.asarray(image.convert('RGB'))
            self.probability=np.where(pixels.min(axis=2)<180,245,0).astype(np.uint8)

    def derive(self,rgb,detections,member_ids,**kwargs):
        if member_ids in self.forced_members:raise a.AutomaticSkip('synthetic_native_failure','exercise optional evidence route')
        return self.original_derive(rgb,detections,member_ids,**kwargs)

    def run_case(self,name,probability=None,configured=True):
        backend=Mock(return_value=SimpleNamespace(raw=self.probability if probability is None else probability,
            evidence={'forwards':1,'scope':'synthetic probability array; no learned inference'}))
        with patch.object(segmentation,'model_spec',return_value=self.spec), \
             patch.object(segmentation,'TextMask',backend), \
             patch.object(a,'derive_plan',side_effect=self.derive), \
             patch.object(best_effort,'native_plan',side_effect=a.AutomaticSkip('synthetic_native_failure','exercise semantic route')):
            report=self.pipeline.run_pipeline(name,render_policy='best-effort',replacement_policy='confidence-gated',
                **({'text_mask_model':self.model} if configured else {}))
        manifest=json.loads((self.pipeline.root/name/'rendering/manifest.json').read_bytes())
        return report,manifest,backend

    def test_explicit_evidence_reaches_uncertain_region_after_prior_strict_edit(self):
        before,_,_=self.run_case('without-model',configured=False)
        after,manifest,backend=self.run_case('with-model')
        self.assertEqual(before['coverage']['rendered'],1,before)
        self.assertEqual(after['coverage']['rendered'],2,after)
        self.assertEqual(manifest['coverage']['coverage_rendered'],1)
        backend.assert_called_once()
        row=next(row for row in manifest['units'] if row['unit_id']==self.coverage_id)
        self.assertEqual(row['cleanup_method'],'flat_fill_semantic')
        self.assertEqual(row['text_mask_method'],segmentation.POLICY)
        self.assertEqual(row['status'],'automatic_composed')
        self.assertEqual(manifest['text_mask_inference']['forwards'],1)
        for suffix in ('foreground','applied-erase','text'):
            name=f'unit-000-{suffix}.png'
            self.assertEqual((self.pipeline.root/'without-model/rendering'/name).read_bytes(),
                             (self.pipeline.root/'with-model/rendering'/name).read_bytes())
        self.assertEqual(manifest['pixel_audit']['protected_composite_changed_pixels'],0)

    def test_unsupported_probability_does_not_authorize_coarse_erasure(self):
        report,manifest,backend=self.run_case('unsupported',np.zeros_like(self.probability))
        self.assertEqual(report['coverage']['rendered'],1,report)
        backend.assert_called_once()
        row=next(row for row in manifest['units'] if row['unit_id']==self.coverage_id)
        self.assertEqual(row['reason_code'],'replacement_cleanup_confidence')
        self.assertTrue(any(attempt['method']=='explicit_semantic_mask' and attempt['status']=='failed'
                            for attempt in row['best_effort_attempts']))

    def test_without_explicit_model_backend_is_never_constructed(self):
        report,manifest,backend=self.run_case('not-configured',configured=False)
        backend.assert_not_called()
        self.assertEqual(report['coverage']['rendered'],1,report)
        self.assertNotIn('text_mask_model',manifest)

    def all_coverage(self):
        engine=self.pipeline.fixture.engine;batch=engine.automatic_unit_proposals[0]
        engine.automatic_unit_proposals=[replace(batch,proposals=tuple(replace(p,
            structural_reasons=tuple(dict.fromkeys(p.structural_reasons+('unknown_region_ownership',))))
            for p in batch.proposals))]
        self.forced_members=[list(p.member_ids) for p in batch.proposals]

    def test_multiple_uncertain_units_share_one_page_backend(self):
        self.all_coverage()
        report,manifest,backend=self.run_case('shared')
        self.assertEqual(report['coverage']['rendered'],2,report)
        self.assertEqual(manifest['coverage']['coverage_rendered'],2)
        backend.assert_called_once()

    def test_failed_backend_is_cached_and_translations_are_retained(self):
        self.all_coverage()
        with patch.object(segmentation,'model_spec',return_value=self.spec), \
             patch.object(segmentation,'TextMask',side_effect=ValueError('synthetic backend failure')) as backend, \
             patch.object(a,'derive_plan',side_effect=self.derive), \
             patch.object(best_effort,'native_plan',side_effect=a.AutomaticSkip('synthetic_native_failure','fixture')):
            report=self.pipeline.run_pipeline('failed-backend',render_policy='best-effort',
                replacement_policy='confidence-gated',text_mask_model=self.model)
        backend.assert_called_once()
        self.assertEqual(report['coverage']['total_translations'],2,report)
        self.assertEqual(report['coverage']['rendered'],0,report)
        self.assertIsNone(report['error'])


if __name__=='__main__':unittest.main()
