"""Portable original-pixel vertical grouping, without artwork or model calls."""
from dataclasses import asdict, replace
from io import BytesIO
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image, ImageDraw

from tralens.ocr import quality_groups as g, quality_reread as q
from tralens.ocr.automatic_units import propose_automatic_units, propose_page_from_boxes, crop_conflicts
from tralens.ocr.recognition_units import PageProvenance, snapshot_detections, image_fingerprint
from tralens.ocr.types import Text_Box
from tralens.translation.contracts import automatic_units
from tralens.translation.coverage import rejection_reasons


class VerticalGroupTests(unittest.TestCase):
    def fixture(self, *, short=True, third=False):
        image=Image.new('RGB',(370,270),'white')
        draw=ImageDraw.Draw(image)
        draw.rectangle((30,20,340,240),outline='black',width=3)
        boxes=[]
        for index,x in enumerate((90,126,162,202,50) if third else (90,126,162,202)):
            bottom=94 if short and index==3 else 180
            boxes.append(Text_Box(index,.96,[[x,60],[x+26,60],[x+26,bottom],[x,bottom]],'あ' if bottom==94 else 'いうえ'))
            for y in range(65,bottom-10,30):
                draw.rectangle((x+4,y,x+19,y+15),fill='black')
        content=BytesIO();image.save(content,format='PNG')
        page=PageProvenance('synthetic','synthetic.png',hashlib.sha256(content.getvalue()).hexdigest(),
                            '0'*64,image_fingerprint(image))
        batch=propose_automatic_units(snapshot_detections(boxes,page),image)
        template=batch.proposals[0]
        def proposal(ids,direction='vertical'):
            points=[point for i in ids for point in boxes[i].paddle_poly]
            xs,ys=zip(*points);bounds=(min(xs),min(ys),max(xs),max(ys))
            return replace(template,proposal_id='fixture-'+str(ids),main_ids=ids,member_ids=ids,annotation_ids=(),
                annotation_parents=(),direction=direction,crop_bounds=bounds,crop_conflicts=(),crop_safe=True,
                structural_reasons=('automatic_not_human_reviewed','unknown_region_ownership'),word_reasons=())
        proposals=[proposal((2,1,0)),proposal((3,),'unknown' if short else 'vertical')]
        if third:proposals.append(proposal((4,)))
        batch=replace(batch,proposals=tuple(proposals),regions=())
        return image,batch

    def group(self,image,batch):
        return g.vertical_groups(batch,q.enclosure_map(image,batch.detections),image)

    def test_short_leading_column_joins_exactly_once_in_right_to_left_order(self):
        image,batch=self.fixture();before=asdict(batch);pixels=image.tobytes()
        updated,records=self.group(image,batch)
        self.assertEqual(len(updated.proposals),1)
        merged=updated.proposals[0]
        self.assertEqual(merged.main_ids,(3,2,1,0))
        self.assertEqual(automatic_units(updated)[0].selected.text,'あいうえいうえいうえ')
        self.assertEqual(asdict(batch),before)
        self.assertEqual(image.tobytes(),pixels)
        self.assertEqual(updated.detections,batch.detections)
        self.assertEqual(updated.evidence_sha256,batch.evidence_sha256)
        self.assertFalse(records[0]['erase_permission'])
        self.assertIn(g.REASON,merged.structural_reasons)
        self.assertFalse(rejection_reasons(automatic_units(updated)[0]))
        self.assertEqual(self.group(image,updated)[0],updated)

    def test_full_column_continuation_and_proposal_order_invariance(self):
        image,batch=self.fixture(short=False)
        updated,records=self.group(image,batch)
        reversed_batch=replace(batch,proposals=tuple(reversed(batch.proposals)))
        other,_=self.group(image,reversed_batch)
        self.assertEqual(len(records),1)
        self.assertEqual(updated.proposals[0].proposal_id,other.proposals[0].proposal_id)
        self.assertEqual(updated.proposals[0].main_ids,other.proposals[0].main_ids)

    def test_same_component_with_narrow_neck_is_not_enough(self):
        image,batch=self.fixture()
        draw=ImageDraw.Draw(image)
        # Shared white label via a gap below the lettering, but a real outline
        # crosses the facing source seam. Never infer continuity via that gap.
        draw.line((194,23,194,200),fill='black',width=2)
        owners=q.enclosure_map(image,batch.detections)
        self.assertEqual(len({owners[i][0] for i in (0,1,2,3)}),1)
        self.assertEqual(g.vertical_groups(batch,owners,image),(batch,[]))

    def test_distinct_enclosures_annotations_and_new_intervening_source_reject(self):
        image,batch=self.fixture();owners=q.enclosure_map(image,batch.detections)
        owners[3]=(999,.99)
        self.assertEqual(g.vertical_groups(batch,owners,image),(batch,[]))
        annotated=replace(batch,proposals=(replace(batch.proposals[0],annotation_ids=('reading',)),batch.proposals[1]))
        self.assertEqual(self.group(image,annotated),(annotated,[]))
        extra=replace(batch.detections[3],source_id='intervening',polygon=((192,70),(200,70),(200,100),(192,100)))
        intervening=replace(batch,detections=batch.detections+(extra,))
        self.assertEqual(self.group(image,intervening),(intervening,[]))

    def test_prior_corner_conflict_is_retained_never_cleared_or_newly_introduced(self):
        image,batch=self.fixture()
        extra=replace(batch.detections[0],source_id='corner',polygon=((87,178),(95,178),(95,192),(87,192)))
        detections=batch.detections+(extra,)
        host=batch.proposals[0]
        host=replace(host,crop_conflicts=crop_conflicts(host.crop_bounds,host.main_ids,detections),crop_safe=False,
                     structural_reasons=host.structural_reasons+('crop_isolation_unresolved',))
        batch=replace(batch,detections=detections,proposals=(host,batch.proposals[1]))
        updated,records=self.group(image,batch)
        self.assertEqual(len(records),1)
        self.assertFalse(updated.proposals[0].crop_safe)
        self.assertEqual(updated.proposals[0].crop_conflicts[0].source_id,'corner')
        self.assertIn('crop_isolation_unresolved',updated.proposals[0].structural_reasons)

    def test_ambiguous_neighbors_do_not_chain(self):
        image,batch=self.fixture(third=True)
        updated,records=self.group(image,batch)
        # The host has two eligible continuations: no greedy merge or chaining.
        self.assertEqual(updated,batch)
        self.assertFalse(records)

    def test_separate_paragraphs_or_distant_columns_are_not_continuations(self):
        image,batch=self.fixture()
        for dx,dy in ((0,70),(70,0)):
            old=batch.detections[3]
            moved=replace(old,polygon=tuple((x+dx,y+dy) for x,y in old.polygon))
            changed=replace(batch,detections=batch.detections[:3]+(moved,))
            self.assertEqual(self.group(image,changed),(changed,[]))

    def test_refine_keeps_original_partition_even_when_no_decoders_are_allowed(self):
        image,batch=self.fixture()
        stream=BytesIO();image.save(stream,format='PNG')
        engine=SimpleNamespace(automatic_unit_proposals=(batch,))
        reader=Mock(side_effect=AssertionError('no model calls allowed'))
        with tempfile.TemporaryDirectory() as directory:
            stage=q.refine(engine,image,stream.getvalue(),vision=reader,manga=reader,model_identity='fixture',
                           output=Path(directory),max_units=0,max_fallback_calls=0)
        self.assertEqual(engine.original_unit_proposals,(batch,))
        self.assertEqual(len(engine.automatic_unit_proposals[0].proposals),1)
        self.assertEqual(stage['vision_calls'],0)
        self.assertEqual(stage['grouping'][0]['policy'],g.POLICY)
        reader.assert_not_called()

    def test_full_pipeline_and_offline_review_preserve_partition_and_raw_ids(self):
        import main
        import json
        from tests.test_pipeline import PipelineTests
        from tools.translation_review import run as review
        fixture=PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        image,batch=self.fixture()
        path=fixture.root/'grouped-input.png';image.save(path)
        boxes=[Text_Box(d.source_id,d.confidence,d.polygon,d.text) for d in batch.detections]
        fresh=propose_page_from_boxes(boxes,path)
        batch=replace(batch,page=fresh.page,detections=fresh.detections,evidence_sha256=fresh.evidence_sha256,
                      proposals=tuple(replace(p,page=fresh.page) for p in batch.proposals))
        engine=SimpleNamespace(automatic_unit_proposals=(batch,),process_images=Mock(),ocr=SimpleNamespace(close=Mock()),
            raw_detections=[boxes])
        reader=Mock(side_effect=AssertionError('no rereading needed'))
        def refine(engine, *, output):
            q.refine(engine,image,path.read_bytes(),vision=reader,manga=reader,model_identity='fixture',
                     output=output,max_units=0,max_fallback_calls=0)
        output=fixture.root/'grouped-run'
        result=main.run_pipeline(path,output,fonts=fixture.fonts,device='cpu',model_root=fixture.fixture.weights,
            run_local_ocr=True,run_local_model=True,ocr_factory=Mock(return_value=engine),
            runtime=SimpleNamespace(generate=fixture.provider),render_policy='best-effort',
            replacement_policy='confidence-gated',ocr_refine=refine)
        self.assertEqual(result['coverage']['rendered'],1,result)
        saved=json.loads((output/'preview/preview.json').read_bytes())
        self.assertEqual(len(saved['ocr']['original_unit_proposals'][0]['proposals']),2)
        self.assertEqual(len(saved['ocr']['batches'][0]['proposals']),1)
        self.assertEqual(saved['units'][0]['unit']['order'],[3,2,1,0])
        report=review(output/'preview/preview.json',path,fixture.root/'grouped-review')
        self.assertEqual(report['raw_detections'],4)
        reader.assert_not_called()


if __name__=='__main__':unittest.main()
