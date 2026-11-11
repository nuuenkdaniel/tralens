"""Visibility opt-in through real HTTP/coordinator, with synthetic providers only."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tralens.devui import server as s


class VisibleViewerTests(unittest.TestCase):
    def setUp(self):
        from tests.test_dev_frontend import ViewerTests
        self.http_fixture=ViewerTests();self.http_fixture.setUp();self.addCleanup(self.http_fixture.doCleanups)
        self.root=self.http_fixture.root

    def test_explicit_option_validation_and_no_asset_or_font_overrides(self):
        defaults=s.options({},render_enabled=True)
        self.assertEqual(defaults['visibility_policy'],'always-show')
        self.assertEqual(defaults['removal_policy'],'aggressive')
        for bad in ({'visibility_policy':True},{'visibility_policy':'unknown'},
                    {'mode':'ocr','visibility_policy':'always-show'},{'fallback_font':'/etc/passwd'}):
            with self.assertRaises(s.RequestError):s.options(bad,render_enabled=True)
        settings=s.options({'mode':'render','render_policy':'conservative','visibility_policy':'always-show'},render_enabled=True)
        self.assertEqual(settings['replacement_policy'],'all')
        self.assertEqual(settings['render_policy'],'conservative')

    def test_http_upload_retry_history_and_separate_counts(self):
        import main
        from tests.test_pipeline import PipelineTests
        from tralens.ocr.automatic_units import propose_page_from_boxes
        fixture=PipelineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        f=fixture.fixture
        def factory(image,**kwargs):
            return SimpleNamespace(raw_detections=[f.boxes],process_images=Mock(),ocr=SimpleNamespace(close=Mock()),
                automatic_unit_proposals=[propose_page_from_boxes(f.boxes,image,complementary_ownership=True,regroup_companions=True)])
        original=main.run_pipeline
        def pipeline(*args,**kwargs):
            return original(*args,**kwargs,ocr_factory=factory,runtime=SimpleNamespace(generate=fixture.provider))
        config=replace(self.http_fixture.config,fonts=fixture.fonts,model_root=f.weights)
        app=s.App(config);http=self.http_fixture.serve(app)
        with patch.object(main,'run_pipeline',side_effect=pipeline):
            status,_,body=self.http_fixture.request(http,'POST','/api/jobs',f.image.read_bytes(),
                {'X-Tralens-Token':app.token,'X-Tralens-Options':json.dumps({'mode':'render',
                    'visibility_policy':'always-show','removal_policy':'legacy','replacement_policy':'confidence-gated'})})
            self.assertEqual(status,202,body)
            identity=json.loads(body)['id'];self.assertTrue(app.job(identity)['done'].wait(15))
            view=app.snapshot(identity)
            self.assertIsNone(view['error'],view)
            self.assertEqual(view['model_calls'],2)
            self.assertEqual(view['display_summary']['clean_replacements'],0)
            self.assertEqual(view['display_summary']['displayed_translations'],2)
            self.assertTrue(all(u['display_status']in('patch','caption')for u in view['units']))
            self.assertIn('run/rendering/patch.png',view['artifacts'])
            olddir=app.job(identity)['directory']
            old={p.relative_to(olddir):p.read_bytes()for p in olddir.rglob('*')if p.is_file()}
            status,_,body=self.http_fixture.request(http,'POST',f'/api/jobs/{identity}/retry',b'{}',
                {'X-Tralens-Token':app.token,'Content-Type':'application/json'})
            self.assertEqual(status,202,body)
            retry=json.loads(body)['id'];self.assertTrue(app.job(retry)['done'].wait(15))
            self.assertEqual(app.job(retry)['options']['visibility_policy'],'always-show')
        for p,data in old.items():self.assertEqual((olddir/p).read_bytes(),data)
        runner=Mock(side_effect=AssertionError('history must not infer'))
        restored=s.App(config,runner=runner)
        self.assertEqual(restored.snapshot(identity)['visibility_policy'],'always-show')
        self.assertEqual(restored.job(identity)['options']['visibility_policy'],'always-show')
        runner.assert_not_called()

    def test_legacy_pipeline_report_restores_mode_without_rewriting_old_job(self):
        from tests.test_dev_frontend import image_bytes
        data=image_bytes();directory=self.http_fixture.config.root/('e'*32);(directory/'run').mkdir(parents=True)
        (directory/'input.png').write_bytes(data)
        (directory/'run/pipeline.json').write_text(json.dumps({'status':'partial','visibility_policy':'always-show',
            'source':{'sha256':s.hashlib.sha256(data).hexdigest(),'size':[100,100]}}))
        restored=s.App(self.http_fixture.config,runner=Mock())
        self.assertEqual(restored.job('e'*32)['options']['visibility_policy'],'always-show')
        self.assertFalse((directory/'viewer.json').exists())

    def test_frontend_does_not_count_captions_as_clean_replacements(self):
        node=shutil.which('node')
        if node is None:self.skipTest('optional Node.js unavailable')
        script=r'''
const assert=require('node:assert/strict');
const {qualityNotice,previewStates}=require(process.argv[1]);
const data={status:'partial',options:{mode:'render',visibility_policy:'always-show'},visibility_policy:'always-show',
 detections:[],units:[],artifacts:['run/rendering/composite.png'],rendering_coverage:{rendered:1},
 display_summary:{clean_replacements:1,patch_backed_translations:2,nearby_captions:3,intentionally_unchanged:4,
 available_not_displayed:5,missing_translations:6,displayed_translations:6}};
assert.match(qualityNotice(data).copy,/fallback patches may cover artwork/);
assert.match(qualityNotice(data).copy,/1 clean replacements/);
assert.match(qualityNotice(data).copy,/3 nearby captions/);
assert.match(qualityNotice(data).copy,/5 available but not displayed/);
assert.equal(qualityNotice(data).action,null);
assert.equal(previewStates(data).final.label,'6 displayed');
'''
        result=subprocess.run([node,'-e',script,str(s.STATIC/'app.js')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
