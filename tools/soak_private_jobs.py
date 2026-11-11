"""Bounded synthetic-provider coordinator/viewer soak. Never loads real models."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def synthetic(config,job):
    from tests.test_pipeline import PipelineTests
    from tralens.ocr.automatic_units import propose_page_from_boxes
    from tralens.translation.runtime import Generation
    import main
    fixture=PipelineTests();fixture.setUp()
    try:
        fonts=json.loads(fixture.fonts.read_text())
        for key in ('path','license_path'):
            old=Path(fonts['fonts'][0][key]);destination=job['directory']/old.name
            destination.write_bytes(old.read_bytes());fonts['fonts'][0][key]=str(destination)
        retained_fonts=job['directory']/'fonts.json';retained_fonts.write_text(json.dumps(fonts))
        fixture.fixture.engine.automatic_unit_proposals=[propose_page_from_boxes(fixture.fixture.boxes,str(job['image']),
            complementary_ownership=True,regroup_companions=True)]
        report=main.run_pipeline(job['image'],job['directory']/'run',fonts=retained_fonts,device='cpu',
            model_root=fixture.fixture.weights,run_local_ocr=True,run_local_model=True,
            ocr_factory=fixture.fixture.factory,runtime=SimpleNamespace(generate=lambda **_:Generation('Hi!','{"done":true}')),
            removal_policy='aggressive',visibility_policy='always-show')
        report['synthetic_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return report
    finally:fixture.doCleanups()


def child(request):
    from tralens.devui import server,job_worker
    with patch.object(server,'execute_job',side_effect=synthetic), \
            patch('socket.socket',side_effect=AssertionError('no network/model provider')):
        job_worker.main(request)


def run(root,count):
    import numpy as np
    from PIL import Image
    from tralens.devui import server,jobs
    from tralens.processes import group_members
    from tralens.translation.workflow import _atomic_json
    root.mkdir();os.environ['TRALENS_STATE_ROOT']=str(root/'state')
    from tests.ocr.test_automatic_recognition import AutomaticRecognitionTests
    source,content,_=AutomaticRecognitionTests().fixture(text='こんにちは',two=True)
    source.close();image=io.BytesIO(content)
    _atomic_json(root/'plan.json',{'jobs_per_mode':count,'wall_limit_seconds':180,'job_limit_seconds':20,
        'runtime':'synthetic providers only; real coordinator/viewer/renderer/worker bootstrap',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(content).hexdigest(),'real_inference_calls':0,
        'restart_at_session_limit':server.MAX_JOBS,'criteria':['exact original and artifact hashes',
            'exact recomposition/outside-support equality','two exact bound transcripts',
            'no surviving worker group','restored completed history','no descriptor growth']})
    records=[];started=time.monotonic();fd_before=len(list(Path('/proc/self/fd').iterdir()))
    for isolated in (True,False):
        app=server.App(server.Config(root=root/('isolated' if isolated else 'embedding'),model_root=root,
            fonts=root/'explicit-synthetic-fonts',isolated_workers=isolated,job_seconds=20),runner=synthetic)
        original=jobs.supervise
        def supervised(command,*args,**kwargs):
            # Keep supervisor, worker resource bootstrap and coordinator real;
            # replace only provider construction in the owned child.
            return original([sys.executable,str(Path(__file__).resolve()),'--child',command[-1]],*args,**kwargs)
        with patch.object(jobs,'supervise',side_effect=supervised), \
                patch('socket.socket',side_effect=AssertionError('no network/model provider')):
            for index in range(count):
                if time.monotonic()-started>180:raise TimeoutError('synthetic soak deadline')
                if app.started_jobs==server.MAX_JOBS:
                    app=server.App(app.config,runner=synthetic)
                    if len(app.jobs)!=index or app.recovery_blocked:raise AssertionError('batch restart restoration')
                clock=time.monotonic();identity=app.start(image.getvalue(),{'mode':'render','removal_policy':'aggressive','visibility_policy':'always-show'})
                job=app.jobs[identity]
                if not job['done'].wait(25):raise TimeoutError('synthetic viewer job')
                if job['status'] not in ('complete','partial'):raise AssertionError(job)
                folder=job['directory'];report=json.loads((folder/'run/pipeline.json').read_bytes())
                manifest=json.loads((folder/'run/rendering/manifest.json').read_bytes())
                if manifest['pixel_audit']['outside_cleanup_changed_pixels']!=0:raise AssertionError('locality')
                if len(manifest['transcript'])!=2:raise AssertionError('transcript unavailable')
                rendering=folder/'run/rendering'
                assert (rendering/'original-file.bin').read_bytes()==content
                for name,digest in manifest['artifacts'].items():
                    assert hashlib.sha256((rendering/name).read_bytes()).hexdigest()==digest,name
                layers={}
                for name in ('source','clean','text','patch','composite','erase-mask'):
                    with Image.open(rendering/(name+'.png')) as im:layers[name]=im.copy()
                final=Image.alpha_composite(Image.alpha_composite(layers['clean'].convert('RGBA'),layers['patch']),layers['text']).convert('RGB')
                assert np.array_equal(np.asarray(final),np.asarray(layers['composite']))
                support=(np.asarray(layers['erase-mask'])>0)|(np.asarray(layers['text'])[:,:,3]>0)|(np.asarray(layers['patch'])[:,:,3]>0)
                assert np.array_equal(np.asarray(final)[~support],np.asarray(layers['source'])[~support])
                bound=[row for row in manifest['units'] if row.get('binding')]
                assert [row['binding'] for row in bound]==[row['binding'] for row in manifest['transcript']]
                if isolated:
                    state=json.loads((folder/'worker-state.json').read_bytes())
                    if group_members(state['process_group']):raise AssertionError('worker group survived')
                records.append({'isolated':isolated,'job':identity,'status':job['status'],
                    'seconds':time.monotonic()-clock,'rendered':manifest['coverage']['rendered'],
                    'artifact_hashes_checked':len(manifest['artifacts']),
                    'parent_resident_kib':int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')//1024,
                    'worker_peak_rss_kib':json.loads((folder/'worker-result.json').read_bytes())['synthetic_peak_rss_kib'] if isolated else None})
                _atomic_json(root/'progress.json',records,replace_owned=True)
        restored=server.App(app.config,runner=synthetic)
        if len(restored.jobs)!=count or restored.recovery_blocked:raise AssertionError('history restoration')
    result={'jobs':records,'count':len(records),'seconds':time.monotonic()-started,
        'fd_before':fd_before,'fd_after':len(list(Path('/proc/self/fd').iterdir())),
        'retained_bytes':jobs.tree_bytes(root),'real_inference_calls':0,
        'scope':'synthetic provider soak; not a live-model or long-duration production soak'}
    assert result['fd_after']==fd_before
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path)
    parser.add_argument('--jobs-per-mode',type=int,default=20);parser.add_argument('--child',type=Path)
    args=parser.parse_args()
    if args.child:child(args.child)
    elif args.root and 1<=args.jobs_per_mode<=20:run(args.root.absolute(),args.jobs_per_mode)
    else:parser.error('explicit new root and 1..20 jobs per mode required')


if __name__=='__main__':main()
