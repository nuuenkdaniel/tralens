"""Offline pilot budgeting and occupancy guards; no models or source references."""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch,Mock


TOOLS=Path(__file__).resolve().parents[2]/'tools'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.cli=load('language_review_cli_test',TOOLS/'language_review.py')

    def test_global_reader_translation_caps_are_subtracted_between_pages(self):
        with patch.dict(sys.modules,{'language_review':self.cli}):
            runner=load('language_review_evaluator_test',TOOLS/'evaluate_language_review.py')
        with TemporaryDirectory() as temp:
            root=Path(temp);model=root/'models';model.mkdir();cases=root/'cases.json'
            cases.write_text(json.dumps([{'name':f'page-{i}','preview':'preview','image':'image','inference':True}for i in range(5)]))
            limits=[]
            def run(**options):
                limits.append(tuple(options[k] for k in ('max_reviews','max_reader_calls','max_translation_calls')))
                counts={'attempted':4,'corroborating_reader_calls':min(3,options['max_reader_calls']),
                    'translation_calls':min(3,options['max_translation_calls'])}
                options['output'].mkdir();(options['output']/'review.json').write_text(json.dumps({'counts':counts}))
                return {'status':'complete_candidates_only','counts':counts}
            with patch('main.run_language_review',side_effect=run),patch.object(runner,'IdleGuard'),\
                    patch.object(runner,'paddle_reader_factory',return_value=Mock()):
                runner.main(['--cases',str(cases),'--output',str(root/'out'),'--model-root',str(model),'--run-local-model'])
            self.assertEqual(limits,[(4,8,8),(4,5,5),(4,2,2),(4,0,0),(4,0,0)])
            summary=json.loads((root/'out/summary.json').read_bytes())
            self.assertEqual(summary['calls'],{'source_calls':20,'reader_calls':8,'translation_calls':8})

    def test_occupancy_guard_refuses_initial_residency_even_experiment_model(self):
        opener=Mock()
        def result(value):
            response=Mock();response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=None)
            response.read.return_value=json.dumps(value).encode();return response
        opener.open.side_effect=[result([]),result({'models':[{'name':'qwen3.8:latest'}]})]
        with patch('urllib.request.build_opener',return_value=opener):
            with self.assertRaisesRegex(RuntimeError,'occupied model server'):
                self.cli.IdleGuard('http://127.0.0.1:11434')()

    def test_occupancy_guard_never_unloads_active_viewer_work(self):
        opener=Mock();response=Mock();response.__enter__=Mock(return_value=response)
        response.__exit__=Mock(return_value=None);response.read.return_value=b'[{"status":"running"}]'
        opener.open.return_value=response
        with patch('urllib.request.build_opener',return_value=opener):
            with self.assertRaisesRegex(RuntimeError,'active viewer inference'):
                self.cli.IdleGuard('http://127.0.0.1:11434')()
        self.assertEqual(opener.open.call_count,1)
