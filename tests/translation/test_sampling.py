"""Explicit visual sampling and retained incomplete replies; no live model I/O."""
import json
import unittest
from unittest.mock import Mock

from tralens.translation.runtime import GenerationError, GenerationOptions, OllamaRuntime


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.transport = Mock(return_value={'done': True, 'message': {'content': 'visible'}})
        self.runtime = OllamaRuntime('http://127.0.0.1:11434', transport=self.transport)
        self.payload = {'messages': [{'role': 'user', 'content': 'Read the image.'}],
                        'format': {'type': 'object'}, **GenerationOptions().transport_fields()}

    def generate(self):
        return self.runtime.generate(model='qwen3.8:latest', payload=self.payload, timeout=30)

    def test_explicit_seeded_sampling_reaches_transport_without_mutating_evidence(self):
        self.payload['options'].update(temperature=.7, seed=42)
        before = json.dumps(self.payload, sort_keys=True)
        self.assertEqual(self.generate().text, 'visible')
        body = self.transport.call_args.args[1]
        self.assertEqual(body['options'], self.payload['options'])
        self.assertFalse(body['think'])
        self.assertEqual(json.dumps(self.payload, sort_keys=True), before)
        self.assertEqual(GenerationOptions().transport_fields()['options']['temperature'], 0)
        self.assertNotIn('seed', GenerationOptions().transport_fields()['options'])

    def test_invalid_sampling_and_unrecognized_options_fail_before_dispatch(self):
        invalid = [{'temperature': v} for v in (True, -.1, 2.1, float('nan'), float('inf'), '0.7')]
        invalid += [{'seed': v} for v in (True, -1, 2**31, 1.5, '42')]
        invalid += [{'repeat_penalty': 2}, {'draft_num_predict': 0}]
        for options in invalid:
            with self.subTest(options=options):
                self.payload['options'] = {**GenerationOptions().transport_fields()['options'], **options}
                with self.assertRaises(ValueError):
                    self.generate()
        self.transport.assert_not_called()

    def test_repetition_abort_is_diagnosed_but_never_accepted_or_retried(self):
        raw = {'done': False, 'message': {'content': '{"text":"1'+'0'*30}}
        self.transport.return_value = raw
        with self.assertRaises(GenerationError) as caught:
            self.generate()
        self.assertEqual(caught.exception.failure_kind, 'incomplete_repetition')
        self.assertIn('possible server repetition abort', str(caught.exception))
        self.assertEqual(json.loads(caught.exception.response_json), raw)
        self.transport.assert_called_once()

    def test_nonrepetitive_incomplete_and_missing_message_are_not_misdiagnosed(self):
        for raw in ({'done': False, 'message': {'content': 'partial'}}, {'done': False},
                    {'done': False, 'message': []}, {'message': {'content': '0'*29}}):
            with self.subTest(raw=raw):
                self.transport.return_value = raw
                with self.assertRaises(GenerationError) as caught:
                    self.generate()
                self.assertEqual(caught.exception.failure_kind, 'incomplete')
                self.assertEqual(json.loads(caught.exception.response_json), raw)

    def test_completed_legitimate_repetition_and_server_errors_stay_distinct(self):
        text = '0'*40
        self.transport.return_value = {'done': True, 'message': {'content': text}}
        self.assertEqual(self.generate().text, text)
        self.transport.return_value = {'error': 'runner failed', 'done': False}
        with self.assertRaises(GenerationError) as caught:
            self.generate()
        self.assertEqual(caught.exception.failure_kind, 'provider')


if __name__ == '__main__':
    unittest.main()
