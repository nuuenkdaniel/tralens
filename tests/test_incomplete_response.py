import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

from tralens import scheduling


class IncompleteResponse(unittest.TestCase):
    def test_incomplete_reply_survives_caller_exit_and_blocks_next_dispatch(self):
        with TemporaryDirectory() as folder,patch.dict(os.environ,TRALENS_STATE_ROOT=folder):
            endpoint='http://127.0.0.1:11434'
            response={'done':False,'error':'worker stopped','message':{'content':'部分'}}
            self.assertEqual(scheduling.remote_call(endpoint,lambda:response),response)
            incident=json.loads(scheduling.remote_state_path(endpoint).read_bytes())
            self.assertEqual(incident['response'],response)
            self.assertFalse(incident['cancellation_confirmed'])
            next_call=Mock()
            with self.assertRaisesRegex(RuntimeError,'may still be running'):
                scheduling.remote_call(endpoint,next_call)
            next_call.assert_not_called()
            # Explicit host-side confirmation may unblock dispatch; it must not
            # destroy the retained reply. This operates only on temporary state.
            scheduling.acknowledge_remote_completion(endpoint,incident['request_id'])
            self.assertFalse(scheduling.remote_state_path(endpoint).exists())
            archived=list(Path(folder).glob('*-acknowledged-*.json'))
            self.assertEqual(len(archived),1)
            self.assertEqual(json.loads(archived[0].read_bytes())['response'],response)


if __name__=='__main__':unittest.main()
