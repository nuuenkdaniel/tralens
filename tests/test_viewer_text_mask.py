from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tralens.devui import server


class ViewerTextMask(unittest.TestCase):
    def test_explicit_mask_reaches_coordinator_without_loading_it_in_viewer(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);model=root/'explicit.onnx'
            config=server.Config(root=root,model_root=root,fonts=root/'fonts.json',text_mask_model=model)
            app=server.App(config)
            self.assertTrue(app.public_config()['text_mask_available'])
            job={'image':root/'input.png','directory':root,
                'options':server.options({'mode':'render'},render_enabled=True)}
            with patch('main.run_pipeline',return_value={'status':'complete'}) as run:
                server.execute_job(config,job)
            self.assertEqual(run.call_args.kwargs['text_mask_model'],model)
            self.assertFalse(model.exists())


if __name__=='__main__':unittest.main()
