"""Acquisition is explicit, bounded, and offline in tests."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from tools import download_render_examples as d


class AcquisitionTests(unittest.TestCase):
    def test_originals_and_licenses_are_saved_without_pixel_rewrites(self):
        stream = BytesIO()
        Image.new('RGB', (40, 60), '#856744').save(stream, format='PNG')
        pixels = stream.getvalue()
        def retrieve(request, timeout):
            url = request.full_url
            data = (b'https://creativecommons.org/licenses/by-nc/3.0/' if url.endswith('license_en.txt')
                else b'Public domain' if 'commons.wikimedia.org/w/' in url else pixels)
            response = BytesIO(data)
            response.url = url
            return response
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'examples'
            with patch.object(d, 'urlopen', side_effect=retrieve) as fetch, patch.object(d, 'PINS', {}):
                report = d.acquire(output)
            self.assertEqual(fetch.call_count, 10)
            self.assertEqual(len(report['records']), 10)
            for row in report['records']:
                self.assertEqual(d.hashlib.sha256((output/row['file']).read_bytes()).hexdigest(), row['sha256'])
                if row.get('kind') == 'artwork':
                    self.assertEqual((output/row['file']).read_bytes(), pixels)
                    self.assertTrue((output/row['license_file']).is_file())
            self.assertTrue(json.loads((output/'acquisition.json').read_text())['no_training'])

    def test_existing_destination_is_never_overwritten_or_fetched(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(d, 'urlopen') as fetch:
            with self.assertRaisesRegex(ValueError, 'fresh'):
                d.acquire(Path(temporary))
            fetch.assert_not_called()

    def test_oversized_license_download_is_rejected_before_image_acquisition(self):
        response = BytesIO(b'x'*65537)
        response.url = 'https://raw.githubusercontent.com/test'
        with tempfile.TemporaryDirectory() as temporary, patch.object(d, 'urlopen', return_value=response) as fetch:
            output = Path(temporary)/'examples'
            with self.assertRaisesRegex(ValueError, 'byte budget'):
                d.acquire(output)
            self.assertEqual(fetch.call_count, 1)
            self.assertFalse((output/'acquisition.json').exists())

    def test_drifted_pinned_bytes_never_publish_a_success_ledger(self):
        response = BytesIO(b'https://creativecommons.org/licenses/by-nc/3.0/ altered')
        response.url = 'https://raw.githubusercontent.com/test'
        with tempfile.TemporaryDirectory() as temporary, patch.object(d, 'urlopen', return_value=response):
            output = Path(temporary)/'examples'
            with self.assertRaisesRegex(ValueError, 'digest drift'):
                d.acquire(output)
            self.assertFalse((output/'acquisition.json').exists())


if __name__ == '__main__':
    unittest.main()
