"""Explicit, bounded acquisition of local-only rendering evaluation examples.

Never imported by the pipeline. Saves untouched originals and license evidence;
no OCR, model calls, training, upload, installation or image redistribution.
Ubunchu! is CC BY-NC 3.0: these samples are NOT commercial product assets.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


PIN = 'f3c6473206a8a8b6f1a946e8c60b4cdaff5ea71b'
BASE = f'https://raw.githubusercontent.com/ubunchu-translators/ubunchu/{PIN}/chapters'
NEMO = [
    ('1905-10-15', 'https://upload.wikimedia.org/wikipedia/commons/3/38/Little_Nemo_1905-10-15.jpg',
     'https://commons.wikimedia.org/w/index.php?title=File:Little_Nemo_1905-10-15.jpg&oldid=709615093'),
    ('1905-10-22', 'https://upload.wikimedia.org/wikipedia/commons/5/5c/Little_Nemo_in_Slumberland_%281905-10-22%29.jpeg',
     'https://commons.wikimedia.org/w/index.php?title=File:Little_Nemo_in_Slumberland_(1905-10-22).jpeg&oldid=1215792021'),
]
PINS = {
    'ubunchu04-license.txt': '4c0a52badfdc24f85b8cdc9c456e94021de92fbab37d4fb221d3c732c2c393ab',
    'ubunchu04_0002.png': '52b198b5b0c0200257e2fd7ef76d359e58778a7d5cf349832d3213920b3b30bd',
    'ubunchu04_0005.png': 'e31ab019c7c1411e41addcc74bf4cbb6a90c2a1962d7cd41d865756932a93a65',
    'ubunchu06-license.txt': '89f106b87d1d31131cf60acde2255f8c985e49b84946f8d2a9aa26805d410254',
    'ubunchu06_0002.png': '200070f15b5a9479c59eecf14222b4af002bb7b4d161a0006915190b15f84105',
    'ubunchu06_0005.png': '106be93711dd23a1ba3f101126f6ec5be6361b6046ff5457d654b327c104c542',
    'nemo-1905-10-15.jpg': '339d7b1228ca24bbffa55afe46efb4203db484afb6875eaa5888684694fce9d6',
    'nemo-1905-10-22.jpg': 'fc4e5a8fbcfbcc287e8bc5415d2ae1f1c52e3fa490fc39a853499f8b3c2ce165',
}


def acquire(output):
    output = output.absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)) or not output.parent.is_dir():
        raise ValueError('fresh nonsymlink output with existing parent required')
    output.mkdir(mode=0o700)
    records, total = [], 0

    def get(name, url, *, limit=10*1024*1024):
        nonlocal total
        with urlopen(Request(url, headers={'User-Agent': 'Tralens-local-render-evaluation/1.0'}), timeout=40) as response:
            data = response.read(limit+1)
            final_url = response.url
        if len(data) > limit or total+len(data) > 40*1024*1024:
            raise ValueError('example acquisition byte budget')
        total += len(data)
        with (output/name).open('xb') as stream:
            stream.write(data)
        row = dict(file=name, url=url, final_url=final_url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if name in PINS and row['sha256'] != PINS[name]:
            raise ValueError('acquired example digest drift; retained bytes are not accepted as the frozen source')
        records.append(row)
        return row, data

    for chapter in ('04', '06'):
        prefix = f'{BASE}/{chapter}/ubunchu{chapter}_ja-release'
        license_row, license_data = get(f'ubunchu{chapter}-license.txt', prefix+'/license_en.txt', limit=65536)
        if b'creativecommons.org/licenses/by-nc/3.0' not in license_data:
            raise ValueError('expected Ubunchu license not found; inspect retained evidence')
        for page in ('0002', '0005'):
            row, data = get(f'ubunchu{chapter}_{page}.png', f'{prefix}/ubunchu{chapter}_{page}.png')
            with Image.open(BytesIO(data)) as image:
                row['size'] = list(image.size)
                row['format'] = image.format
            row.update(kind='artwork', license='CC BY-NC 3.0', license_file=license_row['file'],
                attribution='Hiroshi Seo / ASCII MEDIA WORKS Inc.', upstream_tree=PIN,
                use='local noncommercial rendering evaluation only; no training or redistribution')
    for date, url, license_url in NEMO:
        license_row, license_data = get(f'nemo-{date}-license.html', license_url, limit=2*1024*1024)
        if b'Public domain' not in license_data:
            raise ValueError('expected public-domain statement not found; inspect retained evidence')
        row, data = get(f'nemo-{date}.jpg', url)
        with Image.open(BytesIO(data)) as image:
            row['size'] = list(image.size)
            row['format'] = image.format
        row.update(kind='artwork', license='Public domain (Commons evidence retained)',
            license_file=license_row['file'], attribution='Winsor McCay, Little Nemo in Slumberland',
            use='local rendering evaluation; original upload URL plus acquired digest, not publisher-authenticated')
    manifest = dict(purpose='additional_local_rendering_evaluation_examples', schema_version=1,
        acquired_utc=datetime.now(timezone.utc).isoformat(), total_bytes=total, records=records,
        no_training=True, no_upload=True, no_provider_calls=True,
        acquisition_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'acquisition.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'acquisition-source.py').write_bytes(Path(__file__).read_bytes())
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = acquire(args.output)
    for row in report['records']:
        print(row['file'], row['bytes'], row['sha256'], row.get('size', 'license evidence'))
