"""Experiment: find and translate lettering outside a saved page's detector polygons."""
import argparse
import hashlib
import json
from pathlib import Path


MODELS = ('gpt-6-luna', 'gpt-6.1-terra', 'gpt-6.2-sol')


def discover(preview, model, output):
    doc = json.loads(preview.read_bytes())
    image = Path(doc['input']['path'])
    if hashlib.sha256(image.read_bytes()).hexdigest() != doc['input']['sha256']:
        raise ValueError('source image changed')
    detections = doc['ocr']['raw_detections'][0]
    from tralens.translation.discovery import propose
    regions = propose(image, [b['polygon'] for b in detections], model, 260)
    output.write_text(json.dumps(dict(model=model,preview=str(preview),image_sha256=doc['input']['sha256'],
                                      regions=regions),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preview',type=Path)
    parser.add_argument('model',choices=MODELS)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    discover(args.preview, args.model, args.output)
