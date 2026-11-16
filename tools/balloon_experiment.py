"""Opt-in balloon-level experiment from immutable saved OCR/translation layers."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preview','image','fonts','baseline','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-local-model',action='store_true',help='explicit bounded existing Qwen proposals and fresh grouped-source TranslateGemma calls; no page OCR')
    parser.add_argument('--replay',type=Path,help='saved bound experiment.json; no inference and no manual masks')
    parser.add_argument('--endpoint',default='http://100.121.133.88:11434')
    parser.add_argument('--visibility-policy',choices=('replacement-only','always-show'),default='always-show')
    args = parser.parse_args(argv)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from main import run_balloon_experiment
    state = run_balloon_experiment(**vars(args))
    print(state['status'],'new clean dialogues:',state['new_clean_dialogues'],
        'native preserved:',state['native_preserved'],'atomically superseded:',state['native_atomically_superseded'],
        'calls:',state['proposal_calls'],state['translation_calls'])


if __name__ == '__main__':
    main()
