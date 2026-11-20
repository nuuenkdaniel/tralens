"""Inspect a lost remote request; acknowledge only after server-side completion."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from tralens.translation.runtime import local_endpoint
    from tralens.scheduling import remote_state_path,acknowledge_remote_completion
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',required=True)
    parser.add_argument('--acknowledge-stopped',metavar='REQUEST_ID',
        help='assert remote completion was independently confirmed; a local timeout or /api/ps is insufficient')
    args=parser.parse_args();endpoint=local_endpoint(args.endpoint)
    if args.acknowledge_stopped:acknowledge_remote_completion(endpoint,args.acknowledge_stopped)
    path=remote_state_path(endpoint)
    print(path.read_text() if path.exists() else 'No pending application request recorded. This does not describe unrelated clients.')


if __name__=='__main__':main()
