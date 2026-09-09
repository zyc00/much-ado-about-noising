"""Retrieve probe arrays through short Kubernetes streams, checking SHA256.

Large kubectl cp streams reset on this connection. Small read-only dd requests
are reassembled locally; a destination is replaced only after its hash matches.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pod',default='yuchen-vla-util')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    tasks=json.loads(args.summary.read_text())['tasks']
    args.output.mkdir(parents=True,exist_ok=True)
    def fetch(task):
        source=task['source']
        dest=args.output/Path(source).name
        expected=task['source_sha256']
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:
            return dest.name+' already verified'
        stat=subprocess.run(['kubectl','exec',args.pod,'--','stat','-c','%s',source],
                            capture_output=True,check=True,timeout=30)
        length=int(stat.stdout)
        chunks=[]
        for offset in range(0,length,65536):
            want=min(65536,length-offset)
            for attempt in range(5):
                try:
                    result=subprocess.run(['kubectl','exec',args.pod,'--','dd',f'if={source}',
                        'bs=65536',f'skip={offset//65536}','count=1','status=none'],
                        capture_output=True,timeout=30)
                    if result.returncode==0 and len(result.stdout)==want:
                        chunks.append(result.stdout)
                        break
                except subprocess.TimeoutExpired:
                    pass
            else:
                raise RuntimeError(f'Cannot transfer {source} offset {offset}')
        payload=b''.join(chunks)
        assert len(payload)==length and hashlib.sha256(payload).hexdigest()==expected, source
        tmp=dest.with_suffix('.npz.download')
        tmp.write_bytes(payload)
        tmp.replace(dest)
        return dest.name+' SHA256 verified'
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(fetch,tasks):
            print(result,flush=True)


if __name__=='__main__':
    main()
