"""Accelerate exact-source staging with public assets; PFS checksums remain final.

Only paths from the source inventory are requested. A public asset whose size
differs is NOT installed. All successful downloads are still checked against
PFS by the parent coordinator before any completion claim.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

REPOS={
    'widowx': ('IPEC-COMMUNITY/bridge_orig_lerobot','0e9d76d07e9df3ea3eba257b2520d4913833fad2',''),
    'fractal': ('IPEC-COMMUNITY/fractal20220817_data_lerobot','91bf7d7f7ce50770a1ba5c6db14b8d1c0815122e',''),
    'gr1': ('nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim','09c6de8af50168090e7e9cc01e1ec3bce788de24','LeRobot/'),
}


def main(name, local_dir, inventory):
    local=Path(local_dir)
    repo,revision,prefix=REPOS[name]
    with gzip.open(inventory,'rt') as f:
        files=json.load(f)['files']
    files=[(p,n) for p,n in files if ('data' in Path(p).parts or 'videos' in Path(p).parts) and not p.endswith('.jpgpack')]
    def one(row):
        relative,size=row
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        dest=local/relative
        if dest.is_file() and dest.stat().st_size==size:
            return 'already_staged',relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        original=Path('/home/jigu/data/groot/lerobot/LeRobot')/relative
        if name=='gr1' and original.is_file() and original.stat().st_size==size:
            shutil.copyfile(original,str(dest)+'.seed-partial')
            os.replace(str(dest)+'.seed-partial',dest)
            return 'local_seed',relative
        url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/'+urllib.parse.quote(prefix+relative,safe='/')
        error=None
        for attempt in range(4):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'yuchen-parcc-dataset-setup/1.0'})
                with urllib.request.urlopen(req,timeout=90) as response, open(str(dest)+'.seed-partial','wb') as output:
                    shutil.copyfileobj(response,output,1024*1024)
                if os.path.getsize(str(dest)+'.seed-partial')!=size:
                    return 'size_mismatch_fallback',relative
                os.replace(str(dest)+'.seed-partial',dest)
                return 'public_seed',relative
            except urllib.error.HTTPError as exc:
                error=f'HTTP {exc.code}'
                if exc.code in (401,403,404):
                    break
                retry_after=exc.headers.get('Retry-After','10')
                delay=min(120,int(retry_after) if retry_after.isdigit() else 30) if exc.code==429 else 2**attempt
                time.sleep(delay)
            except Exception as exc:
                error=type(exc).__name__
                time.sleep(2**attempt)
        return 'download_fallback',relative+': '+str(error)
    counts={}
    failed=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(one,row) for row in files]
        for i,job in enumerate(as_completed(jobs),1):
            status,path=job.result()
            counts[status]=counts.get(status,0)+1
            if 'fallback' in status:
                failed.append([status,path])
                if len(failed)<=10:
                    print(json.dumps(dict(first_fallback=status,path=path)),flush=True)
            if i%100==0 or i==len(jobs):
                print(json.dumps(dict(dataset=name,done=i,total=len(jobs),counts=counts)),flush=True)
    print(json.dumps(dict(dataset=name,repo=repo,revision=revision,counts=counts,failures=failed)),flush=True)


if __name__=='__main__':
    main(*sys.argv[1:])
