"""Resumable PFS -> local staging -> existing PARCC SSH multiplex tunnel.

No source deletion, no rsync --delete. Dataset roots are dedicated to this
transfer. Download caches excluded; symlinks dereferenced for portability.
State/logs persist independently of the interactive Codex session.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import fcntl
import gzip

HERE = Path(__file__).resolve().parent
CACHE = Path('/home/jigu/tmp/parcc-datasets-20260908')
AUDIT = Path('/home/jigu/projects/much-ado-about-noising/analysis/diagnostics/parcc_dataset_setup')
DEST = '/vast/projects/jiayuanm/mao-lab/yuchen/data'
SSH = ['ssh', '-S', '/tmp/parcc-jigu-control', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
       '-o', 'ServerAliveCountMax=6', 'parcc']
RSH = 'ssh -S /tmp/parcc-jigu-control -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6'
SOURCES = {
    'widowx': '/mnt/pfs/yuchen/groot/bridge_orig_lerobot',
    'fractal': '/mnt/pfs/yuchen/groot/fractal_lerobot',
    'gr1': '/mnt/pfs/yuchen/groot/lerobot/LeRobot',
}
EXCLUDES = ['--exclude=.cache/', '--exclude=._____temp/', '--exclude=.msc',
            '--exclude=.mv', '--exclude=__pycache__/', '--exclude=.DS_Store',
            '--exclude=*.seed-partial', '--exclude=*.jpgpack']


def now():
    return datetime.now(timezone.utc).isoformat()


def state(name, phase, **extra):
    value = dict(dataset=name, phase=phase, utc=now(), source=SOURCES[name],
                 staging=str(CACHE/name), destination=f'{DEST}/{name}', **extra)
    temp = AUDIT/f'{name}.status.tmp'
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(AUDIT/f'{name}.status.json')
    print(json.dumps(value), flush=True)


def run_logged(name, phase, command):
    state(name, phase, command=command)
    with (AUDIT/f'{name}.{phase}.log').open('a') as stream:
        stream.write('\n'+now()+' '+repr(command)+'\n')
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def transfer(name):
    local = CACHE/name
    local.mkdir(exist_ok=True)
    remote_src = f'rsync://127.0.0.1:28731/{name}/'
    remote_dst = f'parcc:{DEST}/{name}/'
    source_base = ['rsync', '-rtL', '--partial', '--partial-dir=.rsync-partial',
                   '--timeout=600', '--info=progress2,stats2', '--outbuf=L',
                   '--compress', *EXCLUDES]
    dest_base = ['rsync', '-rt', '--partial', '--partial-dir=.rsync-partial',
                 '--timeout=600', '--info=progress2,stats2', '--outbuf=L', '-e', RSH,
                 '--exclude=.rsync-partial/', '--exclude=*.seed-partial', '--exclude=*.jpgpack']
    last_error = None
    for attempt in range(1, 4):
        try:
            subprocess.run(SSH+[f'mkdir -p {DEST}/{name}'], check=True)
            # Make metadata available early; omit payload directories, not tasks.
            run_logged(name, 'source_metadata', source_base+['--exclude=data/', '--exclude=videos/', remote_src, str(local)+'/'])
            run_logged(name, 'metadata_to_parcc', dest_base+['--exclude=data/', '--exclude=videos/', str(local)+'/', remote_dst])
            if attempt==1:
                inventory=AUDIT/f'{name}.inventory.json.gz'
                for poll in range(120):
                    state(name,'fetch_inventory',poll=poll)
                    result=subprocess.run(['rsync','-rt','--timeout=120',
                        f'rsync://127.0.0.1:28731/inventory/{name}.inventory.json.gz',str(inventory)],
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    if result.returncode==0:
                        try:
                            with gzip.open(inventory,'rt') as stream:
                                inventory_data=json.load(stream)
                            assert inventory_data['source']==SOURCES[name]
                            break
                        except (EOFError,ValueError,OSError):
                            pass
                    time.sleep(10)
                else:
                    raise RuntimeError('Source inventory not ready within 20 minutes')
                run_logged(name,'public_asset_seed', ['/usr/bin/python','-u',str(HERE/'parcc_public_seed.py'),
                    name,str(local),str(inventory)])
            # Seed is only a bandwidth optimization; exact PFS bytes win.
            run_logged(name, 'source_payload', source_base+['--checksum',remote_src, str(local)+'/'])
            run_logged(name, 'payload_to_parcc', dest_base+[str(local)+'/', remote_dst])
            # Full content-checksum comparisons on both legs, not size-only checks.
            checks = [
                ('source_checksum', ['rsync','-rLnc','--timeout=600','--out-format=%i %n',
                    *EXCLUDES,remote_src,str(local)+'/']),
                ('parcc_checksum', ['rsync','-rnc','--timeout=600','--out-format=%i %n',
                    '-e',RSH,'--exclude=.rsync-partial/','--exclude=*.seed-partial','--exclude=*.jpgpack',str(local)+'/',remote_dst]),
            ]
            for phase, command in checks:
                state(name,phase,attempt=attempt)
                with (AUDIT/f'{name}.{phase}.stderr.log').open('a') as err:
                    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=err,text=True,check=True)
                (AUDIT/f'{name}.{phase}.diff').write_text(result.stdout)
                if result.stdout.strip():
                    # Repair changed/same-size files by content before retrying.
                    if phase=='source_checksum':
                        run_logged(name,'source_repair',source_base+['--checksum',remote_src,str(local)+'/'])
                    else:
                        run_logged(name,'parcc_repair',dest_base+['--checksum',str(local)+'/',remote_dst])
                    raise RuntimeError(f'{phase}: differences repaired; retrying full verification')
            infos=[]
            for p in sorted(local.rglob('meta/info.json')):
                info=json.loads(p.read_text())
                infos.append(dict(root=str(p.parent.parent.relative_to(local)),
                                  episodes=info['total_episodes'], frames=info.get('total_frames'),
                                  robot_type=info.get('robot_type')))
            file_count=0
            byte_count=0
            for p in local.rglob('*'):
                if p.is_file() and '.rsync-partial' not in p.parts and not p.name.endswith(('.seed-partial','.jpgpack')):
                    file_count+=1
                    byte_count+=p.stat().st_size
            manifest=dict(dataset=name,utc=now(),source=SOURCES[name],destination=f'{DEST}/{name}',
                          verification='both rsync full-content checksum dry runs returned zero differences',
                          files=file_count,bytes=byte_count,datasets=infos,
                          excluded_download_cache=EXCLUDES)
            output=AUDIT/f'{name}.complete.json'
            output.write_text(json.dumps(manifest,indent=2)+'\n')
            subprocess.run(['rsync','-rt','-e',RSH,str(output),f'parcc:{DEST}/_setup_20260908/'],check=True)
            state(name,'complete',manifest=str(output),files=file_count,bytes=byte_count,
                  episodes=sum(x['episodes'] for x in infos),tasks=len(infos))
            return
        except Exception as exc:
            last_error=repr(exc)
            state(name,'retry_wait',attempt=attempt,error=last_error)
            if attempt < 3:
                time.sleep(30)
    state(name,'failed',error=last_error)
    raise RuntimeError(f'{name}: {last_error}')


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    AUDIT.mkdir(parents=True,exist_ok=True)
    lock=(AUDIT/'transfer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    subprocess.run(SSH+[f'test -d {DEST} && mkdir -p {DEST}/_setup_20260908'],check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(transfer,name) for name in SOURCES]
        for job in jobs:
            job.result()
    print('ALL_THREE_DATASETS_VERIFIED',flush=True)


if __name__=='__main__':
    main()
