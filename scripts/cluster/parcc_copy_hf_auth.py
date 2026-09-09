"""User-authorized one-time HF token transfer via SSH stdin, never argv/logs.

Does not start downloads. Refuses to overwrite an existing remote token.
"""
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import json, os, stat, sys, urllib.request, urllib.error
from pathlib import Path

def main():
    token = sys.stdin.buffer.read().decode().strip()
    if not token.startswith('hf_') or any(c.isspace() for c in token):
        raise ValueError('Unexpected token format')
    base = Path(os.environ.get('HF_HOME', os.path.join(
        os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'huggingface')))
    dest = Path(os.environ.get('HF_TOKEN_PATH', str(base/'token')))
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('Remote token already exists; refusing overwrite')
    req = urllib.request.Request('https://huggingface.co/api/whoami-v2',
        headers={'Authorization': 'Bearer '+token})
    with urllib.request.urlopen(req, timeout=30) as response:
        identity = json.load(response)
    dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(token+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    assert dest.read_text().strip() == token
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600
    print(json.dumps(dict(username=identity.get('name'), account_type=identity.get('type'),
        is_pro=identity.get('isPro'), token_path=str(dest), permissions='600',
        authenticated_from_parcc=True, download_started=False)))

try:
    main()
except urllib.error.HTTPError as exc:
    print(json.dumps(dict(error='HF authentication request failed',status=exc.code)), file=sys.stderr)
    sys.exit(1)
except Exception as exc:
    print(json.dumps(dict(error_type=type(exc).__name__,error='Credential transfer or validation failed; no secret displayed')), file=sys.stderr)
    sys.exit(1)
'''


def main():
    source = Path('/home/jigu/.cache/huggingface/token')
    with source.open('rb') as stream:
        result = subprocess.run([
            'ssh', '-S', '/tmp/parcc-jigu-control', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=10', 'parcc', 'python3 -c '+shlex.quote(REMOTE),
        ], stdin=stream, timeout=60)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
