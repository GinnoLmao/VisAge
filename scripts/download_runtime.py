"""Restore the published runtime archive only after its pinned SHA-256 matches."""
import hashlib
import json
import urllib.request
import zipfile
import sys
from pathlib import Path
if sys.platform == 'win32':
    import truststore
    truststore.inject_into_ssl()
ROOT = Path(__file__).resolve().parents[1]


def main():
    record = json.loads((ROOT / 'configs/runtime_release.json').read_text())
    if not record['url'].startswith('https://github.com/GinnoLmao/VisAge/releases/download/'):
        raise ValueError('Unrecognized runtime release origin.')
    temporary = ROOT / '.cache/runtime_download.zip'
    temporary.parent.mkdir(parents=True, exist_ok=True)
    digest, total = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(urllib.request.Request(record['url'], headers={'User-Agent':'VisAge-runtime'}), timeout=120) as response, temporary.open('wb') as output:
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > record['bytes']: raise ValueError('Runtime archive exceeds its recorded size.')
                digest.update(block); output.write(block)
        if total != record['bytes'] or digest.hexdigest() != record['sha256']:
            raise ValueError('Runtime archive checksum mismatch.')
        with zipfile.ZipFile(temporary) as archive:
            manifest = json.loads(archive.read('runtime_manifest.json'))
            expected = {f['path'] for f in manifest['files']} | {'runtime_manifest.json'}
            if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
                raise ValueError('Runtime archive inventory mismatch.')
            for item in manifest['files']:
                target = (ROOT / item['path']).resolve()
                if not target.is_relative_to(ROOT) or not item['path'].startswith(('artifacts/pretrained/serving/', '.cache/face_detector/', 'reports/pretrained/', 'licenses/pretrained/')):
                    raise ValueError('Unsafe runtime archive path.')
                data = archive.read(item['path'])
                if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise ValueError('Runtime archive entry checksum mismatch.')
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        print('Verified runtime models, detector and evaluation evidence are ready.')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__': main()
