"""Archive original software copyright notices alongside weight-license cards."""
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
import truststore
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import sha256_file, write_json, utc_now
truststore.inject_into_ssl()


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'VisAge-academic-provenance/2'}), timeout=45) as response:
        payload = response.read(100001)
    if len(payload) > 100000:
        raise ValueError('Unexpected source-notice payload size.')
    return payload


def main():
    target = ROOT / 'licenses/pretrained/original_sources'
    target.mkdir(parents=True, exist_ok=True)
    marker = target / 'provenance.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        for record in saved['repositories'].values():
            if sha256_file(ROOT / record['path']) != record['sha256']:
                raise ValueError('Original source notice changed.')
        print('Original copyright notices verified.', flush=True)
        return
    records = {}
    for repository in ('facebookresearch/ConvNeXt-V2', 'tensorflow/tpu', 'microsoft/Swin-Transformer',
        'huggingface/pytorch-image-models'):
        info = json.loads(fetch(f'https://api.github.com/repos/{repository}'))
        branch = info['default_branch']
        revision = json.loads(fetch(f'https://api.github.com/repos/{repository}/git/ref/heads/{branch}'))['object']['sha']
        url = f'https://raw.githubusercontent.com/{repository}/{revision}/LICENSE'
        payload = fetch(url)
        if len(payload) < 100 or b'<html' in payload.lower():
            raise ValueError('Unexpected original license payload.')
        destination = target / (repository.replace('/', '_') + '_LICENSE.txt')
        destination.write_bytes(payload)
        records[repository] = {'revision': revision, 'url': url, 'path': destination.relative_to(ROOT).as_posix(),
            'sha256': sha256_file(destination), 'bytes': len(payload)}
    write_json(marker, {'retrieved_utc': utc_now(), 'repositories': records,
        'note': 'Original software source copyright notices and licenses. The separate publisher model cards remain authoritative evidence of the selected weight license tags; software notices do not replace or broaden weight/dataset terms.'})
    print(json.dumps({'ok': True, 'repositories': len(records)}), flush=True)


if __name__ == '__main__':
    main()
