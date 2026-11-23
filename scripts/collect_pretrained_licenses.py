"""Archive license texts and publisher attribution without altering weight pins."""
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
import truststore
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.model_registry import METHODS
from visage.runtime import sha256_file, write_json, utc_now
truststore.inject_into_ssl()


def main():
    directory = ROOT / 'licenses/pretrained'
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'provenance.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        for path, digest in record['files'].items():
            if sha256_file(ROOT / path) != digest:
                raise ValueError('Archived license evidence changed.')
        print('License archive verified.', flush=True)
        return
    def fetch(url):
        with urlopen(Request(url, headers={'User-Agent': 'VisAge-academic-provenance/2'}), timeout=45) as response:
            return response.read(100001)
    revision = json.loads(fetch('https://api.github.com/repos/spdx/license-list-data/git/ref/heads/main'))['object']['sha']
    licenses = {'cc-by-nc-4.0': 'CC-BY-NC-4.0', 'apache-2.0': 'Apache-2.0', 'mit': 'MIT'}
    records, files = {}, {}
    for method in METHODS:
        source = ROOT / '.cache/pretrained_weights' / method
        metadata = json.loads((source / 'provenance.json').read_text())
        identifier = licenses[metadata['license']]
        destination = directory / f'{identifier}.txt'
        url = f'https://raw.githubusercontent.com/spdx/license-list-data/{revision}/text/{identifier}.txt'
        if not destination.exists():
            payload = fetch(url)
            if not 100 < len(payload) < 100000 or b'<html' in payload.lower():
                raise ValueError('Unexpected license text payload.')
            destination.write_bytes(payload)
        files[destination.relative_to(ROOT).as_posix()] = sha256_file(destination)
        target = directory / method
        target.mkdir(parents=True, exist_ok=True)
        for name in ('README.md', 'config.json', 'provenance.json'):
            payload = (source / name).read_bytes()
            path = target / name
            path.write_bytes(payload)
            files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        records[method] = {'publisher': metadata['repository'], 'revision': metadata['revision'],
            'weight_license_tag': metadata['license'], 'archived_text': destination.relative_to(ROOT).as_posix(),
            'license_text_source': url, 'publisher_attribution': 'See the exact model card and its authors and original weight source.'}
    write_json(marker, {'retrieved_utc': utc_now(), 'license_text_repository_revision': revision,
        'methods': records, 'files': files, 'note': 'Model-card license tags are preserved. This archive does not grant additional model or dataset permissions.'})
    print(json.dumps({'ok': True, 'models': len(records)}), flush=True)


if __name__ == '__main__':
    main()
