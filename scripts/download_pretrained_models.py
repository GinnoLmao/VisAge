"""Resolve immutable Hub revisions and verify checkpoint LFS hashes."""
from pathlib import Path
import json
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.model_registry import REGISTRY
from visage.runtime import sha256_file, utc_now, write_json


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'VisAge-Academic-Research/2.0'}), timeout=120)


def main():
    import truststore
    truststore.inject_into_ssl()
    directory = ROOT / '.cache/pretrained_weights'
    directory.mkdir(parents=True, exist_ok=True)
    for method, spec in REGISTRY.items():
        target = directory / method
        target.mkdir(exist_ok=True)
        previous = target / 'provenance.json'
        recorded = json.loads(previous.read_text()) if previous.is_file() else None
        repo = f'timm/{spec.checkpoint}'
        api = f'https://huggingface.co/api/models/{repo}'
        if recorded:
            api += '/revision/' + recorded['revision']
        with fetch(api + '?blobs=true') as response:
            metadata = json.load(response)
        revision = metadata['sha']
        sibling = next(item for item in metadata['siblings'] if item['rfilename'] == 'model.safetensors')
        expected = sibling['lfs']['sha256']
        weights = target / 'model.safetensors'
        if not weights.exists():
            temporary = target / 'model.safetensors.partial'
            with fetch(f'https://huggingface.co/{repo}/resolve/{revision}/model.safetensors') as source, temporary.open('wb') as output:
                size = 0
                while block := source.read(1024 * 1024):
                    output.write(block); size += len(block)
                    if size > 200 * 1024**2:
                        raise ValueError('Unexpected checkpoint size.')
            if sha256_file(temporary) != expected:
                raise ValueError('Downloaded checkpoint differs from Hub LFS SHA-256.')
            temporary.replace(weights)
        if sha256_file(weights) != expected or weights.stat().st_size != sibling['size']:
            raise ValueError('Cached checkpoint does not match immutable Hub metadata.')
        for name in ('config.json', 'README.md'):
            with fetch(f'https://huggingface.co/{repo}/resolve/{revision}/{name}') as response:
                (target / name).write_bytes(response.read())
        config = json.loads((target / 'config.json').read_text())
        cfg = config['pretrained_cfg']
        if tuple(cfg['mean']) != spec.mean or tuple(cfg['std']) != spec.std:
            raise ValueError('Registry normalization differs from published configuration.')
        record = {'repository': repo, 'revision': revision, 'sha256': expected, 'bytes': weights.stat().st_size,
            'relative_path': weights.relative_to(ROOT).as_posix(), 'license': metadata['cardData']['license'],
            'source_url': f'https://huggingface.co/{repo}/resolve/{revision}/model.safetensors',
            'retrieved_utc': utc_now(), 'config_sha256': sha256_file(target / 'config.json'),
            'model_card_sha256': sha256_file(target / 'README.md'), 'configuration': config}
        if recorded and any(record[key] != recorded[key] for key in ('revision', 'sha256', 'repository')):
            raise ValueError('Pinned pretrained identity changed.')
        write_json(previous, record)
        print(json.dumps({'method': method, 'revision': revision, 'sha256': expected, 'bytes': weights.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
