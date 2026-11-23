"""Build a photo-free runtime archive with original model bytes and compact curves."""
import json
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import sha256_file, write_json, utc_now
from visage.model_registry import ARTIFACTS
from visage.evaluation_registry import load_compact_record
from visage.faces import DETECTOR_FILENAME


def main():
    destination = ROOT / 'output/VisAge_Runtime_Assets_v0.4.0.zip'
    if destination.exists():
        raise FileExistsError('Do not replace an existing published runtime archive.')
    reports = ROOT / 'reports/pretrained'
    entries = {}
    for p in ARTIFACTS.rglob('*'):
        if p.is_file(): entries[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    for name in (DETECTOR_FILENAME, 'LICENSE', 'provenance.json'):
        p = ROOT / '.cache/face_detector' / name
        entries[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    for name, kind in [('classification_roc.json', 'classification'), ('detection_ap.json', 'detection')]:
        p = reports / 'evaluation' / name
        entries[p.relative_to(ROOT).as_posix()] = json.dumps(load_compact_record(p, kind), allow_nan=False).encode()
    for path in ('reports/pretrained/evaluation/application_profile.json', 'reports/pretrained/face_pipeline_validation.json'):
        entries[path] = (ROOT / path).read_bytes()
    for p in (ROOT / 'licenses/pretrained').rglob('*'):
        if p.is_file(): entries[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    import hashlib
    manifest = {'created_utc': utc_now(), 'scope': 'Exactly three serving models, detector, compact dashboard evidence and attribution; no photos or optimizer states.',
        'files': [{'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in sorted(entries.items())]}
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for name, payload in entries.items(): archive.writestr(name, payload)
        archive.writestr('runtime_manifest.json', json.dumps(manifest, indent=2))
    write_json(ROOT / 'configs/runtime_release.json', {'version': 'v0.4.0',
        'url': 'https://github.com/GinnoLmao/VisAge/releases/download/v0.4.0/VisAge_Runtime_Assets_v0.4.0.zip',
        'bytes': destination.stat().st_size, 'sha256': sha256_file(destination), 'file_count': len(entries)})
    write_json(reports / 'runtime_assets.json', {'ok': True, 'path': str(destination), 'sha256': sha256_file(destination), 'files': len(entries), 'photos_included': False})
    print(json.dumps({'ok': True, 'bytes': destination.stat().st_size, 'files': len(entries)}), flush=True)


if __name__ == '__main__': main()
