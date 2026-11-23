"""Package the active six-run local app, excluding photos and historical caches."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.runtime import sha256_file, write_json, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'output/VisAge_Complete_Web_App_v0.4.0.zip')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Do not overwrite a delivered archive.')
    runtime = ROOT / 'output/VisAge_Runtime_Assets_v0.4.0.zip'
    record = json.loads((ROOT / 'configs/runtime_release.json').read_text())
    if sha256_file(runtime) != record['sha256']: raise ValueError('Runtime archive changed.')
    acceptance = json.loads((ROOT / 'reports/pretrained/ui_refinement/acceptance.json').read_text())
    if not acceptance['ok']: raise ValueError('The active interface must pass acceptance before packaging.')
    entries = {}
    for folder in ('src', 'scripts', 'tests', 'configs', 'licenses'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.log','.pid','.partial'):
                entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ('README.md','pyproject.toml','package.json','requirements.txt','.python-version','vercel.json','Dockerfile','.dockerignore','.vercelignore','THIRD_PARTY_NOTICES.md','api/index.py','docs/Deployment.md'):
        entries[name] = (ROOT / name).read_bytes()
    with zipfile.ZipFile(runtime) as archive:
        for name in archive.namelist():
            if name != 'runtime_manifest.json': entries[name] = archive.read(name)
    for name in ('selection.json','two_seed_summary.json','two_seed_comparison.csv','two_seed_validation_means.csv','two_seed_per_class.csv','two_seed_test_per_class.csv','hardware_acceptance.json','pilot_scope.json','ui_refinement/acceptance.json'):
        path = ROOT / 'reports/pretrained' / name
        if path.exists(): entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    manifest = {'created_utc':utc_now(),'scope':'Standalone local inference delivery: three pretrained seed-42 models, active web UI/API, detector and offline evaluation. Research reproduction still requires separate permitted data.',
        'seeds':[42,43],'training_runs':6,'default_method':'swinv2_tiny','photos_included':False,
        'environment_included':False,'full_report':'Delivered alongside this archive to avoid self-referential package hashes.',
        'files':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in sorted(entries.items())]}
    forbidden = ('data/raw/','data/cache/','.venv/','artifacts/pretrained/runs/')
    if any(name.startswith(forbidden) or name.endswith(('last.pt','.jpg','.jpeg','.heic')) for name in entries): raise ValueError('Forbidden private/research package content.')
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for name,data in entries.items(): archive.writestr(name,data)
        archive.writestr('delivery_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip(): raise ValueError('Delivery CRC validation failed.')
        for item in manifest['files']:
            if hashlib.sha256(archive.read(item['path'])).hexdigest()!=item['sha256']:raise ValueError('Delivery entry changed.')
    result={'ok':True,'created_utc':utc_now(),'path':str(args.output),'bytes':args.output.stat().st_size,'sha256':sha256_file(args.output),'files':len(entries),'runtime_archive_sha256':record['sha256'],'fresh_extraction':'Pending separate isolated startup and prediction checks.'}
    write_json(ROOT/'reports/pretrained/delivery/package.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
