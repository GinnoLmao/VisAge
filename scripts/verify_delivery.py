"""Verify fresh ZIP extraction using only its source and serving artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile
import httpx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from visage.runtime import write_json,sha256_file,utc_now
from benchmark_backend import wait_for_server,stop_owned_server


def main():
    report=ROOT/'reports/pretrained/delivery';record=json.loads((report/'package.json').read_text())
    archive_path=Path(record['path']);target=ROOT/'tmp'/f'final_delivery_{uuid.uuid4().hex[:10]}'
    target.mkdir(parents=True)
    if sha256_file(archive_path)!=record['sha256']:raise ValueError('Delivery archive changed.')
    with zipfile.ZipFile(archive_path) as archive:
        manifest=json.loads(archive.read('delivery_manifest.json'))
        expected={i['path'] for i in manifest['files']}|{'delivery_manifest.json'}
        if set(archive.namelist())!=expected or len(expected)!=len(archive.namelist()):raise ValueError('Unexpected ZIP inventory.')
        for item in manifest['files']:
            destination=(target/item['path']).resolve()
            if not destination.is_relative_to(target):raise ValueError('Unsafe ZIP path.')
            data=archive.read(item['path'])
            if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('ZIP entry differs from its manifest.')
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
    environment=os.environ.copy();environment.pop('PYTHONPATH',None)
    environment.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
    code='import runpy,sys;from pathlib import Path;r=Path.cwd();sys.path.insert(0,str(r/"src"));import visage;assert Path(visage.__file__).resolve().is_relative_to(r);sys.argv=["serve_api.py","--port","8767"];runpy.run_path(str(r/"scripts/serve_api.py"),run_name="__main__")'
    logpath=report/'extracted_server.log';checks={}
    with logpath.open('w') as log:
        server=subprocess.Popen([sys.executable,'-I','-u','-c',code],cwd=target,env=environment,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            with httpx.Client(base_url='http://127.0.0.1:8767',trust_env=False,timeout=60) as client:
                wait_for_server(server,client,logpath,8767,120)
                shell=client.get('/');shell.raise_for_status();assert 'Academic prototype' not in shell.text and 'score-explanation' not in shell.text
                models=client.get('/v1/models');models.raise_for_status();models=models.json();assert models['default_method']=='swinv2_tiny'
                evaluation=client.get('/v1/evaluation');evaluation.raise_for_status();evaluation=evaluation.json()
                for value in evaluation['classification']['methods'].values():assert value['runtime_median_ms']>0 and value['parameters_millions']>0 and value['training_minutes']>0
                fixture=ROOT/'.cache/web_qa/single.jpg'
                predicted=client.post('/v1/predict?method=all&face_index=0',content=fixture.read_bytes(),headers={'Content-Type':'image/jpeg'});predicted.raise_for_status();predicted=predicted.json()
                baseline=json.loads((ROOT/'reports/pretrained/score_feature/live_scores.json').read_text())
                assert {p['method']:p['class_id'] for p in predicted['predictions']}=={p['method']:p['class_id'] for p in baseline['predictions']}
                for value in predicted['predictions']:assert len(value['class_scores'])==3 and abs(sum(c['score'] for c in value['class_scores'])-1)<1e-6
                heic=client.post('/v1/faces',content=(ROOT/'.cache/web_qa/single.heic').read_bytes(),headers={'Content-Type':'image/heic'});heic.raise_for_status();assert heic.json()['preview']['content_type']=='image/jpeg'
                checks={'shell':True,'models':True,'evaluation_cost_columns':True,'three_predictions_and_scores':True,'labels_match_workspace':True,'heic_preview':True}
        finally:cleanup=stop_owned_server(server,8767)
    result={'ok':True,'verified_utc':utc_now(),'package_sha256':record['sha256'],'extraction':str(target),'files_verified':len(manifest['files']),
        'isolated_import_mode':True,'checks':checks,'cleanup':cleanup,'environment_scope':'Fresh extraction and isolated source import using the verified existing Python dependencies. A separate clean-machine dependency installation is not claimed.',
        'fixture_scope':'Existing public validation fixture supplied externally; no face photos are in the ZIP.'}
    write_json(report/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
