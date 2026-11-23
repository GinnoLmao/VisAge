"""Real loopback HTTP functional checks; image bytes stay in memory only."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from benchmark_backend import validation_input,wait_for_server,stop_owned_server,validate_prediction,artifact_hashes
from visage.runtime import write_json,utc_now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/api_http_smoke.json')
    parser.add_argument('--artifacts-dir',type=Path,default=ROOT/'artifacts')
    args=parser.parse_args()
    if not 1<=args.port<=65535:
        parser.error('Use a valid TCP port.')
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Choose a fresh report path inside the project.')
    log_path=args.output.with_suffix('.server.log')
    if log_path.exists():
        raise FileExistsError('Preserve the earlier server log; choose a new report path.')
    row,data,manifest_sha=validation_input()
    hashes=artifact_hashes(args.artifacts_dir)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',args.port))
    environment=os.environ.copy()
    environment.update({'OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OMP_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'})
    command=[sys.executable,'-u',str(ROOT/'scripts/serve_api.py'),'--port',str(args.port),
             '--artifacts-dir',str(args.artifacts_dir),'--device','cpu']
    import httpx
    from PIL import Image
    process=None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('w',encoding='utf-8') as log:
        try:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',timeout=60,trust_env=False) as client:
                health=wait_for_server(process,client,log_path,args.port,180)
                models=client.get('/v1/models'); models.raise_for_status()
                faces=client.post('/v1/faces',content=data,headers={'Content-Type':'image/jpeg'})
                faces.raise_for_status()
                prediction=client.post('/v1/predict?method=all',content=data,headers={'Content-Type':'image/jpeg'})
                prediction.raise_for_status()
                signature=validate_prediction(prediction.json(),manifest_sha)
                evaluation=client.get('/v1/evaluation'); evaluation.raise_for_status()
                if any(evaluation.json()[key]['status']!='complete' for key in ('classification','detection','face_pipeline')):
                    raise ValueError('All aggregate evaluation components must be complete before final smoke.')
                if prediction.headers.get('cache-control')!='no-store':
                    raise ValueError('Prediction HTTP responses must disable caching.')
                checks={}
                for name,content,kind,status,code in (
                    ('malformed',b'not image data','image/jpeg',400,'INVALID_IMAGE'),
                    ('wrong_content_type',data,'text/plain',415,'UNSUPPORTED_MEDIA_TYPE')):
                    response=client.post('/v1/predict',content=content,headers={'Content-Type':kind})
                    assert response.status_code==status and response.json()['error']['code']==code
                    checks[name]={'status':status,'code':code}
                blank=BytesIO()
                Image.new('RGB',(500,300),(127,127,127)).save(blank,format='PNG')
                response=client.post('/v1/predict',content=blank.getvalue(),headers={'Content-Type':'image/png'})
                assert response.status_code==422 and response.json()['error']['code']=='NO_FACE'
                checks['blank']={'status':422,'code':'NO_FACE'}
                record={'ok':True,'recorded_utc':utc_now(),'transport':'real loopback HTTP through Uvicorn',
                    'host':'127.0.0.1','port':args.port,'device':'cpu','sample_id':row['sample_id'],
                    'input_sha256':row['file_sha256'],'health':health,'models':models.json(),
                    'artifact_sha256':hashes,'artifacts_dir':str(args.artifacts_dir.resolve()),
                    'faces':faces.json(),'prediction':prediction.json(),'consistent_result':signature,
                    'evaluation':{'classification_status':evaluation.json()['classification']['status'],
                        'detection_status':evaluation.json()['detection']['status'],'face_pipeline_status':evaluation.json()['face_pipeline']['status']},
                    'errors':checks,'privacy':'No uploaded bytes or photos saved; only structured outputs recorded.',
                    'timing_note':'Functional smoke; concurrent training may run, so response times are not isolated benchmark evidence.'}
        finally:
            if process is not None:
                cleanup=stop_owned_server(process,args.port)
    record['cleanup']=cleanup
    if artifact_hashes(args.artifacts_dir)!=hashes:
        raise ValueError('Model artifacts changed during HTTP smoke.')
    write_json(args.output,record)
    print(json.dumps({'ok':True,'cleanup':cleanup,'output':str(args.output)}),flush=True)


if __name__=='__main__':
    main()
