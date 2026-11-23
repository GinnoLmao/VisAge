"""Set a conservative gross-blur gate from training images, without age labels."""
from __future__ import annotations
import csv
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from visage.faces import FaceConfig,FaceDetector,FaceError
from visage.runtime import sha256_file,write_json,utc_now


def main():
    cv2.setNumThreads(2)
    with (ROOT/'data/processed/manifest.csv').open(newline='',encoding='utf-8') as stream:
        rows=[row for row in csv.DictReader(stream) if row['split']=='train']
    chosen=[rows[i] for i in np.linspace(0,len(rows)-1,500,dtype=int)]
    detector=FaceDetector(config=FaceConfig(min_blur_variance=0))
    records=[]
    rejected={}
    for row in chosen:
        path=ROOT/'data/raw/UTKFace'/row['relative_path']
        if sha256_file(path)!=row['file_sha256']:
            raise ValueError('Frozen training image changed.')
        with Image.open(path) as original:
            image=ImageOps.exif_transpose(original).convert('RGB')
            try:
                face=detector.prepare(image)
                records.append({'sample_id':row['sample_id'],'laplacian_variance':face.quality['laplacian_variance']})
                face.crop.close()
            except FaceError as exc:
                rejected[exc.code]=rejected.get(exc.code,0)+1
            finally:
                image.close()
    if len(records)<400:
        raise ValueError('Too few valid training faces for gross-blur calibration.')
    values=np.array([r['laplacian_variance'] for r in records])
    # Low percentile avoids treating ordinary smooth skin as severe blur.
    # At most1.0: a Laplacian variance proxy alone is content-dependent and must
    # not become an aggressive age/skin-texture admission gate.
    threshold=float(min(1.0,np.quantile(values,.005)))
    write_json(ROOT/'reports/face_quality_fit.json',{'ok':True,'split':'train','age_labels_used':False,
        'classifier_outputs_used':False,'attempted':500,'samples':len(records),'rejected_other_reasons':rejected,
        'min_blur_variance':threshold,'selection_rule':'Minimum of1.0 and training0.5th percentile; conservative gross-blur guard only.',
        'quantiles':{str(q):float(np.quantile(values,q)) for q in (0,.005,.01,.05,.5,.95,1)},
        'old_threshold10_would_reject':int((values<10).sum()),'new_threshold_would_reject':int((values<threshold).sum()),
        'records':records,'completed_utc':utc_now(),'manifest_sha256':sha256_file(ROOT/'data/processed/manifest.csv'),
        'limitation':'Engineering calibration, not human-labeled image quality. Independent local-photo validation remains pending.'})
    print('Training gross-blur threshold',threshold,'quantiles',np.quantile(values,[0,.005,.01,.05,.5]),flush=True)


if __name__=='__main__':
    main()
