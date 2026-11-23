"""Estimate canonical eye geometry from fixed training crops, without age labels.

This aligns ordinary-photo preprocessing with the original training-crop domain.
No validation/test labels, classifier outputs or source photos are retained.
"""
from __future__ import annotations
import csv
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from visage.faces import FaceDetector
from visage.runtime import sha256_file,write_json,utc_now


def main():
    import cv2
    cv2.setNumThreads(2)
    manifest=ROOT/'data/processed/manifest.csv'
    with manifest.open(newline='',encoding='utf-8') as stream:
        rows=[row for row in csv.DictReader(stream) if row['split']=='train']
    chosen=[rows[index] for index in np.linspace(0,len(rows)-1,500,dtype=int)]
    detector=FaceDetector()
    geometry=[]
    for i,row in enumerate(chosen):
        path=ROOT/'data/raw/UTKFace'/row['relative_path']
        if sha256_file(path)!=row['file_sha256']:
            raise ValueError('Frozen training image changed.')
        with Image.open(path) as source:
            image=ImageOps.exif_transpose(source).convert('RGB')
            faces=detector.detect(image)
            if len(faces)==1:
                eyes=np.array(faces[0].landmarks[:2])
                midpoint=eyes.mean(axis=0)
                distance=float(np.linalg.norm(eyes[1]-eyes[0]))
                if distance>=10:
                    geometry.append({'sample_id':row['sample_id'],'eye_mid_x':float(midpoint[0]/image.width),
                        'eye_mid_y':float(midpoint[1]/image.height),'eye_distance_fraction':distance/image.width,
                        'bbox_width_fraction':faces[0].bbox[2]/image.width,'bbox_height_fraction':faces[0].bbox[3]/image.height})
            image.close()
        if (i+1)%100==0:
            print(f'Training geometry {i+1}/500',flush=True)
    if len(geometry)<400:
        raise ValueError('Too few unambiguous training detections for stable geometry fit.')
    columns=('eye_mid_x','eye_mid_y','eye_distance_fraction','bbox_width_fraction','bbox_height_fraction')
    targets={key:float(np.median([row[key] for row in geometry])) for key in columns}
    record={'ok':True,'fit_split':'train','labels_used':False,'classifier_predictions_used':False,
        'manifest_sha256':sha256_file(manifest),'detector_sha256':detector.info()['sha256'],
        'sampling':'500 evenly spaced frozen training-row positions; one face and >=10px interocular distance required',
        'attempted':len(chosen),'accepted':len(geometry),'target_medians':targets,
        'quantiles':{key:np.quantile([row[key] for row in geometry],[.1,.5,.9]).tolist() for key in columns},
        'training_geometry':geometry,'completed_utc':utc_now(),
        'scope':'Geometry calibration only; age labels and validation/test images unused. Live-photo performance still needs independent pilot data.'}
    write_json(ROOT/'reports/face_geometry_fit.json',record)
    print(targets,flush=True)


if __name__=='__main__':
    main()
