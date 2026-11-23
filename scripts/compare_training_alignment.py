"""Compare two/five-landmark alignment by training-pixel reconstruction only.

No age labels, classifier scores, validation images or test images are read.
The reference is the original training crop resized to224; reconstruction is a
domain-consistency diagnostic, not an independent age-performance measure.
"""
from __future__ import annotations
import csv
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from visage.faces import FaceDetector
from visage.runtime import sha256_file,write_json,utc_now


def similarity(source,target):
    source,target=np.asarray(source,dtype=float),np.asarray(target,dtype=float)
    src_mean,tgt_mean=source.mean(0),target.mean(0)
    x,y=source-src_mean,target-tgt_mean
    denominator=float((x*x).sum())
    if denominator<1e-8:
        raise ValueError('Degenerate alignment landmarks.')
    a=float((x*y).sum()/denominator)
    b=float((x[:,0]*y[:,1]-x[:,1]*y[:,0]).sum()/denominator)
    rotation_scale=np.array([[a,-b],[b,a]])
    return np.c_[rotation_scale,tgt_mean-rotation_scale@src_mean]


def main():
    cv2.setNumThreads(2)
    with (ROOT/'data/processed/manifest.csv').open(newline='',encoding='utf-8') as stream:
        rows=[row for row in csv.DictReader(stream) if row['split']=='train']
    chosen=[rows[i] for i in np.linspace(0,len(rows)-1,500,dtype=int)]
    detector=FaceDetector()
    records=[]
    for row in chosen:
        path=ROOT/'data/raw/UTKFace'/row['relative_path']
        if sha256_file(path)!=row['file_sha256']:
            raise ValueError('Frozen training image changed.')
        with Image.open(path) as original:
            image=ImageOps.exif_transpose(original).convert('RGB')
            faces=detector.detect(image)
            if len(faces)==1:
                points=np.array(faces[0].landmarks,dtype=float)
                points[:2]=points[:2][np.argsort(points[:2,0])]
                points[3:]=points[3:][np.argsort(points[3:,0])]
                if np.linalg.norm(points[1]-points[0])>=10:
                    records.append({'sample_id':row['sample_id'],'relative_path':row['relative_path'],
                        'points':points.tolist(),'normalized':(points/np.array(image.size)).tolist()})
            image.close()
    target=np.median([r['normalized'] for r in records],axis=0)
    errors={'two_eye_v2':[],'five_point_candidate':[]}
    import json
    fitted=json.loads((ROOT/'reports/face_geometry_fit.json').read_text())['target_medians']
    for record in records:
        with Image.open(ROOT/'data/raw/UTKFace'/record['relative_path']) as original:
            image=ImageOps.exif_transpose(original).convert('RGB')
            pixels=np.asarray(image)
            reference=np.asarray(image.resize((224,224),Image.Resampling.BICUBIC),dtype=float)
            points=np.array(record['points'])
            delta=points[1]-points[0]
            midpoint=points[:2].mean(0)
            side=np.linalg.norm(delta)/fitted['eye_distance_fraction']
            matrix=cv2.getRotationMatrix2D(tuple(midpoint),np.degrees(np.arctan2(delta[1],delta[0])),224/side)
            matrix[:,2]+=np.array([224*fitted['eye_mid_x'],224*fitted['eye_mid_y']])-midpoint
            matrices={'two_eye_v2':matrix,'five_point_candidate':similarity(points,target*224)}
            for name,matrix in matrices.items():
                changed=cv2.warpAffine(pixels,matrix,(224,224),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                errors[name].append(float(np.abs(changed.astype(float)-reference).mean()))
            image.close()
    aggregate={name:{'mean_absolute_RGB_error':float(np.mean(v)),'median_absolute_RGB_error':float(np.median(v))} for name,v in errors.items()}
    selected=min(aggregate,key=lambda name:aggregate[name]['mean_absolute_RGB_error'])
    write_json(ROOT/'reports/training_alignment_comparison.json',{'ok':True,'fit_split':'train','age_labels_used':False,
        'classifier_predictions_used':False,'validation_or_test_images_used':False,'samples':len(records),
        'manifest_sha256':sha256_file(ROOT/'data/processed/manifest.csv'),'five_point_template_normalized':target.tolist(),
        'methods':aggregate,'selected_by_training_reconstruction':selected,'selected_geometry_is_five_point':selected=='five_point_candidate',
        'selection_rule':'Lower mean absolute RGB reconstruction error against resized original training crops; no age-performance selection.',
        'completed_utc':utc_now(),'training_sample_ids':[row['sample_id'] for row in records]})
    print(aggregate,selected,flush=True)


if __name__=='__main__':
    main()
