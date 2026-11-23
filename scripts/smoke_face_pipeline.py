"""Reproducible real YuNet smoke checks on validation inputs only."""
from __future__ import annotations
import csv
from io import BytesIO
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from PIL import Image
import cv2
from visage.faces import FaceDetector,FaceError,decode_image
from visage.runtime import write_json,sha256_file,utc_now


def main():
    cv2.setNumThreads(2)
    with (ROOT/'data/processed/manifest.csv').open(newline='',encoding='utf-8') as stream:
        rows=[row for row in csv.DictReader(stream) if row['split']=='validation'][:30]
    detector=FaceDetector()
    samples=[]
    accepted_image=None
    for row in rows:
        path=ROOT/'data/raw/UTKFace'/row['relative_path']
        if sha256_file(path)!=row['file_sha256']:
            raise ValueError('Frozen validation image changed.')
        image=decode_image(path.read_bytes())
        try:
            prepared=detector.prepare(image)
            samples.append({'sample_id':row['sample_id'],'status':'accepted','crop_sha256':prepared.metadata['pixel_sha256']})
            prepared.crop.close()
            if accepted_image is None:
                accepted_image=image.copy()
        except FaceError as exc:
            samples.append({'sample_id':row['sample_id'],'status':'rejected','error_code':exc.code})
        finally:
            image.close()
    if accepted_image is None:
        raise RuntimeError('No usable real validation face for multiple-selection smoke test.')
    blank=Image.new('RGB',(500,300),(127,127,127))
    assert not detector.detect(blank)
    blank.close()
    pair=Image.new('RGB',(500,260),(127,127,127))
    pair.paste(accepted_image.resize((200,200)),(20,30))
    pair.paste(accepted_image.resize((200,200)),(280,30))
    candidates=detector.detect(pair)
    assert len(candidates)==2
    try:
        detector.prepare(pair,candidates=candidates)
    except FaceError as exc:
        assert exc.code=='MULTIPLE_FACES'
    else:
        raise AssertionError('Multiple faces must require explicit selection.')
    prepared=detector.prepare(pair,face_index=1,candidates=candidates)
    prepared.crop.close(); pair.close(); accepted_image.close()
    write_json(ROOT/'reports/face_pipeline_smoke.json',{'ok':True,'recorded_utc':utc_now(),
        'detector':detector.info(),'validation_samples':len(samples),'accepted':sum(s['status']=='accepted' for s in samples),
        'samples':samples,'blank_no_face':True,'multiple_requires_explicit_index':True,'explicit_selection_works':True,
        'scope':'Real validation-face functional check; not an accuracy, AP or independent local-pilot claim. No photos saved.'})
    print('Real face/blank/multiple-selection smoke passed.')


if __name__=='__main__':
    main()
