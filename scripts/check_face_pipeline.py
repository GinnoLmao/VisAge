"""Independent count/metric verification of the full shared-crop diagnostic."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from visage.runtime import sha256_file,write_json,utc_now
from visage.data import load_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/face_pipeline_validation.json')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/face_pipeline_independent_review.json')
    args=parser.parse_args()
    record=json.loads(args.report.read_text())
    spec=importlib.util.spec_from_file_location('independent_research_arithmetic',ROOT/'scripts/check_research.py')
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    path=ROOT/record['predictions_csv']
    assert sha256_file(path)==record['predictions_sha256']
    with path.open(newline='',encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    manifest=ROOT/'data/processed/manifest.csv'
    frozen=[row for row in load_manifest(manifest) if row['split']=='validation']
    if len(rows)!=len(frozen) or any(a['sample_id']!=b['sample_id'] or int(a['label'])!=b['label'] for a,b in zip(rows,frozen,strict=True)):
        raise ValueError('Pipeline predictions differ from frozen validation rows.')
    for relative,expected in record['source_sha256'].items():
        if sha256_file(ROOT/relative)!=expected:
            raise ValueError('Pipeline source/evidence changed after execution.')
    accepted=[row for row in rows if row['status']=='accepted']
    rejected=[row for row in rows if row['status']=='rejected']
    if len(accepted)+len(rejected)!=len(rows):
        raise ValueError('Unknown pipeline outcome.')
    helper.close_values(record,{'attempted':len(rows),'accepted':len(accepted),'rejected':len(rejected),
        'coverage':len(accepted)/len(rows),'rejection_codes':dict(Counter(row['error_code'] for row in rejected))},'coverage')
    checks={}
    for method in ('classical','visagenet','convnext_ordinal'):
        truth=[int(row['label']) for row in accepted]
        predicted=[int(row[method]) for row in accepted]
        original=[int(row['baseline_'+method]) for row in accepted]
        now=helper.independent_metrics(truth,predicted) if accepted else None
        before=helper.independent_metrics(truth,original) if accepted else None
        correct=sum(a==b for a,b in zip(truth,predicted,strict=True))
        expected={'accepted_image_metrics':now,'original_aligned_metrics_same_accepted_images':before,
            'end_to_end_correct_count':correct,'end_to_end_correct_rate':correct/len(rows),
            'prediction_changes_on_accepted':sum(a!=b for a,b in zip(predicted,original,strict=True))}
        helper.close_values(record['methods'][method],expected,method)
        checks[method]={'accepted_accuracy':now['accuracy'] if now else None,
            'accepted_macro_f1':now['macro_f1'] if now else None,'end_to_end_correct_rate':correct/len(rows)}
    write_json(args.output,{'ok':True,'checked_utc':utc_now(),'manifest_sha256':sha256_file(manifest),
        'report_sha256':sha256_file(args.report),'predictions_sha256':sha256_file(path),
        'samples':len(rows),'accepted':len(accepted),'coverage':len(accepted)/len(rows),'methods':checks,
        'scope':'Independent Counter arithmetic and frozen row/source identity; no image inference or test-image access.'})
    print(json.dumps({'ok':True,'coverage':len(accepted)/len(rows),'methods':checks}),flush=True)


if __name__=='__main__':
    main()
