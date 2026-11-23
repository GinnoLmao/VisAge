import json
from copy import deepcopy
import pytest
from visage.deployment import default_artifacts_dir, operational_decision


def test_missing_and_prepared_bundles_leave_original_models_active(tmp_path):
    assert default_artifacts_dir(tmp_path)==tmp_path/'artifacts'
    bundle=tmp_path/'artifacts/serving'
    bundle.mkdir(parents=True)
    (bundle/'deployment.json').write_text(json.dumps({'status':'prepared'}))
    assert default_artifacts_dir(tmp_path)==tmp_path/'artifacts'


def test_only_complete_active_bundle_is_selected(tmp_path):
    bundle=tmp_path/'artifacts/serving'
    bundle.mkdir(parents=True)
    marker=bundle/'deployment.json'
    marker.write_text(json.dumps({'status':'active','methods':{'visagenet':{}}}))
    with pytest.raises(ValueError,match='all three'):
        default_artifacts_dir(tmp_path)
    marker.write_text(json.dumps({'status':'active','methods':{m:{} for m in ('classical','visagenet','convnext_ordinal')}}))
    assert default_artifacts_dir(tmp_path)==bundle
    marker.write_text(json.dumps({'status':'unknown'}))
    with pytest.raises(ValueError,match='status'):
        default_artifacts_dir(tmp_path)


def comparison():
    metrics={'macro_f1':.84,'under18_predicted_older_rate':.1,
             'per_class':[{}, {}, {'recall':.8}]}
    baseline={'ok':True,'manifest_sha256':'freeze','face_config':{'policy':3},'attempted':2,
              'artifact_sha256':{'classical':'a','visagenet':'b','convnext_ordinal':'c'},
              'methods':{'visagenet':{'accepted_image_metrics':metrics}}}
    candidate=deepcopy(baseline)
    candidate['artifact_sha256']['visagenet']='new-b'
    candidate['methods']['visagenet']['accepted_image_metrics']['macro_f1']=.85
    rows=[{'sample_id':str(i),'label':str(i),'status':'accepted','error_code':'','crop_sha256':'pixels'+str(i)} for i in range(2)]
    def benchmark(record, milliseconds):
        return {'ok':True,'no_concurrent_training':True,'sample_id':'0',
            'cleanup':{'server_process_stopped':True,'port_closed':True},
            'artifact_sha256':record['artifact_sha256'],
            'method_preprocessing_and_inference':{'visagenet':{'mean_ms':milliseconds}}}
    protocol={'pipeline_macro_f1_drop_max':.005,'pipeline_minor_error_increase_max':.01,
              'pipeline_senior_recall_drop_max':.01}
    return baseline,candidate,rows,deepcopy(rows),benchmark(baseline,10),benchmark(candidate,11),protocol


def test_operational_gates_accept_improvement_with_identical_faces():
    result=operational_decision(*comparison())
    assert result['qualified'] and all(result['checks'].values())


def test_model_study_success_cannot_override_pipeline_safety_or_latency_regression():
    values=comparison()
    values[1]['methods']['visagenet']['accepted_image_metrics']['under18_predicted_older_rate']=.12
    values[5]['method_preprocessing_and_inference']['visagenet']['mean_ms']=20
    result=operational_decision(*values)
    assert not result['qualified']
    assert not result['checks']['pipeline_minor_error'] and not result['checks']['latency']


@pytest.mark.parametrize('field,value',[('crop_sha256','other-pixels'),('status','rejected'),('sample_id','other-id')])
def test_operational_comparison_refuses_different_selected_faces(field,value):
    values=comparison()
    values[3][0][field]=value
    with pytest.raises(ValueError,match='crop hashes'):
        operational_decision(*values)


def test_operational_comparison_refuses_concurrent_or_unowned_benchmark():
    values=comparison()
    values[5]['no_concurrent_training']=False
    with pytest.raises(ValueError,match='Isolated'):
        operational_decision(*values)
