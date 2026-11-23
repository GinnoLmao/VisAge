"""Audit the approved six-run activation without training or rerunning inference."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.model_registry import METHODS, ARTIFACTS
from visage.pretrained_two_seed import freeze_amendment, freeze_evaluation_amendment
from visage.runtime import sha256_file, write_json, utc_now


def main():
    reports = ROOT / 'reports/pretrained'
    def read(name):
        return json.loads((reports / name).read_text(encoding='utf-8'))
    amendment = freeze_amendment()  # Also verifies original frozen inputs/evidence.
    freeze_evaluation_amendment()
    for excluded in amendment['excluded_runs']:
        directory = ROOT / 'artifacts/pretrained/runs' / excluded['run_id']
        for name, digest in excluded['retained_files_sha256'].items():
            if sha256_file(directory / name) != digest:
                raise ValueError('Excluded unfinished-run evidence changed.')
    verified, reloads = read('verified_runs.json'), read('reload_verification.json')
    if not verified['ok'] or verified['count'] != 6 or len(reloads['runs']) != 6:
        raise ValueError('Six complete verified reloads are required.')
    if not all(r['all_validation_labels_reproduced'] and r['samples'] == 3492 for r in reloads['runs']):
        raise ValueError('Validation reload evidence is incomplete.')
    selection = read('selection.json')
    bundle = json.loads((ARTIFACTS / 'bundle.json').read_text())
    if selection['seeds'] != [42, 43] or selection['test_results_used'] or bundle['default_method'] != selection['default_method']:
        raise ValueError('Default choice violates the approved validation-only rule.')
    test, robust, pipeline = read('test_summary.json'), read('robustness.json'), read('face_pipeline_validation.json')
    roc, devices = read('evaluation/classification_roc.json'), read('device_verification.json')
    selection_sha = sha256_file(reports / 'selection.json')
    if any(r['selection_sha256'] != selection_sha for r in (bundle, test, pipeline)):
        raise ValueError('Bundle or evaluation belongs to a different default decision.')
    if not all(r['ok'] for r in (test, robust, pipeline, roc, devices)) or test['samples'] != 3493 or robust['samples'] != 3492:
        raise ValueError('Evaluation scope is incomplete.')
    hashes = {}
    for method in METHODS:
        selected = next(r for r in verified['runs'] if r['method'] == method and r['seed'] == 42)
        digest = sha256_file(ARTIFACTS / method / 'best.pt')
        hashes[method] = digest
        records = [selected, bundle['methods'][method], test['methods'][method], robust['methods'][method],
            roc['methods'][method], devices['methods'][method]]
        if any(r['artifact_sha256'] != digest for r in records) or pipeline['artifact_sha256'][method] != digest:
            raise ValueError('Serving artifact/evaluation checksums disagree.')
        if len(robust['methods'][method]['conditions']) != 7:
            raise ValueError('All seven validation conditions are required.')
        if bundle['methods'][method]['seed'] != 42 or bundle['methods'][method]['selected_stage'] != selected['selected_stage']:
            raise ValueError('Serving checkpoint stage or seed differs from the audit.')
    lineage = read('evaluation/detection_lineage.json')
    acceptance = read('web_app/automated_acceptance.json')
    visual = read('web_app/visual_review.json')
    if not lineage['ok'] or not acceptance['ok'] or not visual['ok']:
        raise ValueError('Detector lineage, acceptance and visual review must pass.')
    for name, expected in acceptance['counts'].items():
        xml = ET.parse(reports / 'web_app' / name)
        cases = xml.findall('.//testcase')
        if len(cases) != expected or any(case.find(tag) is not None for case in cases for tag in ('failure', 'error', 'skipped')):
            raise ValueError('Regression results are incomplete.')
    events = read('web_app/browser_events_final.json')
    suite = next(e for e in events if e.get('event') == 'upload-integration')
    if not suite['ok'] or len(suite['passed']) < 31:
        raise ValueError('Browser workflow checks are incomplete.')
    for name in ('web_app/production_cleanup.json', 'web_app/headless_cleanup.json'):
        record = read(name)
        server = record.get('server', record)
        if not server['port_closed'] or server['remaining_owned_process_ids']:
            raise ValueError('Temporary HTTP process cleanup is incomplete.')
    benchmark = read('benchmark/backend_benchmark.json')
    if not benchmark['ok'] or not benchmark['cleanup']['port_closed'] or benchmark['iterations'] != 30:
        raise ValueError('Isolated HTTP latency evidence is incomplete.')
    activation = read('app_activation.json')
    active_sources = dict(activation['staged'])
    extension = None
    extension_path = reports / 'score_feature/implementation.json'
    if extension_path.exists():
        extension = read('score_feature/implementation.json')
        request = read('score_feature/change_request.json')
        if extension['change_request_sha256'] != sha256_file(reports / 'score_feature/change_request.json') or request['base_activation_sha256'] != sha256_file(reports / 'app_activation.json'):
            raise ValueError('Application extension is not bound to its original activation.')
        if sha256_file(ROOT / 'src/visage/pretrained_inference.py') != request['preserved_research_adapter_sha256']:
            raise ValueError('The frozen research inference adapter changed.')
        for path, digest in extension['source_sha256'].items():
            if path in active_sources and request['previous_source_sha256'].get(path) != active_sources[path]:
                raise ValueError('Extension baseline differs from the activated source.')
            if path not in request['previous_source_sha256'] and path not in extension['added_source_files']:
                raise ValueError('An unrecorded application source was added.')
            active_sources[path] = digest
        feature = read('score_feature/acceptance.json')
        if not feature['ok'] or sha256_file(reports / 'score_feature/acceptance.json') != extension['acceptance_sha256']:
            raise ValueError('The score display extension has not passed acceptance.')
        for path, digest in feature['evidence_sha256'].items():
            if sha256_file(ROOT / path) != digest:
                raise ValueError('Score feature acceptance evidence changed.')
        for name, count in feature['counts'].items():
            xml = ET.parse(reports / 'score_feature' / name)
            cases = xml.findall('.//testcase')
            if len(cases) != count or any(case.find(tag) is not None for case in cases for tag in ('failure', 'error', 'skipped')):
                raise ValueError('Score feature regression checks are incomplete.')
        if extension['artifact_sha256'] != hashes:
            raise ValueError('A score display change replaced a selected model.')
    refinement_path = reports / 'ui_refinement/implementation.json'
    refinement = None
    if refinement_path.exists():
        refinement = read('ui_refinement/implementation.json')
        request = read('ui_refinement/change_request.json')
        if refinement['change_request_sha256'] != sha256_file(reports / 'ui_refinement/change_request.json') or request['prior_extension_sha256'] != sha256_file(reports / 'score_feature/implementation.json'):
            raise ValueError('UI refinements are not bound to the preceding score extension.')
        for path, digest in refinement['source_sha256'].items():
            if path in active_sources and request['previous_source_sha256'].get(path) != active_sources[path]:
                raise ValueError('UI refinement baseline differs from the previous activated source.')
            if path not in request['previous_source_sha256'] and path not in refinement['added_source_files']:
                raise ValueError('Unrecorded UI refinement source.')
            active_sources[path] = digest
        feature = read('ui_refinement/acceptance.json')
        if not feature['ok'] or sha256_file(reports / 'ui_refinement/acceptance.json') != refinement['acceptance_sha256']:
            raise ValueError('UI refinements have not passed acceptance.')
        for path, digest in feature['evidence_sha256'].items():
            if sha256_file(ROOT / path) != digest:
                raise ValueError('UI refinement acceptance evidence changed.')
        for name, count in feature['counts'].items():
            xml = ET.parse(reports / 'ui_refinement' / name)
            cases = xml.findall('.//testcase')
            if len(cases) != count or any(case.find(tag) is not None for case in cases for tag in ('failure', 'error', 'skipped')):
                raise ValueError('UI refinement regressions are incomplete.')
        if refinement['artifact_sha256'] != hashes or sha256_file(ROOT / 'src/visage/faces.py') != request['preserved_face_pipeline_sha256']:
            raise ValueError('A UI refinement changed the frozen model or face pipeline.')
    for path, digest in active_sources.items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f'Activated source changed after staging: {path}')
    training_scripts = {'run_pretrained_study.py', 'finish_pretrained_migration.py'}
    for process in psutil.process_iter(['name', 'cmdline']):
        try:
            if 'python' in (process.info['name'] or '').lower() and any(Path(arg).name in training_scripts for arg in process.info['cmdline'] or []):
                if Path(process.cwd()).resolve() == ROOT:
                    raise ValueError('A stopped training/controller process has restarted.')
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    rollback = ROOT / 'output/VisAge_Web_App.zip'
    if sha256_file(rollback) != 'd4b51ef6a076f33c14a03d0a49158d21752a04e006cb02ba816e7fcc9a0fe2dc':
        raise ValueError('The historical rollback archive changed.')
    evidence = ['configs/pretrained/two_seed_amendment.json', 'reports/pretrained/evaluation_protocol_two_seed.json',
        'reports/pretrained/selection.json', 'reports/pretrained/verified_runs.json', 'reports/pretrained/reload_verification.json',
        'reports/pretrained/test_summary.json', 'reports/pretrained/robustness.json', 'reports/pretrained/face_pipeline_validation.json',
        'reports/pretrained/evaluation/detection_lineage.json', 'reports/pretrained/device_verification.json',
        'reports/pretrained/web_app/automated_acceptance.json', 'reports/pretrained/web_app/visual_review.json',
        'reports/pretrained/benchmark/backend_benchmark.json', 'reports/pretrained/two_seed_report.html', 'README.md']
    if extension:
        evidence += ['reports/pretrained/score_feature/implementation.json', 'reports/pretrained/score_feature/acceptance.json',
            'reports/pretrained/hardware_acceptance.json']
    if refinement:
        evidence += ['reports/pretrained/ui_refinement/implementation.json', 'reports/pretrained/ui_refinement/acceptance.json',
            'reports/pretrained/evaluation/application_profile.json']
    hardware = read('hardware_acceptance.json') if (reports / 'hardware_acceptance.json').exists() else None
    physical_pending = [] if hardware and hardware['webcam']['status'] == 'user_confirmed_working' and hardware['phone']['status'] == 'not_required_by_user' else ['physical webcam/phone acceptance']
    delivery = read('delivery/verification.json') if (reports / 'delivery/verification.json').exists() else None
    deployment = read('deployment_status.json') if (reports / 'deployment_status.json').exists() else None
    delivery_complete = bool(delivery and delivery.get('ok') and delivery.get('isolated_import_mode')
        and all(delivery.get('checks', {}).values()) and delivery.get('files_verified') == 240
        and delivery.get('cleanup', {}).get('port_closed'))
    if delivery_complete:
        package = json.loads((reports / 'delivery/package.json').read_text())
        if sha256_file(ROOT / 'output/VisAge_Complete_Web_App_v0.4.0.zip') != delivery['package_sha256'] or package['sha256'] != delivery['package_sha256']:
            raise ValueError('Verified fresh delivery archive changed.')
        evidence += ['reports/pretrained/delivery/verification.json', 'reports/pretrained/delivery/package.json']
    if deployment:
        evidence += ['reports/pretrained/deployment_status.json']
    hosting_complete = bool(deployment and deployment.get('vercel', {}).get('hosted_inference_verified'))
    write_json(reports / 'two_seed_activation.json', {'ok': True, 'completed_utc': utc_now(),
        'scope': 'The five user-approved post-training steps; six completed full-data runs only.',
        'steps': { '1_protocol_amendment': 'complete', '2_artifact_verification': 'complete',
            '3_validation_selection_and_metrics': 'complete', '4_inference_evaluation_and_detector_lineage': 'complete',
            '5_activate_and_verify_local_app': 'complete'},
        'models': list(METHODS), 'seeds': [42, 43], 'completed_runs': 6, 'serving_seed': 42,
        'default_method': selection['default_method'], 'artifact_sha256': hashes,
        'test_counts': feature['counts'] if extension else acceptance['counts'], 'browser_assertions': feature['browser_assertions'] if extension else len(suite['passed']),
        'score_display_extension': bool(extension), 'ui_refinements': bool(refinement), 'hardware_acceptance': hardware,
        'robustness_conditions_per_family': 7, 'pipeline_attempted': pipeline['attempted'],
        'pipeline_accepted': pipeline['accepted'], 'training_restarted': False,
        'original_protocol_and_protected_evidence_preserved': True, 'unfinished_seed44_preserved_and_excluded': True,
        'historical_rollback_zip_preserved': True, 'evidence_sha256': {p: sha256_file(ROOT / p) for p in evidence},
        'fresh_delivery_verified': delivery_complete, 'deployment_status': deployment,
        'remaining_separate_scope': [*physical_pending, *([] if (reports / 'pilot_scope.json').exists() else ['independent consented-participant pilot']),
            *([] if hosting_complete else ['Vercel hosted inference: awaiting team authorization and deployment verification']),
            *([] if delivery_complete else ['fresh delivery ZIP and isolated extraction']), 'nine deferred supporting experiments (outside current delivery scope)']})
    print(json.dumps({'ok': True, 'default_method': selection['default_method'], 'steps_completed': 5}), flush=True)


if __name__ == '__main__':
    main()
