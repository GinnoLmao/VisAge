"""Independent metrics and strictly validation-only default selection."""
import csv
import json
from types import SimpleNamespace
import numpy as np
import pytest
import visage.pretrained_evaluation as module
from visage.model_registry import METHODS


def test_rank_auc_ties_and_complete_three_class_verification():
    assert module.independent_auc([1, 0, 1, 0], [1, 1, 1, 1]) == .5
    assert module.independent_auc([1, 0, 1, 0], [.9, .2, .7, .1]) == 1
    truth = np.array([0, 1, 2, 0, 1, 2])
    scores = np.full((6, 3), .1)
    scores[np.arange(6), truth] = .8
    record = module.verified_roc(truth, scores)
    assert record['independent_rank_auc_verified'] and record['macro_auc_mean'] == 1


def test_saved_prediction_order_probabilities_and_argmax_are_checked(tmp_path):
    rows = [{'sample_id': str(i), 'label': i, 'relative_path': f'{i}.jpg', 'age': 20} for i in range(3)]
    scores = np.eye(3)
    path = tmp_path / 'predictions.csv'
    module.save_predictions(path, rows, scores)
    metrics, values = module.checked_predictions(path, rows)
    assert metrics['macro_f1'] == 1 and metrics['weighted_f1'] == 1
    np.testing.assert_array_equal(values, scores)
    with pytest.raises(ValueError, match='order'):
        module.checked_predictions(path, list(reversed(rows)))
    data = list(csv.DictReader(path.open(newline='')))
    data[0]['prediction'] = 1
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=data[0])
        writer.writeheader(); writer.writerows(data)
    with pytest.raises(ValueError, match='argmax'):
        module.checked_predictions(path, rows)


def test_near_tie_uses_predeclared_boundary_rule_and_never_test_scores(tmp_path, monkeypatch):
    reports = tmp_path / 'reports'
    artifacts = tmp_path / 'artifacts'
    reports.mkdir(); artifacts.mkdir()
    protocol = tmp_path / 'protocol.json'
    protocol.write_text('{}')
    (reports / 'selection_cpu_latency.json').write_text('{}')
    monkeypatch.setattr(module, 'REPORTS', reports)
    monkeypatch.setattr(module, 'ARTIFACTS', artifacts)
    monkeypatch.setattr(module, 'PROTOCOL', protocol)
    records = []
    # EfficientNet is within0.005 and has fewer under18->older mistakes;
    # Swin falls outside the near-tie set despite a lower boundary error.
    for method, f1, boundary in zip(METHODS, [.90, .897, .89], [.05, .04, .01], strict=True):
        directory = artifacts / method
        directory.mkdir(); (directory / 'best.pt').write_bytes(method.encode())
        for seed in (42, 43, 44):
            records.append({'run_id': f'{method}_{seed}', 'method': method, 'seed': seed, 'experiment': 'full',
                'artifact_sha256': str(seed), 'predictions_sha256': str(seed),
                'verified_metrics': {'macro_f1': f1, 'under18_predicted_older_rate': boundary,
                    'per_class': [{}, {}, {'recall': .9}]}, 'test_macro_f1': 1 if method == METHODS[2] else 0})
    latency = {'methods': {m: {'median_ms': 1} for m in METHODS}}
    choice = module.choose_default(records, latency)
    assert choice['default_method'] == METHODS[1]
    assert not choice['test_results_used']
    assert choice['near_tied_candidates'] == list(METHODS[:2])
    assert json.loads((artifacts / 'bundle.json').read_text())['default_method'] == METHODS[1]
    # A later attempt to reinterpret the decision cannot overwrite it.
    records[0]['verified_metrics']['macro_f1'] = .99
    with pytest.raises(ValueError, match='would change'):
        module.choose_default(records, latency)
