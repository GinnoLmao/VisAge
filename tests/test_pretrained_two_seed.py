"""Two-seed selection must reject partial/extra seeds and retain fixed tie rules."""
from copy import deepcopy
import pytest
from visage.model_registry import METHODS
from visage.pretrained_two_seed import rank_families


def records():
    return [{'method': m, 'seed': s, 'experiment': 'full',
        'verified_metrics': {'macro_f1': f1, 'accuracy': .9, 'under18_predicted_older_rate': error,
            'per_class': [{}, {}, {'recall': .9}]}, 'test_score': 1 if m == METHODS[2] else 0}
        for m, f1, error in zip(METHODS, [.90, .897, .89], [.05, .04, .01], strict=True) for s in (42, 43)]


def test_two_seed_validation_rule_uses_boundary_tie_and_ignores_test():
    values = records()
    chosen, means, candidates = rank_families(values, {'methods': {m: {'median_ms': 1} for m in METHODS}})
    assert chosen == METHODS[1] and candidates == list(METHODS[:2])
    assert all(v['macro_f1_std_sample'] == 0 for v in means.values())


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'seed44'])
def test_partial_unpaired_or_extra_seeds_cannot_enter_the_comparison(change):
    values = records()
    if change == 'missing': values.pop()
    elif change == 'duplicate': values[-1]['seed'] = 42
    elif change == 'extra': values.append(deepcopy(values[-1]))
    else: values[-1]['seed'] = 44
    with pytest.raises(ValueError, match='Exactly seeds42 and43'):
        rank_families(values, {'methods': {m: {'median_ms': 1} for m in METHODS}})
