import pytest

from melanoma_research import binary_metrics, paired_bootstrap


def test_binary_metrics_include_confusion_matrix_and_f_scores():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.2])

    assert result.accuracy == 0.5
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5
    assert result.f2 == 0.5
    assert (result.true_negative, result.false_positive) == (1, 1)
    assert (result.false_negative, result.true_positive) == (1, 1)


def test_paired_bootstrap_is_deterministic_and_prefers_better_predictions():
    labels = [0, 0, 0, 1, 1, 1]
    baseline = [0.1, 0.7, 0.8, 0.2, 0.4, 0.9]
    candidate = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    first = paired_bootstrap(labels, baseline, candidate, iterations=200, seed=7)
    second = paired_bootstrap(labels, baseline, candidate, iterations=200, seed=7)

    assert first == second
    assert first.f1_delta > 0
    assert first.f2_delta > 0
    assert 0 < first.f1_non_improvement_rate < 0.1


def test_metrics_reject_malformed_probabilities():
    with pytest.raises(ValueError, match="between zero and one"):
        binary_metrics([0, 1], [0.2, 1.2])
