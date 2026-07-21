"""Binary metrics and paired bootstrap comparison without framework dependencies."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass(frozen=True, slots=True)
class BootstrapComparison:
    baseline: BinaryMetrics
    candidate: BinaryMetrics
    f1_delta: float
    f2_delta: float
    f1_interval: tuple[float, float]
    f2_interval: tuple[float, float]
    f1_non_improvement_rate: float
    f2_non_improvement_rate: float
    iterations: int
    seed: int


def _validate_inputs(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> None:
    if not labels:
        raise ValueError("at least one observation is required")
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must contain only zero and one")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("probabilities must be finite values between zero and one")


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0.0:
        return 0.0
    return (1.0 + beta_squared) * precision * recall / denominator


def binary_metrics(
    labels: Iterable[int], probabilities: Iterable[float], *, threshold: float = 0.5
) -> BinaryMetrics:
    """Calculate thresholded binary metrics with an explicit confusion matrix."""

    y_true = tuple(int(value) for value in labels)
    y_probability = tuple(float(value) for value in probabilities)
    _validate_inputs(y_true, y_probability, threshold)
    predictions = tuple(int(value >= threshold) for value in y_probability)

    true_positive = sum(a == 1 and b == 1 for a, b in zip(y_true, predictions, strict=True))
    true_negative = sum(a == 0 and b == 0 for a, b in zip(y_true, predictions, strict=True))
    false_positive = sum(a == 0 and b == 1 for a, b in zip(y_true, predictions, strict=True))
    false_negative = sum(a == 1 and b == 0 for a, b in zip(y_true, predictions, strict=True))

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    return BinaryMetrics(
        accuracy=(true_positive + true_negative) / len(y_true),
        precision=precision,
        recall=recall,
        f1=_fbeta(precision, recall, 1.0),
        f2=_fbeta(precision, recall, 2.0),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    labels: Iterable[int],
    baseline_probabilities: Iterable[float],
    candidate_probabilities: Iterable[float],
    *,
    threshold: float = 0.5,
    iterations: int = 2_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapComparison:
    """Compare two models on the same examples with paired resampling.

    The returned non-improvement rate is the smoothed share of bootstrap
    samples whose metric delta is non-positive. It is descriptive evidence,
    not a substitute for a preregistered clinical hypothesis test.
    """

    y_true = tuple(int(value) for value in labels)
    baseline = tuple(float(value) for value in baseline_probabilities)
    candidate = tuple(float(value) for value in candidate_probabilities)
    _validate_inputs(y_true, baseline, threshold)
    _validate_inputs(y_true, candidate, threshold)
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    baseline_result = binary_metrics(y_true, baseline, threshold=threshold)
    candidate_result = binary_metrics(y_true, candidate, threshold=threshold)
    rng = random.Random(seed)
    f1_deltas: list[float] = []
    f2_deltas: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(len(y_true)) for _ in y_true]
        sampled_labels = [y_true[index] for index in indices]
        baseline_sample = [baseline[index] for index in indices]
        candidate_sample = [candidate[index] for index in indices]
        old = binary_metrics(sampled_labels, baseline_sample, threshold=threshold)
        new = binary_metrics(sampled_labels, candidate_sample, threshold=threshold)
        f1_deltas.append(new.f1 - old.f1)
        f2_deltas.append(new.f2 - old.f2)

    alpha = (1.0 - confidence) / 2.0
    return BootstrapComparison(
        baseline=baseline_result,
        candidate=candidate_result,
        f1_delta=candidate_result.f1 - baseline_result.f1,
        f2_delta=candidate_result.f2 - baseline_result.f2,
        f1_interval=(_quantile(f1_deltas, alpha), _quantile(f1_deltas, 1.0 - alpha)),
        f2_interval=(_quantile(f2_deltas, alpha), _quantile(f2_deltas, 1.0 - alpha)),
        f1_non_improvement_rate=(1 + sum(value <= 0.0 for value in f1_deltas))
        / (iterations + 1),
        f2_non_improvement_rate=(1 + sum(value <= 0.0 for value in f2_deltas))
        / (iterations + 1),
        iterations=iterations,
        seed=seed,
    )
