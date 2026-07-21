"""Group-aware, class-stratified splitting for image manifests."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True, order=True, slots=True)
class Sample:
    sample_id: str
    group_id: str
    label: int
    image_path: str


def read_manifest(
    path: Path,
    *,
    sample_column: str = "sample_id",
    group_column: str = "group_id",
    label_column: str = "label",
    path_column: str = "image_path",
) -> tuple[Sample, ...]:
    """Read a CSV/TSV manifest without accessing the referenced images."""

    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    samples: list[Sample] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {sample_column, group_column, label_column, path_column}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                label = int(row[label_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid label on line {line_number}") from exc
            values = {
                "sample_id": (row[sample_column] or "").strip(),
                "group_id": (row[group_column] or "").strip(),
                "image_path": (row[path_column] or "").strip(),
            }
            if not all(values.values()):
                raise ValueError(f"empty required value on line {line_number}")
            samples.append(Sample(label=label, **values))
    if not samples:
        raise ValueError("manifest is empty")
    return tuple(samples)


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if any(value <= 0.0 for value in ratios):
        raise ValueError("all split ratios must be positive")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")


def _partition_score(
    partition: str,
    counts: dict[str, int],
    targets: dict[str, float],
    item_count: int,
) -> tuple[float, int]:
    projected = dict(counts)
    projected[partition] += item_count
    error = sum(
        ((projected[name] - targets[name]) / max(targets[name], 1.0)) ** 2
        for name in PARTITIONS
    )
    return (error, PARTITIONS.index(partition))


def group_stratified_split(
    samples: Iterable[Sample],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, tuple[Sample, ...]]:
    """Assign every group to one partition while approximating class ratios."""

    _validate_ratios(ratios)
    materialized = tuple(samples)
    if not materialized:
        raise ValueError("at least one sample is required")

    group_labels: dict[str, int] = {}
    groups: dict[tuple[int, str], list[Sample]] = defaultdict(list)
    for sample in materialized:
        if sample.label not in (0, 1):
            raise ValueError("binary labels must contain only zero and one")
        existing = group_labels.setdefault(sample.group_id, sample.label)
        if existing != sample.label:
            raise ValueError(f"group {sample.group_id!r} contains conflicting labels")
        groups[(sample.label, sample.group_id)].append(sample)

    result: dict[str, list[Sample]] = {name: [] for name in PARTITIONS}
    for label in sorted({sample.label for sample in materialized}):
        label_groups = [
            (group_id, tuple(sorted(items)))
            for (group_label, group_id), items in groups.items()
            if group_label == label
        ]
        if len(label_groups) < len(PARTITIONS):
            raise ValueError(f"label {label} needs at least three independent groups")

        rng = random.Random(f"{seed}:{label}")
        label_groups.sort(key=lambda item: item[0])
        rng.shuffle(label_groups)
        label_groups.sort(key=lambda item: -len(item[1]))

        total = sum(len(items) for _, items in label_groups)
        targets = {
            name: total * ratio for name, ratio in zip(PARTITIONS, ratios, strict=True)
        }
        counts = {name: 0 for name in PARTITIONS}
        assigned_groups = {name: 0 for name in PARTITIONS}

        for index, (_, items) in enumerate(label_groups):
            remaining = len(label_groups) - index
            empty = [name for name in PARTITIONS if assigned_groups[name] == 0]
            candidates = empty if remaining == len(empty) else list(PARTITIONS)

            selected = min(
                candidates,
                key=lambda partition: _partition_score(
                    partition, counts, targets, len(items)
                ),
            )
            result[selected].extend(items)
            counts[selected] += len(items)
            assigned_groups[selected] += 1

    partitions = {
        name: tuple(sorted(items, key=lambda item: item.sample_id))
        for name, items in result.items()
    }
    group_sets = [
        {sample.group_id for sample in partitions[name]} for name in PARTITIONS
    ]
    if any(group_sets[a] & group_sets[b] for a in range(3) for b in range(a + 1, 3)):
        raise RuntimeError("group leakage detected after splitting")
    return partitions


def write_split_manifest(partitions: dict[str, tuple[Sample, ...]], path: Path) -> None:
    """Write a deterministic split manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "sample_id", "group_id", "label", "image_path"),
        )
        writer.writeheader()
        for name in PARTITIONS:
            for sample in partitions[name]:
                writer.writerow(
                    {
                        "split": name,
                        "sample_id": sample.sample_id,
                        "group_id": sample.group_id,
                        "label": sample.label,
                        "image_path": sample.image_path,
                    }
                )
