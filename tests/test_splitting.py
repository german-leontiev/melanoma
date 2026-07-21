from pathlib import Path

import pytest

from melanoma_research import Sample, group_stratified_split, read_manifest


def samples():
    return [
        Sample(f"sample-{label}-{group}-{image}", f"group-{label}-{group}", label, "image.jpg")
        for label in (0, 1)
        for group in range(8)
        for image in range(1 + group % 2)
    ]


def test_split_is_deterministic_group_disjoint_and_class_complete():
    first = group_stratified_split(samples(), seed=11)
    second = group_stratified_split(reversed(samples()), seed=11)

    assert first == second
    groups = [{sample.group_id for sample in first[name]} for name in first]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert all({sample.label for sample in partition} == {0, 1} for partition in first.values())


def test_conflicting_group_labels_are_rejected():
    invalid = [
        Sample("a", "same", 0, "a.jpg"),
        Sample("b", "same", 1, "b.jpg"),
        *samples(),
    ]
    with pytest.raises(ValueError, match="conflicting"):
        group_stratified_split(invalid)


def test_read_manifest_requires_explicit_group_identifier(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,label,image_path\na,0,a.jpg\n", encoding="utf-8")

    with pytest.raises(ValueError, match="group_id"):
        read_manifest(manifest)
