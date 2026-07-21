"""Command-line entrypoint for deterministic manifest splitting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .splitting import PARTITIONS, group_stratified_split, read_manifest, write_split_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a group-disjoint train/validation/test manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--group-column", default="group_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--path-column", default="image_path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = read_manifest(
        args.manifest,
        sample_column=args.sample_column,
        group_column=args.group_column,
        label_column=args.label_column,
        path_column=args.path_column,
    )
    partitions = group_stratified_split(samples, seed=args.seed)
    write_split_manifest(partitions, args.output)
    summary = {
        name: {
            "samples": len(partitions[name]),
            "groups": len({sample.group_id for sample in partitions[name]}),
            "labels": dict(sorted(Counter(sample.label for sample in partitions[name]).items())),
        }
        for name in PARTITIONS
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
