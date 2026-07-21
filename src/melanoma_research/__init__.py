"""Reproducible helpers for melanoma-screening model research."""

from .metrics import BinaryMetrics, BootstrapComparison, binary_metrics, paired_bootstrap
from .splitting import Sample, group_stratified_split, read_manifest, write_split_manifest

__all__ = [
    "BinaryMetrics",
    "BootstrapComparison",
    "Sample",
    "binary_metrics",
    "group_stratified_split",
    "paired_bootstrap",
    "read_manifest",
    "write_split_manifest",
]

__version__ = "1.0.0"
