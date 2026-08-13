"""Minimum provenance recorded for every persisted result."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from neuroflow.source.base import SourceIdentity


@dataclass(frozen=True)
class ProvenanceRecord:
    neuroflow_version: str
    source: SourceIdentity
    nwb_paths: tuple[str, ...]
    adapter_name: str
    adapter_version: str
    parameters: Mapping[str, object]
    partition_spec: Mapping[str, object]
    output_locations: Mapping[str, str]
    random_seeds: tuple[int, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
