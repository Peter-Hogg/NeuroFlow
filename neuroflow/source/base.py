"""Backend-neutral source contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from neuroflow.selection.query import NWBQuery, Selection


@dataclass(frozen=True)
class SourceSpec:
    uri: str
    version: str | None = None
    storage_options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceIdentity:
    uri: str
    version: str | None
    asset_id: str | None = None
    checksum: str | None = None


@dataclass(frozen=True)
class AssetMetadata:
    asset_id: str
    path: str
    size: int | None = None
    checksum: str | None = None
    content_url: str | None = None
    is_zarr: bool = False


@dataclass(frozen=True)
class NWBObjectSummary:
    path: str
    name: str | None
    neurodata_type: str
    shape: tuple[int, ...]
    dtype: str
    native_chunks: tuple[int, ...] | None
    axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSummary:
    identity: SourceIdentity
    assets: tuple[AssetMetadata, ...] = ()
    capabilities: tuple[str, ...] = ()
    objects: tuple[NWBObjectSummary, ...] = ()


class NWBSource(Protocol):
    @property
    def identity(self) -> SourceIdentity: ...

    def assets(self) -> Sequence[AssetMetadata]: ...

    def select(self, query: "NWBQuery") -> "Selection": ...

    def inspect(self) -> SourceSummary: ...

    def close(self) -> None: ...
