"""Local and fsspec-backed NWB-Zarr metadata access."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import zarr
from hdmf_zarr import NWBZarrIO

from neuroflow.exceptions import (
    AmbiguousSelectionError,
    ObjectNotFoundError,
    SourceResolutionError,
    UnsupportedBackendError,
)
from neuroflow.selection.query import NWBQuery, Selection, SelectionMetadata
from neuroflow.source.base import (
    AssetMetadata,
    NWBObjectSummary,
    SourceIdentity,
    SourceSummary,
)


def _object_path(obj: object) -> str | None:
    data = getattr(obj, "data", None)
    path = getattr(data, "path", None)
    return path.removesuffix("/data") if isinstance(path, str) else None


def _type_names(obj: object) -> set[str]:
    return {cls.__name__ for cls in type(obj).__mro__}


class LocalNWBZarrSource:
    """Metadata facade over an NWB-Zarr hierarchy."""

    def __init__(
        self,
        uri: str | Path,
        *,
        version: str | None = None,
        storage_options: dict[str, object] | None = None,
        identity: SourceIdentity | None = None,
    ) -> None:
        self.uri = str(uri)
        self.storage_options = dict(storage_options or {})
        if "://" not in self.uri:
            path = Path(self.uri).expanduser().resolve()
            if not path.exists():
                raise SourceResolutionError(f"source does not exist: {path}")
            if not path.is_dir():
                raise UnsupportedBackendError(
                    "version 0.1 supports NWB-Zarr directories, not NWB-HDF5 files"
                )
            self.uri = str(path)
        try:
            self._io = NWBZarrIO(
                path=self.uri,
                mode="r",
                storage_options=self.storage_options or None,
                load_namespaces=True,
            )
            self._nwbfile = cast(Any, self._io.read())
        except Exception as exc:
            raise SourceResolutionError(
                f"could not open NWB-Zarr source {self.uri}"
            ) from exc
        checksum = hashlib.sha256(self.uri.encode()).hexdigest()
        self._identity = identity or SourceIdentity(
            self.uri, version, checksum=checksum
        )
        self._selections = self._discover_selections()

    @property
    def identity(self) -> SourceIdentity:
        return self._identity

    def _discover_selections(self) -> tuple[Selection, ...]:
        selections: list[Selection] = []
        for obj in self._nwbfile.objects.values():
            data = getattr(obj, "data", None)
            if not isinstance(data, zarr.Array) or data.ndim == 0:
                continue
            path = _object_path(obj)
            if path is None:
                continue
            timestamps = getattr(obj, "timestamps", None)
            rate_value = getattr(obj, "rate", None)
            start_value = getattr(obj, "starting_time", None)
            metadata = SelectionMetadata(
                source=self.identity,
                path=path,
                neurodata_type=type(obj).__name__,
                shape=tuple(int(value) for value in data.shape),
                dtype=str(data.dtype),
                native_chunks=tuple(int(value) for value in data.chunks),
                axes=_infer_axes(obj, data.ndim),
                name=getattr(obj, "name", None),
                rate=float(rate_value) if rate_value is not None else None,
                starting_time=float(start_value) if start_value is not None else None,
                timestamps_path=getattr(timestamps, "path", None),
                attributes={
                    "type_hierarchy": tuple(sorted(_type_names(obj))),
                    "subject_id": getattr(
                        getattr(self._nwbfile, "subject", None), "subject_id", None
                    ),
                    "session_id": getattr(self._nwbfile, "session_id", None),
                },
            )
            timestamp_array = timestamps if isinstance(timestamps, zarr.Array) else None
            selections.append(Selection(metadata, data, timestamp_array))
        return tuple(selections)

    def assets(self) -> tuple[AssetMetadata, ...]:
        size = None
        if "://" not in self.uri:
            size = sum(
                p.stat().st_size for p in Path(self.uri).rglob("*") if p.is_file()
            )
        return (
            AssetMetadata(
                asset_id=self.identity.asset_id or "local",
                path=self.uri,
                size=size,
                checksum=self.identity.checksum,
                content_url=self.uri,
                is_zarr=True,
            ),
        )

    def select(self, query: NWBQuery) -> Selection:
        matches = [item for item in self._selections if _matches(item, query)]
        if not matches:
            raise ObjectNotFoundError(f"no NWB object matched {query!r}")
        if len(matches) > 1:
            paths = ", ".join(item.metadata.path for item in matches)
            raise AmbiguousSelectionError(
                f"query matched multiple NWB objects: {paths}"
            )
        return matches[0]

    def inspect(self) -> SourceSummary:
        return SourceSummary(
            self.identity,
            self.assets(),
            ("metadata", "lazy-array", "nwb-zarr"),
            tuple(
                NWBObjectSummary(
                    path=item.metadata.path,
                    name=item.metadata.name,
                    neurodata_type=item.metadata.neurodata_type,
                    shape=item.metadata.shape,
                    dtype=item.metadata.dtype,
                    native_chunks=item.metadata.native_chunks,
                    axes=item.metadata.axes,
                )
                for item in self._selections
            ),
        )

    def close(self) -> None:
        self._io.close()

    def __enter__(self) -> LocalNWBZarrSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _matches(selection: Selection, query: NWBQuery) -> bool:
    metadata = selection.metadata
    if query.name is not None and metadata.name != query.name:
        return False
    if query.path is not None and metadata.path != query.path.rstrip("/"):
        return False
    if query.asset is not None and metadata.source.asset_id != query.asset:
        return False
    attributes = metadata.attributes or {}
    if query.subject is not None and attributes.get("subject_id") != query.subject:
        return False
    if (
        query.session_id is not None
        and attributes.get("session_id") != query.session_id
    ):
        return False
    if query.neurodata_type is not None:
        requested = (
            query.neurodata_type
            if isinstance(query.neurodata_type, str)
            else query.neurodata_type.__name__
        )
        hierarchy = attributes.get("type_hierarchy", ())
        if not isinstance(hierarchy, (tuple, list, set, frozenset)):
            hierarchy = ()
        if requested not in hierarchy:
            return False
    if query.where:
        if any(attributes.get(key) != value for key, value in query.where.items()):
            return False
    return True


def _infer_axes(obj: Any, ndim: int) -> tuple[str, ...]:
    """Infer conservative axis labels without inventing NWB semantics."""
    first = "time" if hasattr(obj, "timestamps") or hasattr(obj, "rate") else "axis_0"
    if ndim == 4 and "ImageSeries" in _type_names(obj):
        return (first, "y", "x", "z")
    if ndim == 1:
        return (first,)
    suffix = {
        2: ("y",),
        3: ("y", "x"),
        4: ("z", "y", "x"),
    }.get(ndim, tuple(f"axis_{index}" for index in range(1, ndim)))
    return (first, *suffix)
