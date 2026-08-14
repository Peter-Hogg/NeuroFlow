"""Local and seekable remote NWB-HDF5 access without eager data conversion."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import fsspec
import h5py
import remfile
from pynwb import NWBHDF5IO

from neuroflow.exceptions import (
    AmbiguousSelectionError,
    ObjectNotFoundError,
    SourceResolutionError,
)
from neuroflow.selection.query import NWBQuery, Selection, SelectionMetadata
from neuroflow.source.base import (
    AssetMetadata,
    NWBObjectSummary,
    SourceIdentity,
    SourceSummary,
)
from neuroflow.source.local import _infer_axes, _matches, _type_names


def _open_remote_file(uri: str, options: dict[str, object]) -> tuple[Any, str]:
    """Open a bounded remote reader using a PyNWB-documented transport."""
    transport_value = options.pop("transport", "auto")
    if transport_value not in ("auto", "remfile", "fsspec"):
        raise ValueError("transport must be 'auto', 'remfile', or 'fsspec'")
    transport = str(transport_value)
    block_size_value = options.pop("block_size", 1_048_576)
    cache_size_value = options.pop("cache_size", 67_108_864)
    if not isinstance(block_size_value, int) or block_size_value <= 0:
        raise TypeError("block_size must be a positive integer")
    if not isinstance(cache_size_value, int) or cache_size_value <= 0:
        raise TypeError("cache_size must be a positive integer")
    cache_type = str(options.pop("cache_type", "readahead"))

    use_remfile = transport == "remfile" or (
        transport == "auto" and uri.startswith(("http://", "https://"))
    )
    if use_remfile:
        if options:
            names = ", ".join(sorted(options))
            raise ValueError(f"unsupported remfile storage options: {names}")
        remote = remfile.File(
            uri,
            _min_chunk_size=block_size_value,
            _max_cache_size=cache_size_value,
        )
        return remote, "remfile"

    open_file = cast(Any, fsspec.open)(
        uri,
        mode="rb",
        block_size=block_size_value,
        cache_type=cache_type,
        **options,
    )
    return open_file.open(), "fsspec"


def _redacted_uri(uri: str) -> str:
    """Remove user information and query credentials from persisted URLs."""
    parsed = urlsplit(uri)
    if not parsed.scheme:
        return uri
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


class NWBHDF5Source:
    """An NWB-HDF5 file whose datasets remain h5py-backed and sliceable.

    Remote URLs are opened through a seekable fsspec file with a bounded
    in-memory readahead cache. HDF5 metadata access may issue many range
    requests, but this class never copies the complete file to disk or memory.
    """

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
        self._remote_file: Any | None = None
        self._metrics_lock = threading.Lock()
        self._http_responses = 0
        self._response_content_bytes = 0
        self.transport = "local"
        try:
            if "://" in self.uri:
                options = dict(self.storage_options)
                self._remote_file, self.transport = _open_remote_file(self.uri, options)
                session = getattr(self._remote_file, "session", None)
                hooks = getattr(session, "hooks", None)
                if isinstance(hooks, dict):
                    hooks.setdefault("response", []).append(self._record_response)
                h5_file = h5py.File(self._remote_file, mode="r")
                self._io = NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)
            else:
                path = Path(self.uri).expanduser().resolve()
                if not path.exists() or not path.is_file():
                    raise SourceResolutionError(f"source is not a file: {path}")
                self.uri = str(path)
                self._io = NWBHDF5IO(path=self.uri, mode="r", load_namespaces=True)
            self._nwbfile = cast(Any, self._io.read())
        except SourceResolutionError:
            raise
        except Exception as exc:
            self.close()
            raise SourceResolutionError(
                f"could not open NWB-HDF5 source {_redacted_uri(self.uri)}; "
                "remote servers must "
                "support byte-range requests"
            ) from exc
        public_uri = _redacted_uri(self.uri)
        identity_value = public_uri
        if "://" not in self.uri:
            stat = Path(self.uri).stat()
            identity_value = f"{public_uri}\0{stat.st_size}\0{stat.st_mtime_ns}"
        checksum = hashlib.sha256(identity_value.encode()).hexdigest()
        self._identity = identity or SourceIdentity(
            public_uri, version, checksum=checksum
        )
        self._selections = self._discover_selections()

    @property
    def identity(self) -> SourceIdentity:
        return self._identity

    def _discover_selections(self) -> tuple[Selection, ...]:
        selections: list[Selection] = []
        for obj in self._nwbfile.objects.values():
            data = getattr(obj, "data", None)
            if not isinstance(data, h5py.Dataset) or data.ndim == 0:
                continue
            if not isinstance(data.name, str):
                continue
            path = data.name.removesuffix("/data")
            timestamps = getattr(obj, "timestamps", None)
            rate_value = getattr(obj, "rate", None)
            start_value = getattr(obj, "starting_time", None)
            metadata = SelectionMetadata(
                source=self.identity,
                path=path,
                neurodata_type=type(obj).__name__,
                shape=tuple(int(value) for value in data.shape),
                dtype=str(data.dtype),
                native_chunks=(
                    tuple(int(value) for value in data.chunks)
                    if data.chunks is not None
                    else None
                ),
                axes=_infer_axes(obj, data.ndim),
                name=getattr(obj, "name", None),
                rate=float(rate_value) if rate_value is not None else None,
                starting_time=float(start_value) if start_value is not None else None,
                timestamps_path=(
                    timestamps.name if isinstance(timestamps, h5py.Dataset) else None
                ),
                attributes={
                    "backend": "nwb-hdf5",
                    "transport": self.transport,
                    "type_hierarchy": tuple(sorted(_type_names(obj))),
                    "subject_id": getattr(
                        getattr(self._nwbfile, "subject", None), "subject_id", None
                    ),
                    "session_id": getattr(self._nwbfile, "session_id", None),
                },
            )
            timestamp_array = (
                timestamps if isinstance(timestamps, h5py.Dataset) else None
            )
            selections.append(Selection(metadata, data, timestamp_array))
        return tuple(selections)

    def assets(self) -> tuple[AssetMetadata, ...]:
        size = None
        if "://" not in self.uri:
            size = Path(self.uri).stat().st_size
        return (
            AssetMetadata(
                asset_id=self.identity.asset_id or "local",
                path=self.uri,
                size=size,
                checksum=self.identity.checksum,
                content_url=self.uri,
                is_zarr=False,
            ),
        )

    def _record_response(
        self, response: object, *args: object, **kwargs: object
    ) -> None:
        headers = getattr(response, "headers", {})
        raw_length = headers.get("Content-Length", 0)
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            length = 0
        with self._metrics_lock:
            self._http_responses += 1
            self._response_content_bytes += max(0, length)

    def io_stats(self) -> dict[str, object]:
        """Return observed HTTP response counts without consuming response bodies."""
        with self._metrics_lock:
            return {
                "transport": self.transport,
                "http_responses": self._http_responses,
                "response_content_bytes": self._response_content_bytes,
            }

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
            ("metadata", "bounded-array", "nwb-hdf5"),
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
        io = getattr(self, "_io", None)
        if io is not None:
            io.close()
            self._io = None
        remote = getattr(self, "_remote_file", None)
        if remote is not None:
            remote.close()
            self._remote_file = None

    def __enter__(self) -> NWBHDF5Source:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
