"""DANDI asset discovery and NWB-Zarr delegation."""

from __future__ import annotations

import json
from typing import Literal
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from neuroflow.exceptions import (
    AmbiguousSelectionError,
    ObjectNotFoundError,
    SourceResolutionError,
    UnsupportedBackendError,
)
from neuroflow.selection.query import NWBQuery, Selection
from neuroflow.source.base import AssetMetadata, SourceIdentity, SourceSummary
from neuroflow.source.hdf5 import NWBHDF5Source
from neuroflow.source.local import LocalNWBZarrSource

API_URL = "https://api.dandiarchive.org/api"
_MAX_ASSET_PAGES = 1000


def _validate_dandi_url(url: str, *, api_only: bool = False) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise SourceResolutionError("DANDI returned an unsupported URL")
    host = (parsed.hostname or "").lower()
    api_host = host == "api.dandiarchive.org"
    storage_host = host in {
        "dandiarchive.s3.amazonaws.com",
        "dandiarchive.s3.us-west-2.amazonaws.com",
    }
    if not api_host if api_only else not (api_host or storage_host):
        raise SourceResolutionError("DANDI returned a URL outside approved hosts")
    return url


def _get_json(
    url: str, headers: dict[str, str], timeout: float = 30.0
) -> dict[str, object]:
    _validate_dandi_url(url, api_only=True)
    request = Request(url, headers={"Accept": "application/json", **headers})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            value = json.load(response)
    except Exception as exc:
        raise SourceResolutionError(f"DANDI request failed: {url}") from exc
    if not isinstance(value, dict):
        raise SourceResolutionError(f"DANDI returned an invalid response: {url}")
    return value


class DandiNWBSource:
    """A Dandiset delegating numerical access to a selected NWB asset."""

    def __init__(
        self,
        dandiset_id: str,
        *,
        version: str | None = None,
        backend: Literal["auto", "lindi", "remfile"] = "auto",
        storage_options: dict[str, object] | None = None,
    ) -> None:
        self.dandiset_id = dandiset_id.zfill(6)
        options = dict(storage_options or {})
        token = options.pop("token", None) or options.pop("api_key", None)
        transport = options.pop("transport", None)
        if transport is not None and transport not in ("auto", "lindi", "remfile"):
            raise ValueError("DANDI HDF5 transport must be auto, lindi, or remfile")
        if backend != "auto" and transport not in (None, "auto", backend):
            raise ValueError("backend conflicts with storage_options['transport']")
        self.backend = backend if backend != "auto" else str(transport or "auto")
        self._headers = {"Authorization": f"token {token}"} if token else {}
        self._storage_options = options
        self.version = version or self._resolve_version()
        self._assets = self._load_assets()
        self._children: dict[str, LocalNWBZarrSource | NWBHDF5Source] = {}
        self._identity = SourceIdentity(
            uri=f"DANDI:{self.dandiset_id}",
            version=self.version,
        )

    def _resolve_version(self) -> str:
        metadata = _get_json(f"{API_URL}/dandisets/{self.dandiset_id}/", self._headers)
        published = metadata.get("most_recent_published_version")
        if isinstance(published, dict) and isinstance(published.get("version"), str):
            return published["version"]
        return "draft"

    def _load_assets(self) -> tuple[AssetMetadata, ...]:
        query = urlencode({"page_size": 100})
        url: str | None = (
            f"{API_URL}/dandisets/{self.dandiset_id}/versions/"
            f"{self.version}/assets/?{query}"
        )
        assets: list[AssetMetadata] = []
        visited: set[str] = set()
        while url:
            if url in visited or len(visited) >= _MAX_ASSET_PAGES:
                raise SourceResolutionError(
                    "DANDI asset pagination is cyclic or too long"
                )
            visited.add(url)
            page = _get_json(url, self._headers)
            results = page.get("results")
            if not isinstance(results, list):
                raise SourceResolutionError("DANDI asset response has no results list")
            for raw in results:
                if not isinstance(raw, dict):
                    continue
                asset_id = raw.get("asset_id")
                path = raw.get("path")
                zarr_id = raw.get("zarr")
                if not isinstance(asset_id, str) or not isinstance(path, str):
                    continue
                content_url = (
                    f"s3://dandiarchive/zarr/{zarr_id}"
                    if isinstance(zarr_id, str)
                    else f"{API_URL}/assets/{asset_id}/download/"
                )
                assets.append(
                    AssetMetadata(
                        asset_id=asset_id,
                        path=path,
                        size=raw.get("size")
                        if isinstance(raw.get("size"), int)
                        else None,
                        checksum=None,
                        content_url=content_url,
                        is_zarr=zarr_id is not None,
                    )
                )
            next_url = page.get("next")
            url = (
                _validate_dandi_url(next_url, api_only=True)
                if isinstance(next_url, str)
                else None
            )
        return tuple(assets)

    @property
    def identity(self) -> SourceIdentity:
        return self._identity

    def assets(self) -> tuple[AssetMetadata, ...]:
        return self._assets

    def _resolve_asset(self, requested: str | None) -> AssetMetadata:
        candidates = [
            asset
            for asset in self._assets
            if asset.is_zarr or asset.path.lower().endswith(".nwb")
        ]
        if requested is not None:
            candidates = [
                asset
                for asset in candidates
                if requested in (asset.asset_id, asset.path)
            ]
        if not candidates:
            if requested is None:
                raise ObjectNotFoundError(
                    "the Dandiset contains no supported NWB assets"
                )
            raise ObjectNotFoundError(f"no supported NWB asset matched {requested!r}")
        if len(candidates) > 1:
            raise AmbiguousSelectionError(
                "multiple NWB assets are available; set NWBQuery(asset=...)"
            )
        return candidates[0]

    def select(self, query: NWBQuery) -> Selection:
        asset = self._resolve_asset(query.asset)
        child = self._children.get(asset.asset_id)
        if child is None:
            if asset.content_url is None:
                raise UnsupportedBackendError(f"asset {asset.path} has no content URL")
            details = _get_json(
                f"{API_URL}/dandisets/{self.dandiset_id}/versions/{self.version}/"
                f"assets/{asset.asset_id}/",
                self._headers,
            )
            content_url = asset.content_url
            if not asset.is_zarr:
                urls = details.get("contentUrl")
                if isinstance(urls, list):
                    direct = [
                        value
                        for value in urls
                        if isinstance(value, str)
                        and "api.dandiarchive.org" not in value
                    ]
                    if direct:
                        content_url = _validate_dandi_url(direct[-1])
            digest = details.get("digest")
            checksum = None
            if isinstance(digest, dict):
                for key in (
                    "dandi:sha2-256",
                    "dandi:zarr-checksum",
                    "dandi:dandi-etag",
                ):
                    value = digest.get(key)
                    if isinstance(value, str):
                        checksum = value
                        break
            identity = SourceIdentity(
                uri=self.identity.uri,
                version=self.version,
                asset_id=asset.asset_id,
                checksum=checksum,
            )
            source_class = LocalNWBZarrSource if asset.is_zarr else NWBHDF5Source
            options = (
                {"anon": True, **self._storage_options}
                if asset.is_zarr
                else {"transport": self.backend, **self._storage_options}
            )
            child = source_class(
                content_url,
                version=self.version,
                storage_options=options,
                identity=identity,
            )
            self._children[asset.asset_id] = child
        object_query = NWBQuery(
            neurodata_type=query.neurodata_type,
            name=query.name,
            path=query.path,
            subject=query.subject,
            session_id=query.session_id,
            where=query.where,
        )
        return child.select(object_query)

    def inspect(self) -> SourceSummary:
        return SourceSummary(
            self.identity,
            self.assets(),
            ("metadata", "bounded-array", "nwb-zarr", "nwb-hdf5", "dandi"),
        )

    def io_stats(self) -> dict[str, object]:
        """Aggregate observed HTTP data responses from opened child assets."""
        responses = 0
        response_bytes = 0
        counters_available = True
        for child in self._children.values():
            stats = getattr(child, "io_stats", None)
            if not callable(stats):
                continue
            value = stats()
            if not isinstance(value, dict):
                continue
            count = value.get("http_responses")
            size = value.get("response_content_bytes")
            if isinstance(count, int):
                responses += count
            elif count is None:
                counters_available = False
            if isinstance(size, int):
                response_bytes += size
            elif size is None:
                counters_available = False
        return {
            "backend": self.backend,
            "http_responses": responses if counters_available else None,
            "response_content_bytes": response_bytes if counters_available else None,
        }

    def close(self) -> None:
        for child in self._children.values():
            child.close()
        self._children.clear()
