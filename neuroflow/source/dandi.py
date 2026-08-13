"""DANDI asset discovery and NWB-Zarr delegation."""

from __future__ import annotations

import json
from urllib.parse import urlencode
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


def _get_json(
    url: str, headers: dict[str, str], timeout: float = 30.0
) -> dict[str, object]:
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
        storage_options: dict[str, object] | None = None,
    ) -> None:
        self.dandiset_id = dandiset_id.zfill(6)
        options = dict(storage_options or {})
        token = options.pop("token", None) or options.pop("api_key", None)
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
        while url:
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
            url = next_url if isinstance(next_url, str) else None
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
                        content_url = direct[-1]
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
                else self._storage_options
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

    def close(self) -> None:
        for child in self._children.values():
            child.close()
        self._children.clear()
