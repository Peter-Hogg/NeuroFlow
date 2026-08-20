from typing import cast

import pytest

from neuroflow.exceptions import AmbiguousSelectionError
from neuroflow.selection import NWBQuery
from neuroflow.source.base import SourceIdentity
from neuroflow.source.dandi import DandiNWBSource


def test_dandi_open_lists_metadata_without_opening_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def response(
        url: str, headers: dict[str, str], timeout: float = 30.0
    ) -> dict[str, object]:
        calls.append(url)
        return {
            "results": [
                {
                    "asset_id": "asset-1",
                    "path": "one.nwb.zarr",
                    "size": 10,
                    "zarr": "zarr-1",
                },
                {
                    "asset_id": "asset-2",
                    "path": "two.nwb.zarr",
                    "size": 20,
                    "zarr": "zarr-2",
                },
            ],
            "next": None,
        }

    monkeypatch.setattr("neuroflow.source.dandi._get_json", response)
    source = DandiNWBSource("123", version="0.250101.0000")
    assert source.identity.uri == "DANDI:000123"
    assert len(source.assets()) == 2
    assert len(calls) == 1
    with pytest.raises(AmbiguousSelectionError):
        source.select(NWBQuery(name="movie"))


def test_dandi_routes_selected_blob_asset_to_hdf5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def response(
        url: str, headers: dict[str, str], timeout: float = 30.0
    ) -> dict[str, object]:
        if url.endswith("assets/?page_size=100"):
            return {
                "results": [
                    {
                        "asset_id": "blob-1",
                        "path": "subject/session.nwb",
                        "size": 123,
                        "zarr": None,
                    }
                ],
                "next": None,
            }
        return {"digest": {"dandi:sha2-256": "abc"}}

    class FakeHDF5Source:
        def __init__(self, uri: str, **kwargs: object) -> None:
            calls.append((uri, kwargs))

        def select(self, query: NWBQuery) -> str:
            return query.name or ""

        def close(self) -> None:
            pass

    monkeypatch.setattr("neuroflow.source.dandi._get_json", response)
    monkeypatch.setattr("neuroflow.source.dandi.NWBHDF5Source", FakeHDF5Source)
    source = DandiNWBSource(
        "49", version="0.230223.1424", backend="lindi"
    )

    assert source.select(NWBQuery(asset="blob-1", name="speed")) == "speed"
    assert calls[0][0] == "https://api.dandiarchive.org/api/assets/blob-1/download/"
    kwargs = cast(dict[str, object], calls[0][1])
    assert cast(SourceIdentity, kwargs["identity"]).asset_id == "blob-1"
    assert cast(dict[str, object], kwargs["storage_options"])["transport"] == "lindi"


def test_open_dandi_exposes_backend_without_low_level_cache_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeDandiSource:
        def __init__(self, dandiset_id: str, **kwargs: object) -> None:
            captured.update(dandiset_id=dandiset_id, **kwargs)

    monkeypatch.setattr("neuroflow.api.DandiNWBSource", FakeDandiSource)

    source = __import__("neuroflow").open_dandi(
        "DANDI:350@0.240822.1759", backend="lindi"
    )

    assert source is not None
    assert captured == {
        "dandiset_id": "350",
        "version": "0.240822.1759",
        "backend": "lindi",
        "storage_options": None,
    }
