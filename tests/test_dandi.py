import pytest

from neuroflow.exceptions import AmbiguousSelectionError
from neuroflow.selection import NWBQuery
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
