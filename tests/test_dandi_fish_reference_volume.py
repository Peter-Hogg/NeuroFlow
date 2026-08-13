from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from examples.dandi_fish_reference_volume import (
    ReferenceVolumeConfig,
    compute_reference_volume,
    parse_args,
    temporal_median_plane,
)


def test_temporal_median_plane() -> None:
    values = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5, 1)
    actual = temporal_median_plane(
        da.from_array(values, chunks=(1, 4, 5, 1))  # pyright: ignore[reportArgumentType]
    ).compute()
    assert actual.shape == (4, 5)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.median(values[..., 0], axis=0))


def test_compute_reference_volume_writes_zyx_npy(tmp_path: Path) -> None:
    values = np.arange(3 * 4 * 5 * 2, dtype=np.uint16).reshape(3, 4, 5, 2)
    output = tmp_path / "reference.npy"
    result = compute_reference_volume(
        da.from_array(values, chunks=(1, 4, 5, 1)),  # pyright: ignore[reportArgumentType]
        output,
    )
    assert output.exists()
    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(
        np.load(output), np.median(values, axis=0).transpose(2, 0, 1)
    )


def test_reference_defaults_are_bounded_and_cover_all_z() -> None:
    config = parse_args([])
    assert config == ReferenceVolumeConfig()
    assert config.frames == 9
    assert (config.y_size, config.x_size) == (512, 512)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--frames", "0"],
        ["--frames", "51"],
        ["--start-frame", "3060", "--frames", "9"],
        ["--y-start", "700", "--y-size", "200"],
        ["--x-start", "2000", "--x-size", "100"],
        ["--block-size", "65535"],
        ["--cache-size-mib", "513"],
    ],
)
def test_reference_rejects_invalid_or_excessive_work(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_no_view_and_optional_tiff_paths(tmp_path: Path) -> None:
    config = parse_args(
        [
            "--no-view",
            "--output",
            str(tmp_path / "v.npy"),
            "--tiff",
            str(tmp_path / "v.tif"),
        ]
    )
    assert not config.view
    assert config.tiff == tmp_path / "v.tif"
