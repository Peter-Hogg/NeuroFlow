from pathlib import Path

import numpy as np
import pytest
import zarr

import neuroflow
from examples.dandi_fish_projection import (
    DEFAULT_OUTPUT,
    DEFAULT_PREVIEW,
    FishProjectionConfig,
    build_projection,
    parse_args,
)
from neuroflow.source.array import ArraySource


def test_fish_projection_defaults_cover_full_volume() -> None:
    config = parse_args([])

    assert config.frames == 50
    assert config.output == DEFAULT_OUTPUT
    assert config.preview == DEFAULT_PREVIEW
    assert config.output.name == "fish-projection-numpy-t50-full-volume.zarr"
    assert config.preview.name == "fish-projection-numpy-t50-full-volume-z14.png"
    assert (config.tile_y, config.tile_x) == (256, 256)
    assert config.max_workers == 1


def test_projection_expression_has_numpy_semantics(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")[:5]
    projection = build_projection(movie)

    assert projection.shape == (3, 4)
    assert projection.axes == ("y", "x")
    assert projection.dtype == np.dtype("float32")
    np.testing.assert_array_equal(
        projection.compute(), np.median(nwb_zarr[1][:5], axis=0)
    )
    movie.close()


def test_integer_fish_projection_is_explicitly_float32(tmp_path: Path) -> None:
    values = np.arange(4 * 3 * 2 * 2, dtype=np.int16).reshape(4, 3, 2, 2)
    path = tmp_path / "integer-movie.zarr"
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("movie", data=values, chunks=(1, 3, 2, 1))
    source = ArraySource(path, component="movie", axes=("time", "y", "x", "z"))
    movie = neuroflow.NeuroArray(source, source.select())

    projection = build_projection(movie)

    assert projection.dtype == np.dtype("float32")
    np.testing.assert_array_equal(
        projection.compute(), np.median(values, axis=0).astype(np.float32)
    )
    movie.close()


def test_fish_projection_arguments_remain_bounded(tmp_path: Path) -> None:
    output = tmp_path / "fish.zarr"
    preview = tmp_path / "fish.png"
    config = parse_args(
        [
            "--frames",
            "5",
            "--tile-y",
            "128",
            "--tile-x",
            "512",
            "--max-workers",
            "2",
            "--output",
            str(output),
            "--preview",
            str(preview),
        ]
    )

    assert config == FishProjectionConfig(
        frames=5,
        tile_y=128,
        tile_x=512,
        max_workers=2,
        output=output,
        preview=preview,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--frames", "51"],
        ["--tile-y", "63"],
        ["--tile-y", "889"],
        ["--tile-x", "63"],
        ["--tile-x", "2049"],
        ["--max-workers", "0"],
        ["--max-workers", "5"],
        ["--block-size", "65535"],
        ["--cache-size-mib", "513"],
    ],
)
def test_fish_projection_rejects_work_outside_caps(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(arguments)
