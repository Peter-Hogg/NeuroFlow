from pathlib import Path

from examples.dandi_hdf5 import (
    ASSET_ID,
    DANDISET,
    OBJECT_NAME,
    build_query,
    parse_args,
)


def test_remote_example_defaults_are_pinned_and_bounded() -> None:
    config = parse_args([])
    query = build_query(config)

    assert config.source == DANDISET
    assert query.asset == ASSET_ID
    assert query.name == OBJECT_NAME
    assert config.block_size == 262_144


def test_remote_example_accepts_safe_scaling_parameters(tmp_path: Path) -> None:
    output = tmp_path / "result.zarr"
    config = parse_args(
        ["--factor", "2", "--block-size", "524288", "--output", str(output)]
    )

    assert config.factor == 2.0
    assert config.block_size == 524_288
    assert config.output == output
