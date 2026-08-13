import json
from pathlib import Path

import numpy as np

import neuroflow


def test_numpy_projection_agrees_across_zarr_and_hdf5_backends(
    nwb_zarr: tuple[Path, np.ndarray],
    nwb_hdf5: tuple[Path, np.ndarray],
    tmp_path: Path,
) -> None:
    reference = json.loads(
        (Path(__file__).parent / "data" / "projection_reference.json").read_text()
    )
    expected = np.asarray(reference["values"], dtype=reference["dtype"])
    np.testing.assert_array_equal(expected, np.median(nwb_zarr[1][:5], axis=0))
    outputs: list[np.ndarray] = []
    for name, path in (("zarr", nwb_zarr[0]), ("hdf5", nwb_hdf5[0])):
        movie = neuroflow.load(path, name="movie").isel(time=slice(0, 5))
        projection = movie.median(
            "time",
            output=tmp_path / f"{name}-projection.zarr",
            chunks=(2, 2),
            max_workers=1,
            memory_limit="16 MiB",
        )
        outputs.append(projection.compute())
        assert projection.workflow is not None
        assert projection.workflow.verify().valid
        projection.close()
        movie.close()

    np.testing.assert_array_equal(outputs[0], expected)
    np.testing.assert_array_equal(outputs[1], expected)
    np.testing.assert_array_equal(outputs[0], outputs[1])
