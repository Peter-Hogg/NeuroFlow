import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import neuroflow
from neuroflow.expression import expression_from_dict, expression_to_dict
from neuroflow.provenance import capture_environment


def test_workflow_spec_round_trip_plan_and_reproduction(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    shifted = movie + np.float32(1)
    normalized = shifted / shifted.max()
    original_output = tmp_path / "declared.zarr"
    spec = normalized.to_spec(
        original_output,
        chunks=(2, 3, 4),
        memory_limit="64 MiB",
        max_workers=1,
    )
    workflow_path = tmp_path / "workflow.json"

    first_json = spec.to_json(workflow_path)
    loaded = neuroflow.WorkflowSpec.from_json(workflow_path)

    assert loaded.to_json() == first_json
    assert (
        expression_to_dict(expression_from_dict(loaded.expression)) == loaded.expression
    )
    dry_run = cast(dict[str, Any], loaded.plan().to_dict())
    assert dry_run["bounded"]["value"] is True
    assert dry_run["stages"][0]["operation"] == "max"
    assert dry_run["stages"][0]["task_count"] == 5

    reproduced_output = tmp_path / "reproduced.zarr"
    result = neuroflow.reproduce(loaded, output=reproduced_output)
    actual = result.arrays["result"].as_dask_array().compute()
    expected = (nwb_zarr[1] + np.float32(1)) / np.max(nwb_zarr[1] + np.float32(1))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
    assert result.verify().valid
    assert result.provenance is not None
    result_provenance = cast(dict[str, Any], result.provenance)
    assert len(result_provenance["stages"]) == 1
    assert result_provenance["execution_metrics"]["completed_task_count"] == 5
    result.source.close()
    movie.close()


def test_staged_reduction_resume_detects_and_repairs_one_corrupt_partial(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    maximum = movie.max()
    normalized = movie / maximum + maximum / maximum
    persisted = normalized.persist(
        tmp_path / "normalized.zarr",
        chunks=(2, 3, 4),
        memory_limit="64 MiB",
        max_workers=1,
    )
    workflow = persisted.workflow
    provenance = cast(dict[str, Any], workflow.provenance)
    assert provenance is not None
    assert len(provenance["stages"]) == 1
    stage_id = provenance["stages"][0]["stage_id"]
    partials = sorted(
        (
            tmp_path
            / "normalized.zarr"
            / ".neuroflow"
            / "stages"
            / stage_id
            / "partials"
        ).glob("*.json")
    )
    assert len(partials) == 5

    damaged = json.loads(partials[2].read_text())
    damaged["checksum"] = "0" * 64
    partials[2].write_text(json.dumps(damaged))
    assert not workflow.verify().valid

    workflow.resume()
    repaired = cast(dict[str, Any], workflow.provenance)
    assert repaired is not None
    assert repaired["stage_execution"][0]["computed_partitions"] == 1
    assert repaired["stage_execution"][0]["skipped_partitions"] == 4
    assert repaired["execution_metrics"]["resumed_task_count"] == 5
    assert workflow.verify().valid
    np.testing.assert_allclose(
        persisted.compute(), nwb_zarr[1] / nwb_zarr[1].max() + 1, rtol=1e-6
    )
    persisted.close()
    movie.close()


def test_workflow_spec_rejects_code_and_unsafe_file_behaviour(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    spec = (movie + 1).to_spec(tmp_path / "output.zarr")
    value = spec.to_dict()
    value["adapter"]["identifier"] = "python.eval"
    with pytest.raises(neuroflow.WorkflowSpecError, match="allowlisted"):
        neuroflow.WorkflowSpec.from_dict(value)

    invalid = spec.to_json().replace(
        '"schema_version": "1"', '"schema_version": "99"', 1
    )
    with pytest.raises(neuroflow.WorkflowSpecError, match="migration"):
        neuroflow.WorkflowSpec.from_json(invalid)

    path = tmp_path / "workflow.json"
    spec.to_json(path)
    with pytest.raises(neuroflow.WorkflowSpecError, match="already exists"):
        spec.to_json(path)
    link = tmp_path / "workflow-link.json"
    link.symlink_to(path)
    with pytest.raises(neuroflow.WorkflowSpecError, match="symlink"):
        spec.to_json(link, overwrite=True)
    movie.close()


def test_environment_capture_is_machine_readable_and_private_by_default() -> None:
    environment = cast(dict[str, Any], capture_environment())
    json.dumps(environment, allow_nan=False)
    assert environment["neuroflow_version"] == neuroflow.__version__
    assert "hostname" not in environment
    assert environment["python"]["version"]
    assert "numpy" in environment["dependencies"]
