import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import neuroflow
from neuroflow.cli import app

runner = CliRunner()


def test_installed_cli_version_and_inspect(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    version = runner.invoke(app, ["version"])
    assert version.exit_code == 0
    assert version.stdout.strip() == neuroflow.__version__

    inspected = runner.invoke(app, ["inspect", str(nwb_zarr[0])])
    assert inspected.exit_code == 0
    value = json.loads(inspected.stdout)
    objects = {item["name"]: item for item in value["objects"]}
    assert objects["movie"]["axes"] == ["time", "y", "x"]


def test_cli_status_and_verify(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    projection = movie.median("time", output=tmp_path / "result.zarr")

    status = runner.invoke(app, ["status", str(tmp_path / "result.zarr")])
    assert status.exit_code == 0
    assert json.loads(status.stdout)["state"] == "complete"
    verification = runner.invoke(app, ["verify", str(tmp_path / "result.zarr")])
    assert verification.exit_code == 0
    assert json.loads(verification.stdout)["valid"] is True
    projection.close()
    movie.close()


def test_cli_plan_reproduce_and_environment(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    spec_path = tmp_path / "workflow.json"
    (movie / movie.max()).to_spec(
        tmp_path / "declared.zarr",
        chunks=(2, 3, 4),
        max_workers=1,
        memory_limit="64 MiB",
    ).to_json(spec_path)

    planned = runner.invoke(app, ["plan", str(spec_path)])
    assert planned.exit_code == 0, planned.output
    plan = json.loads(planned.stdout)
    assert plan["bounded"]["value"] is True
    assert plan["stages"][0]["operation"] == "max"

    output = tmp_path / "reproduced.zarr"
    reproduced = runner.invoke(
        app, ["reproduce", str(spec_path), "--output", str(output)]
    )
    assert reproduced.exit_code == 0, reproduced.output
    summary = json.loads(reproduced.stdout)
    assert summary["status"] == "complete"
    assert summary["integrity_verified"] is True

    environment = runner.invoke(app, ["environment"])
    assert environment.exit_code == 0
    assert "hostname" not in json.loads(environment.stdout)
    movie.close()
