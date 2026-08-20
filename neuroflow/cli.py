"""Command-line entry point."""

import json
from pathlib import Path

import typer

from neuroflow import WorkflowSpec, __version__, open_result, open_source, reproduce
from neuroflow.provenance import capture_environment
from neuroflow.results.workflow import WorkflowResult

app = typer.Typer(help="Lazy execution for archive-scale NWB analysis.")


@app.command()
def version() -> None:
    """Print the installed NeuroFlow version."""
    typer.echo(__version__)


@app.command("inspect")
def inspect_source(source: str, version: str | None = None) -> None:
    """Print source and asset metadata without reading numerical datasets."""
    opened = open_source(source, version=version)
    try:
        summary = opened.inspect()
        typer.echo(
            json.dumps(
                {
                    "identity": {
                        "uri": summary.identity.uri,
                        "version": summary.identity.version,
                    },
                    "capabilities": summary.capabilities,
                    "assets": [
                        {
                            "asset_id": asset.asset_id,
                            "path": asset.path,
                            "size": asset.size,
                            "is_zarr": asset.is_zarr,
                        }
                        for asset in summary.assets
                    ],
                    "objects": [
                        {
                            "path": item.path,
                            "name": item.name,
                            "neurodata_type": item.neurodata_type,
                            "shape": item.shape,
                            "dtype": item.dtype,
                            "native_chunks": item.native_chunks,
                            "axes": item.axes,
                        }
                        for item in summary.objects
                    ],
                },
                indent=2,
            )
        )
    finally:
        opened.close()


@app.command()
def status(uri: str) -> None:
    """Print persisted workflow completion and failure status."""
    result = open_result(uri)
    value = result.status
    typer.echo(
        json.dumps(
            {
                "state": value.state,
                "completed_partitions": len(value.completed_partitions),
                "failed_partitions": list(value.failed_partitions),
            },
            indent=2,
        )
    )


@app.command()
def verify(
    uri: str,
    checksums: bool = typer.Option(
        True,
        "--checksums/--no-checksums",
        help="Read each bounded partition and validate its checksum.",
    ),
) -> None:
    """Audit manifests and optionally verify persisted partition bytes."""
    report = open_result(uri).verify(checksums=checksums)
    typer.echo(
        json.dumps(
            {
                "valid": report.valid,
                "checked_partitions": len(report.checked_partitions),
                "errors": report.errors,
            },
            indent=2,
        )
    )
    if not report.valid:
        raise typer.Exit(code=1)


@app.command("plan")
def plan_workflow(workflow: Path) -> None:
    """Validate a portable workflow and print a metadata-only dry-run report."""
    try:
        spec = WorkflowSpec.from_json(workflow)
        report = spec.plan()
    except (OSError, ValueError) as exc:
        typer.echo(f"workflow planning failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("reproduce")
def reproduce_workflow(
    workflow: Path,
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Use a fresh output path instead of the path stored in the workflow.",
    ),
) -> None:
    """Execute an allowlisted portable workflow and verify its result."""
    result: WorkflowResult | None = None
    try:
        result = reproduce(workflow, output=output)
        verification = result.verify()
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"workflow reproduction failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    finally:
        if result is not None:
            result.source.close()
    if result is None:  # pragma: no cover - every failure exits above
        raise typer.Exit(code=2)
    typer.echo(
        json.dumps(
            {
                "workflow_id": result.plan.workflow_id,
                "output": result.output.uri,
                "status": result.status.state,
                "integrity_verified": verification.valid,
                "verification_errors": list(verification.errors),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not verification.valid:
        raise typer.Exit(code=1)


@app.command("environment")
def environment() -> None:
    """Print privacy-conscious machine and software metadata as JSON."""
    typer.echo(json.dumps(capture_environment(), indent=2, sort_keys=True))
