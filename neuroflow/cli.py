"""Command-line entry point."""

import json

import typer

from neuroflow import __version__, open_result, open_source

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
