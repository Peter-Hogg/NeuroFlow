"""Build dual-channel reference volumes and native-space cell candidate tables.

This is the first half of an atlas-mapping case study.  NeuroFlow performs the
bounded, resumable temporal reductions.  A deliberately simple 3-D blob detector
then produces *candidate* centroids for quality control; these candidates must not
be interpreted as neurons or radial astrocytes until channel metadata and detector
accuracy have been validated.

The output tables retain native voxel coordinates.  A later example will register
the reference volume, transform these points, assign atlas regions, and render the
result.  Internet access is required for DANDI sources.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import neuroflow
from neuroflow.exceptions import ProvenanceMismatchError

DEFAULT_SOURCE = "DANDI:000350@0.240822.1759"
DEFAULT_OUTPUT = Path(__file__).parent / "_output" / "dual-channel-cells-numpy"


@dataclass(frozen=True)
class ChannelConfig:
    """One experimentally identified image channel."""

    name: str
    cell_class: str
    asset: str | None = None
    sigma: float = 1.0
    percentile: float = 99.5
    minimum_distance: int = 2


@dataclass(frozen=True)
class DualChannelConfig:
    source: str
    neuron: ChannelConfig
    glia: ChannelConfig
    frames: int = 50
    output: Path = DEFAULT_OUTPUT
    max_workers: int = 1
    detect: bool = False
    detection_memory_limit: int = 1024 * 1024 * 1024


def detect_blob_candidates(
    volume: np.ndarray,
    *,
    axes: tuple[str, ...],
    cell_class: str,
    sigma: float,
    percentile: float,
    minimum_distance: int,
) -> pd.DataFrame:
    """Return conservative local-maximum candidates in native voxel coordinates.

    This detector is intended to exercise the data path and create annotations for
    manual review.  It is not a validated neuron or radial-astrocyte classifier.
    """
    if volume.ndim != 3 or len(axes) != 3:
        raise ValueError("candidate detection requires a three-dimensional volume")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if sigma < 0.0 or minimum_distance < 1:
        raise ValueError("sigma must be non-negative and minimum_distance positive")

    image_volume = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(image_volume)
    if not finite.any():
        return _empty_candidates(axes)
    fill = float(np.nanmedian(image_volume[finite]))
    filtered = ndimage.gaussian_filter(
        np.where(finite, image_volume, fill), sigma=sigma, mode="nearest"
    )
    threshold = float(np.percentile(filtered[finite], percentile))
    width = 2 * minimum_distance + 1
    maxima = filtered == ndimage.maximum_filter(filtered, size=width, mode="nearest")
    candidate_mask = maxima & finite & (filtered > threshold)
    plateau_labels, plateau_count = ndimage.label(candidate_mask)
    representatives: list[np.ndarray] = []
    for plateau_id in range(1, plateau_count + 1):
        plateau_coordinates = np.argwhere(plateau_labels == plateau_id)
        plateau_values = filtered[tuple(plateau_coordinates.T)]
        representatives.append(plateau_coordinates[int(np.argmax(plateau_values))])
    coordinates = _separated_coordinates(
        np.asarray(representatives, dtype=np.int64), filtered, minimum_distance
    )
    if coordinates.size == 0:
        return _empty_candidates(axes)

    coordinate_columns = pd.Index([f"{axis}_voxel" for axis in axes])
    frame = pd.DataFrame(coordinates, columns=coordinate_columns)
    frame.insert(0, "candidate_id", np.arange(1, len(frame) + 1, dtype=np.int64))
    frame.insert(1, "cell_class", cell_class)
    frame["intensity"] = filtered[tuple(coordinates.T)].astype(np.float32)
    frame["threshold"] = np.float32(threshold)
    return frame


def _separated_coordinates(
    coordinates: np.ndarray, filtered_volume: np.ndarray, minimum_distance: int
) -> np.ndarray:
    """Keep strongest candidates with deterministic Chebyshev separation."""
    if not len(coordinates):
        return np.empty((0, filtered_volume.ndim), dtype=np.int64)
    strengths = filtered_volume[tuple(coordinates.T)]
    order = sorted(
        range(len(coordinates)),
        key=lambda index: (-float(strengths[index]), *coordinates[index].tolist()),
    )
    selected: list[np.ndarray] = []
    for index in order:
        candidate = coordinates[index]
        if all(
            np.max(np.abs(candidate - existing)) >= minimum_distance
            for existing in selected
        ):
            selected.append(candidate)
    return np.asarray(selected, dtype=np.int64)


def _empty_candidates(axes: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=pd.Index(
            [
                "candidate_id",
                "cell_class",
                *(f"{axis}_voxel" for axis in axes),
                "intensity",
                "threshold",
            ]
        )
    )


def build_reference(
    config: DualChannelConfig, channel: ChannelConfig
) -> neuroflow.NeuroArray:
    """Compute or resume one channel's temporal-median reference volume."""
    movie = neuroflow.load(
        config.source,
        name=channel.name,
        asset=channel.asset,
        storage_options={
            "transport": "remfile",
            "block_size": 262_144,
            "cache_size": 64 * 1024 * 1024,
        },
    )
    if "time" not in movie.axes:
        movie.close()
        raise ValueError(f"{channel.name!r} has no time axis")
    frame_count = min(config.frames, movie.shape[movie.axes.index("time")])
    bounded = movie.isel(time=slice(0, frame_count))
    output = config.output / f"{channel.cell_class}-reference.zarr"
    try:
        reference = np.median(  # type: ignore[call-overload]
            bounded, axis="time"
        ).astype(np.float32)
        try:
            return reference.persist(
                output,
                chunks=_reference_chunks(bounded.axes, bounded.shape),
                max_workers=config.max_workers,
                memory_limit="2 GiB",
            )
        except ProvenanceMismatchError as exc:
            raise RuntimeError(
                f"{output} contains a different NeuroFlow workflow. Keep it for "
                "reproducibility and rerun with a fresh --output directory."
            ) from exc
    finally:
        movie.close()


def _reference_chunks(axes: tuple[str, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    remaining = [
        (axis, size) for axis, size in zip(axes, shape, strict=True) if axis != "time"
    ]
    return tuple(
        min(size, 256 if axis in {"x", "y"} else 1) for axis, size in remaining
    )


def run_example(config: DualChannelConfig) -> dict[str, object]:
    """Create both references and, when requested, native candidate tables."""
    config.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"source": config.source, "frames": config.frames}
    for channel in (config.neuron, config.glia):
        reference = build_reference(config, channel)
        try:
            prefix = channel.cell_class
            summary[f"{prefix}_reference"] = str(
                config.output / f"{prefix}-reference.zarr"
            )
            summary[f"{prefix}_shape"] = reference.shape
            summary[f"{prefix}_axes"] = reference.axes
            if config.detect:
                volume_bytes = (
                    int(np.prod(reference.shape))
                    * np.dtype(reference.selection.metadata.dtype).itemsize
                )
                # Gaussian filtering, finite masks, and maxima require several
                # simultaneous arrays; reserve a conservative six-volume budget.
                required_bytes = 6 * volume_bytes
                if required_bytes > config.detection_memory_limit:
                    raise MemoryError(
                        f"candidate detection requires about {required_bytes} bytes; "
                        "raise --detection-memory-mib explicitly or omit --detect"
                    )
                candidates = detect_blob_candidates(
                    reference.compute(),
                    axes=reference.axes,
                    cell_class=channel.cell_class,
                    sigma=channel.sigma,
                    percentile=channel.percentile,
                    minimum_distance=channel.minimum_distance,
                )
                table = config.output / f"{prefix}-candidates.csv"
                candidates.to_csv(table, index=False)
                summary[f"{prefix}_candidates"] = len(candidates)
                summary[f"{prefix}_table"] = str(table)
        finally:
            reference.close()
    return summary


def parse_args(argv: list[str] | None = None) -> DualChannelConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--neuron-name", required=True)
    parser.add_argument("--glia-name", required=True)
    parser.add_argument("--neuron-asset")
    parser.add_argument("--glia-asset")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--detection-memory-mib", type=int, default=1024)
    parser.add_argument("--neuron-percentile", type=float, default=99.5)
    parser.add_argument("--glia-percentile", type=float, default=99.5)
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 500:
        parser.error("--frames must be between 1 and 500")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if not 64 <= args.detection_memory_mib <= 16384:
        parser.error("--detection-memory-mib must be between 64 and 16384")
    for option in (args.neuron_percentile, args.glia_percentile):
        if not 0.0 < option < 100.0:
            parser.error("detection percentiles must be between 0 and 100")
    return DualChannelConfig(
        source=args.source,
        neuron=ChannelConfig(
            args.neuron_name,
            "neuron",
            args.neuron_asset,
            percentile=args.neuron_percentile,
        ),
        glia=ChannelConfig(
            args.glia_name,
            "radial_astrocyte",
            args.glia_asset,
            percentile=args.glia_percentile,
        ),
        frames=args.frames,
        output=args.output,
        max_workers=args.max_workers,
        detect=args.detect,
        detection_memory_limit=args.detection_memory_mib * 1024 * 1024,
    )


def main() -> None:
    for key, value in run_example(parse_args()).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
