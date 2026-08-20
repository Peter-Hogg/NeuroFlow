"""Deterministically reconstruct the retained compact Cellpose input."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np

REFERENCE = Path(__file__).parents[1] / "tests" / "data" / "cellpose_reference.json"


def load_reference(path: Path = REFERENCE) -> tuple[np.ndarray, dict[str, object]]:
    metadata = cast(dict[str, object], json.loads(path.read_text()))
    shape = tuple(_integer(value) for value in _list(metadata["shape"], "shape"))
    y, x = np.mgrid[: shape[0], : shape[1]]
    projection = np.full(shape, _number(metadata["background"]), dtype=np.float64)
    for item in _list(metadata["centers"], "centers"):
        raw = _list(item, "center")
        if len(raw) != 4:
            raise ValueError("each reference center requires four values")
        center_y, center_x, sigma, amplitude = (_number(value) for value in raw)
        projection += amplitude * np.exp(
            -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma**2)
        )
    rng = np.random.default_rng(_integer(metadata["seed"]))
    projection += rng.normal(
        0,
        _number(metadata["noise_standard_deviation"]),
        projection.shape,
    )
    result = np.clip(projection, 0, None).astype(str(metadata["dtype"]))
    checksum = hashlib.sha256(result.tobytes(order="C")).hexdigest()
    if checksum != metadata["sha256"]:
        raise RuntimeError("Cellpose reference reconstruction checksum mismatch")
    return result, metadata


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"Cellpose reference {name} must be a list")
    return value


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Cellpose reference values must be numeric")
    return float(value)


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Cellpose reference values must be integers")
    return value
