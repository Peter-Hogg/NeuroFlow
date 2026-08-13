"""Reference metrics for expert-reviewed instance segmentations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class SegmentationMetrics:
    """Object- and foreground-level agreement with a reference label image."""

    iou_threshold: float
    predicted_objects: int
    reference_objects: int
    matched_objects: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    foreground_dice: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def compare_segmentations(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> SegmentationMetrics:
    """Compare positive instance labels using one-to-one IoU matching.

    Label zero is background. Labels need not be consecutive, and the arrays may
    be 2-D or 3-D, but their shapes must match. Object matches maximize total IoU
    before the requested threshold is applied.
    """
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    if predicted.shape != reference.shape:
        raise ValueError(
            f"predicted shape {predicted.shape} does not match {reference.shape}"
        )
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not np.issubdtype(predicted.dtype, np.integer) or not np.issubdtype(
        reference.dtype, np.integer
    ):
        raise TypeError("segmentations must contain integer labels")
    if np.any(predicted < 0) or np.any(reference < 0):
        raise ValueError("segmentation labels must be non-negative")

    predicted_ids = np.unique(predicted[predicted > 0])
    reference_ids = np.unique(reference[reference > 0])
    intersections = np.zeros((len(predicted_ids), len(reference_ids)), dtype=np.int64)
    predicted_sizes = np.array(
        [np.count_nonzero(predicted == item) for item in predicted_ids], dtype=np.int64
    )
    reference_sizes = np.array(
        [np.count_nonzero(reference == item) for item in reference_ids], dtype=np.int64
    )
    for row, predicted_id in enumerate(predicted_ids):
        overlapping, counts = np.unique(
            reference[predicted == predicted_id], return_counts=True
        )
        for reference_id, count in zip(overlapping, counts, strict=True):
            if reference_id > 0:
                column = int(np.searchsorted(reference_ids, reference_id))
                intersections[row, column] = count

    unions = (
        predicted_sizes[:, np.newaxis]
        + reference_sizes[np.newaxis, :]
        - intersections
    )
    ious = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    if ious.size:
        rows, columns = linear_sum_assignment(ious, maximize=True)
        assigned = ious[rows, columns]
        matched_ious = assigned[assigned >= iou_threshold]
    else:
        matched_ious = np.array([], dtype=np.float64)

    matches = len(matched_ious)
    predicted_count = len(predicted_ids)
    reference_count = len(reference_ids)
    precision = (
        matches / predicted_count if predicted_count else float(reference_count == 0)
    )
    recall = (
        matches / reference_count if reference_count else float(predicted_count == 0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    predicted_foreground = predicted > 0
    reference_foreground = reference > 0
    foreground_total = np.count_nonzero(predicted_foreground) + np.count_nonzero(
        reference_foreground
    )
    foreground_dice = (
        2
        * np.count_nonzero(predicted_foreground & reference_foreground)
        / foreground_total
        if foreground_total
        else 1.0
    )
    return SegmentationMetrics(
        iou_threshold=iou_threshold,
        predicted_objects=predicted_count,
        reference_objects=reference_count,
        matched_objects=matches,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=float(np.mean(matched_ious)) if matches else 0.0,
        foreground_dice=foreground_dice,
    )
