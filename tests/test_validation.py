import numpy as np
import pytest

from neuroflow import compare_segmentations


def test_segmentation_metrics_use_one_to_one_instance_matching() -> None:
    reference = np.array([[1, 1, 0, 2], [1, 1, 0, 2]], dtype=np.uint16)
    predicted = np.array([[7, 7, 0, 8], [7, 7, 0, 0]], dtype=np.uint16)
    metrics = compare_segmentations(predicted, reference, iou_threshold=0.5)
    assert metrics.matched_objects == 2
    assert metrics.precision == 1
    assert metrics.recall == 1
    assert metrics.mean_matched_iou == pytest.approx(0.75)
    assert metrics.foreground_dice == pytest.approx(10 / 11)


def test_segmentation_metrics_report_detection_errors() -> None:
    reference = np.array([[1, 1, 0], [0, 0, 2]], dtype=np.uint16)
    predicted = np.array([[3, 3, 0], [4, 0, 0]], dtype=np.uint16)
    metrics = compare_segmentations(predicted, reference)
    assert metrics.matched_objects == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)


def test_segmentation_metrics_validate_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        compare_segmentations(np.zeros((2, 2), dtype=int), np.zeros((3, 2), dtype=int))
    with pytest.raises(TypeError, match="integer"):
        compare_segmentations(np.zeros((2, 2)), np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match="iou_threshold"):
        compare_segmentations(
            np.zeros((2, 2), dtype=int),
            np.zeros((2, 2), dtype=int),
            iou_threshold=0,
        )
