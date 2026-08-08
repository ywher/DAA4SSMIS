import numpy as np

from util.eval_utils import (
    calculate_dice_jc_per_sample,
    calculate_distance_metrics_medpy,
    compute_metric_means,
)


def test_overlap_metrics_handle_empty_and_exact_masks():
    empty = np.zeros((8, 8), dtype=np.uint8)
    assert calculate_dice_jc_per_sample(empty, empty, 1) == (100.0, 100.0)

    mask = empty.copy()
    mask[2:6, 2:6] = 1
    dice, jaccard = calculate_dice_jc_per_sample(mask, mask, 1)
    np.testing.assert_allclose((dice, jaccard), (100.0, 100.0))
    assert calculate_distance_metrics_medpy(mask, mask, 1) == (0.0, 0.0)


def test_metric_means_exclude_background():
    mean_all, mean_foreground = compute_metric_means([90.0, 70.0])
    assert mean_all == 80.0
    assert mean_foreground == 70.0
