"""Metrics used by the 2D medical segmentation examples."""

from medpy import metric
import numpy as np
import torch
import torch.distributed as dist


def calculate_dice_jc_per_sample(pred_mask, gt_mask, class_id):
    """Calculate per-sample Dice and Jaccard scores on a 0-100 scale."""
    pred_binary = pred_mask == class_id
    gt_binary = gt_mask == class_id
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    if pred_sum == 0 and gt_sum == 0:
        return 100.0, 100.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0, 0.0
    intersection = (pred_binary & gt_binary).sum()
    union = (pred_binary | gt_binary).sum()
    dice = 200.0 * intersection / (pred_sum + gt_sum + 1e-10)
    jaccard = 100.0 * intersection / (union + 1e-10)
    return float(dice), float(jaccard)


def calculate_distance_metrics_medpy(pred_mask, gt_mask, class_id):
    """Calculate ASD and HD95, using the image diagonal as an empty-mask penalty."""
    pred_binary = (pred_mask == class_id).astype(np.uint8)
    gt_binary = (gt_mask == class_id).astype(np.uint8)
    penalty = float(sum(size * size for size in pred_mask.shape) ** 0.5)
    if pred_binary.sum() == 0 and gt_binary.sum() == 0:
        return 0.0, 0.0
    if pred_binary.sum() == 0 or gt_binary.sum() == 0:
        return penalty, penalty
    try:
        return (
            float(metric.binary.asd(pred_binary, gt_binary)),
            float(metric.binary.hd95(pred_binary, gt_binary)),
        )
    except (RuntimeError, ValueError, ZeroDivisionError):
        return penalty, penalty


def aggregate_metrics_across_gpus(metrics_per_class):
    """Return the global per-class means from distributed per-sample values."""
    device = torch.device("cuda", torch.cuda.current_device())
    aggregated = np.zeros(len(metrics_per_class), dtype=np.float64)
    for class_id, values in enumerate(metrics_per_class):
        metric_sum = torch.tensor(sum(values), dtype=torch.float64, device=device)
        metric_count = torch.tensor(len(values), dtype=torch.float64, device=device)
        dist.all_reduce(metric_sum)
        dist.all_reduce(metric_count)
        aggregated[class_id] = (
            metric_sum.item() / metric_count.item()
            if metric_count.item() > 0
            else np.nan
        )
    return aggregated


def compute_metric_means(values):
    """Return the means over all classes and foreground classes."""
    values = np.asarray(values, dtype=np.float64)
    valid = values[~np.isnan(values)]
    mean_all = float(valid.mean()) if len(valid) else np.nan
    foreground = values[1:] if len(values) > 1 else values
    foreground = foreground[~np.isnan(foreground)]
    mean_foreground = float(foreground.mean()) if len(foreground) else np.nan
    return mean_all, mean_foreground
