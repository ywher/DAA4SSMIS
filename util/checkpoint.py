"""Evaluation result serialization."""

import csv
from pathlib import Path

import numpy as np


def _mean_without_background(values):
    values = np.asarray(values[1:] if len(values) > 1 else values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else np.nan


def save_iou_csv(
    csv_path,
    miou,
    iou_class,
    mdice,
    dice_class,
    mhd95,
    hd95_class,
    masd,
    asd_class,
    class_names,
):
    """Save aggregate and per-class IoU, Dice, ASD, and HD95 values."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ["JC (%)", miou, _mean_without_background(iou_class), *iou_class],
        ["Dice (%)", mdice, _mean_without_background(dice_class), *dice_class],
        ["ASD (px)", masd, _mean_without_background(asd_class), *asd_class],
        ["HD95 (px)", mhd95, _mean_without_background(hd95_class), *hd95_class],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Mean (all)", "Mean (no bg)", *class_names])
        writer.writerows(rows)
