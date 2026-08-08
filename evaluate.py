"""Distributed evaluation entry point for the BUSI and ISIC examples."""

import argparse
import logging
import os
from pathlib import Path
import pprint

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from dataset.semi import SemiDataset
from util.checkpoint import save_iou_csv
from util.classes import CLASSES
from util.color_map import color_map as COLOR_MAP
from util.dist_helper import setup_distributed
from util.eval_utils import (
    aggregate_metrics_across_gpus,
    calculate_dice_jc_per_sample,
    calculate_distance_metrics_medpy,
    compute_metric_means,
)
from util.model import build_model
from util.utils import count_params, init_log


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a 2D segmentation model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-path", required=True)
    parser.add_argument("--model-name", required=True, choices=["best", "latest"])
    parser.add_argument("--test-set", action="store_true")
    parser.add_argument("--backbone", default=None)
    parser.add_argument(
        "--weights", choices=["model", "model_ema"], default="model_ema"
    )
    parser.add_argument("--save-pred", action="store_true")
    parser.add_argument("--save-compare", action="store_true")
    parser.add_argument(
        "--eval-resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional common resolution for all metrics",
    )
    parser.add_argument("--local-rank", "--local_rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def get_color_map(dataset):
    colors = np.zeros((256, 3), dtype=np.uint8)
    for index, color in COLOR_MAP[dataset].items():
        colors[index] = color
    return colors


def denormalize(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.cpu().numpy().transpose(1, 2, 0)
    return np.clip((image * std + mean) * 255, 0, 255).astype(np.uint8)


def save_comparison(image, mask, prediction, colors, path):
    valid = mask != 255
    errors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    errors[(prediction == mask) & valid] = (0, 255, 0)
    errors[(prediction != mask) & valid] = (255, 0, 0)
    errors[~valid] = (128, 128, 128)
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = (
        (denormalize(image), "Input"),
        (colors[mask], "Ground Truth"),
        (colors[prediction], "Prediction"),
        (errors, "Green=correct, red=error"),
    )
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def sample_name(sample_id):
    image_path = (
        sample_id.split(",", 1)[0] if "," in sample_id else sample_id.split()[0]
    )
    return Path(image_path).stem


def prepare_output_dirs(root, enabled, comparisons):
    if not enabled:
        return None
    directories = {
        "ids": root / "pred_id",
        "colors": root / "pred_color",
    }
    if comparisons:
        directories["comparisons"] = root / "pred_compare"
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


@torch.inference_mode()
def evaluate(model, loader, cfg, local_rank, args, output_root):
    model.eval()
    nclass = cfg["nclass"]
    metric_lists = {
        name: [[] for _ in range(nclass)] for name in ("jc", "dice", "asd", "hd95")
    }
    colors = get_color_map(cfg["dataset"])
    save_outputs = args.save_pred or args.save_compare
    directories = prepare_output_dirs(
        output_root, save_outputs and dist.get_rank() == 0, args.save_compare
    )

    for images, masks, sample_ids in tqdm(loader, desc="Evaluating"):
        images = images.cuda(local_rank, non_blocking=True)
        predictions = model(images).argmax(1)
        metric_predictions, metric_masks = predictions, masks
        if args.eval_resize is not None:
            size = tuple(args.eval_resize)
            metric_predictions = F.interpolate(
                predictions.unsqueeze(1).float(), size=size, mode="nearest"
            ).squeeze(1)
            metric_masks = F.interpolate(
                masks.unsqueeze(1).float(), size=size, mode="nearest"
            ).squeeze(1)

        pred_numpy = metric_predictions.cpu().numpy()
        mask_numpy = metric_masks.numpy()
        for batch_index in range(len(pred_numpy)):
            for class_id in range(nclass):
                dice, jc = calculate_dice_jc_per_sample(
                    pred_numpy[batch_index], mask_numpy[batch_index], class_id
                )
                asd, hd95 = calculate_distance_metrics_medpy(
                    pred_numpy[batch_index], mask_numpy[batch_index], class_id
                )
                metric_lists["dice"][class_id].append(dice)
                metric_lists["jc"][class_id].append(jc)
                metric_lists["asd"][class_id].append(asd)
                metric_lists["hd95"][class_id].append(hd95)

        if directories is not None:
            original_predictions = predictions.cpu().numpy().astype(np.uint8)
            original_masks = masks.numpy().astype(np.uint8)
            for index, identifier in enumerate(sample_ids):
                name = sample_name(identifier)
                prediction = original_predictions[index]
                Image.fromarray(prediction).save(directories["ids"] / f"{name}.png")
                Image.fromarray(colors[prediction]).save(
                    directories["colors"] / f"{name}.png"
                )
                if args.save_compare:
                    save_comparison(
                        images[index],
                        original_masks[index],
                        prediction,
                        colors,
                        directories["comparisons"] / f"{name}.png",
                    )

    return {
        name: aggregate_metrics_across_gpus(values)
        for name, values in metric_lists.items()
    }


def log_results(logger, metrics, class_names):
    means = {name: compute_metric_means(values) for name, values in metrics.items()}
    logger.info("***** Evaluation Results *****")
    logger.info(
        "JC %.2f (foreground %.2f), Dice %.2f (foreground %.2f)",
        *means["jc"],
        *means["dice"],
    )
    logger.info(
        "ASD %.2f (foreground %.2f), HD95 %.2f (foreground %.2f)",
        *means["asd"],
        *means["hd95"],
    )
    for index, class_name in enumerate(class_names):
        logger.info(
            "%d %-12s JC %.2f Dice %.2f ASD %.2f HD95 %.2f",
            index,
            class_name,
            metrics["jc"][index],
            metrics["dice"][index],
            metrics["asd"][index],
            metrics["hd95"][index],
        )
    return means


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["dataset"] not in CLASSES:
        raise ValueError("Initial release supports only busi and isic")
    if args.backbone is not None:
        cfg["backbone"] = args.backbone

    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ["LOCAL_RANK"])
    logger = init_log("global", logging.INFO)
    logger.propagate = False
    if rank == 0:
        logger.info(
            "%s", pprint.pformat({**cfg, **vars(args), "world_size": world_size})
        )

    cudnn.enabled = True
    cudnn.benchmark = True
    model = build_model(cfg)
    if rank == 0:
        logger.info("Parameters: %.1fM", count_params(model))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )

    checkpoint_path = Path(args.exp_path) / f"{args.model_name}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if args.weights not in checkpoint:
        raise KeyError(f"Checkpoint does not contain {args.weights!r}")
    model.load_state_dict(checkpoint[args.weights])

    split = "test" if args.test_set else "val"
    dataset = SemiDataset(
        cfg["dataset"], cfg["data_root"], split, pre_resize=cfg.get("pre_resize")
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=1)
    output_root = Path(args.exp_path) / f"pred_{split}"
    metrics = evaluate(model, loader, cfg, local_rank, args, output_root)

    if rank == 0:
        means = log_results(logger, metrics, CLASSES[cfg["dataset"]])
        csv_path = Path(args.exp_path) / f"{args.model_name}_{split}_metrics.csv"
        save_iou_csv(
            csv_path,
            means["jc"][0],
            metrics["jc"],
            means["dice"][0],
            metrics["dice"],
            means["hd95"][0],
            metrics["hd95"],
            means["asd"][0],
            metrics["asd"],
            CLASSES[cfg["dataset"]],
        )
        logger.info("Results saved to %s", csv_path)


if __name__ == "__main__":
    main()
