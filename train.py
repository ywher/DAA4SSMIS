"""UniMatch V2 + BCP training entry point for the initial code release."""

import argparse
from copy import deepcopy
import logging
import os
from pathlib import Path
import pprint
import random

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # TensorBoard logging is optional at runtime.
    SummaryWriter = None

from dataset.semi import SemiDataset
from util.cases import assemble_volumes, validate_partition
from util.case_sampler import CaseSampler
from util.eval_utils import calculate_dice_jc_per_sample, aggregate_metrics_across_gpus
from util.bcp import BCPModule, bcp_pretrain, init_ema_from_student
from util.dist_helper import setup_distributed
from util.model import build_model
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import AverageMeter, count_params, init_log


def parse_args():
    parser = argparse.ArgumentParser(
        description="UniMatch V2 + BCP medical segmentation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--labeled-id-path", required=True)
    parser.add_argument("--unlabeled-id-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--pretrained-path", default="pretrained/dinov2_small.pth")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bcp-u-weight", type=float, default=0.5)
    parser.add_argument("--bcp-loss-weight", type=float, default=1.0)
    parser.add_argument("--bcp-pretrain-epochs", type=int, default=0)
    parser.add_argument("--no-bcp", action="store_true", help="Run plain UniMatch V2")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_optimizer(model, cfg):
    base = model.module if hasattr(model, "module") else model
    return AdamW(
        [
            {
                "params": [p for p in base.backbone.parameters() if p.requires_grad],
                "lr": cfg["lr"],
            },
            {
                "params": [
                    p for name, p in base.named_parameters() if "backbone" not in name
                ],
                "lr": cfg["lr"] * cfg["lr_multi"],
            },
        ],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )


def create_criterion(cfg, local_rank):
    name, kwargs = cfg["criterion"]["name"], cfg["criterion"]["kwargs"]
    if name == "CELoss":
        return nn.CrossEntropyLoss(**kwargs).cuda(local_rank)
    if name == "OHEM":
        return ProbOhemCrossEntropy2d(**kwargs).cuda(local_rank)
    raise ValueError(f"Unsupported supervised criterion: {name}")


@torch.inference_mode()
def validate(model, loader, nclass, local_rank, cfg=None):
    """Checkpoint metric: global Dice for 2D; mean case Dice for 3D."""
    model.eval()
    if cfg and cfg.get("case_level", False):
        records = []
        for images, masks, identifiers in loader:
            predictions = model(images.cuda(local_rank, non_blocking=True)).argmax(1)
            size = cfg.get("eval_resize")
            if size is not None:
                predictions = F.interpolate(
                    predictions[:, None].float(), size=size, mode="nearest"
                )[:, 0]
                masks = F.interpolate(
                    masks[:, None].float(), size=size, mode="nearest"
                )[:, 0]
            records.extend(zip(identifiers, predictions.cpu().numpy(), masks.numpy()))
        values = [[] for _ in range(nclass)]
        for _, prediction, target in assemble_volumes(records):
            for class_id in range(nclass):
                values[class_id].append(
                    calculate_dice_jc_per_sample(prediction, target, class_id)[0]
                )
        dice = aggregate_metrics_across_gpus(values)
        return float(dice[1:].mean()), dice
    device = torch.device("cuda", local_rank)
    intersection = torch.zeros(nclass, dtype=torch.float64, device=device)
    prediction = torch.zeros_like(intersection)
    target = torch.zeros_like(intersection)
    for images, masks, _ in loader:
        images = images.cuda(local_rank, non_blocking=True)
        masks = masks.cuda(local_rank, non_blocking=True)
        outputs = model(images).argmax(1)
        valid = masks != 255
        for class_id in range(nclass):
            pred_class = (outputs == class_id) & valid
            target_class = (masks == class_id) & valid
            intersection[class_id] += (pred_class & target_class).sum()
            prediction[class_id] += pred_class.sum()
            target[class_id] += target_class.sum()
    dist.all_reduce(intersection)
    dist.all_reduce(prediction)
    dist.all_reduce(target)
    dice = 200.0 * intersection / (prediction + target).clamp_min(1.0)
    foreground = dice[1:] if nclass > 1 else dice
    return foreground.mean().item(), dice.cpu().numpy()


def update_ema(student, teacher, iteration):
    ratio = min(1.0 - 1.0 / (iteration + 1), 0.996)
    for parameter, ema_parameter in zip(student.parameters(), teacher.parameters()):
        ema_parameter.copy_(ema_parameter * ratio + parameter.detach() * (1.0 - ratio))
    for buffer, ema_buffer in zip(student.buffers(), teacher.buffers()):
        ema_buffer.copy_(ema_buffer * ratio + buffer.detach() * (1.0 - ratio))


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("case_level", False):

        def read_ids(path):
            return Path(path).read_text(encoding="utf-8").splitlines()

        validate_partition(
            read_ids(f"splits/{cfg['dataset']}/train.txt"),
            read_ids(args.labeled_id_path),
            read_ids(args.unlabeled_id_path),
        )
    seed_everything(args.seed)
    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ["LOCAL_RANK"])
    logger = init_log("global", logging.INFO)
    logger.propagate = False

    save_path = Path(args.save_path)
    writer = None
    if rank == 0:
        save_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(save_path) if SummaryWriter is not None else None
        if SummaryWriter is None:
            logger.warning("TensorBoard is unavailable; scalar logging is disabled")
        logger.info(
            "%s", pprint.pformat({**cfg, **vars(args), "world_size": world_size})
        )

    cudnn.enabled = True
    cudnn.benchmark = True
    model = build_model(cfg, args.pretrained_path)
    if rank == 0:
        logger.info("Parameters: %.1fM", count_params(model))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    teacher = deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    optimizer = create_optimizer(model, cfg)
    criterion_l = create_criterion(cfg, local_rank)
    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)
    bcp = (
        None
        if args.no_bcp
        else BCPModule(
            cfg["nclass"], u_weight=args.bcp_u_weight, loss_weight=args.bcp_loss_weight
        ).cuda(local_rank)
    )

    pre_resize = cfg.get("pre_resize")
    train_u = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
        pre_resize=pre_resize,
    )
    train_l = SemiDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(train_u),
        pre_resize=pre_resize,
    )
    val = SemiDataset(cfg["dataset"], cfg["data_root"], "val", pre_resize=pre_resize)
    if not train_l.ids or not train_u.ids:
        raise ValueError("Both labeled and unlabeled splits must be non-empty")

    sampler_l = torch.utils.data.distributed.DistributedSampler(train_l, shuffle=True)
    sampler_u = torch.utils.data.distributed.DistributedSampler(train_u, shuffle=True)
    sampler_v = (
        CaseSampler(val)
        if cfg.get("case_level", False)
        else torch.utils.data.distributed.DistributedSampler(val, shuffle=False)
    )
    loader_l = DataLoader(
        train_l,
        cfg["batch_size"],
        sampler=sampler_l,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_u = DataLoader(
        train_u,
        cfg["batch_size"],
        sampler=sampler_u,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_v = DataLoader(
        val, 1, sampler=sampler_v, num_workers=args.workers, pin_memory=True
    )
    if len(loader_u) == 0 or len(loader_l) == 0:
        raise ValueError(
            "Per-rank batch count is zero; reduce batch size or number of GPUs"
        )

    start_epoch, best_dice = 0, -1.0
    latest = save_path / "latest.pth"
    if latest.is_file():
        checkpoint = torch.load(latest, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        teacher.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", -1.0)
        logger.info("Resumed from epoch %d", start_epoch)
    elif args.bcp_pretrain_epochs > 0:
        bcp_pretrain(
            model,
            optimizer,
            loader_l,
            cfg,
            args,
            writer,
            rank,
            local_rank,
            pretrain_epochs=args.bcp_pretrain_epochs,
        )
        init_ema_from_student(model, teacher)
        optimizer = create_optimizer(model, cfg)

    total_iterations = len(loader_u) * cfg["epochs"]
    for epoch in range(start_epoch, cfg["epochs"]):
        sampler_l.set_epoch(epoch)
        sampler_u.set_epoch(epoch)
        model.train()
        meters = {
            name: AverageMeter()
            for name in ("loss", "supervised", "unimatch", "bcp", "mask_ratio")
        }

        for step, ((image_l, mask_l), batch_u) in enumerate(zip(loader_l, loader_u)):
            image_u_w, image_u_s1, image_u_s2, ignore, box1, box2 = batch_u
            image_l, mask_l = (
                image_l.cuda(non_blocking=True),
                mask_l.cuda(non_blocking=True),
            )
            image_u_w = image_u_w.cuda(non_blocking=True)
            image_u_s1, image_u_s2 = (
                image_u_s1.cuda(non_blocking=True),
                image_u_s2.cuda(non_blocking=True),
            )
            ignore, box1, box2 = (
                ignore.cuda(non_blocking=True),
                box1.cuda(non_blocking=True),
                box2.cuda(non_blocking=True),
            )

            with torch.no_grad():
                weak_logits = teacher(image_u_w)
                confidence, pseudo = weak_logits.softmax(1).max(1)

            for strong, box in ((image_u_s1, box1), (image_u_s2, box2)):
                region = box.unsqueeze(1).expand_as(strong).bool()
                strong[region] = strong.flip(0)[region]
            pseudo1, pseudo2 = pseudo.clone(), pseudo.clone()
            conf1, conf2 = confidence.clone(), confidence.clone()
            ignore1, ignore2 = ignore.clone(), ignore.clone()
            for target_mask, target_conf, target_ignore, box in (
                (pseudo1, conf1, ignore1, box1),
                (pseudo2, conf2, ignore2, box2),
            ):
                region = box.bool()
                target_mask[region] = pseudo.flip(0)[region]
                target_conf[region] = confidence.flip(0)[region]
                target_ignore[region] = ignore.flip(0)[region]

            pred_l = model(image_l)
            pred_s1, pred_s2 = model(
                torch.cat((image_u_s1, image_u_s2)), comp_drop=True
            ).chunk(2)
            supervised_loss = criterion_l(pred_l, mask_l)
            unsupervised = []
            for prediction_s, target_mask, target_conf, target_ignore in (
                (pred_s1, pseudo1, conf1, ignore1),
                (pred_s2, pseudo2, conf2, ignore2),
            ):
                pixel_loss = criterion_u(prediction_s, target_mask)
                valid = target_ignore != 255
                pixel_loss = pixel_loss * ((target_conf >= cfg["conf_thresh"]) & valid)
                unsupervised.append(pixel_loss.sum() / valid.sum().clamp_min(1))
            unimatch_loss = sum(unsupervised) / 2.0
            loss = (supervised_loss + unimatch_loss) / 2.0
            bcp_loss = torch.zeros((), device=image_l.device)
            if bcp is not None:
                bcp_loss, _, _ = bcp(model, image_l, mask_l, image_u_w, pseudo)
                loss = loss + bcp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration = epoch * len(loader_u) + step
            lr = cfg["lr"] * (1.0 - iteration / total_iterations) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]
            update_ema(model, teacher, iteration)

            mask_ratio = (
                ((confidence >= cfg["conf_thresh"]) & (ignore != 255)).sum()
                / (ignore != 255).sum().clamp_min(1)
            ).item()
            for name, value in (
                ("loss", loss),
                ("supervised", supervised_loss),
                ("unimatch", unimatch_loss),
                ("bcp", bcp_loss),
            ):
                meters[name].update(value.item())
            meters["mask_ratio"].update(mask_ratio)
            if rank == 0 and writer is not None:
                for name, meter in meters.items():
                    writer.add_scalar(f"train/{name}", meter.val, iteration)
            if rank == 0 and step % max(len(loader_u) // 8, 1) == 0:
                logger.info(
                    "Epoch %d step %d/%d loss %.3f (sup %.3f, uni %.3f, bcp %.3f)",
                    epoch,
                    step,
                    len(loader_u),
                    meters["loss"].avg,
                    meters["supervised"].avg,
                    meters["unimatch"].avg,
                    meters["bcp"].avg,
                )

        should_evaluate = (epoch + 1) % cfg.get(
            "eval_interval", 10
        ) == 0 or epoch + 1 == cfg["epochs"]
        if should_evaluate:
            student_dice, _ = validate(model, loader_v, cfg["nclass"], local_rank, cfg)
            ema_dice, class_dice = validate(
                teacher, loader_v, cfg["nclass"], local_rank, cfg
            )
            if rank == 0:
                logger.info(
                    "Validation Dice: student %.2f, EMA %.2f, classes %s",
                    student_dice,
                    ema_dice,
                    np.round(class_dice, 2),
                )
                checkpoint = {
                    "model": model.state_dict(),
                    "model_ema": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_dice": max(best_dice, ema_dice),
                    "config": cfg,
                }
                torch.save(checkpoint, latest)
                if ema_dice >= best_dice:
                    best_dice = ema_dice
                    torch.save(checkpoint, save_path / "best.pth")
                if writer is not None:
                    writer.add_scalar("validation/dice", ema_dice, epoch)
        elif rank == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_ema": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "config": cfg,
                },
                latest,
            )
        dist.barrier()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
