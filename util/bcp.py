"""
BCP (Bidirectional Copy-Paste) 模块

参考论文: BCP: Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation (CVPR 2023)

核心思想:
    BCP 通过双向复制粘贴策略增强半监督训练：
    1. 预训练阶段 (--use-pretrain): 在有标签数据上进行 copy-paste 混合训练，
       然后用预训练好的模型初始化 EMA 教师模型
    2. 半监督阶段 (--use-bcp): 在有标签和无标签数据之间进行双向 copy-paste:
       - 方向 1: 将有标签图像区域粘贴到无标签图像上 (unlabeled base + labeled patch)
       - 方向 2: 将无标签图像区域粘贴到有标签图像上 (labeled base + unlabeled patch)
    3. 使用 mask dice loss + mask CE loss 进行训练

损失函数:
    - Masked Dice Loss: 在指定区域计算 per-class Dice Loss
    - Masked CE Loss: 在指定区域计算交叉熵损失
    - 有标签区域权重 = 1.0，无标签（伪标签）区域权重 = u_weight (默认 0.5)
"""

import logging
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.utils import AverageMeter


class BCPDiceLoss(nn.Module):
    """
    BCP 专用 Masked Dice Loss

    对每个类别分别计算 Dice，支持空间 mask 限制计算区域。

    Args:
        n_classes: 语义类别数
    """

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        """将类别标签 (B, 1, H, W) 转为 one-hot 编码 (B, C, H, W)"""
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        """无 mask 的单类 Dice Loss"""
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def _dice_mask_loss(self, score, target, mask):
        """带 mask 的单类 Dice Loss"""
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, mask=None):
        """
        Args:
            inputs: softmax 概率, (B, C, H, W)
            target: 类别标签, (B, 1, H, W)
            mask: 可选二值掩码, (B, 1, H, W)

        Returns:
            dice loss (scalar)
        """
        target = self._one_hot_encoder(target)
        loss = 0.0
        if mask is not None:
            mask = mask.repeat(1, self.n_classes, 1, 1).float()
            for i in range(self.n_classes):
                loss += self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
        else:
            for i in range(self.n_classes):
                loss += self._dice_loss(inputs[:, i], target[:, i])
        return loss / self.n_classes


def generate_bcp_mask(img):
    """
    生成 BCP Copy-Paste 空间掩码

    在图像上生成一个矩形区域（约占面积 4/9），该区域设为 0，其余为 1。
    相同 mask 应用于整个 batch。

    Args:
        img: 输入图像张量, (B, C, H, W)

    Returns:
        mask: 二值掩码, (B, H, W), long 类型
              mask=1: 保留原始图像
              mask=0: 用另一张图像的对应区域替换
    """
    B, C, H, W = img.shape
    mask = torch.ones(B, H, W, device=img.device)
    patch_h = int(H * 2 / 3)
    patch_w = int(W * 2 / 3)
    h_start = np.random.randint(0, max(H - patch_h, 1))
    w_start = np.random.randint(0, max(W - patch_w, 1))
    mask[:, h_start : h_start + patch_h, w_start : w_start + patch_w] = 0
    return mask.long()


def compute_bcp_mix_loss(
    output,
    img_label,
    patch_label,
    mask,
    dice_loss_fn,
    l_weight=1.0,
    u_weight=0.5,
    unlab=False,
):
    """
    计算 BCP 双向 Copy-Paste 混合损失 (Dice + CE)

    混合图像由两部分组成:
      - mask=1 区域: 来自 "image" (对应标签为 img_label)
      - mask=0 区域: 来自 "patch" (对应标签为 patch_label)

    Args:
        output: 模型 logits, (B, C, H, W)
        img_label: mask=1 区域的标签, (B, H, W)
        patch_label: mask=0 区域的标签, (B, H, W)
        mask: 空间掩码, (B, H, W), 1=image 区域, 0=patch 区域
        dice_loss_fn: BCPDiceLoss 实例
        l_weight: 有标签 (GT) 区域的损失权重
        u_weight: 无标签 (伪标签) 区域的损失权重
        unlab: True 时 mask=1 是无标签区域，mask=0 是有标签区域
               False 时 mask=1 是有标签区域，mask=0 是无标签区域

    Returns:
        loss_dice, loss_ce
    """
    CE = nn.CrossEntropyLoss(reduction="none", ignore_index=255)
    img_label = img_label.long()
    patch_label = patch_label.long()
    output_soft = F.softmax(output, dim=1)

    # 根据区域属性（有标签 / 无标签）分配权重
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    else:
        image_weight, patch_weight = l_weight, u_weight

    mask_float = mask.float()
    patch_mask = 1 - mask_float

    # 排除 ignore 像素 (label=255) 的有效 mask
    valid_img = (img_label != 255).float()
    valid_patch = (patch_label != 255).float()

    # Masked Dice Loss（同时排除 ignore 像素）
    dice_mask_img = mask.unsqueeze(1).float() * valid_img.unsqueeze(1)
    dice_mask_patch = (1 - mask).unsqueeze(1).float() * valid_patch.unsqueeze(1)

    loss_dice = (
        dice_loss_fn(output_soft, img_label.unsqueeze(1), dice_mask_img.long())
        * image_weight
    )
    loss_dice += (
        dice_loss_fn(output_soft, patch_label.unsqueeze(1), dice_mask_patch.long())
        * patch_weight
    )

    # Masked CE Loss（ignore_index=255 已自动将 255 像素的 loss 置 0）
    effective_mask_img = mask_float * valid_img
    effective_mask_patch = patch_mask * valid_patch
    loss_ce = (
        image_weight
        * (CE(output, img_label) * effective_mask_img).sum()
        / (effective_mask_img.sum() + 1e-16)
    )
    loss_ce += (
        patch_weight
        * (CE(output, patch_label) * effective_mask_patch).sum()
        / (effective_mask_patch.sum() + 1e-16)
    )

    return loss_dice, loss_ce


class BCPModule(nn.Module):
    """
    BCP 半监督训练模块

    在半监督训练中执行双向 Copy-Paste 混合并计算对应损失。
    每次 forward 会对学生模型进行两次额外的前向传播：
      - 方向 1 (unlabeled base + labeled patch): 无标签图像为主体，粘贴有标签图像区域
      - 方向 2 (labeled base + unlabeled patch): 有标签图像为主体，粘贴无标签图像区域

    Args:
        nclass: 语义类别数
        u_weight: 无标签（伪标签）区域的损失权重，默认 0.5
        loss_weight: BCP 总损失的权重系数，默认 1.0
    """

    def __init__(self, nclass, u_weight=0.5, loss_weight=1.0):
        super().__init__()
        self.nclass = nclass
        self.u_weight = u_weight
        self.loss_weight = loss_weight
        self.dice_loss = BCPDiceLoss(nclass)

    def forward(self, model, img_x, mask_x, img_u, pseudo_u):
        """
        计算双向 Copy-Paste 损失

        Args:
            model: 学生模型 (DDP 包装)
            img_x: 有标签图像, (B_l, C, H, W)
            mask_x: GT 标签, (B_l, H, W)
            img_u: 无标签图像 (弱增强), (B_u, C, H, W)
            pseudo_u: 无标签图像的伪标签, (B_u, H, W)

        Returns:
            loss_bcp: 加权后的 BCP 总损失 (scalar)
            loss_dice: Dice Loss 分量
            loss_ce: CE Loss 分量
        """
        # 匹配 batch size
        B = min(img_x.shape[0], img_u.shape[0])
        img_x, mask_x = img_x[:B], mask_x[:B]
        img_u, pseudo_u = img_u[:B], pseudo_u[:B]

        # 生成空间掩码
        bcp_mask = generate_bcp_mask(img_x)  # (B, H, W)
        mask_4d = bcp_mask.unsqueeze(1).float()  # (B, 1, H, W)

        # ---- 方向 1: Unlabeled base + Labeled patch ----
        # mask=1 区域来自无标签 (伪标签), mask=0 区域来自有标签 (GT)
        net_input_unl = img_u * mask_4d + img_x * (1 - mask_4d)
        out_unl = model(net_input_unl)
        unl_dice, unl_ce = compute_bcp_mix_loss(
            out_unl,
            pseudo_u,
            mask_x,
            bcp_mask,
            self.dice_loss,
            l_weight=1.0,
            u_weight=self.u_weight,
            unlab=True,
        )

        # ---- 方向 2: Labeled base + Unlabeled patch ----
        # mask=1 区域来自有标签 (GT), mask=0 区域来自无标签 (伪标签)
        net_input_l = img_x * mask_4d + img_u * (1 - mask_4d)
        out_l = model(net_input_l)
        l_dice, l_ce = compute_bcp_mix_loss(
            out_l,
            mask_x,
            pseudo_u,
            bcp_mask,
            self.dice_loss,
            l_weight=1.0,
            u_weight=self.u_weight,
            unlab=False,
        )

        # 合并双向损失
        loss_dice = unl_dice + l_dice
        loss_ce = unl_ce + l_ce
        loss_bcp = (loss_dice + loss_ce) / 2.0 * self.loss_weight

        return loss_bcp, loss_dice, loss_ce


def bcp_pretrain(
    model,
    optimizer,
    trainloader_l,
    cfg,
    args,
    writer,
    rank,
    local_rank,
    pretrain_epochs,
):
    """
    BCP 预训练: 在有标签数据上进行 Copy-Paste 混合训练

    将有标签 batch 分成两半 (img_a, img_b)，生成空间掩码后混合，
    使用 Dice + CE 损失在 GT 标签上进行监督训练。
    预训练完成后应调用 init_ema_from_student() 用学生模型初始化 EMA 教师。

    Args:
        model: DDP 包装的学生模型
        optimizer: 优化器
        trainloader_l: 有标签数据加载器
        cfg: 配置字典
        args: 命令行参数
        writer: TensorBoard writer
        rank: 进程 rank
        local_rank: 本地 GPU rank
        pretrain_epochs: 预训练 epoch 数
    """
    logger = logging.getLogger("global")
    nclass = cfg["nclass"]
    dice_loss_fn = BCPDiceLoss(nclass).cuda(local_rank)

    if rank == 0:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"BCP Pretraining: {pretrain_epochs} epochs on labeled data")
        logger.info(f"{'=' * 60}")

    model.train()
    total_pretrain_iters = pretrain_epochs * len(trainloader_l)
    global_iter = 0
    pretrain_start_time = time.time()

    for epoch in range(pretrain_epochs):
        # 使用偏移 epoch 避免与主训练相同的 shuffle 顺序
        trainloader_l.sampler.set_epoch(epoch + 10000)
        loss_meter = AverageMeter()

        for i, (img_x, mask_x) in enumerate(trainloader_l):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            B = img_x.shape[0]
            half = B // 2
            if half == 0:
                continue

            # 将 batch 分成两半
            img_a, img_b = img_x[:half], img_x[half : 2 * half]
            lab_a, lab_b = mask_x[:half], mask_x[half : 2 * half]

            # 生成掩码并混合
            bcp_mask = generate_bcp_mask(img_a)  # (half, H, W)
            mask_4d = bcp_mask.unsqueeze(1).float()

            net_input = img_a * mask_4d + img_b * (1 - mask_4d)
            out = model(net_input)

            # 预训练中两半均为 GT 标签，u_weight=1.0
            loss_dice, loss_ce = compute_bcp_mix_loss(
                out,
                lab_a,
                lab_b,
                bcp_mask,
                dice_loss_fn,
                l_weight=1.0,
                u_weight=1.0,
                unlab=True,
            )
            loss = (loss_dice + loss_ce) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 预训练内的学习率调度
            lr = cfg["lr"] * (1 - global_iter / max(total_pretrain_iters, 1)) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            loss_meter.update(loss.item())
            global_iter += 1

            if writer is not None and rank == 0:
                writer.add_scalar("pretrain/loss", loss.item(), global_iter)
                writer.add_scalar("pretrain/loss_dice", loss_dice.item(), global_iter)
                writer.add_scalar("pretrain/loss_ce", loss_ce.item(), global_iter)
                writer.add_scalar("pretrain/lr", lr, global_iter)

            # 定期打印进度和 ETA
            if rank == 0 and (i % max(len(trainloader_l) // 8, 1) == 0):
                elapsed = time.time() - pretrain_start_time
                remaining = (
                    elapsed * (total_pretrain_iters - global_iter) / max(global_iter, 1)
                )
                eta_h, eta_m, eta_s = (
                    int(remaining // 3600),
                    int((remaining % 3600) // 60),
                    int(remaining % 60),
                )
                logger.info(
                    f"Pretrain Epoch [{epoch + 1}/{pretrain_epochs}] "
                    f"Iter [{i}/{len(trainloader_l)}] "
                    f"LR: {lr:.7f}, Loss: {loss_meter.avg:.4f}, "
                    f"Dice: {loss_dice.item():.4f}, CE: {loss_ce.item():.4f}, "
                    f"ETA: {eta_h:02d}h{eta_m:02d}m{eta_s:02d}s"
                )

        if rank == 0:
            elapsed = time.time() - pretrain_start_time
            remaining = (
                elapsed * (total_pretrain_iters - global_iter) / max(global_iter, 1)
            )
            eta_h, eta_m = int(remaining // 3600), int((remaining % 3600) // 60)
            logger.info(
                f"Pretrain Epoch [{epoch + 1}/{pretrain_epochs}] "
                f"Loss: {loss_meter.avg:.4f}, "
                f"ETA: {eta_h:02d}h{eta_m:02d}m"
            )

    if rank == 0:
        total_time = time.time() - pretrain_start_time
        t_h, t_m, t_s = (
            int(total_time // 3600),
            int((total_time % 3600) // 60),
            int(total_time % 60),
        )
        logger.info(
            f"BCP Pretraining complete. Total iterations: {global_iter}, "
            f"Time: {t_h:02d}h{t_m:02d}m{t_s:02d}s\n"
        )


def init_ema_from_student(model, model_ema):
    """
    用预训练后的学生模型初始化 EMA 教师模型

    将学生模型的所有参数和 buffer（包括 BatchNorm 统计量）复制到 EMA 模型。

    Args:
        model: DDP 包装的学生模型
        model_ema: EMA 教师模型
    """
    logger = logging.getLogger("global")

    for param_ema, param in zip(model_ema.parameters(), model.parameters()):
        param_ema.data.copy_(param.data)
    for buffer_ema, buffer in zip(model_ema.buffers(), model.buffers()):
        buffer_ema.data.copy_(buffer.data)

    logger.info("EMA teacher model initialized from pretrained student model.")
