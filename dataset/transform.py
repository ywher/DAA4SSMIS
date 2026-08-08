import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def fixed_resize(img, mask, size):
    """Resize image and mask to a fixed size [height, width]."""
    h, w = size
    img = img.resize((w, h), Image.BILINEAR)
    mask = mask.resize((w, h), Image.NEAREST)
    return img, mask


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(
        int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1])
    )

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(
    img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3
):
    # 创建一个全 0 的二值 mask（大小为 img_size × img_size），后续用 1 表示 CutMix 区域
    mask = torch.zeros(img_size, img_size)

    # 以概率 (1 - p) 不进行 CutMix，直接返回全 0 mask
    if random.random() > p:
        return mask

    # 随机采样 CutMix 区域的面积比例：从 [size_min, size_max] 中采样
    # 这里先得到“面积比例 × 总像素数”，即 CutMix 区域的目标像素面积 size
    size = np.random.uniform(size_min, size_max) * img_size * img_size

    # 反复采样，直到生成的矩形框不会超出图像边界
    while True:
        # 随机采样宽高比 ratio（h/w 或 w/h 的一种定义，取决于下面的推导方式），应该就是h/w
        ratio = np.random.uniform(ratio_1, ratio_2)

        # 由面积 size 和宽高比 ratio 计算矩形的宽 cutmix_w、高 cutmix_h
        # 令 cutmix_w * cutmix_h = size，并结合 ratio 约束得到下面的形式
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))

        # 随机采样矩形左上角坐标 (x, y)
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        # 判断矩形是否完全落在图像内部（不越界）；若越界则继续采样
        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    # 将矩形区域置为 1，表示 CutMix 掩码区域（后续可用于混合两张图像/标签）
    mask[y : y + cutmix_h, x : x + cutmix_w] = 1

    # 返回二值 mask
    return mask
