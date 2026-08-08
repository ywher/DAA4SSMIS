"""DPT/DINOv2 model construction shared by training and evaluation."""

from pathlib import Path

import torch

from model.semseg.dpt import DPT


MODEL_CONFIGS = {
    "small": {
        "encoder_size": "small",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "base": {
        "encoder_size": "base",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "large": {
        "encoder_size": "large",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "giant": {
        "encoder_size": "giant",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


def build_model(cfg, pretrained_path=None):
    """Build the paper's DINOv2+DPT network and optionally load encoder weights."""
    backbone = cfg["backbone"]
    try:
        backbone_type, encoder_size = backbone.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid backbone name: {backbone}") from exc
    if backbone_type != "dinov2" or encoder_size not in MODEL_CONFIGS:
        raise ValueError("Supported backbones: dinov2_small/base/large/giant")

    model = DPT(
        **MODEL_CONFIGS[encoder_size],
        nclass=cfg["nclass"],
        backbone_type=backbone_type,
        img_size=cfg["crop_size"],
    )
    if pretrained_path is not None:
        path = Path(pretrained_path)
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 weights not found: {path}")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.backbone.load_state_dict(state)
    if cfg.get("lock_backbone", False):
        model.lock_backbone()
    return model
