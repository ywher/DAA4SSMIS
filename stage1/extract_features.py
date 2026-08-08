"""Extract hierarchical DINOv2 embeddings for the 2D Stage 1 example."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model.backbone.dinov2 import DINOv2


LAYER_INDICES = {
    "small": [2, 5, 8, 11],
    "base": [2, 5, 8, 11],
    "large": [4, 11, 17, 23],
    "giant": [9, 19, 29, 39],
}


def parse_image_path(line):
    """Return the image path from a comma- or whitespace-separated split line."""
    return line.split(",", 1)[0].strip() if "," in line else line.split()[0]


def read_split(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


class ImageList(Dataset):
    def __init__(self, root, split_lines, image_size):
        self.root = Path(root)
        self.image_paths = [parse_image_path(line) for line in split_lines]
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.root / self.image_paths[index]).convert("RGB")
        return self.transform(image)


def load_backbone(size, image_size, weights, device):
    model = DINOv2(model_name=size, img_size=image_size)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.eval().to(device)


@torch.inference_mode()
def encode(model, loader, layer_indices, device):
    chunks = []
    for images in tqdm(loader, desc="Extracting DINOv2 features"):
        images = images.to(device, non_blocking=True)
        outputs = model.get_intermediate_layers(
            images, n=layer_indices, return_class_token=True, reshape=False, norm=True
        )
        descriptor = torch.cat(
            [patch_tokens.mean(1) for patch_tokens, _ in outputs], dim=1
        )
        chunks.append(F.normalize(descriptor, p=2, dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 hierarchical DINOv2 feature extraction"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--id-path",
        required=True,
        help="All training samples, one image/mask pair per line",
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--weights", default="pretrained/dinov2_small.pth")
    parser.add_argument(
        "--backbone-size", choices=sorted(LAYER_INDICES), default="small"
    )
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lines = read_split(args.id_path)
    dataset = ImageList(args.data_root, lines, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = load_backbone(args.backbone_size, args.image_size, args.weights, device)
    embeddings = encode(model, loader, LAYER_INDICES[args.backbone_size], device)

    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", embeddings)
    (output / "sample_ids.txt").write_text(
        "\n".join(dataset.image_paths) + "\n", encoding="utf-8"
    )
    metadata = {
        "backbone": f"dinov2_{args.backbone_size}",
        "layers": LAYER_INDICES[args.backbone_size],
        "pooling": "patch_token_gap",
        "normalized": True,
        "image_size": args.image_size,
        "num_samples": len(dataset),
        "embedding_dim": int(embeddings.shape[1]),
    }

    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
