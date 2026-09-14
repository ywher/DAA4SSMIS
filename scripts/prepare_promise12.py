"""Convert labeled PROMISE12 .mhd volumes to ordered PNG slices.

Reference preprocessing for this release, not a reconstruction of undocumented
historical intensity preprocessing. Writes only to a new output directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_volume(volume):
    if not np.isfinite(volume).all():
        raise ValueError("Non-finite image intensities")
    low, high = np.percentile(volume, [0.5, 99.5])
    if high <= low:
        return np.zeros(volume.shape, dtype=np.uint8)
    return (np.clip((volume - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def prepare(raw_root, output, split_root, size=518):
    import SimpleITK as sitk

    raw_root, output, split_root = Path(raw_root), Path(output), Path(split_root)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if size < 1:
        raise ValueError("Size must be positive")
    cases = {}
    for split in ("train", "val", "test"):
        lines = (split_root / f"{split}.txt").read_text().splitlines()
        for line in lines:
            case = Path(line.split()[0]).parent.name
            if case in cases and cases[case] != split:
                raise ValueError(f"Case appears in multiple splits: {case}")
            cases[case] = split
    sources = {}
    for case in sorted(cases):
        image_matches = list(raw_root.rglob(f"{case}.mhd"))
        mask_matches = list(raw_root.rglob(f"{case}_segmentation.mhd"))
        if len(image_matches) != 1 or len(mask_matches) != 1:
            raise ValueError(f"Expected exactly one image and mask for {case}")
        sources[case] = image_matches[0], mask_matches[0]
    output.mkdir(parents=True)
    metadata = {
        "normalization": "per-volume 0.5/99.5 percentile clip to uint8",
        "slice_axis": "SimpleITK z (array axis 0); native orientation",
        "size": size,
        "cases": {},
    }
    generated = {split: [] for split in ("train", "val", "test")}
    for case, (image_path, mask_path) in sources.items():
        image, mask = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(mask_path))
        if image.GetDimension() != 3 or mask.GetDimension() != 3:
            raise ValueError(f"Expected 3D volumes for {case}")
        if image.GetSize() != mask.GetSize() or any(
            not np.allclose(a, b)
            for a, b in (
                (image.GetSpacing(), mask.GetSpacing()),
                (image.GetOrigin(), mask.GetOrigin()),
                (image.GetDirection(), mask.GetDirection()),
            )
        ):
            raise ValueError(f"Image/mask geometry mismatch: {case}")
        volume = normalize_volume(sitk.GetArrayFromImage(image).astype(np.float32))
        labels = sitk.GetArrayFromImage(mask)
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError(f"Expected binary 0/1 labels: {case}")
        split = cases[case]
        for kind in ("image", "label"):
            (output / kind / split / case).mkdir(parents=True)
        for index in range(len(volume)):
            image_rel, label_rel = (
                f"{kind}/{split}/{case}/{index:03d}.png" for kind in ("image", "label")
            )
            Image.fromarray(volume[index]).resize(
                (size, size), Image.Resampling.BICUBIC
            ).save(output / image_rel)
            Image.fromarray(labels[index].astype(np.uint8)).resize(
                (size, size), Image.Resampling.NEAREST
            ).save(output / label_rel)
            generated[split].append(f"{image_rel} {label_rel}")
        metadata["cases"][case] = {
            "split": split,
            "slices": len(volume),
            "source_spacing_xyz": image.GetSpacing(),
            "source_direction": image.GetDirection(),
            "source_origin": image.GetOrigin(),
        }
    for split, lines in generated.items():
        expected = (split_root / f"{split}.txt").read_text().splitlines()
        if set(lines) != set(expected):
            raise ValueError(
                f"Generated {split} slices differ from bundled manifest; inspect {output}"
            )
    (output / "preprocessing.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", default="data/promise12")
    parser.add_argument("--split-root", default="splits/promise12")
    parser.add_argument("--size", type=int, default=518)
    args = parser.parse_args()
    metadata = prepare(args.raw_root, args.output, args.split_root, args.size)
    print(f"Prepared {len(metadata['cases'])} cases in {args.output}")


if __name__ == "__main__":
    main()
