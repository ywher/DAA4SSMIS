"""Case identity, slice ordering, and volume assembly shared by 3D workflows."""

from pathlib import PurePosixPath
import re

import numpy as np


def image_path(line):
    return line.replace(",", " ").split()[0]


def case_id(line):
    path = PurePosixPath(image_path(line))
    if len(path.parts) < 3:
        raise ValueError(f"Expected image/<split>/<case>/<slice>, got {line!r}")
    return path.parent.name


def slice_key(line):
    return tuple(
        int(s) if s.isdigit() else s for s in re.split(r"(\d+)", image_path(line))
    )


def group_cases(lines):
    groups = {}
    paths = [image_path(line) for line in lines]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate slice image paths")
    for index, line in enumerate(lines):
        groups.setdefault(case_id(line), []).append(index)
    return {
        name: sorted(groups[name], key=lambda i: slice_key(lines[i]))
        for name in sorted(groups)
    }


def assemble_volumes(records):
    """Group (split entry, prediction, target) records and sort slices numerically."""
    groups = group_cases([record[0] for record in records])
    for name, indices in groups.items():
        yield (
            name,
            np.stack([records[i][1] for i in indices]),
            np.stack([records[i][2] for i in indices]),
        )


def validate_partition(all_lines, labeled, unlabeled):
    all_paths = {image_path(line) for line in all_lines}
    for lines in (all_lines, labeled, unlabeled):
        group_cases(lines)
    left, right = (
        {image_path(line) for line in lines} for lines in (labeled, unlabeled)
    )
    if not left or not right or left & right or left | right != all_paths:
        raise ValueError(
            "Labeled/unlabeled lists must partition the complete training pool"
        )
    if {case_id(line) for line in labeled} & {case_id(line) for line in unlabeled}:
        raise ValueError("A case occurs in both labeled and unlabeled partitions")
