"""Density-K-Center selection for the offline Stage 1."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from stage1.extract_features import parse_image_path, read_split
from stage1.case_sequences import (
    centered_sequences,
    sequence_distances,
    sequence_density_kcenter,
)
from util.cases import case_id, validate_partition


def l2_normalize(features):
    return features / np.clip(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None
    )


def density_weights(features, knn=20, lower=0.3):
    """Eq. (3): inverse mean squared distance to the k nearest neighbors."""
    if len(features) < 2:
        return np.ones(len(features), dtype=np.float32)
    neighbors = min(knn + 1, len(features))
    distances, _ = (
        NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
        .fit(features)
        .kneighbors(features)
    )
    rho = np.square(distances[:, 1:]).mean(axis=1)
    density = 1.0 / (rho + 1e-6)
    median = np.median(density)
    weights = density / (density + median + 1e-12)
    return np.clip(weights, lower, 1.0).astype(np.float32)


def density_kcenter(features, budget, knn=20):
    """Eqs. (4)-(5), using cosine distance on L2-normalized descriptors."""
    features = l2_normalize(np.asarray(features, dtype=np.float32))
    count = len(features)
    if not 0 < budget <= count:
        raise ValueError(f"budget must be in [1, {count}], got {budget}")
    if budget == count:
        return list(range(count))

    weights = density_weights(features, knn)
    mean = l2_normalize(features.mean(axis=0, keepdims=True))[0]
    first = int(np.argmax(1.0 - features @ mean))
    selected = [first]
    min_distance = 1.0 - features @ features[first]
    min_distance[first] = -np.inf
    while len(selected) < budget:
        index = int(np.argmax(min_distance * weights))
        selected.append(index)
        min_distance = np.minimum(min_distance, 1.0 - features @ features[index])
        min_distance[selected] = -np.inf
    return selected


def parse_ratio(value):
    if "_" in value:
        numerator, denominator = value.split("_", 1)
        return float(numerator) / float(denominator)
    return float(value)


def select_split(
    feature_dir,
    all_id_path,
    ratio,
    output_dir,
    knn=20,
    case_sequence=False,
    sequence_length=None,
):
    feature_dir, output_dir = Path(feature_dir), Path(output_dir)
    lines = read_split(all_id_path)
    fraction = parse_ratio(ratio)
    if not 0.0 < fraction < 1.0:
        raise ValueError("ratio must be between 0 and 1")

    features = np.load(feature_dir / "embeddings.npy")
    sample_ids = (
        (feature_dir / "sample_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    if len(features) != len(sample_ids):
        raise ValueError("Feature and sample ID counts differ")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate feature sample IDs")
    by_image = {parse_image_path(line): line for line in lines}
    if len(by_image) != len(lines):
        raise ValueError("The all-sample split contains duplicate image paths")
    if set(sample_ids) != set(by_image):
        raise ValueError("Feature sample IDs do not match the all-sample split")
    case_metadata = {}
    if case_sequence:
        sequences, cases, mapping = centered_sequences(
            features, sample_ids, sequence_length
        )
        budget = max(1, int(len(cases) * fraction))
        selected_indices = sequence_density_kcenter(
            sequence_distances(sequences), budget, knn
        )
        selected_cases = {cases[i] for i in selected_indices}
        selected_ids = {
            identifier
            for identifier in sample_ids
            if case_id(identifier) in selected_cases
        }
        case_metadata = {
            "sampling_unit": "case",
            "num_cases": len(cases),
            "num_labeled_cases": budget,
            "selected_cases": sorted(selected_cases),
            "sequence_length": sequences.shape[1],
            "sequence_strategy": "middle_with_cyclic_padding",
            "distance": "aligned_mean_cosine",
            "initialization": "largest_mean_distance_to_pool",
            "budget_rounding": "floor",
            "case_mapping": mapping,
        }
    else:
        budget = max(1, int(round(len(sample_ids) * fraction)))
        selected_indices = density_kcenter(features, budget, knn)
        selected_ids = {sample_ids[index] for index in selected_indices}
    labeled = [
        by_image[sample_id] for sample_id in sample_ids if sample_id in selected_ids
    ]
    unlabeled = [
        by_image[sample_id] for sample_id in sample_ids if sample_id not in selected_ids
    ]
    if case_sequence:
        validate_partition(lines, labeled, unlabeled)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labeled.txt").write_text("\n".join(labeled) + "\n", encoding="utf-8")
    (output_dir / "unlabeled.txt").write_text(
        "\n".join(unlabeled) + "\n", encoding="utf-8"
    )
    metadata = {
        "strategy": "density-kcenter",
        "ratio": ratio,
        "knn": knn,
        "num_labeled": len(labeled),
        "num_unlabeled": len(unlabeled),
        **case_metadata,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return labeled, unlabeled, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Density-K-Center selection")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--all-id-path", required=True)
    parser.add_argument("--ratio", required=True, help="Examples: 1_16, 1_8, 0.25")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument(
        "--case-sequence", action="store_true", help="Select complete 3D cases"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Default: median training-case depth",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, metadata = select_split(
        args.feature_dir,
        args.all_id_path,
        args.ratio,
        args.output_dir,
        knn=args.knn,
        case_sequence=args.case_sequence,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
