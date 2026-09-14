"""Structural centering and Density-K-Center for complete 3D cases.

The encoder remains 2D. Cases are compared through aligned slice descriptors;
no masks or foreground statistics are used for selecting cases.
"""

import numpy as np

from util.cases import group_cases


def centered_sequences(features, sample_ids, length=None):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(sample_ids) or not len(features):
        raise ValueError(
            "Expected a nonempty [slices, channels] feature matrix aligned with IDs"
        )
    if not np.isfinite(features).all():
        raise ValueError("Non-finite features")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero descriptors cannot define cosine distance")
    features = features / norms
    groups = group_cases(sample_ids)
    length = (
        int(np.median([len(indices) for indices in groups.values()]))
        if length is None
        else length
    )
    if length < 1:
        raise ValueError("Sequence length must be positive")
    sequences, mapping = [], {}
    for name, indices in groups.items():
        if len(indices) >= length:
            start = (len(indices) - length) // 2
            selected = indices[start : start + length]
        else:
            selected = [indices[i % len(indices)] for i in range(length)]
        sequences.append(features[selected])
        mapping[name] = {"original_slices": len(indices), "selected_indices": selected}
    return np.stack(sequences), list(groups), mapping


def sequence_distances(sequences):
    """Mean cosine distance at corresponding sequence positions, not all slice pairs."""
    sequences = np.asarray(sequences, dtype=np.float32)
    norms = np.linalg.norm(sequences, axis=-1, keepdims=True)
    if (
        sequences.ndim != 3
        or not np.isfinite(sequences).all()
        or np.any(norms <= 1e-12)
    ):
        raise ValueError(
            "Expected finite, nonzero [cases, positions, channels] descriptors"
        )
    sequences = sequences / norms
    distances = np.clip(
        1.0 - np.einsum("imd,jmd->ij", sequences, sequences) / sequences.shape[1], 0, 2
    )
    np.fill_diagonal(distances, 0)
    return distances


def sequence_density_kcenter(distances, budget, knn=20):
    """DKC on case distances; initialize by greatest mean distance to the pool."""
    count = len(distances)
    if distances.shape != (count, count) or not 0 < budget <= count or knn < 1:
        raise ValueError("Invalid distance matrix, budget, or neighbor count")
    if budget == count:
        return list(range(count))
    k = min(knn, count - 1)
    # Same density transform as the research case-sequence implementation.
    rho = np.sort(distances, axis=1)[:, 1 : k + 1].mean(axis=1)
    density = 1.0 / (rho + 1e-12)
    weights = np.clip(density / (density + np.median(density) + 1e-12), 0.3, 1.0)
    first = int(np.argmax(distances.mean(axis=1)))
    selected = [first]
    nearest = distances[:, first].copy()
    nearest[first] = -np.inf
    while len(selected) < budget:
        index = int(np.argmax(nearest * weights))
        selected.append(index)
        nearest = np.minimum(nearest, distances[:, index])
        nearest[selected] = -np.inf
    return selected
