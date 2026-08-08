import numpy as np

from stage1.select_samples import density_kcenter, select_split


def test_density_kcenter_is_deterministic_and_unique():
    rng = np.random.RandomState(7)
    features = rng.randn(20, 6).astype(np.float32)
    first = density_kcenter(features, budget=7, knn=5)
    second = density_kcenter(features, budget=7, knn=5)
    assert first == second
    assert len(first) == len(set(first)) == 7


def test_select_split_covers_all_samples(tmp_path):
    ids = [f"image/{index}.png label/{index}.png" for index in range(8)]
    split = tmp_path / "train.txt"
    split.write_text("\n".join(ids) + "\n", encoding="utf-8")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "embeddings.npy", np.eye(8, dtype=np.float32))
    (feature_dir / "sample_ids.txt").write_text(
        "\n".join(f"image/{index}.png" for index in range(8)) + "\n", encoding="utf-8"
    )
    labeled, unlabeled, metadata = select_split(
        feature_dir, split, "1_4", tmp_path / "out", knn=3
    )
    assert len(labeled) == 2
    assert len(unlabeled) == 6
    assert set(labeled).isdisjoint(unlabeled)
    assert set(labeled + unlabeled) == set(ids)
    assert metadata["strategy"] == "density-kcenter"
