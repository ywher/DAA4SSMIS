from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stage1.case_sequences import centered_sequences, sequence_distances
from stage1.select_samples import select_split
from util.cases import assemble_volumes, case_id, validate_partition
from util.case_sampler import CaseSampler


def entry(case, index, split="train"):
    return f"image/{split}/{case}/{index:03d}.png label/{split}/{case}/{index:03d}.png"


def test_centering_sorts_slices_and_pads_short_cases():
    ids = [entry("long", i) for i in (4, 0, 3, 1, 2)] + [entry("short", 0)]
    features = np.eye(6, dtype=np.float32)
    sequences, names, mapping = centered_sequences(features, ids, length=3)
    assert names == ["long", "short"]
    np.testing.assert_array_equal(sequences[0], features[[3, 4, 2]])
    np.testing.assert_array_equal(sequences[1], features[[5, 5, 5]])
    assert mapping["long"]["original_slices"] == 5


def test_sequence_distance_preserves_order_and_reduces_to_2d():
    sequences = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    # Same mean vector, but different ordered anatomy.
    assert sequence_distances(sequences)[0, 1] == pytest.approx(1)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(
        sequence_distances(vectors[:, None]), 1 - vectors @ vectors.T
    )


def test_case_selection_floors_budget_and_expands_all_slices(tmp_path):
    ids = [entry(f"Case{c:02d}", s) for c in range(35) for s in range(c % 3 + 1)]
    path = tmp_path / "train.txt"
    path.write_text("\n".join(ids) + "\n")
    np.save(tmp_path / "embeddings.npy", np.random.RandomState(1).randn(len(ids), 6))
    (tmp_path / "sample_ids.txt").write_text("\n".join(line.split()[0] for line in ids))
    labeled, unlabeled, metadata = select_split(
        tmp_path, path, "1_10", tmp_path / "out", case_sequence=True
    )
    assert metadata["num_labeled_cases"] == 3  # floor(35/10), not round(3.5)
    validate_partition(ids, labeled, unlabeled)
    repeat = select_split(
        tmp_path, path, "1_10", tmp_path / "repeat", case_sequence=True
    )
    assert repeat[:2] == (labeled, unlabeled)
    with pytest.raises(ValueError, match="case occurs"):
        validate_partition(ids, ids[:2], ids[2:])


def test_case_sampler_never_splits_or_duplicates_volumes():
    ids = [entry("CaseA", 0), entry("CaseB", 1), entry("CaseA", 1), entry("CaseB", 0)]
    partitions = [list(CaseSampler(SimpleNamespace(ids=ids), r, 3)) for r in range(3)]
    assert sorted(i for indices in partitions for i in indices) == list(range(4))
    assert partitions[2] == []
    assert [{case_id(ids[i]) for i in indices} for indices in partitions] == [
        {"CaseA"},
        {"CaseB"},
        set(),
    ]


def test_volume_assembly_orders_numerically():
    records = [
        (entry("CaseA", i), np.full((2, 2), i), np.zeros((2, 2))) for i in (10, 2)
    ]
    _, prediction, _ = next(assemble_volumes(records))
    assert list(prediction[:, 0, 0]) == [2, 10]


def test_released_split_has_35_5_10_cases_and_no_leakage():
    root = Path(__file__).parents[1] / "splits/promise12"
    lines = {
        split: (root / f"{split}.txt").read_text().splitlines()
        for split in ("train", "val", "test")
    }
    groups = [{case_id(line) for line in lines[split]} for split in lines]
    assert list(map(len, groups)) == [35, 5, 10]
    assert len(set.union(*groups)) == 50
    labeled, unlabeled = [
        (root / "dkc/1_16" / f"{kind}.txt").read_text().splitlines()
        for kind in ("labeled", "unlabeled")
    ]
    validate_partition(lines["train"], labeled, unlabeled)
    assert {case_id(line) for line in labeled} == {"Case02", "Case14"}


def test_3d_evaluation_and_training_validation_use_volume_dice(monkeypatch, tmp_path):
    import evaluate as evaluation
    import train

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(evaluation.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(evaluation.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        evaluation.dist,
        "all_gather_object",
        lambda output, rows: output.__setitem__(0, rows),
    )

    def aggregate(values):
        return np.array([np.mean(v) for v in values])

    monkeypatch.setattr(evaluation, "aggregate_metrics_across_gpus", aggregate)
    monkeypatch.setattr(train, "aggregate_metrics_across_gpus", aggregate)

    class Model(torch.nn.Module):
        def forward(self, images):
            # Slice 0 perfect foreground, slice 1 misses all foreground.
            return torch.stack([1 - images[:, 0], images[:, 0]], dim=1)

    images = torch.zeros(2, 3, 2, 2)
    images[0] = 1
    loader = [
        (
            images,
            torch.ones(2, 2, 2, dtype=torch.long),
            [entry("CaseA", 0), entry("CaseA", 1)],
        )
    ]
    cfg = {
        "nclass": 2,
        "dataset": "promise12",
        "case_level": True,
        "eval_resize": [2, 2],
    }
    args = SimpleNamespace(save_pred=True, save_compare=False, eval_resize=[2, 2])
    metrics = evaluation.evaluate(Model(), loader, cfg, 0, args, tmp_path)
    assert metrics["dice"][1] == pytest.approx(200 / 3)
    assert train.validate(Model(), loader, 2, 0, cfg)[0] == pytest.approx(200 / 3)
    assert (tmp_path / "pred_id/CaseA/000.png").is_file()
    assert (tmp_path / "case_metrics.csv").is_file()


def test_reference_preprocessing_on_synthetic_mhd(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from PIL import Image
    from scripts.prepare_promise12 import prepare

    raw, splits, output = tmp_path / "raw", tmp_path / "splits", tmp_path / "data"
    raw.mkdir()
    splits.mkdir()
    for c, split in enumerate(("train", "val", "test")):
        case = f"Case{c:02d}"
        pixels = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        for name, array in (
            (case, pixels),
            (case + "_segmentation", (pixels > 10).astype(np.uint8)),
        ):
            sitk.WriteImage(sitk.GetImageFromArray(array), str(raw / f"{name}.mhd"))
        (splits / f"{split}.txt").write_text(
            "\n".join(entry(case, i, split) for i in range(3))
        )
    metadata = prepare(raw, output, splits, size=14)
    assert len(metadata["cases"]) == 3
    mask = np.array(Image.open(output / "label/train/Case00/001.png"))
    assert mask.shape == (14, 14)
    assert set(np.unique(mask)) <= {0, 1}
    with pytest.raises(FileExistsError):
        prepare(raw, output, splits, size=14)
