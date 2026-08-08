from pathlib import Path


ROOT = Path(__file__).parents[1]
RATIOS = {
    "busi": ("1_16", "1_8", "1_4"),
    "isic": ("1_80", "1_40", "1_20"),
}


def read_lines(path):
    return {line for line in path.read_text(encoding="utf-8").splitlines() if line}


def test_paper_splits_are_disjoint_and_complete():
    for dataset, ratios in RATIOS.items():
        train = read_lines(ROOT / "splits" / dataset / "train.txt")
        assert train
        for ratio in ratios:
            split_dir = ROOT / "splits" / dataset / "dkc" / ratio
            labeled = read_lines(split_dir / "labeled.txt")
            unlabeled = read_lines(split_dir / "unlabeled.txt")
            assert labeled
            assert unlabeled
            assert labeled.isdisjoint(unlabeled)
            assert labeled | unlabeled == train
