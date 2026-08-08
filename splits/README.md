# Data splits

`train.txt` and `val.txt` define the dataset partitions used in the paper. Each
line contains an image path and a mask path, both relative to the dataset's
configured `data_root`.

The frozen experiment splits are stored as:

```text
splits/<dataset>/dkc/<ratio>/labeled.txt
splits/<dataset>/dkc/<ratio>/unlabeled.txt
```

This initial release includes two 2D examples: BUSI and ISIC. Other dataset
partitions will be added after their preprocessing and 3D protocols are
consolidated.

The test suite checks that each labeled/unlabeled pair is disjoint and that its
union exactly recovers `train.txt`.
