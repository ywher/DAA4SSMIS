# PROMISE12: 3D processing and case selection

This example runs entirely inside **DAA4SSMIS**. It releases Stage 1 and the
UniMatch V2 + BCP baseline; it does not reproduce the full paper's MCP/Stage 2
results. No images, masks, embeddings, or model weights are distributed.

## Environment and data preparation

Use the environment in the main README. Raw `.mhd` conversion additionally
requires SimpleITK (the sampling, training, and evaluation paths do not):

```bash
pip install SimpleITK
python scripts/prepare_promise12.py --raw-root /path/to/PROMISE12 --output data/promise12
```

Obtain the 50 labeled training volumes through PROMISE12's own distribution.
Each volume is `CaseXX.mhd`, with `CaseXX_segmentation.mhd` and the corresponding
raw payload files. The converter searches recursively and requires exactly one
matching image/mask pair for each case. It uses the research partition included
here: Case00–34 for training, Case35–39 for validation, Case40–49 for testing.
This 35/5/10 partition is our experimental split, not the challenge's hidden test set.

The converter checks image/mask geometry, reads native SimpleITK `[z,y,x]`
arrays, clips image intensities to the per-volume 0.5/99.5 percentiles, and maps
them to uint8. It keeps every slice and saves zero-padded numeric filenames.
Images are resized to 518×518 with bicubic interpolation; binary masks use
nearest-neighbor interpolation. Source spacing, origin, and orientation are
recorded in `preprocessing.json`. Existing output directories are refused.

This is a documented **reference conversion** for running the released example.
The exact intensity conversion used to create historical PNGs was not recovered;
the bundled selection was verified using the cached historical image features.
Newly converted images can therefore produce different selected cases or metrics.
The converter checks its slice manifests against the bundled lists. A mismatch
stops preparation and must be investigated before training.

Already-prepared data can use this layout directly (masks must contain 0/1):

```text
data/promise12/
├── image/train/Case00/000.png
├── label/train/Case00/000.png
├── image/val/Case35/000.png
├── label/val/Case35/000.png
├── image/test/Case40/000.png
└── label/test/Case40/000.png
```

Each `splits/promise12/{train,val,test}.txt` line contains two relative paths.
To use a different data root, set `DATA_ROOT` for Stage 1 and set `data_root` in
a copy of `configs/promise12.yaml` under `configs/local/` for train/evaluate.

## How Stage 1 extends from 2D to 3D

| Component | 2D image selection | PROMISE12 case selection |
| --- | --- | --- |
| Budget unit | Image | Complete patient volume |
| Frozen encoder | DINOv2-S, layers 2/5/8/11 (zero-based) | Same encoder applied to each slice |
| Descriptor | Patch GAP → concatenate → L2 normalization, 1536 dimensions | Same 1536-dimensional descriptor per slice |
| Representation | One image vector | Ordered `[M,1536]` case sequence |
| Distance | Image cosine distance | Mean cosine distance at corresponding sequence positions |
| Selection | Density-weighted K-Center | Same density/diversity principle on case distances |
| Training | 2D UniMatch V2 + BCP | Same model trained on slices of selected/unselected cases |
| Evaluation | Per-image metrics | Stack complete volumes and average case metrics |

```text
volume → ordered 2D slices → frozen four-layer features
       → center to M slices → case distance matrix → DKC selects cases
       → expand ALL original slices → labeled / unlabeled → 2D training → 3D evaluation
```

`stage1/case_sequences.py` sorts slices numerically within each case. The
reference sequence length `M` is the median depth over training cases only;
for the bundled training pool it is 23. Long cases retain the central M slices.
Short cases are cyclically repeated to length M. This is an anatomical heuristic,
not physical registration or a learned 3D encoder. It does not guarantee that
pathology is central, and truncation/padding can lose or repeat information.

The selected central slices define the **selection representation only**.
Once a case is chosen, **all** its original slices enter the labeled set.
No mask, foreground score, or held-out image is used to rank candidates.

For L2-normalized descriptors, case distance is:

```text
d(i,j) = mean_m [1 - e(i,m) · e(j,m)]
```

The sequence is not mean-pooled before measuring distance. Cases with identical
mean descriptors can still have different distances if slice order differs.
With `M=1`, this reduces to ordinary 2D cosine distance. Equivalently, flattening
the M unit descriptors and dividing by √M gives the same cosine geometry.

The 20 nearest case neighbors determine local density. We use inverse mean
neighbor distance, median scaling, and clipping to [0.3,1]. The first case has
the greatest average distance to the pool; subsequent cases maximize
`density_weight × distance_to_nearest_selected_case`. This is the distance-matrix
counterpart of the 2D initialization. The 2D code's explicit centroid and
squared-Euclidean density calculation remain unchanged. Annotation counts use
`floor(N_cases × ratio)`, so 35 cases yield 2/3/7 for 1/16, 1/10, and 1/5.
These are case fractions, not equal slice counts or measured annotation-time budgets.

## Commands

The bundled 1/16 split selects Case02 and Case14: 74 labeled slices and 938
unlabeled slices, from 35 cases / 1012 training slices. It has no case overlap.
These IDs were regenerated with the released sequence selector on cached
historical features; see `splits/promise12/dkc/1_16/selection.json`.

```bash
# Extract features from training images and generate a new complete-case split.
bash scripts/stage1.sh promise12 1_16

# Train using the bundled split; BCP is enabled by default.
bash scripts/train.sh 2 29500 promise12 1_16

# Or train using the newly selected split.
LABELED_ID_PATH=work_dirs/stage1/promise12/splits/1_16/labeled.txt \
UNLABELED_ID_PATH=work_dirs/stage1/promise12/splits/1_16/unlabeled.txt \
bash scripts/train.sh 2 29500 promise12 1_16 work_dirs/promise12/1_16/regenerated

# Evaluate EMA weights on the ten held-out cases.
bash scripts/evaluate.sh 29501 promise12 work_dirs/promise12/1_16/unimatchv2_bcp best test
```

To reuse already-extracted release-format features without encoding again:

```bash
python -m stage1.select_samples \
  --feature-dir work_dirs/stage1/promise12/features \
  --all-id-path splits/promise12/train.txt --case-sequence \
  --ratio 1_16 --knn 20 --output-dir work_dirs/stage1/promise12/splits/1_16
```

Training rejects splits that drop slices, duplicate image paths, or split a
case between labeled/unlabeled sets. Volume evaluation assigns each complete
case to exactly one process, without padding duplicates, including when there
are fewer cases than processes. Predictions are saved beneath case-specific
folders; `case_metrics.csv` contains one row per volume.

Case IoU/Dice/ASD/HD95 are calculated after resizing slice predictions/masks to
224×224 and stacking in numeric order. The default distance metrics use unit
grid spacing, **not millimeters in original MRI coordinates**. Best-checkpoint
selection uses mean foreground case Dice on validation cases. No complete
training or real-image inference run was repeated for this release.

To run the CPU verification suite, including synthetic-volume conversion:

```bash
pip install pytest SimpleITK
python -m pytest -q tests
```

The suite covers ordering, centering, short-case padding, case budgets, complete
partitions, distributed case assignment, and volume Dice versus slice Dice.

## 会场讲解（中文）

可以这样讲：

> 我们把“选择一张图”推广成“选择一整个病例”。先用相同的冻结 DINOv2
> 对每张 MRI 切片提取四层 patch 特征，再将同一病例按切片顺序组成序列，
> 通过中心截取和短序列补齐得到统一长度。之后计算对应位置切片的平均
> 余弦距离，用同样兼顾覆盖性与局部密度的 DKC 思路挑选病例。选中后，
> 该病例的全部切片一起进入标注集合。训练继续使用二维网络，评估时把
> 预测切片重新堆叠成三维体数据，按病例计算指标。

现场需要区分三个概念：**采样单位是病例，网络输入是二维切片，评估单位
是三维 volume**。它们可以不同。不要把本示例称为原生 3D 网络，也不要把
UniMatch V2+BCP 示例的结果等同于完整论文 MCP 的结果。

若被问“与 2D 方法的联系”：共用特征编码与密度加权覆盖准则，新增的主要
是病例表示和病例间距离；`M=1` 时序列距离直接回到二维余弦距离。
若被问“是否使用未标注数据的真值”：选样仅使用训练图像特征，标签用于
选样后的监督训练和独立评估。中心截取不是通过 GT 前景定位。
