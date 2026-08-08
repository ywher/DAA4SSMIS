#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <busi|isic> <ratio>" >&2
  exit 2
fi

dataset=$1
ratio=$2
if [[ "${dataset}" != "busi" && "${dataset}" != "isic" ]]; then
  echo "Initial release supports only busi and isic" >&2
  exit 2
fi
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
feature_dir="${root_dir}/work_dirs/stage1/${dataset}/features"
output_dir="${root_dir}/work_dirs/stage1/${dataset}/splits/${ratio}"
data_root=${DATA_ROOT:-data/${dataset}}
weights=${DINOV2_WEIGHTS:-pretrained/dinov2_small.pth}

cd "${root_dir}"
python -m stage1.extract_features \
  --data-root "${data_root}" \
  --id-path "splits/${dataset}/train.txt" \
  --weights "${weights}" \
  --save-dir "${feature_dir}"

python -m stage1.select_samples \
  --feature-dir "${feature_dir}" \
  --all-id-path "splits/${dataset}/train.txt" \
  --ratio "${ratio}" \
  --output-dir "${output_dir}"
