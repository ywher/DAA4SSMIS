#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <num_gpus> <port> <dataset> <ratio> [save_path]" >&2
  exit 2
fi

num_gpus=$1
port=$2
dataset=$3
ratio=$4
if [[ "${dataset}" != "busi" && "${dataset}" != "isic" && "${dataset}" != "promise12" ]]; then
  echo "Supported datasets: busi, isic, promise12" >&2
  exit 2
fi
save_path=${5:-work_dirs/${dataset}/${ratio}/unimatchv2_bcp}
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config_path=${CONFIG_PATH:-configs/${dataset}.yaml}
weights=${DINOV2_WEIGHTS:-pretrained/dinov2_small.pth}
labeled_path=${LABELED_ID_PATH:-splits/${dataset}/dkc/${ratio}/labeled.txt}
unlabeled_path=${UNLABELED_ID_PATH:-splits/${dataset}/dkc/${ratio}/unlabeled.txt}

cd "${root_dir}"
torchrun --nproc_per_node="${num_gpus}" --master_port="${port}" train.py \
  --config "${config_path}" \
  --labeled-id-path "${labeled_path}" \
  --unlabeled-id-path "${unlabeled_path}" \
  --pretrained-path "${weights}" \
  --save-path "${save_path}"
