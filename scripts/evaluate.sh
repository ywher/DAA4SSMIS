#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <port> <dataset> <exp_path> <best|latest> <val|test>" >&2
  exit 2
fi

port=$1
dataset=$2
exp_path=$3
model_name=$4
split=$5
if [[ "${dataset}" != "busi" && "${dataset}" != "isic" ]]; then
  echo "Initial release supports only busi and isic" >&2
  exit 2
fi
if [[ "${split}" != "val" && "${split}" != "test" ]]; then
  echo "Split must be val or test" >&2
  exit 2
fi
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config_path=${CONFIG_PATH:-configs/${dataset}.yaml}
split_args=()
resize=224
if [[ "${split}" == "test" ]]; then
  split_args+=(--test-set)
fi
cd "${root_dir}"
torchrun --nproc_per_node=1 --master_port="${port}" evaluate.py \
  --config "${config_path}" \
  --exp-path "${exp_path}" \
  --model-name "${model_name}" \
  --eval-resize "${resize}" "${resize}" \
  "${split_args[@]}"
