#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME=HHCAST
SEQ_LEN=96
LABEL_LEN=48
DATA_ROOT=./data
DATA_PATH=ETTh1.csv
DATASET=ETTh1
MODEL_TAG=ETTh1

COMMON_ARGS=(
  --task_name long_term_forecast
  --is_training 1
  --root_path "${DATA_ROOT}"
  --data_path "${DATA_PATH}"
  --model "${MODEL_NAME}"
  --data "${DATASET}"
  --features M
  --seq_len "${SEQ_LEN}"
  --label_len "${LABEL_LEN}"
  --e_layers 2
  --d_layers 1
  --enc_in 7
  --c_out 7
  --dropout 0.8
  --patch_len 2
  --pos 0
  --des Exp
  --learning_rate 0.0001
  --batch_size 32
  --train_epochs 10
  --d_model 128
  --itr 1
)

run_experiment() {
  local pred_len=$1
  shift || true

  python -u run.py \
    "${COMMON_ARGS[@]}" \
    --model_id "${MODEL_TAG}_${SEQ_LEN}_${pred_len}" \
    --pred_len "${pred_len}" \
    "$@"
}

for pred_len in 96 192 336; do
  run_experiment "${pred_len}" --d_ff 256
done

run_experiment 720 --d_ff 128
