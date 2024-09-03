#!/bin/bash

# 実行するPythonスクリプトのパス
SCRIPT_PATH="src.evaluation"

# 使用するモデルパス、データセットパス、クラス数、データセット名
DATASET='cub200_2011'
NUM_CLASSES=200

# 複数のEVAL_RUN_DIRをリストとして定義
EVAL_RUN_DIRS=(
    "./results/run_cub200_2011_FT"
    "./results/run_cub200_2011_HITL_lambda_001"
    "./results/run_cub200_2011_HITL_lambda_001"
    "./results/run_cub200_2011_HITL_lambda_005"
    "./results/run_cub200_2011_HITL_lambda_010"
    "./results/run_cub200_2011_HITL_lambda_050"
    "./results/run_cub200_2011_HITL_lambda_100"
    "./results/run_cub200_2011_HITL_lambda_500"
    "./results/run_cub200_2011_HITL_lambda_1000"
    "./results/run_cub200_2011_HITL_lambda_5000"
    "./results/run_cub200_2011_HITL_lambda_10000"
    "./results/run_cub200_2011_HITL_lambda_15000"
    "./results/run_cub200_2011_HITL_denoise_lambda_001"
    "./results/run_cub200_2011_HITL_denoise_lambda_001"
    "./results/run_cub200_2011_HITL_denoise_lambda_005"
    "./results/run_cub200_2011_HITL_denoise_lambda_010"
    "./results/run_cub200_2011_HITL_denoise_lambda_050"
    "./results/run_cub200_2011_HITL_denoise_lambda_100"
    "./results/run_cub200_2011_HITL_denoise_lambda_500"
    "./results/run_cub200_2011_HITL_denoise_lambda_1000"
    "./results/run_cub200_2011_HITL_denoise_lambda_5000"
    "./results/run_cub200_2011_HITL_denoise_lambda_10000"
    "./results/run_cub200_2011_HITL_denoise_lambda_15000"
)

# 各EVAL_RUN_DIRに対してPythonスクリプトを実行
for EVAL_RUN_DIR in "${EVAL_RUN_DIRS[@]}"; do
    echo "Running evaluation for $EVAL_RUN_DIR"
    CUDA_VISIBLE_DEVICES=0 python3 -m $SCRIPT_PATH \
        --eval_run_dir $EVAL_RUN_DIR \
        --dataset $DATASET \
        --num_classes $NUM_CLASSES
done