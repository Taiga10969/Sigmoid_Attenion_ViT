#!/bin/bash

# 実行するPythonスクリプトのパス
SCRIPT_PATH="src.evaluation"

# 使用するモデルパス、データセットパス、クラス数、データセット名
EVAL_RUN_DIR="./results/run_cub200_2011_FT"
DATASET='cub200_2011'
NUM_CLASSES=200

# Pythonスクリプトを引数付きで実行
CUDA_VISIBLE_DEVICES=0 python3 -m $SCRIPT_PATH \
    --eval_run_dir $EVAL_RUN_DIR \
    --dataset $DATASET \
    --num_classes $NUM_CLASSES
