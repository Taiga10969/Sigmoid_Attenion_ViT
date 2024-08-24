#!/bin/bash

# 実行するPythonスクリプトのパス
SCRIPT_PATH="eval_acc.py"

# 使用するモデルパス、データセットパス、クラス数、データセット名
MODEL_PATH="./result/run_cub200_FT/best_acc.pt"
DATASET_PATH="/taiga/share/ABN_Fine-tuning/dataset/cub200"
NUM_CLASSES=200
DATASET="cub200"

# Pythonスクリプトを引数付きで実行
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT_PATH \
    --model_pth $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --num_classes $NUM_CLASSES \
    --dataset $DATASET