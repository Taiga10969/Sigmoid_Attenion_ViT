#!/bin/bash

# ViT-Trainingの設定
PROJECTS_NAME="Sigmoid_ViT_Training"
RUNS_NAME="cifar10_FT"
DATASET="cifar10"
BATCH_SIZE=512
MODEL_NAME="vit_small_patch16_224"
IMG_SIZE=224
IS_DATAPARALLEL=true
LR=1e-4
OPT="adam"
EPOCHS=50
# option====================
WANDB=false
WANDB_KEY="your_wandb_key"

# コマンド実行
CUDA_VISIBLE_DEVICES=0,1 python3 vit_train.py \
    --projects_name $PROJECTS_NAME \
    --runs_name $RUNS_NAME \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --model_name $MODEL_NAME \
    --img_size $IMG_SIZE \
    $( [ $IS_DATAPARALLEL = true ] && echo "--is_DataParallel" ) \
    --lr $LR \
    --opt $OPT \
    --epochs $EPOCHS \
    $( [ $WANDB = true ] && echo "--wandb" ) \
    --wandb_key $WANDB_KEY
