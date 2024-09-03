#!/bin/bash

# ViT-Trainingの設定
PROJECTS_NAME="Sigmoid_Attention_ViT"
RUNS_NAME="cub200_2011_FT"
DATASET="cub200_2011" #"cub200"
#DATASET_PATH="/taiga/share/ABN_Fine-tuning/dataset/cub200"
DATASET_PATH='./datasets/CUB_200_2011/images'
BATCH_SIZE=128
MODEL_NAME="vit_small_patch16_224"
IMG_SIZE=224
IS_DATAPARALLEL=false
LR=1e-5
OPT="adam"
EPOCHS=100
WANDB=false # wandbで学習進捗を確認するにはtrueに変更してapiキーを指定
WANDB_KEY="your_api_key"
PRETRAINED=true   # falseにすると，スクラッチからの学習

# コマンド実行
CUDA_VISIBLE_DEVICES=3 python3 -m src.vit_train \
    --projects_name $PROJECTS_NAME \
    --runs_name $RUNS_NAME \
    --dataset $DATASET \
    --dataset_path $DATASET_PATH \
    --batch_size $BATCH_SIZE \
    --model_name $MODEL_NAME \
    --img_size $IMG_SIZE \
    $( [ $IS_DATAPARALLEL = true ] && echo "--is_DataParallel" ) \
    --lr $LR \
    --opt $OPT \
    --epochs $EPOCHS \
    $( [ $WANDB = true ] && echo "--wandb" ) \
    --wandb_key $WANDB_KEY \
    $( [ $PRETRAINED = false ] && echo "--pretrained" ) \
