#!/bin/bash

# ViT-Trainingの設定
PROJECTS_NAME="Sigmoid_Attention_ViT_denoise"
RUNS_NAME="cub200_2011_HITL_lambda"
DATASET="cub200_2011"
DATASET_PATH="./datasets/CUB_200_2011/images"
BUBBLE_PATH="./datasets/CUB_GHA"
BATCH_SIZE=128
MODEL_NAME="vit_small_patch16_224"
IMG_SIZE=224
IS_DATAPARALLEL=false
LR=1e-5
OPT="adam"
EPOCHS=50
WANDB=true
WANDB_KEY="your_api_key"
PRETRAINED=true   # falseにすると，スクラッチからの学習
FIRST_STEP_PRETRAIN_PTH="./results/run_cub200_2011_FT/best_acc.pt"
#ATTN_LAMBDA=20.0
ATTN_LAMBDAS=(0.01 0.05 0.10 0.50 1.00 5.00 10.00 50.00 100.00 150.00)
#ATTN_LAMBDAS=(5.00 10.00 50.00 100.00 150.00)
#ATTN_LAMBDAS=(200.00 300.00 500.00 750.00 1000.00)

## コマンド実行
#CUDA_VISIBLE_DEVICES=0,1 python3 vit_HITL_train.py \
#    --projects_name $PROJECTS_NAME \
#    --runs_name $RUNS_NAME \
#    --dataset $DATASET \
#    --dataset_path $DATASET_PATH \
#    --batch_size $BATCH_SIZE \
#    --model_name $MODEL_NAME \
#    --img_size $IMG_SIZE \
#    $( [ $IS_DATAPARALLEL = true ] && echo "--is_DataParallel" ) \
#    --lr $LR \
#    --opt $OPT \
#    --epochs $EPOCHS \
#    $( [ $WANDB = true ] && echo "--wandb" ) \
#    --wandb_key $WANDB_KEY \
#    $( [ $PRETRAINED = false ] && echo "--pretrained" ) \
#    #--first_step_pretrain_pth $FIRST_STEP_PRETRAIN_PTH \
#    --attn_lambda $ATTN_LAMBDA \


for attn_lambda in "${ATTN_LAMBDAS[@]}"
do
  runs_name_="${RUNS_NAME}_$(echo $attn_lambda | sed 's/\.//')"
  echo "Running training with attn_lambda $attn_lambda and runs_name $runs_name_"

  CUDA_VISIBLE_DEVICES=3 python3 -m src.vit_HITL_train \
    --projects_name $PROJECTS_NAME \
    --runs_name $runs_name_ \
    --dataset $DATASET \
    --dataset_path $DATASET_PATH \
    --bubble_path $BUBBLE_PATH \
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
    --first_step_pretrain_pth $FIRST_STEP_PRETRAIN_PTH \
    --attn_lambda $attn_lambda \

done
