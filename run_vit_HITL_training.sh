#!/bin/bash

# ViT-Trainingの設定
PROJECTS_NAME="Sigmoid_ViT_Training"
#RUNS_NAME="cifar10_FT"
RUNS_NAME="cub200_HITL_Train"
DATASET="cub200"
DATASET_PATH="/taiga/share/ABN_Fine-tuning/dataset/cub200"
BATCH_SIZE=128
MODEL_NAME="vit_small_patch16_224"
IMG_SIZE=224
IS_DATAPARALLEL=false
LR=1e-5
OPT="adam"
EPOCHS=100
WANDB=true
WANDB_KEY="00fe025208d55e3e209f0132d63704ebc4c03b13"
PRETRAINED=true   # falseにすると，スクラッチからの学習
FIRST_STEP_PRETRAIN_PTH="./results/run_cub200_FT/best_acc.pt"
#ATTN_LAMBDA=20.0
#ATTN_LAMBDAS=(1.00 2.00 5.00 10.00 15.00 20.00)
#ATTN_LAMBDAS=(50.00 100.00 150.00 200.00)
ATTN_LAMBDAS=(300.00 500.00 750.00 1000.00)

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

  CUDA_VISIBLE_DEVICES=3 python3 vit_HITL_train.py \
    --projects_name $PROJECTS_NAME \
    --runs_name $runs_name_ \
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
    --first_step_pretrain_pth $FIRST_STEP_PRETRAIN_PTH \
    --attn_lambda $attn_lambda \

done
