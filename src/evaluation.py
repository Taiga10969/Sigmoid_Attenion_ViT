import os
import argparse
import torch
import json
from tqdm import tqdm
import pandas as pd
from timm.layers import PatchEmbed, Mlp
from models.vit import Block
from torch.nn import LayerNorm

from torchvision import datasets, transforms
from datasets.cub200_2011_dataset import CUB2011_Dataset

from models.model_config import load_model_config
from models.vit import VisionTransformer
from models.sigmoid_attention import Sigmoid_Attention


parser = argparse.ArgumentParser(description='evaluation_argparser')
parser.add_argument('--eval_run_dir', type=str, default="./results/run_cub200_2011_FT")
parser.add_argument('--dataset', type=str, default='cub200_2011')
parser.add_argument('--num_classes', type=int, default=200)
args = parser.parse_args()

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU') 
else: print(f'Use {torch.cuda.device_count()} GPUs')


eval_name = args.eval_run_dir.split('/')[-1]
#save_dir = f"./evaluations/eval_{eval_name}"
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)

model_pth = os.path.join(args.eval_run_dir, "best_acc.pt")
config_pth = os.path.join(args.eval_run_dir, "train_data_lists.json")


eval_data = {"run_name": eval_name}


## localモデルでの読み込み
model_config = load_model_config(args.num_classes, PatchEmbed, LayerNorm, Block, Mlp)
model = VisionTransformer(**model_config)
model.blocks[-1].attn = Sigmoid_Attention(dim=model_config['embed_dim'], 
                                          num_heads=model_config['num_heads'], 
                                          qkv_bias=model_config['qkv_bias']
                                          )

state_dict = torch.load(model_pth, map_location=torch.device('cpu'), weights_only=True)

msg = model.load_state_dict(state_dict, strict=False)
print("model.load_state_dict msg : ", msg)
model = model.to(device)



if args.dataset == "cub200":
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4817, 0.4974, 0.4319), (0.2297, 0.2256, 0.2655)),
    ])

    testset = datasets.ImageFolder(root='./datasets/cub200/val', transform=transform_test)
    print(f"dataset : {testset}\nlen(trainset) : {len(testset)}")
    idx_to_class = [k for k, v in testset.class_to_idx.items()]

elif args.dataset == "cub200_2011":
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4859, 0.4996, 0.4318), (0.1750, 0.1738, 0.1859)),
    ])

    testset = CUB2011_Dataset(json_path="./datasets/cub200_2011_dataset.json", 
                              images_dir_path="./datasets/CUB_200_2011/images", 
                              split='test', 
                              img_transform=transform_test)
    print(f"dataset : {testset}\nlen(trainset) : {len(testset)}")
    idx_to_class = testset.label_list


# Nの最大値を指定
max_n = 5
cnt = 0
correct = [0] * max_n  # 各Nに対する正解数を記録するリスト
miss_index = []

model=model.eval()
for i in tqdm(range(len(testset))):
    image, label = testset[i]
    image = image.unsqueeze(0).to(device)

    output = model(image)
    
    # モデルの出力から上位max_n個の予測ラベルを取得
    top_n_predictions = torch.topk(output, max_n).indices.squeeze(0)

    if label in top_n_predictions[:1]:
        pass
    else:
        miss_index.append(i)

    
    # 各Nに対して正解ラベルが含まれているかを確認
    for n in range(max_n):
        if label in top_n_predictions[:n+1]:
            correct[n] += 1
    cnt += 1

title = args.eval_run_dir
for n in range(max_n):
    topN_acc = correct[n] / cnt
    eval_data[f"top{n+1}"] = topN_acc
    print(f"Top-{n+1} Accuracy : {topN_acc:.4f}")


# CSVファイルのパス
csv_file_path = f"./evaluations/eval_data_{args.dataset}.csv"

# CSVファイルの存在を確認し、データを追加または新規作成
if os.path.exists(csv_file_path):
    # 既存のCSVファイルを読み込む
    df = pd.read_csv(csv_file_path)
    # 新しいデータをデータフレームとして作成
    new_data = pd.DataFrame([eval_data])
    # データを結合
    df = pd.concat([df, new_data], ignore_index=True)
else:
    # 新しいCSVファイルを作成
    df = pd.DataFrame([eval_data])

# データフレームをCSVに書き込み
df.to_csv(csv_file_path, index=False)


# JSONファイルにデータを書き込み
json_file_path = f"./evaluations/eval_data_{args.dataset}.json"

# JSONファイルの存在を確認し、データを追加または新規作成
if os.path.exists(json_file_path):
    # 既存のJSONファイルを読み込み
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
else:
    # 新しいJSONデータを作成
    json_data = {}
json_data[eval_name] = miss_index
with open(json_file_path, 'w') as file:
    json.dump(json_data, file, ensure_ascii=False, indent=4)