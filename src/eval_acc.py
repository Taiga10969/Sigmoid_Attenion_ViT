import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
from timm.layers import Mlp, PatchEmbed
from torch.nn import LayerNorm
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vit import Block, VisionTransformer

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU') 
else: print(f'Use {torch.cuda.device_count()} GPUs')

date = datetime.now().strftime("%Y%m%d_%H%M")
save_dir = f"./results/eval_acc{date}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Config
parser = argparse.ArgumentParser(description="Model and Dataset Configuration")
parser.add_argument('--model_pth', type=str, default="./result/run_cub200_FT/best_acc.pt", help="Path to the model file")
parser.add_argument('--dataset_path', type=str, default="/taiga/share/ABN_Fine-tuning/dataset/cub200", help="Path to the dataset directory")
parser.add_argument('--num_classes', type=int, default=200, help="Number of classes in the dataset")
parser.add_argument('--dataset', type=str, default="cub200", help="Name of the dataset")

args = parser.parse_args()

# Save config to text file
result_text = f"""
# Configuration
model_pth = "{args.model_pth}"
dataset_path = "{args.dataset_path}"
num_classes = {args.num_classes}
dataset = "{args.dataset}"

# Result
"""

# Model load
# vit_small_patch16_224 の設定
model_config = {
    "img_size": (224, 224),
    "patch_size": 16,
    "in_chans": 3,
    "num_classes": args.num_classes,
    "global_pool": 'token',
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_norm": False,
    "init_values": None,
    "class_token": True,
    "pos_embed": 'learn',
    "no_embed_class": False,
    "reg_tokens": 0,
    "pre_norm": False,
    "fc_norm": None,
    "dynamic_img_size": False,
    "dynamic_img_pad": False,
    "drop_rate": 0.0,
    "pos_drop_rate": 0.0,
    "patch_drop_rate": 0.0,
    "proj_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0,
    "weight_init": '',  # 空の文字列に設定
    "fix_init": False,
    "embed_layer": PatchEmbed,
    "norm_layer": LayerNorm,
    "act_layer": None,
    "block_fn": Block,
    "mlp_layer": Mlp,
}

model = VisionTransformer(**model_config)

state_dict = torch.load(args.model_pth, map_location=torch.device('cpu'), weights_only=True)

msg = model.load_state_dict(state_dict, strict=False)
print("model.load_state_dict msg : ", msg)

model = model.to(device)

# データセットの用意
if args.dataset == "cifar10":
    
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    print(f"dataset : {testset}\nlen(trainset) : {len(testset)}")

    idx_to_class = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


elif args.dataset == "cub200":

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4817, 0.4974, 0.4319), (0.2297, 0.2256, 0.2655)),
    ])

    testset = datasets.ImageFolder(root=os.path.join(args.dataset_path,'val'), transform=transform_test)
    print(f"dataset : {testset}\nlen(trainset) : {len(testset)}")

    idx_to_class = [k for k, v in testset.class_to_idx.items()]
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)



# Nの最大値を指定
max_n = 5
cnt = 0
correct = [0] * max_n  # 各Nに対する正解数を記録するリスト
mpp_list = []

model=model.eval()
for i in tqdm(range(len(testset))):
    image, label = testset[i]
    image = image.unsqueeze(0).to(device)

    output = model(image)
    
    # モデルの出力から上位max_n個の予測ラベルを取得
    top_n_predictions = torch.topk(output, max_n).indices.squeeze(0)

    probs = torch.softmax(output, dim=1).squeeze(0)[label]
    mpp_list.append(probs.cpu().item())

    
    # 各Nに対して正解ラベルが含まれているかを確認
    for n in range(max_n):
        if label in top_n_predictions[:n+1]:
            correct[n] += 1
    cnt += 1


topN_accs = []
for n in range(max_n):
    topN_acc = correct[n] / cnt
    print(f"Top-{n+1} Accuracy : {topN_acc:.4f}")
    topN_accs.append(f"Top-{n+1} Accuracy : {topN_acc:.4f}")

print("MPP (mean) : ", np.mean(mpp_list))
print("MPP (max) : ", np.max(mpp_list))
print("MPP (min) : ", np.min(mpp_list))

result_text += "\n".join(topN_accs)
result_text += f"\nMPP (mean) = {np.mean(mpp_list)}"
result_text += f"\nMPP (max) = {np.max(mpp_list)}"
result_text += f"\nMPP (min) = {np.min(mpp_list)}"


with open(os.path.join(save_dir, "result.txt"), "w") as config_file:
    config_file.write(result_text)


# KDIをプロット
sns.histplot(mpp_list, kde=True, stat="density", bins=30, color='skyblue', alpha=0.6, edgecolor=None)
plt.title('MPP : Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.savefig(os.path.join(save_dir, "label_prob_MPP.pdf"))
plt.savefig(os.path.join(save_dir, "label_prob_MPP.png"))
plt.show()