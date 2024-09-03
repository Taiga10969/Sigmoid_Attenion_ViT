import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timm.layers import Mlp, PatchEmbed
from torch.nn import LayerNorm
from torch.nn.utils import clip_grad_norm_
from torchvision import datasets, transforms
from tqdm import tqdm

from datasets.cub200_2010_dataset import ImageBubbleFolder
from datasets.cub200_2011_dataset import CUB2011_BubbleDataset, CUB2011_Dataset
from datasets.paired_transforms import (PairedTransforms, RandomHorizontalFlipPair, RandomCropPair, ResizePair)
from models.sigmoid_attention import Sigmoid_Attention
from models.vit import Block, VisionTransformer
from models.model_config import load_model_config


parser = argparse.ArgumentParser(description='ViT-Training')
parser.add_argument('--projects_name', type=str, default="ViT-Training")
parser.add_argument('--runs_name', type=str, default="")
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--dataset_path', type=str, default='./data')
parser.add_argument('--bubble_path', type=str, default='./')
parser.add_argument('--batch_size', type=int, default='512')
parser.add_argument('--model_name', type=str, default="vit_small_patch16_224")
parser.add_argument('--img_size', type=int, default="224")
parser.add_argument('--is_DataParallel', action='store_true')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb_key', type=str, default="")
parser.add_argument('--pretrained', action='store_false') #--pretrainedを渡すとFalseになる
parser.add_argument('--first_step_pretrain_pth', type=str, default="") #1st stepの学習済モデルからスタートする場合に指定
parser.add_argument('--attn_lambda', type=float, default=1.0)

args = parser.parse_args()


def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
fix_seeds(99)


if args.wandb == True:
    import wandb
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.projects_name,
               name=args.runs_name,
               config=args,
               )
    wandb.alert(title=f"from WandB infomation project:vit_cifar10", 
                text=f"start run vit_cifar10"
                )
else:
    wandb = None


# check result directory
result_dir = './results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# check project name
save_dir = f'./results/run_{args.runs_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print("train [info] : This project name is already running")
    raise RuntimeError("Duplicate project name. Program stopped.")

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU') 
else: print(f'Use {torch.cuda.device_count()} GPUs')

print("Registered variables in args:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")


# Logger setup
class Logger:
    def __init__(self, filename="my_program.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        #self.error = sys.stderr  # 元の標準エラー出力を保存

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 標準出力を Logger に置き換え
logger = Logger(filename=os.path.join(save_dir, "train.log"))
sys.stdout = logger


# Dataset  ===========================================================================

if args.dataset == "cifar10":
    
    pass

elif args.dataset == "cub200":

    # 画像とバブル画像に適用するTransform
    paired_transform = PairedTransforms(
    RandomHorizontalFlipPair(p=0.5),
    ResizePair(transforms.Resize((224, 224))),
    RandomCropPair(size=179)#, padding=4)
    # 他のペアで適用するTransformを追加可能
    )

    # 画像用のTransform
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.4817, 0.4974, 0.4319), (0.2297, 0.2256, 0.2655)),  # 画像にはNormalizeを適用
    ])
        
    # バブル画像用のTransform (Normalizeなし)
    bubble_transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.Grayscale(num_output_channels=1),  # 3チャンネルから1チャンネルのグレースケールに変換
    ])
    
    trainset = ImageBubbleFolder(
        image_root=os.path.join(args.dataset_path, 'train_bubbles'),
        bubble_root=os.path.join(args.dataset_path, 'bubbles_att'),
        img_transform=img_transform,
        bubble_transform=bubble_transform, 
        paired_transform=paired_transform
        )

    print(f"dataset : {trainset}\nlen(trainset) : {len(trainset)}")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4817, 0.4974, 0.4319), (0.2297, 0.2256, 0.2655)),
    ])

    testset = datasets.ImageFolder(root=os.path.join(args.dataset_path,'val'), transform=transform_test)
    print(f"dataset : {testset}\nlen(trainset) : {len(testset)}")
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    num_classes = 200

elif args.dataset == "cub200_2011":

    # 画像とバブル画像に適用するTransform
    paired_transform = PairedTransforms(
    RandomHorizontalFlipPair(p=0.5),
    ResizePair(transforms.Resize((224, 224))),
    RandomCropPair(size=179)#, padding=4)
    # 他のペアで適用するTransformを追加可能
    )

    # 画像用のTransform
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.4859, 0.4996, 0.4318), (0.1750, 0.1738, 0.1859)),  # 画像にはNormalizeを適用
    ])
        
    # バブル画像用のTransform (Normalizeなし)
    bubble_transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.Grayscale(num_output_channels=1),  # 3チャンネルから1チャンネルのグレースケールに変換
    ])
    
    trainset = CUB2011_BubbleDataset(json_path = './datasets/cub200_2011_dataset.json', 
                                    images_dir_path = args.dataset_path, 
                                    bubble_dir_path = args.bubble_path, 
                                    split='train', 
                                    img_transform=img_transform, 
                                    bubble_transform=bubble_transform, 
                                    paired_transform=paired_transform,
                                    )

    print(f"dataset : {trainset}\nlen(trainset) : {len(trainset)}")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4859, 0.4996, 0.4318), (0.1750, 0.1738, 0.1859)),
    ])

    testset = CUB2011_Dataset(json_path="./datasets/cub200_2011_dataset.json", 
                              images_dir_path=args.dataset_path, 
                              split='test', 
                              img_transform=transform_test)
    print(f"dataset : {testset}\nlen(trainset) : {len(testset)}")
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    num_classes = 200







# Model の定義 ==============================================================

# vit_small_patch16_224 の設定
model_config = load_model_config(num_classes, PatchEmbed, LayerNorm, Block, Mlp)

model = VisionTransformer(**model_config)
# replace
model.blocks[-1].attn = Sigmoid_Attention(dim=model_config['embed_dim'], 
                                          num_heads=model_config['num_heads'], 
                                          qkv_bias=model_config['qkv_bias']
                                          )

if args.pretrained:
    state_dict = torch.load('./models/vit_small_patch16_224.pt', map_location=torch.device('cpu'), weights_only=True)

    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.size() == model.state_dict()[k].size()}
    msg = model.load_state_dict(filtered_state_dict, strict=False)
    print("args.pretrained : model.load_state_dict msg : ", msg)

if args.first_step_pretrain_pth != "":
    state_dict = torch.load(args.first_step_pretrain_pth, map_location=torch.device('cpu'), weights_only=True)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"args.first_step_pretrain_pth : {args.first_step_pretrain_pth} model.load_state_dict msg : {msg}")

model = model.to(device)

if args.is_DataParallel == True:
    model = nn.DataParallel(model)

print("model : ", model)

criterion = nn.CrossEntropyLoss()
criterion.to(device)

attn_criterion = nn.MSELoss(reduction='none')  # 例えばMSELossを使用
attn_criterion.to(device)

if args.opt == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

# 学習 =======================================================================================

train_loss_list = []
test_loss_list = []
acc_list = []

train_iter = 1
test_iter = 1
best_acc = 0


for epoch in range(args.epochs):
    start = time.time()
    # train ==================================================================================
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    with tqdm(enumerate(trainloader), total=len(trainloader), desc='Training') as pbar:
        for batch_idx, (inputs, targets, bubbles) in pbar:
            inputs, targets, bubbles = inputs.to(device), targets.to(device), bubbles.to(device)

            outputs, attention  = model(inputs,
                                  output_attentions=True,
                                  only_last_attn=True
                                  )
            
            outputs_loss = criterion(outputs, targets)

            bs = bubbles.size(0)
            bubbles = bubbles.reshape(bs, 1, -1)
            bubbles = bubbles.repeat(1, 6, 1)
            #bubbles = bubbles*0.9
            bubbles[bubbles != 0] = -1 #bubbleデータが0以外の部分は無視：ノイズを消す方向性

            #print("loss : ", loss)
            sigmoid_att = torch.sigmoid(attention)

            attn_loss = attn_criterion(sigmoid_att[:, :, 0, 1:], bubbles)
            attn_loss_mask = bubbles != -1
            masked_attn_loss = attn_loss[attn_loss_mask]
            attn_loss = masked_attn_loss.mean()

            total_loss = outputs_loss + (attn_loss * args.attn_lambda)

            optimizer.zero_grad()
            total_loss.backward()

            # 勾配クリップを追加
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(total_loss=train_loss/(batch_idx + 1), acc=100.*correct/total, outputs_loss=outputs_loss.item(), attn_loss=attn_loss.item())

            if args.wandb:
                wandb.log({
                    "train_iter" : train_iter,
                    "train_iter_loss" : total_loss.item(),
                    "attn_loss" : attn_loss.item(),
                    "outputs_loss" : outputs_loss.item(),
                    "lr" : optimizer.param_groups[0]['lr'],
                    })
        
            train_iter += 1
        
        trainloss = train_loss/(batch_idx + 1)
    
    # evaluate ==================================================================================

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(enumerate(testloader), total=len(testloader), desc='Testing') as pbar:
            for batch_idx, (inputs, targets) in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix(loss=test_loss/(batch_idx + 1), acc=100.*correct/total)

                if args.wandb:
                    wandb.log({
                        "val_iter" : test_iter,
                        "val_iter_loss" : loss.item(),
                        })
                
                test_iter += 1
            
            testloss = test_loss/(batch_idx+1)
    
    acc = 100.*correct/total

    if acc > best_acc:
        best_acc = acc
        if args.is_DataParallel == True:
            torch.save(model.module.state_dict(), os.path.join(save_dir, f'best_acc.pt'))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_acc.pt'))
        print(f"Epoch{epoch} [info] : save best_acc.pt")
    
    if args.is_DataParallel == True:
        torch.save(model.module.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pt'))
    
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {testloss:.5f}, acc: {acc:.5f}'
    print(content)

    with open(os.path.join(save_dir,'train_log.txt'), 'a') as appender:
        appender.write(content + "\n")
    
    scheduler.step()

    train_loss_list.append(trainloss)
    test_loss_list.append(testloss)
    acc_list.append(acc)

    # Log training..
    if wandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': testloss, "val_acc": acc, "lr_epoch": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})


# 辞書にまとめる
data = {
    'train_loss_list': train_loss_list,
    'test_loss_list': test_loss_list,
    'acc_list': acc_list,
}

# ファイルに保存
with open(os.path.join(save_dir, f'train_data_lists.json'), 'w') as f:
    json.dump(data, f)

if args.wandb:
    wandb.finish()

    









