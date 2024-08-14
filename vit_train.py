import os
import sys
import json
import time
import torch
import random
import argparse
import numpy as np
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from timm.models import create_model
import torchvision.transforms as transforms

from models.sigmoid_attention import Sigmoid_Attention


parser = argparse.ArgumentParser(description='ViT-Training')
parser.add_argument('--projects_name', type=str, default="ViT-Training")
parser.add_argument('--runs_name', type=str, default="")
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch_size', type=int, default='512')
parser.add_argument('--model_name', type=str, default="vit_small_patch16_224")
parser.add_argument('--img_size', type=int, default="224")
parser.add_argument('--is_DataParallel', action='store_true')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb_key', type=str, default="")

args = parser.parse_args()



def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
result_dir = './result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# check project name
save_dir = f'./result/run_{args.runs_name}'
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
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(args.img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    print(f"dataset : {trainset}\nlen(trainset) : {len(trainset)}")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    print(f"dataset : {testset}\nlen(trainset) : {len(testset)}")
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

elif args.dataset == "cub200":

    pass


# Model の定義 ==============================================================
model = create_model(args.model_name, pretrained=True, num_classes=10)
# replace
model.blocks[-1].attn = Sigmoid_Attention(dim=384, num_heads=6, qkv_bias=True)

state_dict = torch.load('./models/vit_small_patch16_224.pt', map_location=torch.device('cpu'), weights_only=True)
msg = model.load_state_dict(state_dict)
print("model.load_state_dict msg : ", msg)

model = model.to(device)

if args.is_DataParallel == True:
    model = nn.DataParallel(model)

print("model : ", model)

criterion = nn.CrossEntropyLoss()
criterion.to(device)

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
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(loss=train_loss/(batch_idx + 1), acc=100.*correct/total)

            if args.wandb:
                wandb.log({
                    "train_iter" : train_iter,
                    "train_iter_loss" : loss.item(),
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

    









