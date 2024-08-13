import os
import timm
import torch
import argparse
from timm.models import create_model

parser = argparse.ArgumentParser(description='load_pretrained')
parser.add_argument('--model_name', type=str, default="vit_small_patch16_224")
parser.add_argument('--save_dir', type=str, default="./models")
args = parser.parse_args()


model = create_model(args.model_name, pretrained=True, num_classes=10)
torch.save(model.to('cpu').state_dict(),os.path.join(args.save_dir, 'vit_small_patch16_224.pt'))

print("Save model weights successful")
