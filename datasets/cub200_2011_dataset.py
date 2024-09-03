import json
import os
import warnings
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class CUB2011_Dataset(Dataset):
    def __init__(self, json_path, images_dir_path, split='train', img_transform=None):
        # JSONファイルをロード
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        
        self.images_dir_path = images_dir_path
        
        # 'ALL' を指定した場合、train と test を結合
        if split == 'ALL':
            self.data = data['train'] + data['test']
        else:
            # 'train' または 'test' のデータを取得
            self.data = data[split]
        
        self.label_list = data['label_list']

        self.img_transform = img_transform
    
    def __len__(self):
        # データの長さを返す
        return len(self.data)
    
    def __getitem__(self, idx):
        # 指定されたインデックスのデータを取得
        img_path, label, _ = self.data[idx]
        
        # 画像を読み込む
        img_path = os.path.join(self.images_dir_path, img_path)
        img = Image.open(img_path).convert("RGB")  # 画像をRGB形式に変換
        
        if self.img_transform:
                img = self.img_transform(img)
        
        return img, label
    


class CUB2011_BubbleDataset(Dataset):
    def __init__(self, json_path, images_dir_path, bubble_dir_path, split='train', img_transform=None, bubble_transform=None, paired_transform=None):
        # JSONファイルをロード
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        
        self.images_dir_path = images_dir_path
        self.bubble_dir_path = bubble_dir_path
        
        # 'ALL' を指定した場合、train と test を結合
        if split == 'ALL':
            self.data = data['train'] + data['test']
        else:
            # 'train' または 'test' のデータを取得
            self.data = data[split]
        
        self.label_list = data['label_list']

        self.img_transform = img_transform
        self.bubble_transform = bubble_transform
        self.paired_transform = paired_transform

        self.to_tensor=transforms.ToTensor()
    
    def __len__(self):
        # データの長さを返す
        return len(self.data)
    
    def __getitem__(self, idx):
        # 指定されたインデックスのデータを取得
        img_path, label, bubble_data_name = self.data[idx]
        
        # 画像を読み込む
        img_path = os.path.join(self.images_dir_path, img_path)
        img = Image.open(img_path).convert("RGB")  # 画像をRGB形式に変換

        bubble_path = os.path.join(self.bubble_dir_path, f"{bubble_data_name}.jpg")

        try:
            bubble_img = Image.open(bubble_path).convert('RGB')
            # Bubble ImageにNormalize以外のTransformを適用
        except FileNotFoundError:
            # ファイルが存在しない場合の処理
            warnings.warn(f"Warning: Could not find bubble data.  file_path : {bubble_path} Returning data with all elements set to -1.",
                          category=UserWarning)
            bubble_img = Image.fromarray(np.full((224, 224, 3), -1, dtype=np.uint8))
        
        img = self.to_tensor(img)
        bubble_img = self.to_tensor(bubble_img)


        # 画像とバブル画像に同じTransformを適用
        if self.paired_transform:
            img, bubble_img = self.paired_transform(img, bubble_img)

        if self.img_transform:
                img = self.img_transform(img)

        if self.bubble_transform:
                bubble_img = self.bubble_transform(bubble_img)
        
        return img, label, bubble_img