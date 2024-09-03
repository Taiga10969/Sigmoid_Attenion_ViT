import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import torch.nn.functional as F


class ImageBubbleFolder(Dataset):
    def __init__(self, image_root, bubble_root, img_transform=None, bubble_transform=None, paired_transform=None):
        self.image_folder = datasets.ImageFolder(root=image_root, transform=None)
        self.bubble_root = bubble_root
        self.img_transform = img_transform
        self.bubble_transform = bubble_transform
        self.paired_transform = paired_transform

        self.to_tensor=transforms.ToTensor()

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, index):
        # 画像とラベルを取得
        img, label = self.image_folder[index]

        # 対応するバブルデータを取得
        img_path, _ = self.image_folder.samples[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0] + '.png'
        bubble_path = os.path.join(self.bubble_root, self.image_folder.classes[label], img_name)

        try:
            bubble_img = Image.open(bubble_path).convert('RGB')
            # Bubble ImageにNormalize以外のTransformを適用
        except FileNotFoundError:
            # ファイルが存在しない場合の処理
            bubble_img = Image.fromarray(np.full((224, 224, 3), -1, dtype=np.uint8))  # サイズは適宜調整
        
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
    
class PairedTransforms:
    def __init__(self, *pair_transforms):
        self.pair_transforms = pair_transforms

    def __call__(self, img, bubble_img):
        for transform in self.pair_transforms:
            img, bubble_img = transform(img, bubble_img)
        return img, bubble_img

class ResizePair:
    def __init__(self, resize_transformers):
        self.resize_transformers = resize_transformers

    def __call__(self, img, bubble_img):
        img = self.resize_transformers(img)
        bubble_img = self.resize_transformers(bubble_img)
        return img, bubble_img
    
class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bubble_img):
        if torch.rand(1).item() < self.p:
            if isinstance(img, Image.Image) and isinstance(bubble_img, Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT), bubble_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(img, torch.Tensor) and isinstance(bubble_img, torch.Tensor):
                return torch.flip(img, [2]), torch.flip(bubble_img, [2])
        return img, bubble_img

class RandomCropPair:
    def __init__(self, size, padding=None):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, bubble_img):
        if isinstance(img, torch.Tensor) and isinstance(bubble_img, torch.Tensor):
            if self.padding:
                # Add padding to the tensors
                img = F.pad(img, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
                bubble_img = F.pad(bubble_img, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

            _, h, w = img.shape
            th, tw = self.size

            if w == tw and h == th:
                return img, bubble_img

            x1 = torch.randint(0, w - tw + 1, (1,)).item()
            y1 = torch.randint(0, h - th + 1, (1,)).item()

            img = img[:, y1:y1 + th, x1:x1 + tw]
            bubble_img = bubble_img[:, y1:y1 + th, x1:x1 + tw]

        elif isinstance(img, Image.Image) and isinstance(bubble_img, Image.Image):
            if self.padding:
                # Add padding to the images
                w, h = img.size
                img = Image.new(img.mode, (w + 2 * self.padding, h + 2 * self.padding), 0)
                img.paste(img, (self.padding, self.padding))

                w, h = bubble_img.size
                bubble_img = Image.new(bubble_img.mode, (w + 2 * self.padding, h + 2 * self.padding), 0)
                bubble_img.paste(bubble_img, (self.padding, self.padding))

            w, h = img.size
            th, tw = self.size

            if w == tw and h == th:
                return img, bubble_img

            x1 = torch.randint(0, w - tw + 1, (1,)).item()
            y1 = torch.randint(0, h - th + 1, (1,)).item()

            img = img.crop((x1, y1, x1 + tw, y1 + th))
            bubble_img = bubble_img.crop((x1, y1, x1 + tw, y1 + th))

        return img, bubble_img

if __name__ == '__main__':
    import torch
    
    dataset_path = "/taiga/share/ABN_Fine-tuning/dataset/cub200"
    batch_size = 32

    # 画像とバブル画像に適用するTransform
    paired_transform = PairedTransforms([
        RandomHorizontalFlipPair(p=0.5),
        # 他のペアで適用するTransformを追加可能
    ])
    
    # 画像用のTransform (Normalizeは最後に個別で適用)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # バブル画像用のTransform (Normalizeなし)
    bubble_transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
    ])
    
    trainset = ImageBubbleFolder(
        image_root=os.path.join(dataset_path, 'train_bubbles'),
        bubble_root=os.path.join(dataset_path, 'bubbles_att'),
        transform=transform_train,
        bubble_transform=bubble_transform,
        paired_transform=paired_transform
    )
    
    print(f"dataset : {trainset}\nlen(trainset) : {len(trainset)}")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    img, label, bubble_img = trainset[0]
    