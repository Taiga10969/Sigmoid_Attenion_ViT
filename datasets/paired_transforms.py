import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import torch.nn.functional as F

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