#!/usr/bin/env python3
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_root = os.path.join(os.path.dirname(__file__), f'datasets/{dataset_name}')
        self.image_root_path = os.path.join(self.dataset_root, 'image')
        self.pose_root_path = os.path.join(self.dataset_root, 'pose')
        self.image_paths = sorted(os.listdir(self.image_root_path))
        self.pose_paths = sorted(os.listdir(self.pose_root_path))

        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([127.5], [127.5]),
        ])

    def __len__(self):
        return len(self.image_root_path)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_root_path, self.image_paths[idx]))
        image = self.image_transforms(image)
        with open(os.path.join(self.pose_root_path, self.pose_paths[idx]), 'r') as f:
            pose = json.load(f)
        norm = torch.Tensor([(pose['x'] ** 2 + pose['y'] ** 2) ** 0.5]) / 24  # max norm
        theta = torch.atan2(
            torch.Tensor([pose['y']]),
            torch.Tensor([pose['x']]),
        ) / np.pi
        alpha = torch.Tensor([pose['yaw']]) / np.pi

        # print(image.shape, norm.shape, theta.shape, alpha.shape)

        return image, (norm, theta, alpha)