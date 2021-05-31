from torchvision import datasets
from torchvision import transforms as T

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from PIL import Image

import os

transformation = T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

class CelebADataset(Dataset):
    def __init__(self, transformation):
        super(CelebADataset, self).__init__()
        self.root = "Datas/img_align_celeba/"
        self.transformation = transformation
    def __getitem__(self, idx):
        location = f"{self.root}{str(idx+1).zfill(6)}.jpg"
        img = Image.open(location)
        img = self.transformation(img)
        return img
    def __len__(self):
        return len(os.listdir(self.root))

def getDataLoader(batch_size = 1, num_workers =4):
    data = CelebADataset(transformation = transformation)
    loader = DataLoader(data, batch_size = batch_size, shuffle= True, num_workers= num_workers)

    return loader


