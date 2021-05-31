import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader
import numpy as np 
import torch

class MakeContinuous():
    def __init__(self, quantize_level = 5):
        super().__init__()
        self.quantize_level = quantize_level
    def __call__(self, sample):
        #sample CxHxW tensor
        sample *= (2**8-1) #0~255 사이
        # print(sample)
        assert (sample <= 255).all().item() and (sample >= 0).all().item()
        sample //= 2**(8 - self.quantize_level) #0 ~ 2^5-1
        sample /= 2**self.quantize_level
        sample += torch.rand_like(sample)/2**self.quantize_level
        return sample
def getTrainLoader(config):
    if config.data == "CIFAR10":
        transform = T.Compose([T.ToTensor(), MakeContinuous(config.quantize)])
        train_data = CIFAR10("../data/CIFAR10", train = True, download= True, transform = transform)
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        if config.class_name is not None:
            label_idx = classes.index(config.class_name)
            one_class = np.array(train_data.targets) == label_idx
            train_data.targets = np.array(train_data.targets)[one_class]
            train_data.data = train_data.data[one_class]
        train_loader = DataLoader(train_data, config.batch_size)
        config.gen_img_shape = (config.n_row, 3, 32, 32)
        config.n_pixel = 3*32*32
        return train_loader
    if config.data =="CelebA":# CelebA-HQ는 너무 큰 관계로, 64x64로 진행
        train_data = ImageFolder(root = "../data/CelebA-HQ", transform=T.Compose([
            T.Resize((64,64)),
            T.ToTensor(),
            MakeContinuous(quantize_level= config.quantize)
        ]))
        train_loader = DataLoader(train_data, config.batch_size)
        config.gen_img_shape = (config.n_row, 3, 64, 64)
        config.quantize = 2**config.quantize
        config.n_pixel = 3*64*64
        return train_loader

    