import numpy as np
import torch
import torchvision.transforms as T 
from torchvision.datasets import STL10
from torch.utils.data import DataLoader

# random crop with flip and resize
random_crop_and_flip = T.Compose([
    T.RandomResizedCrop((96,96)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),])
# color distortion
def getColorDistortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([
    rnd_color_jitter,
    rnd_gray])
    return color_distort
class getTwoAug():
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, sample):
        aug1 = self.transform(sample)
        aug2 = self.transform(sample)
        return aug1, aug2
def collate_fn(batch):
    aug_ , label = list(zip(*batch))
    aug1, aug2 = list(zip(*aug_))
    aug1, aug2 = torch.stack(aug1), torch.stack(aug2)
    label = torch.tensor(label)
    aug = torch.stack([aug1,aug2])
    aug = aug.transpose(0,1).reshape(-1, aug.shape[-3], aug.shape[-2], aug.shape[-1]) #2xbxcxhxw
    return aug, label

def getDataLoader(config):
    common_transform = T.Compose([
        random_crop_and_flip,
        getColorDistortion(config.distortion_strength),
        # T.RandomApply([T.GaussianBlur(kernel_size = config.kernel_size, sigma=(0.1, config.sigma))], p = 0.5),
        T.ToTensor()
    ])
    simCLR_transform = getTwoAug(common_transform)
    train_loader, valid_loader = None, None
    if config.train_type == "pretrain":
        train = STL10(root = "../data/STL10", download= True, split = "train+unlabeled", transform = simCLR_transform)
        val = STL10(root = "../data/STL10", download= True ,split = "test", transform = simCLR_transform)
        train_loader = DataLoader(train, batch_size = config.batch_size, shuffle = True, collate_fn = collate_fn, num_workers = 4)
        valid_loader = DataLoader(val, batch_size = config.batch_size, shuffle = False, collate_fn = collate_fn, num_workers = 4)
    if config.train_type == "train-aug":
        train = STL10(root = "../data/STL10", download= True, split = "train", transform = common_transform)
        val = STL10(root = "../data/STL10", download= True ,split = "test", transform = T.ToTensor())
        train_loader = DataLoader(train, batch_size = config.batch_size, shuffle = True, num_workers = 4 )
        valid_loader = DataLoader(val, batch_size = config.batch_size, shuffle = False, num_workers = 4)
    if config.train_type == "train":
        train = STL10(root = "../data/STL10", download= True, split = "train", transform = T.ToTensor())
        val = STL10(root = "../data/STL10", download= True ,split = "test", transform = T.ToTensor())
        train_loader = DataLoader(train, batch_size = config.batch_size, shuffle = True, num_workers = 4)
        valid_loader = DataLoader(val, batch_size = config.batch_size, shuffle = False, num_workers = 4)
    return train_loader, valid_loader