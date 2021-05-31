
import numpy as np
from scipy.stats import entropy
import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image

def meanRound(arr:list, digits:int):
    return round(np.mean(arr),digits)

class ImageDataset(Dataset):
    def __init__(self, img_folder, transform):
        super(ImageDataset, self).__init__()
        self.img_list = []
        self.transform = transform
        for elem in os.listdir(img_folder):
            self.img_list.append(os.path.join(img_folder, elem))
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        return self.transform(img)
    def __len__(self):
        return len(self.img_list)

#Calculate KL Divergence of Inception Model
def inceptionScore(img_folder, device:torch.device, batch_size, splits = 1):

    # get DataLoader
    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
    ])
    dataset = ImageDataset(img_folder, transform = transform)
    dataloader = DataLoader(dataset, batch_size = batch_size)

    preds = np.zeros((N, 1000))
    filled = 0
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = inception_model.to(device)
    inception_model.eval()
    for idx, imgs in enumerate(dataloader):
        imgs = imgs.to(device)
        length = imgs.shape[0]
        pred = inception_model(imgs)
        pred = nn.functional.softmax(pred).detach().cpu().numpy()
        preds[filled: filled+ length] = pred
        filled += length

    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)