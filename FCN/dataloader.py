from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

import torch
import numpy as np

from PIL import Image
import pickle

img_transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
class PILToLong:
    def __call__(self, x):
        copy = np.array(x)
        # copy[copy == 255] = -1
        return torch.tensor(copy, dtype = torch.long, requires_grad = False)
    
label_transform = T.Compose([
    T.Resize((256,256)),
    PILToLong()
])
def integrated_transform(x,y):
    return img_transform(x), label_transform(y)

class SBDVOCDataset(Dataset):
    def __init__(self, root, split="train", transforms = None):
        self.dir = root
        self.file_list = []
        self.transforms = transforms
        with open(self.dir+"/"+split+".txt", "rb") as txt:
            for line in txt.readlines():
                self.file_list.append(line.strip().decode('utf-8'))
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        img = Image.open(self.dir+"/img/"+self.file_list[idx]+".jpg")
        label = Image.open(self.dir+"/label/"+self.file_list[idx]+".png")
        return self.transforms(img, label)
def getDataLoader(batch_size = 1, num_workers =4):
    train_data = SBDVOCDataset(root= "Datas/SBD_VOC",split= "train", transforms = integrated_transform)
    val_data = SBDVOCDataset(root= "Datas/SBD_VOC",split= "val", transforms = integrated_transform)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle= True, num_workers= num_workers)
    val_loader = DataLoader(val_data, batch_size= batch_size, shuffle = False, num_workers= num_workers)

    return train_loader, val_loader

def getTestData():
    with open("sample_image_name.p","rb") as f:
        image_list = pickle.load(f)
    image_to_batch = []
    label_to_batch = []
    for name in image_list:
        img = Image.open(f"Datas/SBD_VOC/img/{name}.jpg")
        label = Image.open(f"Datas/SBD_VOC/label/{name}.png")
        image_to_batch.append(img_transform(img).unsqueeze(0))
        label_to_batch.append(label_transform(label).unsqueeze(0))
    img_batch = torch.cat(image_to_batch, dim = 0)
    label_batch =torch.cat(label_to_batch, dim = 0)
    return img_batch, label_batch
        

