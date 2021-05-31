from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

import os 
import random

transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

class PairedImageDataset(Dataset):
    def __init__(self,transform,split = "train"):
        super(PairedImageDataset, self).__init__()
        A_img_list = os.listdir(f"Datas/{split}A")
        B_img_list = os.listdir(f"Datas/{split}B")
        self.A_img_list = [os.path.join(f"Datas/{split}A/",f) for f in A_img_list]
        self.B_img_list = [os.path.join(f"Datas/{split}B/",f) for f in B_img_list]
        self.transform = transform
    def __getitem__(self, idx):
        A_img = Image.open(self.A_img_list[random.randint(0, len(self.A_img_list)-1)])
        B_img = Image.open(self.B_img_list[idx])
        return self.transform(A_img), self.transform(B_img)
    def __len__(self):
        return len(self.B_img_list)
def getDataLoader(batch_size, num_workers):
    train_loader = DataLoader(PairedImageDataset(split="train", transform= transform), batch_size= batch_size, num_workers= num_workers)
    valid_loader = DataLoader(PairedImageDataset(split="test", transform= transform), batch_size= batch_size, num_workers= num_workers)
    
    return train_loader, valid_loader