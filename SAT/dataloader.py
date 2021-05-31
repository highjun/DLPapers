# from torch.utils.data import random_split,Dataset, DataLoader
import os
from PIL import Image
from torchtext import data
# from torchtext import data
import torch
import torchvision.transforms as T

class PreprocImg(data.RawField):
    def __init__(self, transform, img_dir):
        super().__init__()
        self.transform = transform
        self.img_dir = img_dir
    def process(self, x, device):
        output =[]
        for item in x:
            item = Image.open(self.img_dir+ item)
            if self.transform is not None:
                item = self.transform(item)
            output.append(item)
        output = torch.stack(output, dim = 0)
        return output

class Flickr8k(data.TabularDataset):
    def __init__(self, root_dir, field, transform = None):
        super().__init__(root_dir + "/captions.txt","csv", 
        [("img", PreprocImg(img_dir= root_dir+"/Images/", transform = transform)), ("cap", field)])
def getDatas(data_name = "flickr8k", batch_size = 1):
    caption_field= data.Field(tokenize= "spacy", tokenizer_language="en", init_token="<sos>", 
    eos_token="<eos>", lower = True, batch_first= True)
    datas = Flickr8k(root_dir = "../data/Flickr8k", transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.225,0.225,0.225]),
    ]), field = caption_field)
    train_data, valid_data = datas.split(split_ratio= 0.9)
    caption_field.build_vocab(train_data)
    train_iter = data.BucketIterator(train_data, batch_size=batch_size)
    valid_iter = data.BucketIterator(valid_data, batch_size=batch_size)
    return train_iter, valid_iter, caption_field 
# class Flickr8k(Dataset):
#     def __init__(self, root_dir, transform = None):
#         super().__init__()
#         self.transform = transform
#         self.img_dir = root_dir + "/Images/"
#         with open(root_dir+"/captions.txt") as f:
#            self.img_label_list = f.readlines()[1:]
#     def __len__(self):
#         return len(self.img_label_list)
#     def __getitem__(self, idx):
#         img, label = self.img_label_list[idx].strip().split(",")
#         img = Image.open(self.img_dir+ img)
#         if self.transform is not None:
#             img = self.transform(img)
#         label = label.split()
#         label.insert(0,"<sos>")
#         label.append("<eos>")
#         return img, label
#     def getDataSrc(self):
#         src = []
#         for col in self.img_label_list:
#             data = col.strip().split(",")[1].split()
#             src.append(data) 
#         return src
# def getDatas(data_name = "flickr8k", batch_size = 1):
#     caption_field= Field(tokenize= "spacy", tokenizer_language="en", init_token="<sos>", eos_token="<eos>", lower = True, batch_first= True)
#     datas = Flickr8k(root_dir = "../data/Flickr8k", transform = T.Compose([
#         T.Resize(224),
#         T.ToTensor(),
#         T.Normalize([0.5,0.5,0.5],[0.225,0.225,0.225]),
#     ]))
#     n_train= round(len(datas)*0.9)
#     n_valid = len(datas)- n_train
#     caption_field.build_vocab(datas.getDataSrc())
#     train_data, valid_data = random_split(datas, [n_train,n_valid])
#     train_loader = DataLoader(train_data, batch_size= batch_size)
#     valid_loader = DataLoader(valid_data, batch_size= batch_size)
#     return train_loader, valid_loader, caption_field