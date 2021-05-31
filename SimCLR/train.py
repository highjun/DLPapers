import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

import utils
from model import pretrainedModel, downstreamModel
from dataloader import getDataLoader

from time import time
from tqdm import tqdm


config = utils.Args({
    #DataAugmentation
    "sigma": 2.0,
    "kernel_size": 9,
    "distortion_strength": 1.0,
    #training hyperparam
    "epoch": 50,
    "lr": 1e-4,#.3, #for BatchSize 256
    "warmup_step": 10,
    "batch_size": 256,
    "device": torch.device("cuda:1"),
    #pretrain hyper
    "temperature": 1,
    #train_type
    "train_type": "train",
    "experiment": "no_gaussian",
})
config.lr *= config.batch_size/256

model, criterion = None, None
train_loader, valid_loader = getDataLoader(config)
if config.train_type == "pretrain":
    model = pretrainedModel()
    criterion = utils.NTXentLoss(temperature= config.temperature)
if config.train_type == "train" or config.train_type == "train-aug":
    # model = downstreamModel(n_class = 10, use_fc = False)
    model = downstreamModel(pretrain = torch.load(f"best_pretrain_{config.experiment}.pth"), n_class = 10, freeze = False)
    criterion = nn.CrossEntropyLoss()

model = model.to(config.device)

optimizer = Adam(model.parameters(), lr = config.lr)



best_loss = float("inf")
train_loss_arr = []
valid_loss_arr = []
valid_acc_arr = []
n_iter = 0
print("start training")
for ep in range(config.epoch):
    ep_train_loss_arr = []
    ep_val_loss_arr =[]
    model.train()
    start = time()
    # for warming up
    if config.train_type == "pretrain":
        if ep <= config.warmup_step:
            for g in optimizer.param_groups:
                g['lr'] = config.lr* ep/config.warmup_step
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max = config.epoch-config.warmup_step)
    for idx, (img, label) in tqdm(enumerate(train_loader), total = len(train_loader)):
        optimizer.zero_grad()
        img, label = img.to(config.device), label.to(config.device)
        if config.train_type == "pretrain":
            paired_output = model(img)
            loss = criterion(paired_output)
            # for pretrained with small batch size
            # batch_size, calculated = 512, 0 
            # n_smallbatch = (paired_output.shape[0]+batch_size-1)//batch_size
            # loss = 0
            # while calculated < paired_output.shape[0]:
            #     loss += criterion(paired_output[calculated: np.min([calculated+batch_size, paired_output.shape[0]])])
            #     calculated += batch_size
        else:
            pred = model(img)
            loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        n_iter +=1
        ep_train_loss_arr.append(loss.item())
    if ep>10 and config.train_type=="pretrain":
        scheduler.step()
    train_loss_arr += ep_train_loss_arr
    train_loss = np.mean(ep_train_loss_arr)
    model.eval()
    n_data, n_correct, valid_acc = 0,0,0
    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(valid_loader), total = len(valid_loader)):
            img,label = img.to(config.device),label.to(config.device)
            if config.train_type =="pretrain":
                paired_output = model(img)
                loss = criterion(paired_output)
            else:
                pred = model(img)
                loss = criterion(pred, label)
                n_correct += torch.sum(pred.argmax(axis = 1) == label).item()
                n_data += img.shape[0]
            ep_val_loss_arr.append(loss.item())
        valid_loss = np.mean(ep_val_loss_arr)
        valid_loss_arr.append(valid_loss)
        if config.train_type!= "pretrain":
            valid_acc = n_correct/n_data*100
            valid_acc_arr.append(valid_acc)
    model = model.cpu()
    if best_loss > valid_loss:
        torch.save(model.state_dict(),f"best_{config.train_type}_{config.experiment}.pth")
    torch.save({
        "config": list(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "n_iter": n_iter,
        "ep": ep,
        "train_loss_arr": train_loss_arr,
        "valid_loss_arr": valid_loss_arr,
        "valid_acc_arr": valid_acc_arr,
    },f"recent_{config.train_type}_{config.experiment}.pth")
    model = model.to(config.device)
    print(f"[{ep},{n_iter}]: {utils.time_check(start)} train_loss: {train_loss:.4f}, valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}")
print(f"experiment: {config}, maximum acc: {np.max(valid_acc_arr)}")