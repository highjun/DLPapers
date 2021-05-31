
import argparse

from torchvision.transforms.transforms import RandomCrop

import utils
import torch
import numpy as np
from torch.optim import Adam, SGD
import torch.nn as nn
from tqdm import tqdm

from torchvision.models import resnet50, resnet18
from torchvision.datasets import CIFAR100
import torchvision.transforms as T
from model import Teacher, Std

from torch.utils.data import DataLoader

def train(args):
    device = torch.device(f"cuda:{args.device_id}")
    # model = resnet50(num_classes = 100)
    # model = resnet18(num_classes = 100)
    model = Std()
    teacher = Teacher()
    teacher.to(device)
    criterion = nn.CrossEntropyLoss()
    distill = utils.NCELoss(T= args.T)

    model.to(device)
    optimizer = SGD(model.parameters(), lr = args.lr, weight_decay= args.weight_decay, momentum= args.momentum)
    train_data = CIFAR100(root = "./Data", download= True,train= True, transform= T.Compose([
        T.RandomCrop(32, padding = 4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]))
    valid_data = CIFAR100(root = "./Data", download= True, train= False, transform= T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]))
    train_loader = DataLoader(train_data, batch_size= args.batch_size, shuffle= True, num_workers= args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size= args.batch_size, shuffle= False, num_workers= args.num_workers)

    train_loss_arr = []
    valid_loss_arr = []
    valid_acc_arr = []
    valid_top5_arr = []
    n_iter = 0
    best_loss = float('inf')
    best_top1_acc = 0
    best_top5_acc = 0
    for ep in range(args.epoch):
        if ep in [150, 180, 210]:
            for g in optimizer.param_groups:
                g['lr'] /= 10
        model.train()
        for _, (img, label) in tqdm(enumerate(train_loader), total = len(train_loader)):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            # pred = model(img)
            # loss = criterion(pred, label)
            x1, x2 = model(img)
            with torch.no_grad():
                x3 = teacher(img)
            loss = criterion(x2, label)
            loss += distill(x1, x3)
            loss.backward()
            optimizer.step()
            train_loss_arr.append(loss.item())
            n_iter += 1
        model.eval()
        ep_valid_loss_arr = []
        ep_acc_arr = []
        ep_top5_arr =[]
        with torch.no_grad():
            for _, (img, label) in tqdm(enumerate(valid_loader),total= len(valid_loader)):
                img, label = img.to(device), label.to(device)
                # pred = model(img)
                # loss = criterion(pred, label)
                # top1 = torch.topk(pred, k = 1)
                # top5 = torch.topk(pred, k = 5)
                # acc = utils.top_k_acc(k = 1, pred =pred.detach().cpu().numpy(),label =  label.detach().cpu().numpy())
                # acc5 =  utils.top_k_acc(k = 5, pred = pred.detach().cpu().numpy(), label = label.detach().cpu().numpy())
                x1, x2 = model(img)
                x3 = teacher(img)
                loss = criterion(x2, label)
                loss += distill(x1, x3)
                acc = utils.top_k_acc(k = 1, pred =x2.detach().cpu().numpy(),label =  label.detach().cpu().numpy())
                acc5 =  utils.top_k_acc(k = 5, pred = x2.detach().cpu().numpy(), label = label.detach().cpu().numpy())
                ep_acc_arr.append(acc)
                ep_top5_arr.append(acc5)
                ep_valid_loss_arr.append(loss.item())
        valid_loss = np.mean(ep_valid_loss_arr)
        valid_acc = np.mean(ep_acc_arr)
        valid_top5 = np.mean(ep_top5_arr)
        train_loss = np.mean(train_loss_arr[-len(train_loader):])
        valid_loss_arr.append(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_top1_acc = valid_acc
            best_top5_acc = valid_top5
            model.cpu()
            torch.save(model.state_dict(), f"{args.desc}_best.pth")
            model.to(device)
        if (ep+1)%10 == 0:
            model.cpu()
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss_arr,
                "valid_loss": valid_loss_arr,
                "valid_acc": valid_acc_arr,
                "valid_top5": valid_top5_arr,
                "best_loss": best_loss,
                "best_top1_acc": best_top1_acc,
                "best_top5_acc": best_top5_acc,
                "ep" : ep,
                "n_iter": n_iter,
            }, f"{args.desc}_ckp.pth")
            model.to(device)
        print(f"[{ep}, {n_iter}] train: {train_loss:.4f}, valid: {valid_loss:.4f}, acc: {valid_acc:.4f}, top5: {valid_top5:.4f}")
    with open("result.txt", "a+") as f:
        f.write(f"{args.desc}: top1: {best_top1_acc:.3f}, top5: {best_top5_acc:.3f}\n")
if __name__ == "__main__":
   parser = argparse.ArgumentParser()

   parser.add_argument('--device_id', type = int, default = 0)
   parser.add_argument('--num_workers', type = int, default = 8)
   parser.add_argument('--batch_size', type = int, default = 256)

   parser.add_argument('--lr', type =float, default = .05)
   parser.add_argument('--epoch', type = int, default = 240)
   parser.add_argument('--weight_decay', type = float, default = 5e-4)
   parser.add_argument('--momentum', type = float, default = .9)
   parser.add_argument('--T', type = float, default = 1)


   
   parser.add_argument('--desc', type = str, default = "std_crd_256")

   args = parser.parse_args()
   train(args)