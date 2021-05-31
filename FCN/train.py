import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler

import numpy as np
import time
import logging
from tqdm import tqdm
import os
import shutil

from dataloader import getDataLoader
from fcn import FCN
from util import calConfusionMatrix, calMetric

batch_size = 4
num_workers = 4
device = torch.device("cuda:0")
lr = 2e-6
upsample = "32"
weight_decay = 5 ** (-4)
exp = 0
description = """FCN-32s"""
class_num = 21
early_stop = 10

net = FCN(upsample=upsample, class_num=class_num)


def criterion(pred, label):
    label_np = np.array(label.detach().cpu().view(-1))
    mask = np.all([label_np < class_num, label_np >= 0], axis=0)
    pred = pred.permute((0, 2, 3, 1)).contiguous()
    pred = pred.view(-1, class_num)
    loss = nn.functional.cross_entropy(pred[mask], label.view(-1)[mask])
    return loss


# pretrained = FCN(upsample="32", class_num=21)
# pretrained.load_state_dict(torch.load("Training/3/best-FCN.pth"))
# module_list = ['conv1','conv2','conv3','conv4','conv5','fcn']
# for module in module_list:
#     net.__getattr__(module).load_state_dict(pretrained.__getattr__(module).state_dict())
net = net.to(device)

train_loader, val_loader = getDataLoader(batch_size=batch_size, num_workers=num_workers)

optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
scheduler = lr_scheduler.StepLR(optimizer, 10)

if not os.path.exists("Training"):
    os.makedirs("Training")
if os.path.exists(f"Training/{exp}"):
    raise Exception(
        f"{exp} folder exist, please try after remove folder or change experiment number"
    )
os.makedirs(f"Training/{exp}")
logging.basicConfig(
    filename=f"Training/{exp}/training.log", filemode="w", level=logging.INFO
)
logging.info(
    f"""batch_size= {batch_size},num_workers={num_workers},GPU: RTX2070s x1, {device}, dataset: {"SBD_VOC"} \
optimizer:{optimizer} \
scheduler = {scheduler.__class__.__name__},{scheduler.state_dict()} \
description={description}"""
)

best_mean_iu = 0
best_epoch = -1
iters = 0
ep = 0
while True:
    ep += 1
    ep_starting_time = time.time()
    train_loss = []
    val_loss = []
    net.train()
    for _, (img, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
        iters += 1
        optimizer.zero_grad()
        img, label = img.to(device), label.squeeze(1).to(device)
        pred = net(img)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    with torch.no_grad():
        net.eval()
        mat = np.zeros((class_num, class_num))
        for _, (img, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
            img = img.to(device)
            pred = net(img).detach().cpu()
            pred_label = torch.argmax(pred, dim=1)
            mat = calConfusionMatrix(pred_label, label, class_num, mat)
            loss = criterion(pred, label)
            val_loss.append(loss.item())
        pixel_acc, mean_acc, mean_iu, freq_weighted_iu = calMetric(mat)
    ep_end_time = time.time()
    scheduler.step()
    net.cpu(),
    torch.save(
        {
            "epoch": ep,
            "iter": iters,
            "scheduler": scheduler.state_dict(),
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        f"Training/{exp}/recent-FCN.pth",
    )
    net.to(device)
    if best_mean_iu < mean_iu:
        best_mean_iu = mean_iu
        best_epoch = ep
        logging.info(f"save {ep} as best")
        shutil.copy(f"Training/{exp}/recent-FCN.pth", f"Training/{exp}/best-FCN.pth")
    print_string = f"[{ep}: {iters}], {round(ep_end_time- ep_starting_time,2)}s, train_loss: {round(np.mean(train_loss),3)}, valid_loss: {round(np.mean(val_loss),3)}, mean_iu: {round(mean_iu*100,1)}"
    print(print_string)
    logging.info(print_string)
    if best_epoch < ep - early_stop:
        logging.info("Stopped by Early Stopping")
        break
