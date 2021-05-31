import torch
from torch.optim import Adam
from torchvision.utils import make_grid, save_image

import numpy as np
import time
import logging
import os
from tqdm import tqdm
from itertools import chain

from dataloader import getDataLoader
from cyclegan import Image2Image, ImageDiscriminator

from torchvision import transforms
from util import meanRound

import torch.nn as nn

batch_size = 2
num_workers = 4
epoch = 200
device = torch.device("cuda:1")
lr = 2e-4
exp = 0

os.makedirs(f"Training/exp{exp}/progress",exist_ok= True)
logging.basicConfig(filename=f"Training/exp{exp}/training.log",filemode = "w", level= logging.INFO)
logging.info(f'''batch_size={batch_size}, num_workers= {num_workers}, epoch = {epoch}, device = 'RTX2070s x1', lr ={lr}''')

G_A2B = Image2Image()
G_A2B = G_A2B.to(device)
G_B2A = Image2Image()
G_B2A = G_B2A.to(device)
D_A = ImageDiscriminator()
D_B = ImageDiscriminator()
D_A = D_A.to(device)
D_B = D_B.to(device)

train_loader, valid_loader = getDataLoader(batch_size= batch_size, num_workers = num_workers)

optimizerG = Adam(chain(G_A2B.parameters(), G_B2A.parameters()), lr=lr)
optimizerD_A = Adam(D_A.parameters(),lr =lr)
optimizerD_B = Adam(D_B.parameters(),lr = lr)

bce_loss =nn.BCELoss()
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

valid_iter= iter(valid_loader)
sample_img_A,sample_img_B =  next(valid_iter)
save_image(sample_img_A, f"Training/exp{exp}/sampleA.jpg",nrow=2, normalize= True)
save_image(sample_img_B, f"Training/exp{exp}/sampleB.jpg",nrow=2, normalize= True)
sample_img_A, sample_img_B = sample_img_A.to(device), sample_img_B.to(device)

n_iter = 0
best_val_loss = np.inf
for ep in range(epoch):
    ep_start= time.time()
    train_D_A_loss = []
    train_D_B_loss = []
    train_G_loss = []
    for iter, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        img_A, img_B = data
        img_A, img_B = img_A.to(device), img_B.to(device)

        ones = torch.ones((batch_size,1,30,30), device= device)
        zeros = torch.zeros((batch_size,1,30,30),device = device)
        
        optimizerG.zero_grad()
        fake_A2B = G_A2B(img_A)
        fake_B2A = G_B2A(img_B)

        lsgan_loss = mse_loss(D_B(fake_A2B), ones)+ mse_loss(D_A(fake_B2A), ones)
        cycle_loss = l1_loss(G_A2B(fake_B2A), img_A)+ l1_loss(G_B2A(fake_A2B), img_B)
        cycle_loss *= 10
        identity_loss = l1_loss(G_A2B(img_B), img_B)+ l1_loss(G_B2A(img_A), img_A)
        identity_loss *= 5

        loss_G = lsgan_loss+ cycle_loss + identity_loss
        loss_G.backward()
        optimizerG.step()

        optimizerD_A.zero_grad()
        fake_A2B = G_A2B(img_A).detach()
        fake_B2A = G_B2A(img_B).detach()
        loss_D_A = mse_loss(D_A(fake_B2A),zeros) + mse_loss(D_A(img_A), ones)
        loss_D_A /= 2
        loss_D_A.backward()
        optimizerD_A.step()
        
        optimizerD_B.zero_grad()
        loss_D_B = mse_loss(D_B(fake_A2B),zeros) + mse_loss(D_B(img_B), ones)
        loss_D_B /= 2
        loss_D_B.backward()
        optimizerD_B.step()

        train_D_A_loss.append(loss_D_A.item())
        train_D_B_loss.append(loss_D_B.item())
        train_G_loss.append(loss_G.item())

        if n_iter%100 == 0:
            k = n_iter//100
            with torch.no_grad():
                save_image(G_A2B(sample_img_A).detach(), nrow = 2,normalize = True, fp = f"Training/exp{exp}/progress/A2B-{k}.jpg")
                save_image(G_B2A(sample_img_B).detach(), nrow = 2,normalize = True, fp = f"Training/exp{exp}/progress/B2A-{k}.jpg")
        n_iter +=1
    ep_end = time.time()
    if (ep+1)% 10 == 0:
        torch.save(G_A2B.state_dict(), f"Training/exp{exp}/G_A2B-{ep+1}.pth")
        torch.save(G_B2A.state_dict(), f"Training/exp{exp}/G_B2A-{ep+1}.pth")
        torch.save(D_A.state_dict(), f"Training/exp{exp}/D_A-{ep+1}.pth")
        torch.save(D_B.state_dict(), f"Training/exp{exp}/D_B-{ep+1}.pth")
    torch.save(G_A2B.state_dict(), f"Training/exp{exp}/G_A2Brecent.pth")
    torch.save(G_B2A.state_dict(), f"Training/exp{exp}/G_B2Arecent.pth")
    torch.save(D_A.state_dict(), f"Training/exp{exp}/D_Arecent.pth")
    torch.save(D_B.state_dict(), f"Training/exp{exp}/D_Brecent.pth")
    print_string = f'''[{ep+1}]/[{epoch}], {round(ep_end- ep_start,1)}s, D_A:{meanRound(train_D_A_loss,4)}, D_B={meanRound(train_D_B_loss,4)}, G:{meanRound(train_G_loss,4)}'''
    print(print_string)
    logging.info(print_string)