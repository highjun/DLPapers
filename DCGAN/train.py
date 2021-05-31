import torch
from torch.optim import Adam
from torchvision.utils import make_grid, save_image

import numpy as np
import time
import logging
import os
from tqdm import tqdm

from dataloader import getDataLoader
from dcgan import Generator, Discriminator

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from util import meanRound

import torch.nn as nn

batch_size = 128
num_workers = 4
epoch = 30
device = torch.device("cuda:0")
lr = 2e-4
exp = 3
loss_type = "lsgan"
beta1 = 0.5

os.makedirs(f"Training/exp{exp}/progress",exist_ok= True)
logging.basicConfig(filename=f"Training/exp{exp}/training.log",filemode = "w", level= logging.INFO)
logging.info(f'''batch_size={batch_size}, num_workers= {num_workers}, epoch = {epoch}, device = 'RTX2070s x1', lr ={lr},loss_type = {loss_type}, 
beta1={beta1}''')

D = Discriminator(loss_type= loss_type,n_channel=128)
# D.load_state_dict(torch.load(f"Training/exp{1}/recentD.pth"))
D = D.to(device)
G = Generator(n_channel=128)
# G.load_state_dict(torch.load(f"Training/exp{1}/recentG.pth"))
G = G.to(device)

loader = getDataLoader(batch_size= batch_size, num_workers = num_workers)

optimizerD = Adam(D.parameters(), lr = lr, betas= (beta1, 0.999))
optimizerG = Adam(G.parameters(), lr = lr, betas = (beta1, 0.999))

# criterion =nn.BCELoss()
criterion = nn.MSELoss()

n_iter = 0
fixed_noise= torch.randn((10,100, 1, 1), device = device)
best_val_loss = np.inf
for ep in range(epoch):
    ep_start= time.time()
    train_D_loss = []
    train_G_loss = []
    real_img_D = []
    fake_img_D_1 = [] #before train D
    fake_img_D_2 = [] # after train D
    G.train()
    D.train()
    for iter, real_img in tqdm(enumerate(loader), total = len(loader)):
        real_img = real_img.to(device)
        batch_size = real_img.shape[0]

        optimizerD.zero_grad()
        pred_real = D(real_img).squeeze()        
        loss_real = criterion(pred_real, torch.ones(batch_size,dtype = torch.float, device = device))
        # loss_real = torch.mean((pred_real - torch.ones(batch_size,dtype= torch.float, device= device))**2)
        
        fake_img = G(torch.randn((batch_size,100,1,1),device = device))
        fake_img = fake_img.detach()
        pred_fake = D(fake_img).squeeze()
        loss_fake = criterion(pred_fake, torch.zeros(batch_size,dtype = torch.float, device = device))
        # loss_fake = torch.mean((pred_fake - torch.zeros(batch_size,dtype= torch.float, device= device))**2)
        

        loss_D = loss_real + loss_fake
        loss_D /= 2
        loss_D.backward()
        optimizerD.step()

        real_img_D.append(pred_real.mean().item())
        fake_img_D_1.append(pred_fake.mean().item())
        train_D_loss.append(loss_D.item())

        optimizerG.zero_grad()
        fake_img = G(torch.randn((batch_size,100,1,1),device = device))
        pred_fake = D(fake_img).squeeze()
        loss_G = criterion(pred_fake, torch.ones(batch_size, dtype=torch.float,device = device))
        loss_G/= 2
        # loss_G = torch.mean((pred_fake - torch.ones(batch_size,dtype= torch.float, device= device))**2)

        loss_G.backward()
        optimizerG.step()

        train_G_loss.append(loss_G.item())
        fake_img_D_2.append(pred_fake.mean().item())
        
        if n_iter%100 == 0:
            k = n_iter//100
            fake_img= G(fixed_noise)
            save_image(fake_img, nrow = 5,normalize = True, fp = f"Training/exp{exp}/progress/iter-{k}.jpg")
        n_iter +=1
    ep_end = time.time()
    if (ep+1)% 10 == 0:
        torch.save(G.state_dict(), f"Training/exp{exp}/G-{ep+1}.pth")
        torch.save(D.state_dict(), f"Training/exp{exp}/D-{ep+1}.pth")
    torch.save(G.state_dict(), f"Training/exp{exp}/recentG.pth")
    torch.save(D.state_dict(), f"Training/exp{exp}/recentD.pth")
    print_string = f'''[{ep+1}]/[{epoch}], {round(ep_end- ep_start,1)}s, D_loss:{meanRound(train_D_loss,4)}, G_loss:{meanRound(train_G_loss,4)}, D(x)={meanRound(real_img_D,4)}, D(G(z))={meanRound(fake_img_D_1,4)}/{meanRound(fake_img_D_2,4)}'''
    print(print_string)
    logging.info(print_string)