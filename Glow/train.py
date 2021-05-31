import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.utils import save_image

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder

from model import Glow
import utils
from dataloader import getTrainLoader

from time import time
import numpy as np
from tqdm import tqdm

config = utils.Args({
    "n_level": 3,
    "n_flow": 32,
    "epoch": 1000,
    "batch_size": 64,
    "device": torch.device("cuda:1"),
    "lr": 1e-4,
    
    "data": "CIFAR10",#"CelebA",#"CIFAR10"
    "quantize": 5,
    "class_name": None,#"plane",
    "n_row": 10,
    "temp": .7,
})

train_loader  = getTrainLoader(config)

#pre-generated Gaussian distribution for image
z= utils.make_latent(n_level= config.n_level, image_shape= config.gen_img_shape, device = config.device, temp = config.temp)
z_cat = torch.cat([latent.reshape(config.n_row, -1) for latent in z], dim = 1)
#compare true_logp without temperature
true_logp = utils.calc_prior(torch.randn(1, config.n_pixel))[0].item()/config.n_pixel

model = Glow(n_channel= 3, n_flow = config.n_flow, n_level = config.n_level)
model.to(config.device)
optimizer = Adam(model.parameters(), lr = config.lr)

loss_arr = []
logp_arr = []
n_iter = 0
best_loss = float("inf")
for ep in range(config.epoch):
    ep_loss_arr = []
    ep_logp_arr = []
    model.train()
    start = time()
    for idx, (img, label) in tqdm(enumerate(train_loader), total = len(train_loader)):
        img = img.to(config.device)
        batch_size = img.shape[0]
        optimizer.zero_grad()
        output, log_det = model(img)
        output = [latent.reshape(batch_size, -1) for latent in output]
        output = torch.cat(output, dim = 1)
        logp = utils.calc_prior(output)
        loss = logp - log_det + config.n_pixel *np.log(2)* config.quantize
        loss = loss.mean()/config.n_pixel
        loss.backward()
        optimizer.step()
        n_iter += 1
        ep_loss_arr.append(loss.item())
        ep_logp_arr.append(logp.mean().item()/config.n_pixel)
    loss_arr += ep_loss_arr
    logp_arr += ep_logp_arr
    model.eval()
    with torch.no_grad():
        gen_image = model.reverse(z)
        save_image(gen_image, f"Figure/CIFAR10/{ep}.png", nrow = config.n_row)
    train_loss = np.mean(ep_loss_arr)
    print(f"[{ep},{n_iter}]: time: {utils.time_check(start)}, loss: {np.mean(ep_loss_arr):.2f}, logp: {np.mean(ep_logp_arr):.2f}, true_logp:{true_logp:.2f}")
    if train_loss < best_loss:
        torch.save(model.state_dict(),"best.pth")
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "n_iter": n_iter,
        "ep": ep,
        "loss_arr": loss_arr,
        "logp_arr": logp_arr
    },"recent.pth")
    