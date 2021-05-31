import torch
from time import time
import numpy as np

class Args(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def make_latent(n_level = 3, image_shape= (10, 3, 32, 32), device = torch.device("cpu"), temp = .7):
    b, c, h, w = image_shape
    z_arr =[]
    for idx in range(n_level):
        multiple = 2**(idx+1)
        channel = c*multiple
        if n_level -1 == idx:
            channel *= 2
        z_arr.append(torch.randn(b, channel, h//multiple, w//multiple, device = device)*temp)
    return z_arr

def check_model_size(model):
    n_params = np.sum([p.numel() for p in model.parameters()])
    print(f"{n_params* 4 /10**6:2f}MB")

def time_check(start):
    total_time = round(time() - start)
    min_, sec_ = divmod(total_time, 60)
    return "{:02}:{:02}".format(int(min_),int(sec_))

def calc_prior(z):
    #Gaussian distribution logp 계산
    assert len(z.shape) == 2
    return ((z**2+ np.log(2*np.pi))/2).sum(dim = 1)