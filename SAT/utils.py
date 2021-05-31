import torch
from time import time
import numpy as np


def check_model_size(model):
    n_params = np.sum([p.numel() for p in model.parameters()])
    print(f"{n_params* 4 /10**6:2f}MB")

def timeCheck(start):
    total_time = round(time() - start)
    min_, sec_ = divmod(total_time, 60)
    return "{:02}:{:02}".format(int(min_),int(sec_))

def save(model, path):
    device = next(model.parameters()).device
    model = model.cpu()
    torch.save(model.state_dict(),path)
    model = model.to(device)

def freeze(model):
    for param in model.parameters(): 
        param.require_grad = False