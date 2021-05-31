import torch
from time import time
import numpy as np
import torch.nn as nn

class Args(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def check_model_size(model):
    n_params = np.sum([p.numel() for p in model.parameters()])
    print(f"{n_params* 4 /10**6:2f}MB")

def time_check(start):
    total_time = round(time() - start)
    min_, sec_ = divmod(total_time, 60)
    return "{:02}:{:02}".format(int(min_),int(sec_))

class NTXentLoss(nn.Module):
    def __init__(self, temperature = 1):
        super().__init__()
        self.temperature = temperature
    def forward(self, paired_output):
        #paired_output:  2bxdim_
        b, dim_ = paired_output.shape
        assert b%2 ==0
        norm = torch.sum(paired_output ** 2, dim = -1)**0.5#2b
        norm = torch.matmul(norm.view(-1,1), norm.view(1, -1)) #2bx2b
        inner_prod = torch.matmul(paired_output, paired_output.transpose(0, 1))
        cos_matrix = (inner_prod/(norm* self.temperature)).exp()
        loss_matrix = -torch.log(cos_matrix / (cos_matrix.sum(dim = -1) - cos_matrix.diagonal()).view(-1, 1))
        loss = (loss_matrix.diagonal(offset = 1)[::2].sum()* 2)/b
        return loss