import numpy as np
import torch
import torch.nn as nn 

def top_k_acc(k: int,pred:np.ndarray, label:np.ndarray):
    sorted_ = pred.argsort(axis = -1)
    top_k = sorted_[:, -k:]
    acc = 0
    for idx in range(top_k.shape[0]):
        if label[idx] in top_k[idx]:
            acc += 1
    return acc / top_k.shape[0]

class NCELoss(nn.Module):
    def __init__(self, T= 1) -> None:
        super().__init__()
        self.T = T
    def forward(self, output1, output2):
        '''
        output1: BxN
        output2: BxN
        '''
        output1 = output1 / torch.sum(output1*output1, dim = 1, keepdim= True)
        output2 = output2 / torch.sum(output2*output2, dim = 1, keepdim= True)
        sim = torch.mm(output1, output2.T)
        sim = torch.exp(sim/self.T)
        l_matrix = -torch.log(sim / torch.sum(sim- torch.diag(torch.diag(sim)), dim = 1, keepdim= True))
        loss = torch.sum(torch.diag(l_matrix ,1)) + torch.sum(torch.diag(l_matrix ,-1))
        return loss/2/output1.size(0)