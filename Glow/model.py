import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import numpy as np 

class ActNorm(nn.Module):
    def __init__(self, n_channel):
        super().__init__()
        self.n_channel = n_channel

        self.scale = Parameter(torch.Tensor(1,n_channel,1,1))
        self.bias = Parameter(torch.Tensor(1,n_channel,1,1))
        self.initialize = False
    def forward(self, x):
        #x: bxcxhxw Tensor
        #return: output(bxcxhxw Tensor), log_det(b Tensor)
        b, c, h, w = x.shape
        assert x.shape[1] == self.n_channel
        output = torch.clone(x)
        # Initialize into zero-mean, unit variance of mini-batch
        if not self.initialize: 
            data = output.transpose(0,1).reshape(self.n_channel,-1)
            std, mean = torch.std_mean(data, dim = -1)
            std, mean = std.view(1, self.n_channel, 1, 1), mean.view(1, self.n_channel, 1, 1)
            self.scale.data.copy_(1/(std+1e-9))
            self.bias.data.copy_(-mean)
            self.initialize = True
        output += self.bias
        output *= self.scale

        log_det = h * w * self.scale.abs().log().sum()
        log_det = log_det.repeat(b)
        return output, log_det
    def reverse(self, z): 
        output = torch.clone(z)
        output /= self.scale
        output -= self.bias
        return output

class ImageAffineCouplingLayer(nn.Module):
    def __init__(self, n_channel):
        super().__init__()
        self.n_channel = n_channel
        #split along channel
        self.n_split = n_channel // 2

        self.nn = nn.Sequential(
            nn.Conv2d(self.n_split, 512, 3, padding =1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 2*(self.n_channel - self.n_split), 3, padding = 1)
        )
        #init last weight into zero
        nn.init.constant_(self.nn[-1].weight, 0)
        nn.init.constant_(self.nn[-1].bias, 0)
    def forward(self, x):
        # x: bxcxhxw
        b,c,h,w = x.shape
        assert self.n_channel == c
        x_a, x_b = x[:,:self.n_split], x[:,self.n_split:]
        nn_result = self.nn(x_a) #bx2(D-d) x h x w
        #log_s, t: bx(D-d)xhxw
        log_s, t = nn_result[:,0::2,:,:], nn_result[:,1::2,:,:]
        # s = torch.exp(log_s) #log_s는 initially 0
        s = torch.sigmoid(log_s + 2) #torch.exp대문에 잘 안되는 듯??
        y_a, y_b = x_a, s*x_b + t
        y = torch.cat((y_a, y_b), dim = 1)
        log_det = s.view(b, -1).abs().log().sum(dim = 1)

        return y, log_det
    def reverse(self, z):
        # z: bxcxhxw
        z_a, z_b = z[:,:self.n_split], z[:,self.n_split:]
        nn_result = self.nn(z_a) #bx2(D-d) x h x w
        log_s, t = nn_result[:,0::2,:,:], nn_result[:,1::2,:,:]
        # s = torch.exp(log_s)
        s = torch.sigmoid(log_s + 2)
        x_a, x_b = z_a, (z_b-t)/s
        x = torch.cat((x_a, x_b), dim = 1)
        return x

class Invertible1to1Conv(nn.Module):
    def __init__(self, n_channel):
        super().__init__()
        self.n_channel = n_channel
        #LDU decomposition안해도 괜찮은 성능인 듯 하여 안씀
        self.matrix = Parameter(torch.Tensor(self.n_channel, self.n_channel))
        
        #initialize with random permutation matrix
        init_matrix = torch.eye(self.n_channel)
        randperm = torch.randperm(self.n_channel)
        init_matrix = init_matrix[:, randperm]
        self.matrix.data.copy_(init_matrix)
    def forward(self, x):
        #x: bxcxhxw
        b,c,h,w = x.shape 
        output = x.transpose(1, -1) # bxhxwxc
        output = torch.matmul(output, self.matrix) #bxhxwxc
        log_det = h*w*self.matrix.det().abs().log().repeat(b)
        return output.transpose(1, -1), log_det
    def reverse(self, z):
        output = z.transpose(1, -1)
        output = torch.matmul(output, self.matrix.inverse())
        return output.transpose(1, -1)

class GlowBlock(nn.Module):
    def __init__(self, n_channel):
        super().__init__()
        self.step = nn.ModuleList([
            ActNorm(n_channel = n_channel),
            Invertible1to1Conv(n_channel = n_channel),
            ImageAffineCouplingLayer(n_channel = n_channel),
        ])
    def forward(self, x):
        b,c,h,w = x.shape
        output, log_det = x, 0 
        for layer in self.step:
            output, log_det_ = layer(output)
            log_det += log_det_
        return output, log_det
    def reverse(self, z):
        output = z
        for layer in self.step[::-1]:
            output = layer.reverse(output)
        return output
class GlowLevel(nn.Module):
    def __init__(self, n_channel, n_flow, split = True):
        super().__init__()
        self.n_channel, self.n_flow, self.split = n_channel, n_flow, split
        self.step = nn.ModuleList([GlowBlock(n_channel = n_channel* 4) for _ in range(n_flow)])
    def forward(self, x):
        b, c, h, w = x.shape
        c_out, h_out, w_out = c*4, h//2, w//2
        output = x.view(b,c,h_out,2,w_out,2).permute(0,1,3,5,2,4).reshape(b, c_out, h_out, w_out)
        log_det = 0
        for layer in self.step:
            output, log_det_ = layer(output)
            log_det += log_det_
        if self.split:
            z_new, output = output.chunk(2 , dim = 1)
            return (z_new, output), log_det
        else:
            return output, log_det
    def reverse(self, z):
        output = None
        if self.split:
            z1, z2 = z
            output = torch.cat([z1,z2], dim = 1)
        else:
            output = z
        b, c, h, w = output.shape
        for layer in self.step[::-1]:
            output = layer.reverse(output)
        output = output.view(b, c//4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
        output = output.reshape(b , c//4, h*2, w*2)
        return output

class Glow(nn.Module):
    def __init__(self, n_channel, n_flow, n_level):
        super().__init__()
        self.n_level, self.n_flow, self.n_channel = n_level, n_flow, n_channel
        self.blocks = nn.ModuleList([GlowLevel(n_channel = self.n_channel *(2**idx), 
                                               split = idx!= self.n_level-1, n_flow = n_flow) for idx in range(self.n_level)])
        n_outchannel = n_channel *(2** n_level)
    def forward(self, x):
        b,c,h,w = x.shape
        hidden, z_arr, log_det = x, [], 0
        for layer in self.blocks[:-1]:
            (z, hidden), log_det_= layer(hidden)
            z_arr.append(z)
            log_det += log_det_
        z, log_det_ = self.blocks[-1](hidden)
        log_det += log_det_
        z_arr.append(z)
        return z_arr, log_det
    def reverse(self, z):
        hidden = self.blocks[-1].reverse(z[-1])
        for idx in range(2, self.n_level+1):
            hidden = self.blocks[-idx].reverse((z[-idx], hidden))
        return hidden