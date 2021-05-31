import torch.nn as nn
import torch

def weightInit(module:nn.Module):
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer,nn.ConvTranspose2d):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
    # elif type(module) == nn.BatchNorm2d:
    #     nn.init.normal_(module.weight.data, 1.0, 0.02)
    #     nn.init.constant_(module.bias.data, 0)

class Generator(nn.Module):
    def __init__(self,n_channel = 128):
        super(Generator, self).__init__()
        '''
        input: Bx100x1x1
        '''
        self.G= nn.Sequential(
            nn.ConvTranspose2d(100, n_channel*8, 4, bias= False), # 1024x4x4
            nn.BatchNorm2d(n_channel*8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channel*8, n_channel*4, kernel_size = 4, stride= 2,padding= 1,bias = False), #512x8x8,(4-1)x2 + 4 - 2 = 8
            nn.BatchNorm2d(n_channel*4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channel*4,n_channel*2, 4, stride= 2, padding= 1,bias = False), #256x16x16, (4-1)x2+ 4-2 = 16
            nn.BatchNorm2d(n_channel*2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channel*2,n_channel, 4, stride= 2, padding= 1,bias = False), #128x32x32
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channel,3, 4, stride= 2, padding= 1, bias= False), #3x64x64
            nn.Tanh()
        )
        # weightInit(self.G)

    def forward(self, noise):
        return self.G(noise)

class Discriminator(nn.Module):
    def __init__(self, loss_type, n_channel = 128):
        super(Discriminator, self).__init__()
        self.loss_type = loss_type
        self.D = nn.Sequential(
            nn.Conv2d(3,n_channel,4,stride=2, padding= 1, bias= False), #128x32x32 (64-4+2)/2+1 = 32
            nn.LeakyReLU(negative_slope= 0.2),
            nn.Conv2d(n_channel,n_channel*2,4,stride=2, padding= 1, bias=False), #256x16x16
            nn.BatchNorm2d(n_channel*2),
            nn.LeakyReLU(negative_slope= 0.2),
            nn.Conv2d(n_channel*2,n_channel*4,4,stride=2, padding= 1,bias=False), #512x8x8
            nn.BatchNorm2d(n_channel*4),
            nn.LeakyReLU(negative_slope= 0.2),
            nn.Conv2d(n_channel*4,n_channel*8,4,stride=2, padding= 1,bias= False), #1024x4x4
            nn.BatchNorm2d(n_channel*8),
            nn.LeakyReLU(negative_slope= 0.2),
            nn.Conv2d(n_channel*8, 1, 4, bias = False), # 1x1x1
            # nn.Sigmoid()
        )
        # weightInit(self.D)
        # self.loss = nn.BCELoss()
    def forward(self, inputs):
        return self.D(inputs)
    # def criterion(self, pred, label):
    #     if self.loss_type =="gan":
    #         S = nn.Sigmoid()
    #         return self.loss(S(pred),label)
    #     elif self.loss_type=="lsgan":
    #         return torch.mean((pred-label)**2)
