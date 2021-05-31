import torch.nn as nn
import torch

class Image2Image(nn.Module):
    def __init__(self):
        super(Image2Image,self).__init__()
        self.translate = nn.Sequential(
            nn.ReplicationPad2d(3),
            nn.Conv2d(3,64, 7, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128, 3, 2, padding= 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256, 3,2,padding= 1 ),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            nn.ConvTranspose2d(256,128,3, 2, 1,1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,3, 2, 1,1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ReplicationPad2d(3),
            nn.Conv2d(64,3, 7, 1),
            nn.Tanh()
        )
    def forward(self, inputs):
        return self.translate(inputs)
class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3,64, 4, 2, 1,bias= False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,4,2,1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,4,1,1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, padding= 1, bias= False)
        )
    def forward(self, inputs):
        return self.feature(inputs)

class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super(ResnetBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(dim, dim, 3, stride = 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(dim,dim,3),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x: torch.Tensor):
        out =  self.conv_block(x) + x
        return out