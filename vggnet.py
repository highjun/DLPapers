import torch.nn as nn

class ConvBlock2(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ConvBlock2, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.main = nn.Sequential(nn.Conv2d(self.in_dim, self.out_dim, kernel_size=3, padding=1),                    
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_dim,self.out_dim, kernel_size=3, padding=1),                    
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),)
    def forward(self, x):
        out = self.main(x)
        return out
class ConvBlock4(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvBlock4, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.main = nn.Sequential(nn.Conv2d(self.in_dim, self.out_dim, kernel_size=3, padding=1),                    
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1),                    
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1),                    
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1),                    
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2) 
                                 )
        
    def forward(self, x):
        out = self.main(x)
        return out
class ConvBlock3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvBlock3, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.main = nn.Sequential(nn.Conv2d(self.in_dim, self.out_dim, kernel_size=3, padding=1),                    
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1),                    
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1),                    
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2) 
                                 )
        
    def forward(self, x):
        out = self.main(x)
        return out

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        
        self.convlayer1 = ConvBlock2(3, 64) # 64x32x32
        self.convlayer2 = ConvBlock2(64, 128) # 128x16x16
        self.convlayer3 = ConvBlock4(128, 256) # 256x8x8
        self.convlayer4 = ConvBlock4(256, 512) # 512x4x4
        self.convlayer5 = ConvBlock4(512, 512) # 512x2x2
        self.features = nn.Sequential(
            self.convlayer1,
            self.convlayer2,
            self.convlayer3,
            self.convlayer4,
            self.convlayer5
        )
        self.linear = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, input):
        out = self.features(input)
        out = out.view(-1,2048)
        out = self.linear(out)
        return out
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        
        self.convlayer1 = ConvBlock2(3, 64) # 64x32x32
        self.convlayer2 = ConvBlock2(64, 128) # 128x16x16
        self.convlayer3 = ConvBlock3(128, 256) # 256x8x8
        self.convlayer4 = ConvBlock3(256, 512) # 512x4x4
        self.convlayer5 = ConvBlock3(512, 512) # 512x2x2
        self.features = nn.Sequential(
            self.convlayer1,
            self.convlayer2,
            self.convlayer3,
            self.convlayer4,
            self.convlayer5
        )
        self.linear = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, input):
        out = self.features(input)
        out = out.view(-1,2048)
        out = self.linear(out)
        return out