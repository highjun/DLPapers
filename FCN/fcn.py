import torch.nn as nn
import torch
from torchvision.models import vgg16_bn

import numpy as np

def init_upsample_layer(channel, factor = 2):
    out = nn.ConvTranspose2d(channel, channel, kernel_size = 2*factor, stride = factor, padding = factor //2, bias = False)
    kernel = 2* factor
    center = factor - 0.5
    og = np.ogrid[:kernel, :kernel]
    filt = (1 - abs(og[0] - center) / factor) * \
          (1 - abs(og[1] - center) / factor)
    weight = np.zeros((channel, channel, kernel, kernel),
                    dtype=np.float64)
    weight[range(channel), range(channel), :, :] = filt
    out.weight.data.copy_(torch.from_numpy(weight).float())
    return out

class FCN(nn.Module):
    def __init__(self, upsample = "8", class_num = 3):
        '''
        upsample: String 8, 16, 32
        class_num
        '''
        super(FCN, self).__init__()
        self.upsample = upsample
        self.class_num = class_num
        pretrained = vgg16_bn(pretrained = True)
        block = []
        idx = 1
        for name,layer in pretrained.features.named_modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer,nn.ReLU) or isinstance(layer, nn.Conv2d):
                block.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                block.append(layer)
                self.__setattr__(f"conv{idx}", nn.Sequential(*block))
                block =[]
                idx += 1
        original = self.conv1[0]
        self.conv1[0] = nn.Conv2d(3, 64,3, padding= 100) #32k+ 200 - 2= 32k+ 198 
        self.conv1[0].weight.data.copy_(original.weight.data)
        # 32k+198 - 16k+94, 8k+47, 4k+23, 2k+11, k+6 -> k 
        # 256x256x3
        self.fcn = nn.Sequential(
            nn.Conv2d(512,4096, 7,bias= False), # 14-7+1 = 8
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(4096,4096, 1,bias= False),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(4096, class_num,1,bias = False),
        )
        if upsample =="32":
            self.upsample1 = init_upsample_layer( class_num, factor = 32)
        elif upsample =="16":
            self.upsample1 = init_upsample_layer(class_num, factor = 2)
            self.one_conv1 = nn.Conv2d(512,class_num,kernel_size=1,padding=1, bias=False)#14+2
            nn.init.zeros_(self.one_conv1.weight)
            self.upsample2 = init_upsample_layer(class_num, factor = 16)
        elif upsample =="8":
            self.upsample1 = init_upsample_layer( class_num, factor = 2)
            self.one_conv1 = nn.Conv2d(512,class_num,kernel_size=1, padding=1,bias = False)#14+2
            nn.init.constant_(self.one_conv1.weight, 0)
            self.upsample2 = init_upsample_layer(class_num, factor = 2)
            self.one_conv2 = nn.Conv2d(512,class_num,kernel_size=1, padding=2,bias= False)#28+4
            nn.init.constant_(self.one_conv2.weight, 0)
            self.upsample3 = init_upsample_layer(class_num, factor = 8)
        # self.criterion = nn.CrossEntropyLoss()
    def forward(self, input):
        feature1 = self.conv1(input)
        feature2 = self.conv2(feature1)
        feature3 = self.conv3(feature2)
        feature4 = self.conv4(feature3)
        feature5 = self.conv5(feature4)
        out = self.fcn(feature5)
        # print(feature1.shape, feature2.shape, feature3.shape, feature4.shape, feature5.shape)
        # print(self.one_conv1(feature5).shape)
        if self.upsample =="32":
            output = self.upsample1(out)
        elif self.upsample == "16":
            output = self.upsample1(out)
            # print(self.one_conv1(feature5).shape)
            # print(output.shape, feature4.shape)
            output = output + self.one_conv1(feature5)
            output = self.upsample2(output)
        elif self.upsample == "8":
            output = self.upsample1(out)
            output = output + self.one_conv1(feature5)
            output = self.upsample2(output)
            output = output + self.one_conv2(feature4)
            output = self.upsample3(output)
        return output
