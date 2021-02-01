# Imports
from preprocess import *
import torch.nn as nn
import torch.nn.functional as F
import math
#import torch.cuda


#%%


class Residual(nn.Module):
    def __init__(self, kernel_size, filters, reduce=False,s=2):
        super(Residual, self).__init__()

        self.reduce = reduce
        self.kernel_size = kernel_size
        self.s = s
        self.f0 = filters[0]
        self.f1 = filters[1]
        self.f2 = filters[2]
        self.f3 = filters[3]
        self.conv1 = nn.Conv2d(self.f0,
                               self.f3,1,
                               stride=self.s)
        self.batch1 = nn.BatchNorm2d(self.f3)
        self.conv2 = nn.Conv2d(self.f0,
                               self.f1,1,
                               stride=self.s)
        self.batch2 = nn.BatchNorm2d(self.f1)
        self.conv3 = nn.Conv2d(self.f0,
                               self.f1,
                               1,
                               stride=1)
        self.batch3 = nn.BatchNorm2d(self.f1)
        self.conv4 = nn.Conv2d(self.f1,
                               self.f2,
                               self.kernel_size,
                               stride=1,
                               padding=1)
        self.batch4 = nn.BatchNorm2d(self.f2)

        self.conv5 = nn.Conv2d(self.f2,
                               self.f3,
                               1,
                               stride=1)
        self.batch5 = nn.BatchNorm2d(self.f3)


    def forward(self, input):
        x_shortcut = input
        if self.reduce:
            x_shortcut = self.conv1(x_shortcut)
            x_shortcut = self.batch1(x_shortcut)
            x = self.conv2(input)
            x = F.relu(self.batch2(x))
        else:
            x = self.conv3(input)
            x = F.relu(self.batch3(x))
        x = self.conv4(x)
        x = F.relu(self.batch4(x))
        x = self.conv5(x)
        x = F.relu(self.batch5(x))
        x = F.relu(torch.add(x,x_shortcut))
        return x


class ProModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super(ProModel,self).__init__()

        self.conv1 = nn.Conv2d(1,64, 7,2)
        self.batch1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3,2)

        self.residual1 = Residual(3, [64,64,64,256] ,reduce=True,s=1)
        self.residual2 = Residual(3, [256,64, 64, 256])

        self.residual3 = Residual(3, [256,128, 128, 512], reduce=True)
        self.residual4 = Residual(3, [512,128, 128, 512])

        self.residual5 = Residual(3, [512,256, 256, 1024], reduce=True)
        self.residual6 = Residual(3, [1024,256, 256, 1024])

        self.residual7 = Residual(3, [1024,512, 512, 2048], reduce=True)
        self.residual8 = Residual(3, [2048,512, 512, 2048])

        self.avg = nn.AvgPool2d(1)

        self.head_linear = nn.Linear(100352, 4)
        #self.head_linear2 = nn.Linear(9216, 4)
        self.head_softmax = nn.Softmax(dim=1)

    def forward(self, input):
        #347
        x = self.conv1(input)
        x = F.relu(self.batch1(x))#171
        x = self.pool1(x)#85

        x = self.residual1(x)#85
        x = self.residual2(x)#85
        x = self.residual2(x)#85

        x = self.residual3(x)#43
        x = self.residual4(x)#43
        x = self.residual4(x)#43
        x = self.residual4(x)#43

        x = self.residual5(x)#22
        x = self.residual6(x)#22
        x = self.residual6(x)
        x = self.residual6(x)
        x = self.residual6(x)
        x = self.residual6(x)#22

        x = self.residual7(x)
        x = self.residual8(x)
        x = self.residual8(x)

        x = self.avg(x)

        conv_flat = x.view(-1, 100352)
        #conv_flat = output.view(output.size(0), -1)

        linear_output = self.head_linear(conv_flat)
        #linear_output = self.relu(linear_output)
        #linear_output = self.head_linear2(linear_output)

        return linear_output, self.head_softmax(linear_output)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias != None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


