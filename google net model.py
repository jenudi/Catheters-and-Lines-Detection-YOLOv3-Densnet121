from preprocess import *
import torch.nn as nn
import torch.nn.functional as F
import math
#import torch.cuda


#%%
class Inception(nn.Module):
    def __init__(self, input_dim, filter1x1, filter3x3r, filter3x3, filter5x5r, filter5x5, filter_pool):
        super(Inception, self).__init__()

        self.input_dim = input_dim
        self.filter1x1 = filter1x1
        self.filter3x3r = filter3x3r
        self.filter3x3 = filter3x3
        self.filter5x5r = filter5x5r
        self.filter5x5 = filter5x5
        self.filter_pool = filter_pool
        self.conv1x1 = nn.Conv2d(self.input_dim, self.filter1x1, kernel_size=1, padding=0, stride=1, bias=True)
        self.pre_conv3x3 = nn.Conv2d(self.input_dim, self.filter3x3r, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv3x3 = nn.Conv2d(self.filter3x3r, self.filter3x3, kernel_size=3, padding=1, stride=1, bias=True)
        self.pre_conv5x5 = nn.Conv2d(self.input_dim, self.filter5x5r, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5x5 = nn.Conv2d(self.filter5x5r, self.filter5x5, kernel_size=5, padding=2, stride=1, bias=True)
        self.pool_pro_conv = nn.Conv2d(self.input_dim, self.filter_pool, kernel_size=1, stride=1, padding=0, bias=True)
        self.pool_pro = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        output1x1 = F.relu(self.conv1x1(input))
        output3x3 = F.relu(self.pre_conv3x3(input))
        output3x3 = F.relu(self.conv3x3(output3x3))
        output5x5 = F.relu(self.pre_conv5x5(input))
        output5x5 = F.relu(self.conv5x5(output5x5))
        output_pool = F.relu(self.pool_pro_conv(self.pool_pro(input)))
        output = torch.cat((output1x1, output3x3, output5x5, output_pool), 1)
        return output


class ProModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super(ProModel, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,
                                     stride=2)

        self.conv1 = nn.Conv2d(1, 64,
                               kernel_size=5,
                               padding=1,
                               stride=2,
                               bias=True)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               bias=True)

        self.conv3 = nn.Conv2d(64, 192,
                               kernel_size=3,
                               padding=0,
                               stride=1,
                               bias=True)
        self.batchnorm2 = nn.BatchNorm2d(192)

        self.avg = nn.AvgPool2d(kernel_size=7,
                                stride=1)

        self.ins1 = Inception(192, 64, 96, 128, 16, 32, 32)  # 42
        self.ins2 = Inception(256, 128, 128, 192, 32, 96, 64)  # 42

        self.ins3 = Inception(480, 192, 96, 208, 16, 48, 64)  # 21
        self.ins4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.ins5 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.ins6 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.ins7 = Inception(528, 256, 160, 320, 32, 128, 128)  # 21

        self.ins8 = Inception(832, 256, 160, 320, 32, 128, 128)  # 10
        self.ins9 = Inception(832, 384, 192, 384, 48, 128, 128)  # 10

        self.head_linear = nn.Linear(9216, 4)
        # self.head_linear2 = nn.Linear(9216, 4)
        self.head_softmax = nn.Softmax(dim=1)
        self.flat = nn.Flatten()
        self._init_weights()

    def forward(self, input):
        # 349
        x = F.relu(self.conv1(input))  # 175
        x = self.maxpool(x)  # 88
        x = self.batchnorm1(x)

        x = F.relu(self.conv2(x))  # 88
        x = F.relu(self.conv3(x))  # 86
        x = self.batchnorm2(x)
        x = self.maxpool(x)  # 42

        x = self.ins1(x)  # 42
        x = self.ins2(x)  # 42
        x = self.maxpool2(x)  # 21

        x = self.ins3(x)  # 21
        x = self.ins4(x)
        x = self.ins5(x)
        x = self.ins6(x)
        x = self.ins7(x)  # 21
        x = self.maxpool(x)  # 10

        x = self.ins8(x)  # 10
        x = self.ins9(x)  # 10
        x = self.avg(x)  # 4
        # dim 1024
        conv_flat = self.flat(x)

        linear_output = self.head_linear(conv_flat)
        # linear_output = self.relu(linear_output)
        # linear_output = self.head_linear2(linear_output)

        return linear_output, self.head_softmax(linear_output)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias != None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

