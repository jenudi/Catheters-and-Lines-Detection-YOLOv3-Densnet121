import torch.nn as nn
import torch.nn.functional as F
import math


#Base Net
class ProBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super(ProBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               conv_channels,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_channels,
                               conv_channels,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))

        return self.maxpool(x)


class ProModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super(ProModel,self).__init__()

        self.tail_batchnorm = nn.BatchNorm2d(1)
        self.block1 = ProBlock(in_channels,
                               conv_channels)
        self.block2 = ProBlock(conv_channels,
                               conv_channels * 2)
        self.block3 = ProBlock(conv_channels * 2,
                               conv_channels * 4)
        self.block4 = ProBlock(conv_channels * 4,
                               conv_channels * 8)

        self.flat = nn.Flatten()
        self.head_linear = nn.Linear(25600, 4)
        self.head_softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, input):

        x = self.tail_batchnorm(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        conv_flat = self.flat(x)
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear,
                           nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias != None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
