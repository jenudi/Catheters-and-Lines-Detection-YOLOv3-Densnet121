from util.unet import UNet
import torch.nn as nn
import math

#UNet
class Unet(nn.Module):
    def __init__(self, **kwargs):
        super(Unet, self).__init__()

        self.batch = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = Unet(**kwargs)
        self.final = nn.Sigmoid()
        self._init_weights()

    def forward(self, input):

        x = self.batch(input)
        x = self.unet(x)
        x = self.final(x)

        return x

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)