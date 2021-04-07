import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act = True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self,x):
        return self.leaky(self.bn(self.conv(x))) if self.use_bn_act else self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual= True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [nn.Sequential(CNNBlock(channels, channels//2, kernel_size=1),CNNBlock(channels // 2, channels,kernel_size=3,padding=1))]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self,x):
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(CNNBlock(in_channels, 2*in_channels, kernel_size=3,padding=1),CNNBlock(2 * in_channels, 3*(num_classes+5),bn_act=False,kernel_size=1))
        self.num_classes = num_classes

    def forward(self,x):
        return (self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2))


class YOLOv3(nn.Module): # img size = 416
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = 32
        self.layers = nn.ModuleList(
            CNNBlock(self.in_channels,self.out_channels, 3, 1, 1),
            CNNBlock(self.out_channels,self.out_channels * 2, 3, 2, 1),
            ResidualBlock(self.out_channels * 2, 1),
            CNNBlock(self.out_channels * 2,self.out_channels * 4, 3, 2, 1),
            ResidualBlock(self.out_channels * 4, 2),
            ResidualBlock(self.out_channels * 8, 8),
            CNNBlock(self.out_channels * 8,self.out_channels * 16, 3,  2, 1),
            ResidualBlock(self.out_channels * 16, 8),
            CNNBlock(self.out_channels * 16,self.out_channels * 32, 3, 2, 1),
            ResidualBlock(self.out_channels * 32, 4),
            CNNBlock(self.out_channels * 32,self.out_channels * 16, 1, 1, 0),
            ResidualBlock(self.out_channels * 32, False, 1),
            CNNBlock(self.out_channels * 32, self.out_channels * 16, 1),
            ScalePrediction(self.out_channels * 16, self.num_classes),
            CNNBlock(self.out_channels * 16,self.out_channels * 8, 1, 1, 0),
            nn.Upsample(scale_factor=2),
            CNNBlock(self.out_channels * 24,self.out_channels * 8, 1, 1, 0),
            CNNBlock(self.out_channels * 8,self.out_channels * 16, 3, 1, 1),
            ResidualBlock(self.out_channels * 16, False, 1),
            CNNBlock(self.out_channels * 16, self.out_channels * 8, 1),
            ScalePrediction(self.out_channels * 8, self.num_classes),
            CNNBlock(self.out_channels * 8,self.out_channels * 4, 1, 1, 0),
            nn.Upsample(scale_factor=2),
            CNNBlock(self.out_channels * 12,self.out_channels * 4, 1, 1, 0),
            CNNBlock(self.out_channels * 4,self.out_channels * 8, 3, 1, 1),
            ResidualBlock(self.out_channels * 8, False, 1),
            CNNBlock(self.out_channels * 8, self.out_channels * 4, 1),
            ScalePrediction(self.out_channels * 4, self.num_classes),
            )

    def forward(self,x):
        outputs, route_connections = [], []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs


class YOLOLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.mse = nn.MSELoss()
    self.bce = nn.BCEWithLogitsLoss()
    self.sigmoid = nn.Sigmoid()

  def forward(self, predictions, target, anchors):
    obj = target[..., 0] == 1
    noobj = target[..., 0] == 0
    no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))
    anchors = anchors.reshape(1, 3, 1, 1, 2)
    box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
    object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])
    predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
    target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
    box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
    class_loss = self.bce((predictions[..., 5:][obj]), (target[..., 5][obj].long()))
    return (box_loss + object_loss + no_object_loss + class_loss)
