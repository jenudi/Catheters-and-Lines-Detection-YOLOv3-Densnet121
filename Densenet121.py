import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Densenet(nn.Module):
    def __init__(self):
        super(Densenet,self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.dropout1 = nn.Dropout2d(p=0.3)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.head_linear = nn.Linear(1024,100,bias=True)
        self.head_linear2 = nn.Linear(100,4,bias=True)
        self.head_softmax = nn.Softmax(dim=1)

        self.change_model()

    def forward(self, input):

        x = self.model(input)
        x = self.dropout1(x)
        x = F.relu(self.head_linear(x))
        x = self.dropout2(x)
        linear_output = self.head_linear2(x)

        return linear_output, self.head_softmax(linear_output)

    def change_model(self):
      self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
      for index,feature in enumerate(self.model.children()):
        if index == 1:
          break
        for i,v in enumerate(feature):
          if i == 10:
            break
          for param in v.parameters():
            param.requires_grad = False


def clac_param(model):
  print(f"total parameters: {sum(p.numel() for p in model.parameters())}")
  print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Densenet(nn.Module):
    def __init__(self):
        super(Densenet,self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.dropout1 = nn.Dropout2d(p=0.3)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.head_linear = nn.Linear(1024,100,bias=True)
        self.head_linear2 = nn.Linear(100,4,bias=True)
        self.head_softmax = nn.Softmax(dim=1)

        self.change_model()

    def forward(self, input):

        x = self.model(input)
        x = self.dropout1(x)
        x = F.relu(self.head_linear(x))
        x = self.dropout2(x)
        linear_output = self.head_linear2(x)

        return linear_output, self.head_softmax(linear_output)

    def change_model(self):
      self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
      for index,feature in enumerate(self.model.children()):
        if index == 1:
          break
        for i,v in enumerate(feature):
          if i == 10:
            break
          for param in v.parameters():
            param.requires_grad = False


def clac_param(model):
  print(f"total parameters: {sum(p.numel() for p in model.parameters())}")
  print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
